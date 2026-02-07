"""Executor abstraction: host and Docker execution."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SandboxConfig:
    type: str  # "host" or "docker"
    container: str | None = None


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    code: int


class Executor(Protocol):
    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> ExecResult: ...

    def get_workspace_path(self, host_path: str) -> str: ...


class HostExecutor:
    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        MAX_OUTPUT = 10 * 1024 * 1024  # 10 MB

        async def _read_stream(
            stream: asyncio.StreamReader | None,
        ) -> str:
            if stream is None:
                return ""
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total <= MAX_OUTPUT:
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace")

        async def _wait_with_abort() -> tuple[str, str]:
            stdout_task = asyncio.create_task(_read_stream(proc.stdout))
            stderr_task = asyncio.create_task(_read_stream(proc.stderr))

            tasks: list[asyncio.Task[str]] = [stdout_task, stderr_task]

            # Also wait for the process to complete
            wait_task = asyncio.create_task(proc.wait())

            abort_task: asyncio.Task[None] | None = None
            if abort_event is not None:
                abort_task = asyncio.create_task(abort_event.wait())  # type: ignore[arg-type]

            try:
                done: set[asyncio.Task[Any]] = set()
                pending: set[asyncio.Task[Any]] = set()
                all_futures: set[asyncio.Task[Any]] = {wait_task}
                if abort_task is not None:
                    all_futures.add(abort_task)

                done, pending = await asyncio.wait(
                    all_futures,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                timed_out = not done  # nothing completed → timeout
                aborted = abort_task is not None and abort_task in done

                if timed_out or aborted:
                    _kill_process_tree(proc.pid)
                    await proc.wait()
                    stdout = await stdout_task
                    stderr = await stderr_task
                    if aborted:
                        raise RuntimeError(
                            f"{stdout}\n{stderr}\nCommand aborted".strip()
                        )
                    raise RuntimeError(
                        f"{stdout}\n{stderr}\nCommand timed out after {timeout} seconds".strip()
                    )

                stdout = await stdout_task
                stderr = await stderr_task
                return stdout, stderr
            finally:
                if abort_task is not None and not abort_task.done():
                    abort_task.cancel()
                if not wait_task.done():
                    wait_task.cancel()

        stdout, stderr = await _wait_with_abort()
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            code=proc.returncode if proc.returncode is not None else 0,
        )

    def get_workspace_path(self, host_path: str) -> str:
        return host_path


class DockerExecutor:
    def __init__(self, container: str) -> None:
        self._container = container
        self._host = HostExecutor()

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> ExecResult:
        docker_cmd = f"docker exec {self._container} sh -c {_shell_escape(command)}"
        return await self._host.exec(
            docker_cmd, timeout=timeout, abort_event=abort_event
        )

    def get_workspace_path(self, _host_path: str) -> str:
        return "/workspace"


# ── Helpers ──────────────────────────────────────────────────────────


def _kill_process_tree(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


async def _exec_simple(cmd: str, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        cmd,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace") or f"Exit code {proc.returncode}")
    return stdout.decode("utf-8", errors="replace")


def parse_sandbox_arg(value: str) -> SandboxConfig:
    if value == "host":
        return SandboxConfig(type="host")
    if value.startswith("docker:"):
        container = value[len("docker:"):]
        if not container:
            print(
                "Error: docker sandbox requires container name "
                "(e.g., docker:mom-sandbox)",
                file=sys.stderr,
            )
            sys.exit(1)
        return SandboxConfig(type="docker", container=container)
    print(
        f"Error: Invalid sandbox type '{value}'. "
        "Use 'host' or 'docker:<container-name>'",
        file=sys.stderr,
    )
    sys.exit(1)


async def validate_sandbox(config: SandboxConfig) -> None:
    if config.type == "host":
        return

    try:
        await _exec_simple("docker", "--version")
    except Exception:
        print("Error: Docker is not installed or not in PATH", file=sys.stderr)
        sys.exit(1)

    try:
        result = await _exec_simple(
            "docker", "inspect", "-f", "{{.State.Running}}", config.container or ""
        )
        if result.strip() != "true":
            print(
                f"Error: Container '{config.container}' is not running.",
                file=sys.stderr,
            )
            print(
                f"Start it with: docker start {config.container}",
                file=sys.stderr,
            )
            sys.exit(1)
    except RuntimeError:
        print(
            f"Error: Container '{config.container}' does not exist.",
            file=sys.stderr,
        )
        print(
            "Create it with: ./docker.sh create <data-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Docker container '{config.container}' is running.")


def create_executor(config: SandboxConfig) -> Executor:
    if config.type == "host":
        return HostExecutor()
    assert config.container is not None
    return DockerExecutor(config.container)
