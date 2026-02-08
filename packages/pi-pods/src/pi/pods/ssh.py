"""SSH execution utilities. Shells out to the ssh/scp binaries (same approach as the TS version)."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class SSHResult:
    stdout: str
    stderr: str
    exit_code: int


async def ssh_exec(
    ssh_cmd: str,
    command: str,
    *,
    keep_alive: bool = False,
) -> SSHResult:
    """Execute an SSH command and return the result."""
    parts = shlex.split(ssh_cmd)
    ssh_binary = parts[0]
    ssh_args = list(parts[1:])

    if keep_alive:
        ssh_args = ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120", *ssh_args]

    ssh_args.append(command)

    proc = await asyncio.create_subprocess_exec(
        ssh_binary,
        *ssh_args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await proc.communicate()
    return SSHResult(
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        exit_code=proc.returncode or 0,
    )


async def ssh_exec_stream(
    ssh_cmd: str,
    command: str,
    *,
    silent: bool = False,
    force_tty: bool = False,
    keep_alive: bool = False,
) -> int:
    """Execute an SSH command with output streaming to the console. Returns exit code."""
    parts = shlex.split(ssh_cmd)
    ssh_binary = parts[0]
    ssh_args = list(parts[1:])

    if force_tty and "-t" not in parts:
        ssh_args = ["-t", *ssh_args]

    if keep_alive:
        ssh_args = ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120", *ssh_args]

    ssh_args.append(command)

    if silent:
        proc = await asyncio.create_subprocess_exec(
            ssh_binary,
            *ssh_args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            ssh_binary,
            *ssh_args,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    await proc.wait()
    return proc.returncode or 0


async def scp_file(ssh_cmd: str, local_path: str, remote_path: str) -> bool:
    """Copy a file to remote via SCP."""
    parts = shlex.split(ssh_cmd)
    host = ""
    port = "22"
    i = 1  # Skip 'ssh'

    while i < len(parts):
        if parts[i] == "-p" and i + 1 < len(parts):
            port = parts[i + 1]
            i += 2
        elif not parts[i].startswith("-"):
            host = parts[i]
            break
        else:
            i += 1

    if not host:
        print("Could not parse host from SSH command", file=sys.stderr)
        return False

    proc = await asyncio.create_subprocess_exec(
        "scp",
        "-P", port,
        local_path,
        f"{host}:{remote_path}",
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    await proc.wait()
    return proc.returncode == 0
