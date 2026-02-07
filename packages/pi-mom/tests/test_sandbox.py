"""Tests for pi.mom.sandbox."""

import asyncio
import sys

import pytest

from pi.mom.sandbox import (
    DockerExecutor,
    HostExecutor,
    SandboxConfig,
    create_executor,
    parse_sandbox_arg,
)


class TestParseSandboxArg:
    def test_host(self) -> None:
        config = parse_sandbox_arg("host")
        assert config.type == "host"
        assert config.container is None

    def test_docker(self) -> None:
        config = parse_sandbox_arg("docker:my-container")
        assert config.type == "docker"
        assert config.container == "my-container"

    def test_docker_empty_container_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_sandbox_arg("docker:")

    def test_invalid_type_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_sandbox_arg("invalid")


class TestCreateExecutor:
    def test_creates_host_executor(self) -> None:
        config = SandboxConfig(type="host")
        executor = create_executor(config)
        assert isinstance(executor, HostExecutor)

    def test_creates_docker_executor(self) -> None:
        config = SandboxConfig(type="docker", container="test")
        executor = create_executor(config)
        assert isinstance(executor, DockerExecutor)


class TestHostExecutor:
    @pytest.mark.asyncio
    async def test_simple_command(self) -> None:
        executor = HostExecutor()
        result = await executor.exec("echo hello")
        assert result.stdout.strip() == "hello"
        assert result.code == 0

    @pytest.mark.asyncio
    async def test_failing_command(self) -> None:
        executor = HostExecutor()
        result = await executor.exec("false")
        assert result.code != 0

    @pytest.mark.asyncio
    async def test_stderr(self) -> None:
        executor = HostExecutor()
        result = await executor.exec("echo err >&2")
        assert "err" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        executor = HostExecutor()
        with pytest.raises(RuntimeError, match="timed out"):
            await executor.exec("sleep 10", timeout=0.5)

    def test_workspace_path_passthrough(self) -> None:
        executor = HostExecutor()
        assert executor.get_workspace_path("/some/path") == "/some/path"


class TestDockerExecutor:
    def test_workspace_path(self) -> None:
        executor = DockerExecutor("test-container")
        assert executor.get_workspace_path("/any") == "/workspace"
