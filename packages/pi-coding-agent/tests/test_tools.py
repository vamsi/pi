"""Tests for coding agent tools."""

import tempfile
from pathlib import Path

import pytest

from pi.coding.core.tools.bash import execute_bash
from pi.coding.core.tools.edit import execute_edit
from pi.coding.core.tools.find import execute_find
from pi.coding.core.tools.ls import execute_ls
from pi.coding.core.tools.read import execute_read
from pi.coding.core.tools.write import execute_write


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.mark.asyncio
async def test_write_and_read(tmp_dir):
    # Write a file
    result = await execute_write("tc1", {"path": "test.txt", "content": "hello\nworld"}, cwd=tmp_dir)
    assert "Successfully wrote" in result.content[0].text

    # Read it back
    result = await execute_read("tc2", {"path": "test.txt"}, cwd=tmp_dir)
    assert "hello" in result.content[0].text
    assert "world" in result.content[0].text


@pytest.mark.asyncio
async def test_write_creates_directories(tmp_dir):
    result = await execute_write(
        "tc1",
        {"path": "a/b/c/test.txt", "content": "nested"},
        cwd=tmp_dir,
    )
    assert "Successfully wrote" in result.content[0].text
    assert (Path(tmp_dir) / "a" / "b" / "c" / "test.txt").exists()


@pytest.mark.asyncio
async def test_edit(tmp_dir):
    # Create file
    path = Path(tmp_dir) / "edit_test.txt"
    path.write_text("line 1\nline 2\nline 3\n")

    # Edit it
    await execute_edit(
        "tc1",
        {"path": "edit_test.txt", "old_text": "line 2", "new_text": "modified line 2"},
        cwd=tmp_dir,
    )

    content = path.read_text()
    assert "modified line 2" in content
    assert "line 2" not in content or "modified line 2" in content


@pytest.mark.asyncio
async def test_edit_not_found(tmp_dir):
    path = Path(tmp_dir) / "edit_test2.txt"
    path.write_text("hello world")

    with pytest.raises(ValueError, match="Could not find"):
        await execute_edit(
            "tc1",
            {"path": "edit_test2.txt", "old_text": "nonexistent", "new_text": "replacement"},
            cwd=tmp_dir,
        )


@pytest.mark.asyncio
async def test_bash(tmp_dir):
    result = await execute_bash("tc1", {"command": "echo 'hello from bash'"}, cwd=tmp_dir)
    assert "hello from bash" in result.content[0].text


@pytest.mark.asyncio
async def test_bash_exit_code(tmp_dir):
    result = await execute_bash("tc1", {"command": "exit 42"}, cwd=tmp_dir)
    assert "Exit code: 42" in result.content[0].text


@pytest.mark.asyncio
async def test_ls(tmp_dir):
    # Create some files and dirs
    (Path(tmp_dir) / "file1.txt").write_text("a")
    (Path(tmp_dir) / "file2.py").write_text("b")
    (Path(tmp_dir) / "subdir").mkdir()

    result = await execute_ls("tc1", {"path": "."}, cwd=tmp_dir)
    text = result.content[0].text
    assert "file1.txt" in text
    assert "file2.py" in text
    assert "subdir/" in text


@pytest.mark.asyncio
async def test_find(tmp_dir):
    (Path(tmp_dir) / "a.py").write_text("a")
    (Path(tmp_dir) / "b.txt").write_text("b")
    (Path(tmp_dir) / "sub").mkdir()
    (Path(tmp_dir) / "sub" / "c.py").write_text("c")

    result = await execute_find("tc1", {"pattern": "*.py"}, cwd=tmp_dir)
    text = result.content[0].text
    assert "a.py" in text


@pytest.mark.asyncio
async def test_read_nonexistent(tmp_dir):
    with pytest.raises(FileNotFoundError):
        await execute_read("tc1", {"path": "nonexistent.txt"}, cwd=tmp_dir)


@pytest.mark.asyncio
async def test_read_with_offset(tmp_dir):
    path = Path(tmp_dir) / "lines.txt"
    path.write_text("\n".join(f"line {i}" for i in range(1, 11)))

    result = await execute_read("tc1", {"path": "lines.txt", "offset": 5, "limit": 3}, cwd=tmp_dir)
    text = result.content[0].text
    assert "line 5" in text
    assert "line 7" in text
