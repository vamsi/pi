"""Tests for the artifacts module (ArtifactStore + create_artifacts_tool)."""

from __future__ import annotations

import pytest

from pi.web.artifacts import (
    ARTIFACTS_DESCRIPTION,
    ARTIFACTS_SCHEMA,
    ArtifactStore,
    create_artifacts_tool,
)


# ---------------------------------------------------------------------------
# ArtifactStore tests
# ---------------------------------------------------------------------------


class TestArtifactStoreCreate:
    """ArtifactStore.create"""

    def test_create_basic(self) -> None:
        store = ArtifactStore()
        artifact = store.create("hello.html", "<h1>Hello</h1>", "Hello Page")
        assert artifact["filename"] == "hello.html"
        assert artifact["content"] == "<h1>Hello</h1>"
        assert artifact["title"] == "Hello Page"
        assert artifact["version"] == 1

    def test_create_stores_artifact(self) -> None:
        store = ArtifactStore()
        store.create("file.txt", "body")
        assert "file.txt" in store.artifacts
        assert store.artifacts["file.txt"]["content"] == "body"

    def test_create_title_defaults_to_filename(self) -> None:
        store = ArtifactStore()
        artifact = store.create("chart.svg", "<svg/>")
        assert artifact["title"] == "chart.svg"

    def test_create_empty_title_defaults_to_filename(self) -> None:
        store = ArtifactStore()
        artifact = store.create("chart.svg", "<svg/>", title="")
        assert artifact["title"] == "chart.svg"

    def test_create_overwrites_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "old")
        artifact = store.create("f.txt", "new", "New Title")
        assert artifact["content"] == "new"
        assert artifact["title"] == "New Title"
        assert artifact["version"] == 1  # reset because create always sets version 1

    def test_create_multiple_artifacts(self) -> None:
        store = ArtifactStore()
        store.create("a.txt", "aaa")
        store.create("b.txt", "bbb")
        assert len(store.artifacts) == 2


class TestArtifactStoreUpdate:
    """ArtifactStore.update"""

    def test_update_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "v1")
        updated = store.update("f.txt", "v2")
        assert updated is not None
        assert updated["content"] == "v2"
        assert updated["version"] == 2

    def test_update_nonexistent_returns_none(self) -> None:
        store = ArtifactStore()
        result = store.update("missing.txt", "content")
        assert result is None

    def test_update_increments_version(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "v1")
        store.update("f.txt", "v2")
        store.update("f.txt", "v3")
        artifact = store.get("f.txt")
        assert artifact is not None
        assert artifact["version"] == 3

    def test_update_changes_title_when_provided(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body", "Old Title")
        store.update("f.txt", "body2", title="New Title")
        assert store.get("f.txt")["title"] == "New Title"  # type: ignore[index]

    def test_update_preserves_title_when_none(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body", "Original")
        store.update("f.txt", "body2", title=None)
        assert store.get("f.txt")["title"] == "Original"  # type: ignore[index]

    def test_update_preserves_title_when_omitted(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body", "Original")
        store.update("f.txt", "body2")
        assert store.get("f.txt")["title"] == "Original"  # type: ignore[index]


class TestArtifactStoreDelete:
    """ArtifactStore.delete"""

    def test_delete_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        assert store.delete("f.txt") is True
        assert "f.txt" not in store.artifacts

    def test_delete_nonexistent(self) -> None:
        store = ArtifactStore()
        assert store.delete("nope.txt") is False

    def test_delete_only_removes_target(self) -> None:
        store = ArtifactStore()
        store.create("a.txt", "a")
        store.create("b.txt", "b")
        store.delete("a.txt")
        assert "a.txt" not in store.artifacts
        assert "b.txt" in store.artifacts


class TestArtifactStoreGet:
    """ArtifactStore.get / get_all"""

    def test_get_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        artifact = store.get("f.txt")
        assert artifact is not None
        assert artifact["content"] == "body"

    def test_get_nonexistent(self) -> None:
        store = ArtifactStore()
        assert store.get("missing.txt") is None

    def test_get_all_empty(self) -> None:
        store = ArtifactStore()
        assert store.get_all() == []

    def test_get_all_returns_list(self) -> None:
        store = ArtifactStore()
        store.create("a.txt", "a")
        store.create("b.txt", "b")
        all_artifacts = store.get_all()
        assert len(all_artifacts) == 2
        filenames = {a["filename"] for a in all_artifacts}
        assert filenames == {"a.txt", "b.txt"}

    def test_get_all_returns_copy(self) -> None:
        """Mutating the returned list should not affect the store."""
        store = ArtifactStore()
        store.create("f.txt", "body")
        result = store.get_all()
        result.clear()
        assert len(store.get_all()) == 1


class TestArtifactStoreOnChange:
    """ArtifactStore on_change callback."""

    def test_on_change_called_on_create(self) -> None:
        store = ArtifactStore()
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        store.create("f.txt", "body")
        assert calls == ["changed"]

    def test_on_change_called_on_update(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        store.update("f.txt", "new body")
        assert calls == ["changed"]

    def test_on_change_called_on_delete(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        store.delete("f.txt")
        assert calls == ["changed"]

    def test_on_change_not_called_on_failed_update(self) -> None:
        store = ArtifactStore()
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        store.update("missing.txt", "content")
        assert calls == []

    def test_on_change_not_called_on_failed_delete(self) -> None:
        store = ArtifactStore()
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        store.delete("missing.txt")
        assert calls == []

    def test_on_change_not_called_when_not_set(self) -> None:
        """No error when _on_change is None."""
        store = ArtifactStore()
        store.create("f.txt", "body")
        store.update("f.txt", "body2")
        store.delete("f.txt")
        # Should not raise

    def test_set_on_change_replaces_callback(self) -> None:
        store = ArtifactStore()
        first: list[str] = []
        second: list[str] = []
        store.set_on_change(lambda: first.append("1"))
        store.create("a.txt", "a")
        store.set_on_change(lambda: second.append("2"))
        store.create("b.txt", "b")
        assert first == ["1"]
        assert second == ["2"]


class TestArtifactStoreVersioning:
    """Version incrementing across operations."""

    def test_version_starts_at_1(self) -> None:
        store = ArtifactStore()
        artifact = store.create("f.txt", "body")
        assert artifact["version"] == 1

    def test_version_increments_on_each_update(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "v1")
        store.update("f.txt", "v2")
        store.update("f.txt", "v3")
        store.update("f.txt", "v4")
        assert store.get("f.txt")["version"] == 4  # type: ignore[index]

    def test_version_resets_on_recreate(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "v1")
        store.update("f.txt", "v2")
        assert store.get("f.txt")["version"] == 2  # type: ignore[index]
        store.create("f.txt", "fresh")
        assert store.get("f.txt")["version"] == 1  # type: ignore[index]


# ---------------------------------------------------------------------------
# create_artifacts_tool tests
# ---------------------------------------------------------------------------


class TestCreateArtifactsTool:
    """create_artifacts_tool returns a properly configured AgentTool."""

    def test_tool_metadata(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        assert tool.name == "artifacts"
        assert tool.label == "Artifacts"
        assert tool.description == ARTIFACTS_DESCRIPTION
        assert tool.parameters == ARTIFACTS_SCHEMA

    def test_tool_has_execute(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        assert tool.execute is not None
        assert callable(tool.execute)


class TestToolCreateAction:
    """Tool execute with action='create'."""

    async def test_create_success(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "test.html", "content": "<p>hi</p>", "title": "Test"})
        assert len(result.content) == 1
        assert result.content[0].text == "Created artifact: test.html (v1)"
        assert result.details["artifact"]["filename"] == "test.html"
        assert result.details["artifact"]["content"] == "<p>hi</p>"
        assert result.details["artifact"]["title"] == "Test"
        assert result.details["artifact"]["version"] == 1

    async def test_create_stores_in_store(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "hello"})
        assert store.get("f.txt") is not None
        assert store.get("f.txt")["content"] == "hello"  # type: ignore[index]

    async def test_create_missing_content(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt"})
        assert result.content[0].text == "Error: content is required for create"
        assert store.get("f.txt") is None

    async def test_create_empty_content(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": ""})
        assert result.content[0].text == "Error: content is required for create"

    async def test_create_no_title(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "body"})
        assert result.details["artifact"]["title"] == "f.txt"

    async def test_create_with_title(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "body", "title": "My File"})
        assert result.details["artifact"]["title"] == "My File"


class TestToolUpdateAction:
    """Tool execute with action='update'."""

    async def test_update_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "old content", "Title")
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": "new content"})
        assert "Updated artifact: f.txt (v2)" in result.content[0].text
        assert result.details["artifact"]["version"] == 2
        assert result.details["artifact"]["content"] == "new content"

    async def test_update_existing_with_title(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "old", "Old Title")
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": "new", "title": "New Title"})
        assert result.details["artifact"]["title"] == "New Title"

    async def test_update_existing_without_title_preserves(self) -> None:
        """When title arg is empty string, it becomes None via `title or None`, preserving the original."""
        store = ArtifactStore()
        store.create("f.txt", "old", "Original Title")
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": "new"})
        assert store.get("f.txt")["title"] == "Original Title"  # type: ignore[index]

    async def test_update_nonexistent_auto_creates(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "new.txt", "content": "auto"})
        assert "was not found, created new" in result.content[0].text
        assert result.details["artifact"]["version"] == 1
        assert store.get("new.txt") is not None

    async def test_update_nonexistent_auto_creates_with_title(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "new.txt", "content": "auto", "title": "Title"})
        assert result.details["artifact"]["title"] == "Title"

    async def test_update_missing_content(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "f.txt"})
        assert result.content[0].text == "Error: content is required for update"
        # Original should be untouched
        assert store.get("f.txt")["content"] == "body"  # type: ignore[index]

    async def test_update_empty_content(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": ""})
        assert result.content[0].text == "Error: content is required for update"

    async def test_update_increments_version_multiple_times(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "v1")
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": "v2"})
        result = await tool.execute("call-2", {"action": "update", "filename": "f.txt", "content": "v3"})
        assert result.details["artifact"]["version"] == 3


class TestToolDeleteAction:
    """Tool execute with action='delete'."""

    async def test_delete_existing(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "delete", "filename": "f.txt"})
        assert result.content[0].text == "Deleted artifact: f.txt"
        assert store.get("f.txt") is None

    async def test_delete_nonexistent(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "delete", "filename": "missing.txt"})
        assert result.content[0].text == "Artifact not found: missing.txt"


class TestToolUnknownAction:
    """Tool execute with unknown action."""

    async def test_unknown_action(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "rename", "filename": "f.txt"})
        assert result.content[0].text == "Unknown action: rename"

    async def test_empty_action(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"filename": "f.txt"})
        assert result.content[0].text == "Unknown action: "

    async def test_missing_action_key(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"filename": "f.txt"})
        assert "Unknown action" in result.content[0].text


class TestToolResultStructure:
    """Verify the AgentToolResult structure returned by the tool."""

    async def test_result_content_is_text_content(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "body"})
        assert len(result.content) == 1
        tc = result.content[0]
        assert tc.type == "text"
        assert isinstance(tc.text, str)

    async def test_create_result_has_details(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "body"})
        assert result.details is not None
        assert "artifact" in result.details

    async def test_delete_result_has_no_details(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "delete", "filename": "f.txt"})
        assert result.details is None

    async def test_error_result_has_no_details(self) -> None:
        store = ArtifactStore()
        tool = create_artifacts_tool(store)
        result = await tool.execute("call-1", {"action": "create", "filename": "f.txt"})
        assert result.details is None


class TestToolWithOnChangeCallback:
    """Tool operations trigger the on_change callback through the store."""

    async def test_create_triggers_callback(self) -> None:
        store = ArtifactStore()
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "create", "filename": "f.txt", "content": "body"})
        assert len(calls) == 1

    async def test_update_triggers_callback(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "update", "filename": "f.txt", "content": "new"})
        assert len(calls) == 1

    async def test_delete_triggers_callback(self) -> None:
        store = ArtifactStore()
        store.create("f.txt", "body")
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "delete", "filename": "f.txt"})
        assert len(calls) == 1

    async def test_update_auto_create_triggers_callback(self) -> None:
        store = ArtifactStore()
        calls: list[str] = []
        store.set_on_change(lambda: calls.append("changed"))
        tool = create_artifacts_tool(store)
        await tool.execute("call-1", {"action": "update", "filename": "new.txt", "content": "auto"})
        assert len(calls) == 1  # auto-create fires one callback via store.create


# ---------------------------------------------------------------------------
# Schema / constants tests
# ---------------------------------------------------------------------------


class TestArtifactsSchema:
    """Verify the module-level schema and description constants."""

    def test_schema_required_fields(self) -> None:
        assert ARTIFACTS_SCHEMA["required"] == ["action", "filename"]

    def test_schema_action_enum(self) -> None:
        action_prop = ARTIFACTS_SCHEMA["properties"]["action"]
        assert set(action_prop["enum"]) == {"create", "update", "delete"}

    def test_schema_has_all_properties(self) -> None:
        props = set(ARTIFACTS_SCHEMA["properties"].keys())
        assert props == {"action", "filename", "title", "content"}

    def test_description_is_nonempty(self) -> None:
        assert len(ARTIFACTS_DESCRIPTION) > 0
