"""Server-side artifacts tool for the Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent

logger = logging.getLogger(__name__)

# JSON Schema for the artifacts tool
ARTIFACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "update", "delete"],
            "description": "The action to perform on the artifact.",
        },
        "filename": {
            "type": "string",
            "description": "Unique filename for the artifact (e.g., 'chart.html', 'diagram.svg').",
        },
        "title": {
            "type": "string",
            "description": "Human-readable title for the artifact.",
        },
        "content": {
            "type": "string",
            "description": "The full content of the artifact. Required for create and update.",
        },
    },
    "required": ["action", "filename"],
}

ARTIFACTS_DESCRIPTION = """Create, update, or delete artifacts. Artifacts are files that will be displayed in a side panel.

Supported types:
- HTML files (.html) - Rendered in a sandboxed iframe
- SVG files (.svg) - Rendered as vector graphics
- Markdown files (.md) - Rendered as formatted text
- Code/text files (.js, .py, .css, etc.) - Displayed with syntax highlighting
- Image references (.png, .jpg, etc.) - Displayed as images

Use create to make a new artifact, update to modify existing ones, delete to remove.
Always provide the full content for create and update (not diffs)."""


class ArtifactStore:
    """In-memory store for artifacts within a session."""

    def __init__(self) -> None:
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._on_change: Any = None

    def set_on_change(self, callback: Any) -> None:
        self._on_change = callback

    def create(self, filename: str, content: str, title: str = "") -> dict[str, Any]:
        artifact = {
            "filename": filename,
            "content": content,
            "title": title or filename,
            "version": 1,
        }
        self.artifacts[filename] = artifact
        if self._on_change:
            self._on_change()
        return artifact

    def update(self, filename: str, content: str, title: str | None = None) -> dict[str, Any] | None:
        if filename not in self.artifacts:
            return None
        artifact = self.artifacts[filename]
        artifact["content"] = content
        artifact["version"] = artifact.get("version", 1) + 1
        if title is not None:
            artifact["title"] = title
        if self._on_change:
            self._on_change()
        return artifact

    def delete(self, filename: str) -> bool:
        if filename in self.artifacts:
            del self.artifacts[filename]
            if self._on_change:
                self._on_change()
            return True
        return False

    def get(self, filename: str) -> dict[str, Any] | None:
        return self.artifacts.get(filename)

    def get_all(self) -> list[dict[str, Any]]:
        return list(self.artifacts.values())


def create_artifacts_tool(store: ArtifactStore) -> AgentTool:
    """Create the artifacts AgentTool."""

    async def execute(tool_call_id: str, args: dict[str, Any], signal: Any = None) -> AgentToolResult:
        action = args.get("action", "")
        filename = args.get("filename", "")
        content = args.get("content", "")
        title = args.get("title", "")

        if action == "create":
            if not content:
                return AgentToolResult(
                    content=[TextContent(text="Error: content is required for create")]
                )
            artifact = store.create(filename, content, title)
            return AgentToolResult(
                content=[TextContent(text=f"Created artifact: {filename} (v{artifact['version']})")],
                details={"artifact": artifact},
            )

        elif action == "update":
            if not content:
                return AgentToolResult(
                    content=[TextContent(text="Error: content is required for update")]
                )
            artifact = store.update(filename, content, title or None)
            if artifact is None:
                # Auto-create if doesn't exist
                artifact = store.create(filename, content, title)
                return AgentToolResult(
                    content=[TextContent(text=f"Created artifact: {filename} (was not found, created new)")],
                    details={"artifact": artifact},
                )
            return AgentToolResult(
                content=[TextContent(text=f"Updated artifact: {filename} (v{artifact['version']})")],
                details={"artifact": artifact},
            )

        elif action == "delete":
            deleted = store.delete(filename)
            if deleted:
                return AgentToolResult(
                    content=[TextContent(text=f"Deleted artifact: {filename}")]
                )
            return AgentToolResult(
                content=[TextContent(text=f"Artifact not found: {filename}")]
            )

        else:
            return AgentToolResult(
                content=[TextContent(text=f"Unknown action: {action}")]
            )

    return AgentTool(
        name="artifacts",
        description=ARTIFACTS_DESCRIPTION,
        parameters=ARTIFACTS_SCHEMA,
        label="Artifacts",
        execute=execute,
    )
