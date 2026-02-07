"""File attachment (Slack upload) tool."""

from __future__ import annotations

import os
from typing import Any

from pi.agent.types import AgentTool, AgentToolResult
from pi.mom.tools import get_upload_function


def create_attach_tool() -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> AgentToolResult:
        upload_fn = get_upload_function()
        if upload_fn is None:
            raise RuntimeError("Upload function not configured")

        path = args["path"]
        title = args.get("title")

        absolute_path = os.path.abspath(path)
        file_name = title or os.path.basename(absolute_path)

        await upload_fn(absolute_path, file_name)

        return AgentToolResult(
            content=[{"type": "text", "text": f"Attached file: {file_name}"}],
        )

    return AgentTool(
        name="attach",
        label="attach",
        description=(
            "Attach a file to your response. Use this to share files, images, "
            "or documents with the user. Only files from /workspace/ can be attached."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Brief description of what you're sharing (shown to user)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to attach",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the file (defaults to filename)",
                },
            },
            "required": ["label", "path"],
        },
        execute=execute,
    )
