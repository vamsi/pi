"""Tool call validation using JSON Schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jsonschema

if TYPE_CHECKING:
    from pi.ai.types import Tool


def validate_tool_call(
    tools: list[Tool],
    tool_name: str,
    arguments: dict[str, Any],
) -> list[str]:
    """Find a tool by name and validate arguments against its schema.

    Returns a list of validation error messages (empty if valid).
    """
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        return [f"Unknown tool: {tool_name}"]
    return validate_tool_arguments(tool.parameters, arguments)


def validate_tool_arguments(
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> list[str]:
    """Validate arguments against a JSON schema.

    Returns a list of validation error messages (empty if valid).
    """
    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return []
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)"
        return [f"{path}: {e.message}"]
    except jsonschema.SchemaError as e:
        return [f"Invalid schema: {e.message}"]
