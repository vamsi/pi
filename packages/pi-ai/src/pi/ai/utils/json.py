"""JSON parsing utilities for streaming/incomplete JSON."""

from __future__ import annotations

import json
from typing import Any, TypeVar

T = TypeVar("T")


def parse_streaming_json(text: str | None) -> dict[str, Any]:
    """Parse potentially incomplete JSON from a streaming response.

    Attempts standard JSON parsing first, then falls back to
    best-effort partial parsing for incomplete JSON fragments.
    Returns empty dict if all parsing fails.
    """
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Best-effort partial JSON parsing:
    # Try adding closing braces/brackets to make it valid
    for suffix in ["}", "}}", "}}}", "]", "]}", '"}', '"]']:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    return {}
