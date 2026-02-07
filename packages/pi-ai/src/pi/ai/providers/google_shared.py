"""Shared utilities for Google Generative AI, Vertex, and Gemini CLI providers."""

from __future__ import annotations

import re
from typing import Any

from pi.ai.providers.transform import transform_messages
from pi.ai.types import (
    Context,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
)

# Regex to strip surrogate pairs
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE_RE = re.compile(r"^[A-Za-z0-9+/]+=*$")


def _sanitize(text: str) -> str:
    return _SURROGATE_RE.sub("\ufffd", text)


def is_thinking_part(part: dict[str, Any]) -> bool:
    """Check if a streamed Gemini Part should be treated as thinking."""
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """Preserve the last non-empty signature for the current block."""
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_RE.match(signature))


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs."""
    return model_id.startswith("claude-") or model_id.startswith("gpt-oss-")


def convert_messages(model: Model, context: Context) -> list[dict[str, Any]]:
    """Convert internal messages to Gemini Content[] format."""
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(id: str) -> str:
        if not requires_tool_call_id(model.id):
            return id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]

    transformed = transform_messages(context.messages, current_model=model.id, normalize_tool_id=normalize_tool_call_id)

    for msg in transformed:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": _sanitize(msg.content)}]})
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        parts.append({"text": _sanitize(item.text)})
                    elif isinstance(item, ImageContent):
                        parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
                filtered = [p for p in parts if "text" in p] if "image" not in model.input else parts
                if not filtered:
                    continue
                contents.append({"role": "user", "parts": filtered})

        elif msg.role == "assistant":
            parts = []
            is_same = msg.provider == model.provider and msg.model == model.id

            for block in msg.content:
                if isinstance(block, TextContent):
                    if not block.text or not block.text.strip():
                        continue
                    sig = _resolve_thought_signature(is_same, block.text_signature)
                    part: dict[str, Any] = {"text": _sanitize(block.text)}
                    if sig:
                        part["thoughtSignature"] = sig
                    parts.append(part)
                elif isinstance(block, ThinkingContent):
                    if not block.thinking or not block.thinking.strip():
                        continue
                    if is_same:
                        sig = _resolve_thought_signature(is_same, block.thinking_signature)
                        part = {"thought": True, "text": _sanitize(block.thinking)}
                        if sig:
                            part["thoughtSignature"] = sig
                        parts.append(part)
                    else:
                        parts.append({"text": _sanitize(block.thinking)})
                elif isinstance(block, ToolCall):
                    sig = _resolve_thought_signature(is_same, block.thought_signature)
                    is_gemini3 = "gemini-3" in model.id.lower()
                    if is_gemini3 and not sig:
                        args_str = str(block.arguments or {})
                        parts.append(
                            {
                                "text": f'[Historical context: a different model called tool "{block.name}" with arguments: {args_str}. Do not mimic this format - use proper function calling.]'
                            }
                        )
                    else:
                        fc: dict[str, Any] = {
                            "functionCall": {
                                "name": block.name,
                                "args": block.arguments or {},
                                **({"id": block.id} if requires_tool_call_id(model.id) else {}),
                            }
                        }
                        if sig:
                            fc["thoughtSignature"] = sig
                        parts.append(fc)

            if not parts:
                continue
            contents.append({"role": "model", "parts": parts})

        elif msg.role == "tool_result":
            text_items = [c for c in msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(c.text for c in text_items)
            image_items = [c for c in msg.content if isinstance(c, ImageContent)] if "image" in model.input else []

            has_text = len(text_result) > 0
            has_images = len(image_items) > 0
            supports_multimodal_fn = "gemini-3" in model.id

            response_value = _sanitize(text_result) if has_text else ("(see attached image)" if has_images else "")

            image_parts = [{"inlineData": {"mimeType": img.mime_type, "data": img.data}} for img in image_items]

            include_id = requires_tool_call_id(model.id)
            fn_response_part: dict[str, Any] = {
                "functionResponse": {
                    "name": msg.tool_name,
                    "response": {"error": response_value} if msg.is_error else {"output": response_value},
                    **({"parts": image_parts} if has_images and supports_multimodal_fn else {}),
                    **({"id": msg.tool_call_id} if include_id else {}),
                }
            }

            last = contents[-1] if contents else None
            if last and last.get("role") == "user" and any("functionResponse" in p for p in last.get("parts", [])):
                last["parts"].append(fn_response_part)
            else:
                contents.append({"role": "user", "parts": [fn_response_part]})

            if has_images and not supports_multimodal_fn:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})

    return contents


def convert_tools(tools: list[Any]) -> list[dict[str, Any]] | None:
    """Convert tools to Gemini function declarations format."""
    if not tools:
        return None
    return [
        {
            "functionDeclarations": [
                {"name": tool.name, "description": tool.description, "parameters": tool.parameters} for tool in tools
            ]
        }
    ]


def map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini FunctionCallingConfigMode."""
    mapping = {"auto": "AUTO", "none": "NONE", "any": "ANY"}
    return mapping.get(choice, "AUTO")


def map_stop_reason(reason: str) -> StopReason:
    """Map Gemini FinishReason to StopReason."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


def map_stop_reason_string(reason: str) -> StopReason:
    """Map string finish reason to StopReason (for raw API responses)."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"
