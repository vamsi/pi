"""WebSocket message protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- Client -> Server messages ---


@dataclass
class PromptMessage:
    type: str = "prompt"
    text: str = ""
    attachments: list[str] = field(default_factory=list)


@dataclass
class AbortMessage:
    type: str = "abort"


@dataclass
class SetModelMessage:
    type: str = "set_model"
    provider: str = ""
    model_id: str = ""


@dataclass
class SetThinkingLevelMessage:
    type: str = "set_thinking_level"
    level: str = "off"


@dataclass
class LoadSessionMessage:
    type: str = "load_session"
    session_id: str = ""


@dataclass
class NewSessionMessage:
    type: str = "new_session"


@dataclass
class SetApiKeyMessage:
    type: str = "set_api_key"
    provider: str = ""
    key: str = ""


@dataclass
class DeleteSessionMessage:
    type: str = "delete_session"
    session_id: str = ""


def parse_client_message(data: dict[str, Any]) -> Any:
    """Parse a raw dict into a typed client message."""
    msg_type = data.get("type", "")
    match msg_type:
        case "prompt":
            return PromptMessage(
                text=data.get("text", ""),
                attachments=data.get("attachments", []),
            )
        case "abort":
            return AbortMessage()
        case "set_model":
            return SetModelMessage(
                provider=data.get("provider", ""),
                model_id=data.get("modelId", data.get("model_id", "")),
            )
        case "set_thinking_level":
            return SetThinkingLevelMessage(level=data.get("level", "off"))
        case "load_session":
            return LoadSessionMessage(session_id=data.get("sessionId", data.get("session_id", "")))
        case "new_session":
            return NewSessionMessage()
        case "set_api_key":
            return SetApiKeyMessage(
                provider=data.get("provider", ""),
                key=data.get("key", ""),
            )
        case "delete_session":
            return DeleteSessionMessage(
                session_id=data.get("sessionId", data.get("session_id", "")),
            )
        case _:
            return None


# --- Server -> Client message builders ---


def state_message(
    session_id: str,
    model: dict[str, Any] | None,
    thinking_level: str,
    messages: list[dict[str, Any]],
    is_streaming: bool,
) -> dict[str, Any]:
    return {
        "type": "state",
        "sessionId": session_id,
        "model": model,
        "thinkingLevel": thinking_level,
        "messages": messages,
        "isStreaming": is_streaming,
    }


def error_message(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def api_key_required_message(provider: str) -> dict[str, Any]:
    return {"type": "api_key_required", "provider": provider}


def models_message(providers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "models", "providers": providers}


def sessions_message(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "sessions", "sessions": sessions}
