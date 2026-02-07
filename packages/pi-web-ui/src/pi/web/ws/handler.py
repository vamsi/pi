"""WebSocket endpoint handler."""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from pi.web.agent_manager import AgentManager
from pi.web.storage.database import Database
from pi.web.ws.protocol import (
    AbortMessage,
    DeleteSessionMessage,
    LoadSessionMessage,
    NewSessionMessage,
    PromptMessage,
    SetApiKeyMessage,
    SetModelMessage,
    SetThinkingLevelMessage,
    parse_client_message,
)

logger = logging.getLogger(__name__)


async def websocket_handler(websocket: WebSocket, db: Database) -> None:
    """Main WebSocket handler - one per client connection."""
    await websocket.accept()

    manager = AgentManager(db)

    async def send_json(data: dict[str, Any]) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    manager.set_send(send_json)

    # Create initial session
    await manager.new_session()

    # Send initial state + models + sessions
    await send_json(manager.get_state_dict())
    await send_json(await manager.get_models_dict())
    await send_json(await manager.get_sessions_dict())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg = parse_client_message(data)
            if msg is None:
                await send_json({"type": "error", "message": f"Unknown message type: {data.get('type')}"})
                continue

            match msg:
                case PromptMessage():
                    await manager.prompt(msg.text)

                case AbortMessage():
                    manager.abort()

                case SetModelMessage():
                    manager.set_model(msg.provider, msg.model_id)
                    await send_json(manager.get_state_dict())

                case SetThinkingLevelMessage():
                    manager.set_thinking_level(msg.level)
                    await send_json(manager.get_state_dict())

                case LoadSessionMessage():
                    loaded = await manager.load_session(msg.session_id)
                    if loaded:
                        await send_json(manager.get_state_dict())
                    else:
                        await send_json({"type": "error", "message": "Session not found"})

                case NewSessionMessage():
                    await manager.save_session()
                    await manager.new_session()
                    await send_json(manager.get_state_dict())
                    await send_json(await manager.get_sessions_dict())

                case SetApiKeyMessage():
                    await manager.set_api_key(msg.provider, msg.key)
                    await send_json({"type": "api_key_saved", "provider": msg.provider})

                case DeleteSessionMessage():
                    await manager.delete_session(msg.session_id)
                    await send_json(await manager.get_sessions_dict())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        await manager.cleanup()
