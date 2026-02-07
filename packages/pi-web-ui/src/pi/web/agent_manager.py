"""Agent lifecycle management per WebSocket session."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pi.agent import Agent, AgentEvent, ThinkingLevel
from pi.ai import get_model, get_models, get_providers
from pi.ai.types import Model

from pi.web.artifacts import ArtifactStore, create_artifacts_tool
from pi.web.storage.database import Database
from pi.web.storage.provider_keys import ProviderKeyStore
from pi.web.storage.sessions import SessionStore
from pi.web.ws.serializer import serialize_event, serialize_message

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages an Agent instance for a single WebSocket connection."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._sessions = SessionStore(db)
        self._keys = ProviderKeyStore(db)
        self._agent: Agent | None = None
        self._session_id: str = ""
        self._unsubscribe: Any = None
        self._send: Any = None  # async callable to send JSON to WebSocket
        self._artifact_store: ArtifactStore = ArtifactStore()

    @property
    def agent(self) -> Agent | None:
        return self._agent

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_send(self, send_fn: Any) -> None:
        """Set the async send function for WebSocket output."""
        self._send = send_fn

    async def _send_json(self, data: dict[str, Any]) -> None:
        if self._send:
            await self._send(data)

    # --- Session lifecycle ---

    async def new_session(self) -> str:
        """Create a new session with a fresh Agent."""
        session_id = str(uuid.uuid4())
        self._session_id = session_id

        if self._unsubscribe:
            self._unsubscribe()

        self._artifact_store = ArtifactStore()
        self._artifact_store.set_on_change(self._on_artifacts_change)

        self._agent = Agent(
            session_id=session_id,
            get_api_key=self._get_api_key,
        )

        # Set artifacts tool
        artifacts_tool = create_artifacts_tool(self._artifact_store)
        self._agent.set_tools([artifacts_tool])

        # Set default model if available
        providers = get_providers()
        if providers:
            models = get_models(providers[0])
            if models:
                self._agent.set_model(models[0])

        self._unsubscribe = self._agent.subscribe(self._on_event)
        return session_id

    async def load_session(self, session_id: str) -> bool:
        """Load an existing session from storage."""
        data = await self._sessions.load(session_id)
        if data is None:
            return False

        self._session_id = session_id

        if self._unsubscribe:
            self._unsubscribe()

        # Reconstruct model
        model = None
        try:
            model_data = json.loads(data["model_json"])
            if model_data:
                model = Model.model_validate(model_data)
        except Exception:
            pass

        thinking_level: ThinkingLevel = data.get("thinking_level", "off")  # type: ignore[assignment]

        # Reconstruct messages
        messages_list = []
        try:
            raw = json.loads(data["messages_json"])
            from pi.ai.types import AssistantMessage, ToolResultMessage, UserMessage

            for m in raw:
                role = m.get("role", "")
                if role == "user":
                    messages_list.append(UserMessage.model_validate(m))
                elif role == "assistant":
                    messages_list.append(AssistantMessage.model_validate(m))
                elif role in ("tool_result", "toolResult"):
                    messages_list.append(ToolResultMessage.model_validate(m))
        except Exception:
            logger.exception("Failed to deserialize messages for session %s", session_id)

        from pi.agent.types import AgentState

        self._artifact_store = ArtifactStore()
        self._artifact_store.set_on_change(self._on_artifacts_change)

        self._agent = Agent(
            initial_state=AgentState(
                model=model,
                thinking_level=thinking_level,
                messages=messages_list,
            ),
            session_id=session_id,
            get_api_key=self._get_api_key,
        )

        # Set artifacts tool
        artifacts_tool = create_artifacts_tool(self._artifact_store)
        self._agent.set_tools([artifacts_tool])

        self._unsubscribe = self._agent.subscribe(self._on_event)
        return True

    async def save_session(self) -> None:
        """Save current session state to storage."""
        if not self._agent or not self._session_id:
            return

        state = self._agent.state
        model_json = state.model.model_dump_json(by_alias=True) if state.model else "{}"
        messages_data = []
        for msg in state.messages:
            if hasattr(msg, "model_dump"):
                messages_data.append(msg.model_dump(by_alias=True))
        messages_json = json.dumps(messages_data)

        title = SessionStore.extract_title(messages_json)
        preview = SessionStore.extract_preview(messages_json)

        await self._sessions.save(
            self._session_id,
            model_json=model_json,
            thinking_level=state.thinking_level,
            messages_json=messages_json,
            title=title,
            message_count=len(state.messages),
            model_id=state.model.id if state.model else "",
            preview=preview,
        )

    # --- Agent operations ---

    async def prompt(self, text: str) -> None:
        """Send a user prompt to the agent."""
        if not self._agent:
            await self._send_json({"type": "error", "message": "No active session"})
            return

        if not self._agent.state.model:
            await self._send_json({"type": "error", "message": "No model selected"})
            return

        # Check API key
        provider = self._agent.state.model.provider
        key = await self._keys.get(provider)
        if not key:
            await self._send_json({"type": "api_key_required", "provider": provider})
            return

        try:
            await self._agent.prompt(text)
        except Exception as e:
            logger.exception("Error during prompt")
            await self._send_json({"type": "error", "message": str(e)})
        finally:
            await self.save_session()

    def abort(self) -> None:
        """Abort the current agent run."""
        if self._agent:
            self._agent.abort()

    def set_model(self, provider: str, model_id: str) -> None:
        """Set the agent's model."""
        model = get_model(provider, model_id)
        if model and self._agent:
            self._agent.set_model(model)

    def set_thinking_level(self, level: str) -> None:
        """Set the agent's thinking level."""
        if self._agent and level in ("off", "minimal", "low", "medium", "high", "xhigh"):
            self._agent.set_thinking_level(level)  # type: ignore[arg-type]

    async def set_api_key(self, provider: str, key: str) -> None:
        """Store an API key for a provider."""
        await self._keys.set(provider, key)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session from storage."""
        await self._sessions.delete(session_id)

    # --- State serialization ---

    def get_state_dict(self) -> dict[str, Any]:
        """Get full state for sending to client."""
        if not self._agent:
            return {
                "type": "state",
                "sessionId": self._session_id,
                "model": None,
                "thinkingLevel": "off",
                "messages": [],
                "isStreaming": False,
            }

        state = self._agent.state
        model_dict = state.model.model_dump(by_alias=True) if state.model else None
        messages = [serialize_message(m) for m in state.messages]

        return {
            "type": "state",
            "sessionId": self._session_id,
            "model": model_dict,
            "thinkingLevel": state.thinking_level,
            "messages": messages,
            "isStreaming": state.is_streaming,
        }

    async def get_models_dict(self) -> dict[str, Any]:
        """Get available models grouped by provider."""
        provider_list = []
        for provider_name in get_providers():
            models = get_models(provider_name)
            model_dicts = [m.model_dump(by_alias=True) for m in models]
            provider_list.append({"name": provider_name, "models": model_dicts})
        return {"type": "models", "providers": provider_list}

    async def get_sessions_dict(self) -> dict[str, Any]:
        """Get all session metadata."""
        metadata = await self._sessions.get_all_metadata()
        return {"type": "sessions", "sessions": metadata}

    # --- Private ---

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        return self._artifact_store.get_all()

    async def _get_api_key(self, provider: str) -> str | None:
        """Callback for Agent to retrieve API keys."""
        return await self._keys.get(provider)

    def _on_artifacts_change(self) -> None:
        """Send updated artifacts list to client when artifacts change."""
        data = {"type": "artifacts", "artifacts": self._artifact_store.get_all()}
        if self._send:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_json(data))
            except RuntimeError:
                pass

    def _on_event(self, event: AgentEvent) -> None:
        """Forward agent events to WebSocket."""
        data = serialize_event(event)
        if data and self._send:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_json(data))
            except RuntimeError:
                pass

    async def cleanup(self) -> None:
        """Clean up when WebSocket disconnects."""
        if self._agent and self._agent.state.is_streaming:
            self._agent.abort()
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await self.save_session()
