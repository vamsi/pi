"""REST API for session management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pi.web.storage.database import Database
from pi.web.storage.sessions import SessionStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def create_sessions_router(db: Database) -> APIRouter:
    sessions = SessionStore(db)

    @router.get("")
    async def list_sessions():
        return await sessions.get_all_metadata()

    @router.get("/{session_id}")
    async def get_session(session_id: str):
        data = await sessions.load(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

    @router.delete("/{session_id}")
    async def delete_session(session_id: str):
        await sessions.delete(session_id)
        return {"status": "deleted"}

    return router
