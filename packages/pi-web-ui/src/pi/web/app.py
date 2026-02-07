"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from pi.web.config import Config
from pi.web.storage.database import Database
from pi.web.ws.handler import websocket_handler

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or Config()
    db = Database(config.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        # Ensure db directory exists
        db_dir = Path(config.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        await db.connect()
        logger.info("Database connected at %s", config.db_path)
        yield
        await db.close()
        logger.info("Database closed")

    app = FastAPI(title="pi-web-ui", lifespan=lifespan)

    # --- WebSocket endpoint ---

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket_handler(websocket, db)

    # --- REST API endpoints ---

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    from pi.web.api.sessions import create_sessions_router
    from pi.web.api.models_api import create_models_router
    from pi.web.api.upload import create_upload_router

    app.include_router(create_sessions_router(db))
    app.include_router(create_models_router())
    app.include_router(create_upload_router(db))

    # --- Static files (must be last) ---

    static_dir = Path(config.static_dir)
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
