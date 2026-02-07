"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import importlib

import pytest
from httpx import ASGITransport, AsyncClient

from pi.web.config import Config


@pytest.fixture
async def app(tmp_path):
    """Create a FastAPI app with a temporary database.

    The API modules (sessions, models_api, upload) use module-level router
    objects. Calling create_*_router() appends handlers that close over the db
    instance. To avoid stale closures across tests we reload those modules so
    each test gets a fresh module-level router.
    """
    import pi.web.api.sessions as sessions_mod
    import pi.web.api.models_api as models_mod
    import pi.web.api.upload as upload_mod

    importlib.reload(sessions_mod)
    importlib.reload(models_mod)
    importlib.reload(upload_mod)

    from pi.web.app import create_app

    config = Config(db_path=str(tmp_path / "test.db"))
    application = create_app(config)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app):
    """Create an httpx AsyncClient bound to the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health(client: AsyncClient):
    """GET /api/health returns ok status."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_list_sessions_empty(client: AsyncClient):
    """GET /api/sessions returns an empty list initially."""
    response = await client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


async def test_list_models(client: AsyncClient):
    """GET /api/models returns a list."""
    response = await client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_get_session_not_found(client: AsyncClient):
    """GET /api/sessions/{nonexistent} returns 404."""
    response = await client.get("/api/sessions/nonexistent-id")
    assert response.status_code == 404


async def test_delete_session_nonexistent(client: AsyncClient):
    """DELETE /api/sessions/{nonexistent} returns deleted status."""
    response = await client.delete("/api/sessions/nonexistent-id")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}


async def test_upload_and_retrieve(client: AsyncClient):
    """POST /api/upload then GET /api/upload/{id} round-trips a file."""
    file_content = b"hello world from test"
    file_name = "test_file.txt"

    # Upload the file
    upload_response = await client.post(
        "/api/upload",
        files={"file": (file_name, file_content, "text/plain")},
    )
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["fileName"] == file_name
    assert upload_data["mimeType"] == "text/plain"
    assert upload_data["size"] == len(file_content)
    attachment_id = upload_data["id"]

    # Retrieve the file
    get_response = await client.get(f"/api/upload/{attachment_id}")
    assert get_response.status_code == 200
    assert get_response.content == file_content
    assert get_response.headers["content-type"] == "text/plain; charset=utf-8"
    assert file_name in get_response.headers["content-disposition"]


async def test_upload_with_session_id(client: AsyncClient):
    """POST /api/upload with session_id query parameter."""
    file_content = b"session file"
    upload_response = await client.post(
        "/api/upload",
        params={"session_id": "sess-123"},
        files={"file": ("doc.pdf", file_content, "application/pdf")},
    )
    assert upload_response.status_code == 200
    data = upload_response.json()
    assert data["fileName"] == "doc.pdf"
    assert data["mimeType"] == "application/pdf"
    assert data["size"] == len(file_content)


async def test_get_attachment_not_found(client: AsyncClient):
    """GET /api/upload/{nonexistent} returns 404."""
    response = await client.get("/api/upload/nonexistent-id")
    assert response.status_code == 404
