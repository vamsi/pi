"""File upload API endpoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile

from pi.web.storage.database import Database

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def create_upload_router(db: Database) -> APIRouter:

    @router.post("")
    async def upload_file(file: UploadFile, session_id: str = ""):
        if not file.filename:
            raise HTTPException(400, "No filename provided")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")

        attachment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        await db.conn.execute(
            """INSERT INTO attachments (id, session_id, file_name, mime_type, size, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                attachment_id,
                session_id,
                file.filename,
                file.content_type or "application/octet-stream",
                len(content),
                content,
                now,
            ),
        )
        await db.conn.commit()

        return {
            "id": attachment_id,
            "fileName": file.filename,
            "mimeType": file.content_type or "application/octet-stream",
            "size": len(content),
        }

    @router.get("/{attachment_id}")
    async def get_attachment(attachment_id: str):
        cursor = await db.conn.execute(
            "SELECT file_name, mime_type, content FROM attachments WHERE id = ?",
            (attachment_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(404, "Attachment not found")

        from fastapi.responses import Response

        return Response(
            content=row["content"],
            media_type=row["mime_type"],
            headers={"Content-Disposition": f'inline; filename="{row["file_name"]}"'},
        )

    return router
