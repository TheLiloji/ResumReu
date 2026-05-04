"""Documents router — download a meeting's generated report."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.container import Container, get_container

router = APIRouter(prefix="/meetings", tags=["documents"])


@router.get("/{meeting_id}/document")
def download_document(
    meeting_id: UUID, container: Container = Depends(get_container)
) -> FileResponse:
    meeting = container.meeting_repo.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not meeting.document_path or not meeting.document_path.exists():
        raise HTTPException(status_code=404, detail="Document not generated yet")
    return FileResponse(
        path=meeting.document_path,
        filename=meeting.document_path.name,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
    )
