"""Meetings router — upload audio, list meetings, get status."""

from __future__ import annotations

import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.container import Container, get_container
from src.api.schemas.meeting_schemas import (
    MeetingCreatedResponse,
    MeetingListItem,
    MeetingStatusResponse,
)
from src.domain.entities.meeting import Meeting

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post(
    "",
    response_model=MeetingCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_meeting(
    title: str = Form(...),
    audio: UploadFile = File(...),
    container: Container = Depends(get_container),
) -> MeetingCreatedResponse:
    """Upload an audio file and enqueue the processing pipeline."""
    saved_path = _persist_upload(audio, container.settings.app.uploads_dir)
    meeting = Meeting(audio_path=saved_path, title=title)
    container.meeting_repo.add(meeting)

    # Enqueue the pipeline asynchronously (Celery worker handles the heavy lifting).
    from src.workers.tasks.audio_processing_task import run_pipeline

    run_pipeline.delay(str(meeting.id))

    return MeetingCreatedResponse(
        meeting_id=meeting.id, status=meeting.status.value, title=meeting.title
    )


@router.get("", response_model=list[MeetingListItem])
def list_meetings(
    limit: int = 50, container: Container = Depends(get_container)
) -> list[MeetingListItem]:
    return [
        MeetingListItem(
            id=m.id, title=m.title, status=m.status.value, created_at=m.created_at
        )
        for m in container.meeting_repo.list_recent(limit=limit)
    ]


@router.get("/{meeting_id}", response_model=MeetingStatusResponse)
def get_meeting(
    meeting_id: UUID, container: Container = Depends(get_container)
) -> MeetingStatusResponse:
    meeting = container.meeting_repo.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return MeetingStatusResponse(
        id=meeting.id,
        title=meeting.title,
        status=meeting.status.value,
        created_at=meeting.created_at,
        updated_at=meeting.updated_at,
        transcript_id=meeting.transcript_id,
        summary=meeting.summary,
        document_path=str(meeting.document_path) if meeting.document_path else None,
        error_message=meeting.error_message,
    )


def _persist_upload(file: UploadFile, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix or ".bin"
    target = uploads_dir / f"{uuid.uuid4()}{suffix}"
    with target.open("wb") as out:
        while chunk := file.file.read(1 << 20):
            out.write(chunk)
    return target
