"""Pydantic I/O schemas for the meetings API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class MeetingCreatedResponse(BaseModel):
    meeting_id: UUID
    status: str
    title: str


class MeetingStatusResponse(BaseModel):
    id: UUID
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    transcript_id: UUID | None = None
    summary: str | None = None
    document_path: str | None = None
    error_message: str | None = None


class MeetingListItem(BaseModel):
    id: UUID
    title: str
    status: str
    created_at: datetime
