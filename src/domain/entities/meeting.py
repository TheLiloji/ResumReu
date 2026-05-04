"""Meeting aggregate — root entity of the domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4


class MeetingStatus(str, Enum):
    """Lifecycle status of a Meeting."""

    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    GENERATING_DOCUMENT = "generating_document"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Meeting:
    """Aggregate root.

    A Meeting starts as PENDING with just an audio file path. As the pipeline
    progresses, status transitions and the related artifacts (transcript_id,
    summary, document_path) are filled in.
    """

    audio_path: Path
    title: str
    id: UUID = field(default_factory=uuid4)
    status: MeetingStatus = MeetingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    transcript_id: UUID | None = None
    summary: str | None = None
    document_path: Path | None = None
    error_message: str | None = None

    def transition_to(self, new_status: MeetingStatus) -> None:
        if self.status == MeetingStatus.COMPLETED and new_status != MeetingStatus.FAILED:
            raise ValueError("Cannot transition out of COMPLETED")
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def mark_failed(self, message: str) -> None:
        self.error_message = message
        self.transition_to(MeetingStatus.FAILED)

    def attach_transcript(self, transcript_id: UUID) -> None:
        self.transcript_id = transcript_id
        self.updated_at = datetime.utcnow()

    def attach_summary(self, summary: str) -> None:
        if not summary.strip():
            raise ValueError("Summary must not be empty")
        self.summary = summary
        self.updated_at = datetime.utcnow()

    def attach_document(self, path: Path) -> None:
        self.document_path = path
        self.updated_at = datetime.utcnow()
