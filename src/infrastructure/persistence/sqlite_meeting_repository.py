"""SQLAlchemy-backed implementation of MeetingRepository."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.repositories.meeting_repository import MeetingRepository
from src.infrastructure.persistence.database import session_scope
from src.infrastructure.persistence.models import MeetingModel


class SqliteMeetingRepository(MeetingRepository):
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def add(self, meeting: Meeting) -> None:
        with session_scope(self._session_factory) as session:
            session.add(_to_model(meeting))

    def get(self, meeting_id: UUID) -> Meeting | None:
        with session_scope(self._session_factory) as session:
            row = session.get(MeetingModel, str(meeting_id))
            return _to_entity(row) if row else None

    def update(self, meeting: Meeting) -> None:
        with session_scope(self._session_factory) as session:
            row = session.get(MeetingModel, str(meeting.id))
            if row is None:
                raise LookupError(f"Meeting {meeting.id} not found")
            _apply_changes(row, meeting)

    def list_recent(self, limit: int = 50) -> list[Meeting]:
        with session_scope(self._session_factory) as session:
            stmt = (
                select(MeetingModel).order_by(MeetingModel.created_at.desc()).limit(limit)
            )
            rows = session.scalars(stmt).all()
            return [_to_entity(row) for row in rows]

    def delete(self, meeting_id: UUID) -> None:
        with session_scope(self._session_factory) as session:
            row = session.get(MeetingModel, str(meeting_id))
            if row is not None:
                session.delete(row)


def _to_model(meeting: Meeting) -> MeetingModel:
    return MeetingModel(
        id=str(meeting.id),
        title=meeting.title,
        audio_path=str(meeting.audio_path),
        status=meeting.status.value,
        transcript_id=str(meeting.transcript_id) if meeting.transcript_id else None,
        summary=meeting.summary,
        document_path=str(meeting.document_path) if meeting.document_path else None,
        error_message=meeting.error_message,
        created_at=meeting.created_at,
        updated_at=meeting.updated_at,
    )


def _apply_changes(row: MeetingModel, meeting: Meeting) -> None:
    row.title = meeting.title
    row.audio_path = str(meeting.audio_path)
    row.status = meeting.status.value
    row.transcript_id = str(meeting.transcript_id) if meeting.transcript_id else None
    row.summary = meeting.summary
    row.document_path = str(meeting.document_path) if meeting.document_path else None
    row.error_message = meeting.error_message
    row.updated_at = meeting.updated_at


def _to_entity(row: MeetingModel) -> Meeting:
    return Meeting(
        id=UUID(row.id),
        title=row.title,
        audio_path=Path(row.audio_path),
        status=MeetingStatus(row.status),
        transcript_id=UUID(row.transcript_id) if row.transcript_id else None,
        summary=row.summary,
        document_path=Path(row.document_path) if row.document_path else None,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
