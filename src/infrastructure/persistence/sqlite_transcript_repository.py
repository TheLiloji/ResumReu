"""SQLAlchemy-backed implementation of TranscriptRepository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.repositories.transcript_repository import TranscriptRepository
from src.infrastructure.persistence.database import session_scope
from src.infrastructure.persistence.models import TranscriptModel, TranscriptSegmentModel


class SqliteTranscriptRepository(TranscriptRepository):
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def add(self, transcript: Transcript) -> None:
        with session_scope(self._session_factory) as session:
            row = TranscriptModel(
                id=str(transcript.id),
                meeting_id=str(transcript.meeting_id),
                language=transcript.language,
            )
            row.segments = [
                TranscriptSegmentModel(
                    idx=idx,
                    speaker_id=seg.speaker_id,
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    text=seg.text,
                )
                for idx, seg in enumerate(transcript.segments)
            ]
            session.add(row)

    def get(self, transcript_id: UUID) -> Transcript | None:
        with session_scope(self._session_factory) as session:
            row = session.get(TranscriptModel, str(transcript_id))
            return _to_entity(row) if row else None

    def get_by_meeting(self, meeting_id: UUID) -> Transcript | None:
        with session_scope(self._session_factory) as session:
            stmt = select(TranscriptModel).where(
                TranscriptModel.meeting_id == str(meeting_id)
            )
            row = session.scalar(stmt)
            return _to_entity(row) if row else None


def _to_entity(row: TranscriptModel) -> Transcript:
    return Transcript(
        id=UUID(row.id),
        meeting_id=UUID(row.meeting_id),
        language=row.language,
        segments=[
            TranscriptSegment(
                speaker_id=seg.speaker_id,
                start_seconds=seg.start_seconds,
                end_seconds=seg.end_seconds,
                text=seg.text,
            )
            for seg in row.segments
        ],
    )
