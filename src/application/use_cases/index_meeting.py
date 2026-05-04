"""Use case: index a Meeting's transcript and summary in the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.application.ports.vector_store_port import VectorRecord, VectorStorePort
from src.domain.entities.meeting import MeetingStatus
from src.domain.repositories.meeting_repository import MeetingRepository
from src.domain.repositories.transcript_repository import TranscriptRepository


@dataclass(frozen=True, slots=True)
class IndexMeetingInput:
    meeting_id: UUID
    collection: str = "meetings"


@dataclass(frozen=True, slots=True)
class IndexMeetingOutput:
    chunks_indexed: int


class IndexMeetingUseCase:
    """Chunk + upsert into the vector store for later RAG retrieval."""

    CHUNK_SIZE_CHARS = 1500
    CHUNK_OVERLAP_CHARS = 150

    def __init__(
        self,
        vector_store: VectorStorePort,
        meetings: MeetingRepository,
        transcripts: TranscriptRepository,
    ) -> None:
        self._vector_store = vector_store
        self._meetings = meetings
        self._transcripts = transcripts

    def execute(self, payload: IndexMeetingInput) -> IndexMeetingOutput:
        meeting = self._meetings.get(payload.meeting_id)
        if meeting is None:
            raise LookupError(f"Meeting {payload.meeting_id} not found")
        if meeting.transcript_id is None:
            raise ValueError(f"Meeting {meeting.id} has no transcript")
        transcript = self._transcripts.get(meeting.transcript_id)
        if transcript is None:
            raise LookupError(f"Transcript {meeting.transcript_id} not found")

        meeting.transition_to(MeetingStatus.INDEXING)
        self._meetings.update(meeting)

        chunks = self._chunk_text(transcript.as_text(with_timestamps=True))
        records = [
            VectorRecord(
                id=f"{meeting.id}:chunk:{idx}",
                text=chunk,
                metadata={
                    "meeting_id": str(meeting.id),
                    "title": meeting.title,
                    "type": "transcript",
                    "chunk_index": idx,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
        if meeting.summary:
            records.append(
                VectorRecord(
                    id=f"{meeting.id}:summary",
                    text=meeting.summary,
                    metadata={
                        "meeting_id": str(meeting.id),
                        "title": meeting.title,
                        "type": "summary",
                    },
                )
            )

        self._vector_store.upsert(payload.collection, records)
        meeting.transition_to(MeetingStatus.COMPLETED)
        self._meetings.update(meeting)
        return IndexMeetingOutput(chunks_indexed=len(records))

    def _chunk_text(self, text: str) -> list[str]:
        if not text:
            return []
        size = self.CHUNK_SIZE_CHARS
        overlap = self.CHUNK_OVERLAP_CHARS
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size]]
