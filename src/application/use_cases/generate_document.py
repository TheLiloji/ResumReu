"""Use case: render a MeetingDocument to a .docx file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from src.application.ports.document_generator_port import DocumentGeneratorPort
from src.domain.entities.document import MeetingDocument
from src.domain.entities.meeting import MeetingStatus
from src.domain.repositories.meeting_repository import MeetingRepository


@dataclass(frozen=True, slots=True)
class GenerateDocumentInput:
    meeting_id: UUID
    document: MeetingDocument
    output_dir: Path


@dataclass(frozen=True, slots=True)
class GenerateDocumentOutput:
    document_path: Path


class GenerateDocumentUseCase:
    def __init__(
        self,
        generator: DocumentGeneratorPort,
        meetings: MeetingRepository,
    ) -> None:
        self._generator = generator
        self._meetings = meetings

    def execute(self, payload: GenerateDocumentInput) -> GenerateDocumentOutput:
        meeting = self._meetings.get(payload.meeting_id)
        if meeting is None:
            raise LookupError(f"Meeting {payload.meeting_id} not found")

        meeting.transition_to(MeetingStatus.GENERATING_DOCUMENT)
        self._meetings.update(meeting)

        payload.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"compte_rendu_{meeting.id}.docx"
        output_path = payload.output_dir / filename
        produced = self._generator.generate(payload.document, output_path)

        meeting.attach_document(produced)
        self._meetings.update(meeting)
        return GenerateDocumentOutput(document_path=produced)
