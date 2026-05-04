"""Repository interface for Transcript entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.transcript import Transcript


class TranscriptRepository(ABC):
    @abstractmethod
    def add(self, transcript: Transcript) -> None: ...

    @abstractmethod
    def get(self, transcript_id: UUID) -> Transcript | None: ...

    @abstractmethod
    def get_by_meeting(self, meeting_id: UUID) -> Transcript | None: ...
