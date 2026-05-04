"""Repository interface for the Meeting aggregate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.meeting import Meeting


class MeetingRepository(ABC):
    """Persistence boundary for Meeting aggregates."""

    @abstractmethod
    def add(self, meeting: Meeting) -> None: ...

    @abstractmethod
    def get(self, meeting_id: UUID) -> Meeting | None: ...

    @abstractmethod
    def update(self, meeting: Meeting) -> None: ...

    @abstractmethod
    def list_recent(self, limit: int = 50) -> list[Meeting]: ...

    @abstractmethod
    def delete(self, meeting_id: UUID) -> None: ...
