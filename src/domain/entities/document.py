"""Structured meeting-report entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ActionItem:
    """A concrete follow-up assigned during the meeting."""

    description: str
    owner: str | None = None
    due_date: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("ActionItem.description must not be empty")


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """One section of a meeting report (e.g. 'Decisions', 'Risks')."""

    title: str
    content: str

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("DocumentSection.title must not be empty")


@dataclass(slots=True)
class MeetingDocument:
    """Structured meeting report assembled from the LLM summary.

    The infrastructure DocumentGenerator turns this into a .docx file.
    """

    meeting_id: UUID
    title: str
    date: datetime
    participants: tuple[str, ...]
    summary: str
    key_points: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[ActionItem, ...] = ()
    technical_terms: tuple[tuple[str, str], ...] = ()
    extra_sections: tuple[DocumentSection, ...] = ()
    id: UUID = field(default_factory=uuid4)
