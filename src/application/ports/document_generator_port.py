"""Port for document-generation providers (Word, PDF, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.document import MeetingDocument


class DocumentGeneratorPort(ABC):
    """Renders a structured `MeetingDocument` to a file on disk."""

    @abstractmethod
    def generate(self, document: MeetingDocument, output_path: Path) -> Path: ...
