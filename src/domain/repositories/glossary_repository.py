"""Repository interface for the technical glossary."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.value_objects.technical_term import Glossary


class GlossaryRepository(ABC):
    """Loads the technical glossary used to enrich summaries."""

    @abstractmethod
    def load(self) -> Glossary: ...
