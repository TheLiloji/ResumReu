"""Port for embedding providers (used by the RAG pipeline)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedderPort(ABC):
    """Text embedding contract — implemented in `infrastructure/llm`."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...
