"""Port for vector-store providers (used by the RAG pipeline)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    record: VectorRecord
    score: float


class VectorStorePort(ABC):
    """Vector-store contract — implemented in `infrastructure/vector_store`."""

    @abstractmethod
    def upsert(self, collection: str, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def query(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[RetrievalHit]: ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None: ...
