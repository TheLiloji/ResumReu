"""Use case: answer a question grounded in indexed meetings + Cegid docs."""

from __future__ import annotations

from dataclasses import dataclass

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.application.ports.vector_store_port import RetrievalHit, VectorStorePort

_SYSTEM_PROMPT = """Tu es un assistant interne qui répond aux questions sur les comptes rendus
de réunion et la documentation technique de Cegid.
Tu DOIS uniquement utiliser les passages fournis comme contexte. Si le contexte ne contient
pas la réponse, indique-le clairement.
Réponds en français, avec des références aux sources sous la forme [#1], [#2], etc."""


@dataclass(frozen=True, slots=True)
class QueryKnowledgeInput:
    question: str
    top_k: int = 6
    collections: tuple[str, ...] = ("meetings", "cegid_docs")


@dataclass(frozen=True, slots=True)
class CitedSource:
    rank: int
    text: str
    metadata: dict[str, str | int | float | bool]
    score: float


@dataclass(frozen=True, slots=True)
class QueryKnowledgeOutput:
    answer: str
    sources: tuple[CitedSource, ...]


class QueryKnowledgeUseCase:
    def __init__(self, llm: LlmPort, vector_store: VectorStorePort) -> None:
        self._llm = llm
        self._vector_store = vector_store

    def execute(self, payload: QueryKnowledgeInput) -> QueryKnowledgeOutput:
        hits = self._retrieve(payload)
        if not hits:
            return QueryKnowledgeOutput(
                answer="Aucun document pertinent n'a été trouvé pour cette question.",
                sources=(),
            )

        sources = tuple(
            CitedSource(
                rank=idx + 1,
                text=hit.record.text,
                metadata=dict(hit.record.metadata),
                score=hit.score,
            )
            for idx, hit in enumerate(hits)
        )
        context_block = self._format_context(sources)
        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Contexte:\n{context_block}\n\nQuestion: {payload.question}",
            ),
        ]
        answer = self._llm.complete(messages)
        return QueryKnowledgeOutput(answer=answer, sources=sources)

    def _retrieve(self, payload: QueryKnowledgeInput) -> list[RetrievalHit]:
        per_coll = max(1, payload.top_k // max(1, len(payload.collections)))
        all_hits: list[RetrievalHit] = []
        for collection in payload.collections:
            all_hits.extend(
                self._vector_store.query(
                    collection=collection,
                    query_text=payload.question,
                    top_k=per_coll,
                )
            )
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[: payload.top_k]

    @staticmethod
    def _format_context(sources: tuple[CitedSource, ...]) -> str:
        return "\n\n".join(
            f"[#{src.rank}] (source: {src.metadata.get('type', 'unknown')}, "
            f"id: {src.metadata.get('meeting_id') or src.metadata.get('document_id', 'n/a')})\n"
            f"{src.text}"
            for src in sources
        )
