from unittest.mock import MagicMock

from src.application.ports.vector_store_port import RetrievalHit, VectorRecord
from src.application.use_cases.query_knowledge import (
    QueryKnowledgeInput,
    QueryKnowledgeUseCase,
)


def _hit(rid: str, text: str, score: float, meta: dict) -> RetrievalHit:
    return RetrievalHit(
        record=VectorRecord(id=rid, text=text, metadata=meta), score=score
    )


def test_query_returns_no_sources_when_empty() -> None:
    llm = MagicMock()
    store = MagicMock()
    store.query.return_value = []
    out = QueryKnowledgeUseCase(llm, store).execute(
        QueryKnowledgeInput(question="?", collections=("meetings",))
    )
    assert out.sources == ()
    llm.complete.assert_not_called()


def test_query_orders_hits_by_score_and_calls_llm() -> None:
    llm = MagicMock(); llm.complete.return_value = "réponse"
    store = MagicMock()
    store.query.side_effect = [
        [_hit("m1", "T1", 0.4, {"type": "summary"})],
        [_hit("c1", "T2", 0.9, {"type": "doc"})],
    ]
    out = QueryKnowledgeUseCase(llm, store).execute(
        QueryKnowledgeInput(question="?", top_k=2, collections=("meetings", "cegid_docs"))
    )
    assert out.answer == "réponse"
    assert out.sources[0].text == "T2"  # higher score first
    assert out.sources[1].text == "T1"
    assert llm.complete.call_count == 1
