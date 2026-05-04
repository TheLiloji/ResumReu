"""Query router — Q&A across indexed meetings and Cegid docs."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.container import Container, get_container
from src.api.schemas.query_schemas import QueryRequest, QueryResponse, QuerySource
from src.application.use_cases.query_knowledge import QueryKnowledgeInput

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask(
    payload: QueryRequest, container: Container = Depends(get_container)
) -> QueryResponse:
    use_case = container.query_knowledge_use_case()
    result = use_case.execute(
        QueryKnowledgeInput(
            question=payload.question,
            top_k=payload.top_k,
            collections=tuple(payload.collections),
        )
    )
    return QueryResponse(
        answer=result.answer,
        sources=[
            QuerySource(
                rank=s.rank, text=s.text, metadata=s.metadata, score=s.score
            )
            for s in result.sources
        ],
    )
