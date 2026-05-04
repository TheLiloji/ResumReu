"""Pydantic I/O schemas for the Q&A endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=6, ge=1, le=20)
    collections: list[str] = Field(default_factory=lambda: ["meetings", "cegid_docs"])


class QuerySource(BaseModel):
    rank: int
    text: str
    metadata: dict[str, str | int | float | bool]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
