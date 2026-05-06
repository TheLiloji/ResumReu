"""Port for the local LLM (Gemma 4) used for summarization and Q&A."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LlmPort(ABC):
    """LLM contract — implemented in `infrastructure/llm`."""

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the assistant completion for the given conversation."""

    @abstractmethod
    def complete_structured(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> T:
        """Return a Pydantic instance constrained to `output_schema`.

        Implementations must guarantee structural conformity to the schema
        (e.g. via constrained decoding). Semantic validation (`Field`
        constraints) is still applied by Pydantic.
        """
