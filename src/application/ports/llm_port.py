"""Port for the local LLM (Gemma 4) used for summarization and Q&A."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
