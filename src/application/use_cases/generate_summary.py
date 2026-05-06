"""Use case: turn a Transcript into a structured MeetingDocument via the LLM.

The LLM is queried via `LlmPort.complete_structured`, which delegates to a
constrained-decoding adapter (Outlines) — the response is guaranteed to match
`SummarySchema` structurally. Pydantic field constraints (`min_length`, etc.)
still apply and surface as `ValidationError` if the model produces an empty
summary or other semantically-invalid content.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.domain.entities.document import ActionItem, MeetingDocument
from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.entities.transcript import Transcript
from src.domain.repositories.glossary_repository import GlossaryRepository
from src.domain.repositories.meeting_repository import MeetingRepository
from src.domain.repositories.transcript_repository import TranscriptRepository
from src.domain.value_objects.technical_term import Glossary

logger = logging.getLogger(__name__)

MIN_SUMMARY_LINES = 10
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class GenerateSummaryInput:
    meeting_id: UUID


@dataclass(frozen=True, slots=True)
class GenerateSummaryOutput:
    document: MeetingDocument


class SummaryActionItemSchema(BaseModel):
    """Validated action-item structure expected from the LLM."""

    model_config = ConfigDict(extra="ignore")

    description: str = Field(min_length=1)
    owner: str | None = None
    due_date: str | None = None

    @field_validator("description")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("description must not be blank")
        return stripped

    @field_validator("owner", "due_date", mode="before")
    @classmethod
    def _blank_optional_text_to_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class SummarySchema(BaseModel):
    """Validated meeting-summary structure expected from the LLM."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[SummaryActionItemSchema] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary must not be blank")
        lines = _meaningful_lines(stripped)
        if len(lines) >= MIN_SUMMARY_LINES:
            return "\n".join(lines)

        sentences = _meaningful_sentences(stripped)
        if len(sentences) >= MIN_SUMMARY_LINES:
            return "\n".join(sentences)

        raise ValueError(
            f"summary must contain at least {MIN_SUMMARY_LINES} meaningful lines"
        )

    @field_validator("key_points", "decisions", mode="before")
    @classmethod
    def _none_list_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("action_items", mode="before")
    @classmethod
    def _none_action_items_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("key_points", "decisions")
    @classmethod
    def _strip_string_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


def _meaningful_lines(text: str) -> list[str]:
    return [
        line
        for raw_line in text.splitlines()
        if (line := raw_line.strip(" \t-•:;,."))
        and any(char.isalnum() for char in line)
    ]


def _meaningful_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [
        sentence
        for raw_sentence in _SENTENCE_SPLIT_RE.split(normalized)
        if (sentence := raw_sentence.strip(" \t-•:;,."))
        and any(char.isalnum() for char in sentence)
    ]


class GenerateSummaryUseCase:
    """Calls the LLM to produce a structured MeetingDocument from a Transcript."""

    def __init__(
        self,
        llm: LlmPort,
        meetings: MeetingRepository,
        transcripts: TranscriptRepository,
        glossary_repo: GlossaryRepository,
        summary_system_prompt: str,
    ) -> None:
        self._llm = llm
        self._meetings = meetings
        self._transcripts = transcripts
        self._glossary_repo = glossary_repo
        self._system_prompt = summary_system_prompt

    def execute(self, payload: GenerateSummaryInput) -> GenerateSummaryOutput:
        meeting = self._meetings.get(payload.meeting_id)
        if meeting is None:
            raise LookupError(f"Meeting {payload.meeting_id} not found")
        if meeting.transcript_id is None:
            raise ValueError(f"Meeting {meeting.id} has no transcript yet")
        transcript = self._transcripts.get(meeting.transcript_id)
        if transcript is None:
            raise LookupError(f"Transcript {meeting.transcript_id} not found")

        meeting.transition_to(MeetingStatus.SUMMARIZING)
        self._meetings.update(meeting)

        glossary = self._glossary_repo.load()
        document = self._summarize(meeting, transcript, glossary)

        meeting.attach_summary(document.summary)
        self._meetings.update(meeting)
        return GenerateSummaryOutput(document=document)

    def _summarize(
        self, meeting: Meeting, transcript: Transcript, glossary: Glossary
    ) -> MeetingDocument:
        glossary_text = self._format_glossary(glossary)
        transcript_text = transcript.as_text(with_timestamps=True)

        user_prompt = (
            f"Glossaire technique:\n{glossary_text}\n\n"
            f"Transcription diarisée:\n{transcript_text}"
        )
        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        logger.info(
            "Requesting LLM summary (meeting_id=%s, transcript_chars=%s, "
            "segments=%s, speakers=%s, glossary_terms=%s, temperature=0.0)",
            meeting.id,
            len(transcript_text),
            len(transcript.segments),
            len(transcript.speaker_ids()),
            len(glossary.terms),
        )
        started_at = time.perf_counter()
        parsed = self._llm.complete_structured(
            messages=messages,
            output_schema=SummarySchema,
            temperature=0.0,
        )
        elapsed_s = time.perf_counter() - started_at
        logger.info(
            "Parsed LLM summary (meeting_id=%s, duration_s=%.2f, summary_chars=%s, "
            "key_points=%s, decisions=%s, action_items=%s)",
            meeting.id,
            elapsed_s,
            len(parsed.summary),
            len(parsed.key_points),
            len(parsed.decisions),
            len(parsed.action_items),
        )

        detected_terms = glossary.detect_in_text(transcript_text)
        technical_terms = tuple((t.term, t.definition) for t in detected_terms)

        return MeetingDocument(
            meeting_id=meeting.id,
            title=meeting.title,
            date=meeting.created_at,
            participants=transcript.speaker_ids(),
            summary=parsed.summary,
            key_points=tuple(parsed.key_points),
            decisions=tuple(parsed.decisions),
            action_items=tuple(
                ActionItem(
                    description=item.description,
                    owner=item.owner,
                    due_date=item.due_date,
                )
                for item in parsed.action_items
            ),
            technical_terms=technical_terms,
        )

    @staticmethod
    def _format_glossary(glossary: Glossary) -> str:
        if not glossary.terms:
            return "(vide)"
        return "\n".join(
            f"- {t.term}: {t.definition}" for t in glossary.terms
        )
