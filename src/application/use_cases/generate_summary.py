"""Use case: turn a Transcript into a structured MeetingDocument via the LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.domain.entities.document import ActionItem, MeetingDocument
from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.entities.transcript import Transcript
from src.domain.repositories.glossary_repository import GlossaryRepository
from src.domain.repositories.meeting_repository import MeetingRepository
from src.domain.repositories.transcript_repository import TranscriptRepository
from src.domain.value_objects.technical_term import Glossary, TechnicalTerm

_SYSTEM_PROMPT = """Tu es un assistant qui rédige des comptes rendus de réunion en français.
Réponds UNIQUEMENT avec un objet JSON valide, sans commentaire ni texte additionnel.
Le JSON doit suivre exactement ce schéma:
{
  "summary": "résumé synthétique de la réunion en 5 à 10 phrases",
  "key_points": ["point clé 1", "point clé 2", ...],
  "decisions": ["décision 1", ...],
  "action_items": [{"description": "...", "owner": "...", "due_date": "..."}, ...]
}
Utilise les termes techniques du glossaire fourni quand ils apparaissent.
Si une information n'est pas explicite, utilise null pour owner/due_date."""


@dataclass(frozen=True, slots=True)
class GenerateSummaryInput:
    meeting_id: UUID


@dataclass(frozen=True, slots=True)
class GenerateSummaryOutput:
    document: MeetingDocument


class GenerateSummaryUseCase:
    """Calls the LLM to produce a structured MeetingDocument from a Transcript."""

    def __init__(
        self,
        llm: LlmPort,
        meetings: MeetingRepository,
        transcripts: TranscriptRepository,
        glossary_repo: GlossaryRepository,
    ) -> None:
        self._llm = llm
        self._meetings = meetings
        self._transcripts = transcripts
        self._glossary_repo = glossary_repo

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
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
        raw = self._llm.complete(messages)
        parsed = self._parse_json(raw)

        detected_terms = glossary.detect_in_text(transcript_text)
        technical_terms = tuple((t.term, t.definition) for t in detected_terms)

        return MeetingDocument(
            meeting_id=meeting.id,
            title=meeting.title,
            date=meeting.created_at,
            participants=transcript.speaker_ids(),
            summary=parsed.get("summary", ""),
            key_points=tuple(parsed.get("key_points", [])),
            decisions=tuple(parsed.get("decisions", [])),
            action_items=tuple(
                ActionItem(
                    description=item["description"],
                    owner=item.get("owner"),
                    due_date=item.get("due_date"),
                )
                for item in parsed.get("action_items", [])
                if item.get("description")
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

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = raw.strip()
        # Strip optional markdown fences
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        # Take the largest top-level JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"LLM response is not JSON: {raw[:200]}...")
        return json.loads(cleaned[start : end + 1])
