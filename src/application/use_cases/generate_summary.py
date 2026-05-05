"""Use case: turn a Transcript into a structured MeetingDocument via the LLM."""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.domain.entities.document import ActionItem, MeetingDocument
from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.entities.transcript import Transcript
from src.domain.repositories.glossary_repository import GlossaryRepository
from src.domain.repositories.meeting_repository import MeetingRepository
from src.domain.repositories.transcript_repository import TranscriptRepository
from src.domain.value_objects.technical_term import Glossary

logger = logging.getLogger(__name__)

_SUMMARY_MAX_RETRIES = 2
_RETRY_RESPONSE_LIMIT = 4000

_SYSTEM_PROMPT = """### RÔLE
Tu es un secrétaire d'entreprise expert en projets de migration informatique (spécialité ERP Cegid). Ta mission est de synthétiser des transcriptions de réunions en documents JSON structurés.

### CONTEXTE
Le texte source provient d'une transcription traitée par pyannote (diarisation) et Whisper.
- Priorise les aspects techniques : mapping de données, tests d'interface, décommissionnement d'anciens systèmes et modules Cegid.
- Utilise la terminologie professionnelle (Cut-over, Sandbox, scripts SQL, reprise d'historique).

### INSTRUCTIONS
1. ANALYSE : Identifie l'objectif central de la réunion et les rôles des intervenants.
2. EXTRACTION : Concentre-toi sur les décisions actées et les points de blocage techniques.
3. FORMATAGE : Produis UNIQUEMENT un objet JSON valide. Aucun texte avant ou après, aucun bloc Markdown (pas de ```json).

### CONTRAINTES DE SORTIE
- Langue : Rédige le contenu des champs "summary", "key_points" et "decisions" en FRANÇAIS.
- FORMAT : Retourne exclusivement l'objet JSON. Ne jamais ajouter de texte explicatif.
- INTÉGRITÉ : Toutes les clés du schéma ("summary", "key_points", "decisions", "action_items") doivent être présentes dans CHAQUE réponse, sans exception.
- ABSENCE DE DONNÉES : 
    - Si une liste est vide : utilise `[]`.
    - Si une information textuelle manque : utilise `""` ou `null`.
    - Ne jamais omettre une clé.- Valeurs Logiques : Utilise 'null' pour les informations manquantes. N'utilise pas les termes Python None/True/False.
- Synthèse : Le champ "summary" doit faire entre 5 et 10 phrases maximum.
- ABSENCE DE DONNÉES : 
    - Si une liste est vide : utilise `[]`.
    - Si une information textuelle manque : utilise `""` ou `null`.
    - Ne jamais omettre une clé.

### SCHÉMA DE RÉPONSE
{
  "summary": "résumé en français",
  "key_points": ["point 1", "point 2"],
  "decisions": ["décision 1"],
  "action_items": [
    {
      "description": "description de la tâche",
      "owner": "nom ou null",
      "due_date": "date ou null"
    }
  ]
}

### GLOSSAIRE
Respecte strictement les termes techniques du glossaire fourni lors de la rédaction."""


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
        return stripped

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
        parsed = self._complete_valid_summary(
            messages=messages,
            meeting=meeting,
            transcript=transcript,
            transcript_text=transcript_text,
            glossary=glossary,
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

    def _complete_valid_summary(
        self,
        messages: list[ChatMessage],
        meeting: Meeting,
        transcript: Transcript,
        transcript_text: str,
        glossary: Glossary,
    ) -> SummarySchema:
        max_attempts = _SUMMARY_MAX_RETRIES + 1
        for attempt_index in range(max_attempts):
            attempt = attempt_index + 1
            logger.info(
                "Requesting LLM summary (meeting_id=%s, attempt=%s/%s, "
                "transcript_chars=%s, segments=%s, speakers=%s, glossary_terms=%s, "
                "temperature=0.0)",
                meeting.id,
                attempt,
                max_attempts,
                len(transcript_text),
                len(transcript.segments),
                len(transcript.speaker_ids()),
                len(glossary.terms),
            )
            started_at = time.perf_counter()
            raw = self._llm.complete(messages, temperature=0.0)
            elapsed_s = time.perf_counter() - started_at
            logger.info(
                "Received LLM summary response (meeting_id=%s, attempt=%s/%s, "
                "duration_s=%.2f, raw_chars=%s)",
                meeting.id,
                attempt,
                max_attempts,
                elapsed_s,
                len(raw),
            )
            logger.debug(
                "LLM summary raw response preview (meeting_id=%s, attempt=%s/%s, "
                "preview=%r)",
                meeting.id,
                attempt,
                max_attempts,
                _preview(raw),
            )

            try:
                parsed = self._parse_summary(raw)
            except (ValueError, ValidationError) as error:
                retrying = attempt < max_attempts
                logger.warning(
                    "Invalid LLM summary response (meeting_id=%s, attempt=%s/%s, "
                    "retrying=%s, error=%s, raw_chars=%s, preview=%r)",
                    meeting.id,
                    attempt,
                    max_attempts,
                    retrying,
                    _compact_error(error),
                    len(raw),
                    _preview(raw, limit=500),
                )
                if not retrying:
                    raise ValueError(
                        "LLM summary response is invalid after "
                        f"{max_attempts} attempts: {_compact_error(error)}"
                    ) from error
                messages.extend(_retry_messages(raw, error))
                continue

            logger.info(
                "Parsed LLM summary (meeting_id=%s, attempt=%s/%s, summary_chars=%s, "
                "key_points=%s, decisions=%s, action_items=%s)",
                meeting.id,
                attempt,
                max_attempts,
                len(parsed.summary),
                len(parsed.key_points),
                len(parsed.decisions),
                len(parsed.action_items),
            )
            return parsed

        raise RuntimeError("unreachable summary retry state")

    @staticmethod
    def _format_glossary(glossary: Glossary) -> str:
        if not glossary.terms:
            return "(vide)"
        return "\n".join(
            f"- {t.term}: {t.definition}" for t in glossary.terms
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = _strip_markdown_fence(raw.strip())
        object_text = _extract_object(cleaned)
        try:
            parsed = json.loads(object_text)
        except json.JSONDecodeError as json_error:
            logger.warning(
                "LLM response is not strict JSON; trying Python literal fallback "
                "(preview=%r)",
                _preview(object_text, limit=500),
            )
            try:
                parsed = ast.literal_eval(_json_literals_to_python(object_text))
            except (SyntaxError, ValueError) as python_error:
                snippet = object_text.replace("\n", " ")[:500]
                raise ValueError(
                    "LLM response is not valid JSON and could not be recovered: "
                    f"{snippet}..."
                ) from python_error
            if not isinstance(parsed, dict):
                raise ValueError("LLM response root must be a JSON object") from json_error
        if not isinstance(parsed, dict):
            raise ValueError("LLM response root must be a JSON object")
        return parsed

    @staticmethod
    def _parse_summary(raw: str) -> SummarySchema:
        return SummarySchema.model_validate(GenerateSummaryUseCase._parse_json(raw))


def _strip_markdown_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _preview(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _retry_messages(raw: str, error: Exception) -> list[ChatMessage]:
    return [
        ChatMessage(role="assistant", content=_truncate_for_retry(raw)),
        ChatMessage(
            role="user",
            content=(
                "Ta réponse précédente ne respecte pas le format demandé.\n"
                f"Erreur de validation/parsing: {_compact_error(error)}\n"
                "Recommence en répondant UNIQUEMENT avec un objet JSON valide, "
                "sans Markdown ni texte autour. Respecte exactement les clés "
                "`summary`, `key_points`, `decisions`, `action_items`. "
                "Utilise null pour owner/due_date inconnus."
            ),
        ),
    ]


def _truncate_for_retry(text: str) -> str:
    if len(text) <= _RETRY_RESPONSE_LIMIT:
        return text
    return f"{text[:_RETRY_RESPONSE_LIMIT]}..."


def _compact_error(error: Exception, limit: int = 500) -> str:
    return _preview(str(error), limit=limit)


def _extract_object(cleaned: str) -> str:
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"LLM response is not JSON: {cleaned[:200]}...")

    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index, char in enumerate(cleaned[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]

    raise ValueError(f"LLM response has no complete JSON object: {cleaned[:200]}...")


def _json_literals_to_python(text: str) -> str:
    replacements = {"null": "None", "true": "True", "false": "False"}
    output: list[str] = []
    index = 0
    in_string = False
    quote = ""
    escaped = False

    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            index += 1
            continue

        if char in {"'", '"'}:
            in_string = True
            quote = char
            output.append(char)
            index += 1
            continue

        if char.isalpha() or char == "_":
            start = index
            while index < len(text) and (
                text[index].isalnum() or text[index] == "_"
            ):
                index += 1
            word = text[start:index]
            output.append(replacements.get(word, word))
            continue

        output.append(char)
        index += 1

    return "".join(output)
