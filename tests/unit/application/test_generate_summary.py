"""GenerateSummaryUseCase test — LLM port is mocked to return a SummarySchema instance."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.generate_summary import (
    GenerateSummaryInput,
    GenerateSummaryUseCase,
    SummaryActionItemSchema,
    SummarySchema,
)
from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.value_objects.technical_term import Glossary, TechnicalTerm

_TEST_PROMPT = "système prompt de test"


def _summary_text() -> str:
    return "\n".join(
        f"Ligne {index} de synthèse avec un contenu exploitable."
        for index in range(1, 11)
    )


def test_generate_summary_uses_structured_completion_and_attaches_terms(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="src.application.use_cases.generate_summary")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    meeting = Meeting(audio_path=audio, title="Sprint Y2")
    transcript = Transcript(
        meeting_id=meeting.id,
        segments=[
            TranscriptSegment("SPEAKER_00", 0.0, 5.0, "On a livré Y2 cette semaine."),
            TranscriptSegment("SPEAKER_01", 5.0, 9.0, "Action: Léa met à jour Loop demain."),
        ],
    )
    meeting.attach_transcript(transcript.id)

    llm = MagicMock()
    summary = _summary_text()
    llm.complete_structured.return_value = SummarySchema(
        summary=summary,
        key_points=["livraison Y2"],
        decisions=[],
        action_items=[
            SummaryActionItemSchema(
                description="MAJ Loop", owner="Léa", due_date="demain"
            )
        ],
    )

    meetings = MagicMock()
    meetings.get.return_value = meeting
    transcripts = MagicMock()
    transcripts.get.return_value = transcript
    glossary_repo = MagicMock()
    glossary_repo.load.return_value = Glossary(
        terms=(
            TechnicalTerm("Y2", "ERP Cegid Y2"),
            TechnicalTerm("Loop", "Solution expert-comptable"),
        )
    )

    use_case = GenerateSummaryUseCase(
        llm=llm,
        meetings=meetings,
        transcripts=transcripts,
        glossary_repo=glossary_repo,
        summary_system_prompt=_TEST_PROMPT,
    )
    output = use_case.execute(GenerateSummaryInput(meeting_id=meeting.id))

    assert output.document.summary == summary
    assert output.document.action_items[0].owner == "Léa"
    detected = {term for term, _ in output.document.technical_terms}
    assert detected == {"Y2", "Loop"}
    assert meeting.summary == summary

    call = llm.complete_structured.call_args
    assert call.kwargs["temperature"] == 0.0
    assert call.kwargs["output_schema"] is SummarySchema
    system_message = call.kwargs["messages"][0]
    assert system_message.role == "system"
    assert system_message.content == _TEST_PROMPT
    assert "Requesting LLM summary" in caplog.text
    assert "Parsed LLM summary" in caplog.text


def test_generate_summary_propagates_validation_error_from_structured_call(
    tmp_path: Path,
) -> None:
    """Outlines guarantees structure but Pydantic still rejects empty summaries."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    meeting = Meeting(audio_path=audio, title="Empty")
    transcript = Transcript(
        meeting_id=meeting.id,
        segments=[TranscriptSegment("SPEAKER_00", 0.0, 1.0, "Bonjour")],
    )
    meeting.attach_transcript(transcript.id)

    llm = MagicMock()
    llm.complete_structured.side_effect = ValueError("summary must not be blank")

    meetings = MagicMock()
    meetings.get.return_value = meeting
    transcripts = MagicMock()
    transcripts.get.return_value = transcript
    glossary_repo = MagicMock()
    glossary_repo.load.return_value = Glossary(terms=())

    use_case = GenerateSummaryUseCase(
        llm=llm,
        meetings=meetings,
        transcripts=transcripts,
        glossary_repo=glossary_repo,
        summary_system_prompt=_TEST_PROMPT,
    )
    with pytest.raises(ValueError, match="summary must not be blank"):
        use_case.execute(GenerateSummaryInput(meeting_id=meeting.id))

    assert llm.complete_structured.call_count == 1


def test_summary_schema_rejects_punctuation_only_summary() -> None:
    with pytest.raises(ValueError, match="at least 10 meaningful lines"):
        SummarySchema(summary=":", key_points=[], decisions=[], action_items=[])


def test_summary_schema_normalizes_ten_sentences_to_lines() -> None:
    parsed = SummarySchema(
        summary=" ".join(
            f"Phrase {index} de synthèse avec un contenu exploitable."
            for index in range(1, 11)
        ),
        key_points=[],
        decisions=[],
        action_items=[],
    )

    assert len(parsed.summary.splitlines()) == 10
