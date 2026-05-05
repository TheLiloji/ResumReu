"""GenerateSummaryUseCase test — LLM port is mocked to return canned JSON."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.generate_summary import (
    GenerateSummaryInput,
    GenerateSummaryUseCase,
)
from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.value_objects.technical_term import Glossary, TechnicalTerm


def test_generate_summary_parses_json_and_attaches_terms(
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
    llm.complete.return_value = (
        '{"summary": "Bilan sprint", '
        '"key_points": ["livraison Y2"], '
        '"decisions": [], '
        '"action_items": [{"description": "MAJ Loop", "owner": "Léa", "due_date": "demain"}]}'
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
    )
    output = use_case.execute(GenerateSummaryInput(meeting_id=meeting.id))

    assert output.document.summary == "Bilan sprint"
    assert output.document.action_items[0].owner == "Léa"
    detected = {term for term, _ in output.document.technical_terms}
    assert detected == {"Y2", "Loop"}
    assert meeting.summary == "Bilan sprint"
    assert llm.complete.call_args.kwargs["temperature"] == 0.0
    assert "Requesting LLM summary" in caplog.text
    assert "Received LLM summary response" in caplog.text
    assert "Parsed LLM summary" in caplog.text


def test_generate_summary_strips_markdown_fences(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    meeting = Meeting(audio_path=audio, title="x")
    transcript = Transcript(
        meeting_id=meeting.id,
        segments=[TranscriptSegment("SPEAKER_00", 0.0, 1.0, "Hello")],
    )
    meeting.attach_transcript(transcript.id)

    llm = MagicMock()
    llm.complete.return_value = (
        '```json\n{"summary": "ok", "key_points": [], "decisions": [], "action_items": []}\n```'
    )

    meetings = MagicMock()
    meetings.get.return_value = meeting
    transcripts = MagicMock()
    transcripts.get.return_value = transcript
    glossary_repo = MagicMock()
    glossary_repo.load.return_value = Glossary(terms=())

    out = GenerateSummaryUseCase(
        llm=llm, meetings=meetings, transcripts=transcripts, glossary_repo=glossary_repo
    ).execute(GenerateSummaryInput(meeting_id=meeting.id))
    assert out.document.summary == "ok"


def test_parse_json_recovers_python_dict_style_output() -> None:
    parsed = GenerateSummaryUseCase._parse_json(
        "Réponse:\n"
        "{'summary': 'ok', 'key_points': ['p'], 'decisions': [], "
        "'action_items': [{'description': 'todo', 'owner': null, 'due_date': None}]}"
    )

    assert parsed["summary"] == "ok"
    assert parsed["key_points"] == ["p"]
    assert parsed["action_items"][0]["owner"] is None


def test_generate_summary_retries_when_schema_is_invalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="src.application.use_cases.generate_summary")
    llm = MagicMock()
    llm.complete.side_effect = [
        '{"summary": "", "key_points": [], "decisions": [], "action_items": []}',
        '{"summary": "ok", "key_points": ["point"], "decisions": [], "action_items": []}',
    ]
    use_case, meeting = _build_use_case(tmp_path, llm)

    out = use_case.execute(GenerateSummaryInput(meeting_id=meeting.id))

    assert out.document.summary == "ok"
    assert llm.complete.call_count == 2
    retry_messages = llm.complete.call_args_list[1].args[0]
    assert any("réponse précédente" in message.content for message in retry_messages)
    assert "retrying=True" in caplog.text
    assert "attempt=2/3" in caplog.text


def test_generate_summary_fails_after_retry_budget(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.complete.return_value = (
        '{"summary": "", "key_points": [], "decisions": [], "action_items": []}'
    )
    use_case, meeting = _build_use_case(tmp_path, llm)

    with pytest.raises(ValueError, match="invalid after 3 attempts"):
        use_case.execute(GenerateSummaryInput(meeting_id=meeting.id))

    assert llm.complete.call_count == 3


def _build_use_case(
    tmp_path: Path, llm: MagicMock
) -> tuple[GenerateSummaryUseCase, Meeting]:
    audio = tmp_path / "retry.wav"
    audio.write_bytes(b"x")
    meeting = Meeting(audio_path=audio, title="Retry")
    transcript = Transcript(
        meeting_id=meeting.id,
        segments=[TranscriptSegment("SPEAKER_00", 0.0, 1.0, "Bonjour")],
    )
    meeting.attach_transcript(transcript.id)

    meetings = MagicMock()
    meetings.get.return_value = meeting
    transcripts = MagicMock()
    transcripts.get.return_value = transcript
    glossary_repo = MagicMock()
    glossary_repo.load.return_value = Glossary(terms=())
    return (
        GenerateSummaryUseCase(
            llm=llm,
            meetings=meetings,
            transcripts=transcripts,
            glossary_repo=glossary_repo,
        ),
        meeting,
    )
