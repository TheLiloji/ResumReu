"""GenerateSummaryUseCase test — LLM port is mocked to return canned JSON."""

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

from src.application.use_cases.generate_summary import (
    GenerateSummaryInput,
    GenerateSummaryUseCase,
)
from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.value_objects.technical_term import Glossary, TechnicalTerm


def test_generate_summary_parses_json_and_attaches_terms(tmp_path: Path) -> None:
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

    meetings = MagicMock(); meetings.get.return_value = meeting
    transcripts = MagicMock(); transcripts.get.return_value = transcript
    glossary_repo = MagicMock(); glossary_repo.load.return_value = Glossary(terms=())

    out = GenerateSummaryUseCase(
        llm=llm, meetings=meetings, transcripts=transcripts, glossary_repo=glossary_repo
    ).execute(GenerateSummaryInput(meeting_id=meeting.id))
    assert out.document.summary == "ok"
