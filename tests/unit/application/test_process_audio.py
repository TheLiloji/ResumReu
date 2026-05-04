"""ProcessAudioUseCase orchestration test — all ports mocked."""

from pathlib import Path
from uuid import UUID
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.process_audio import (
    ProcessAudioInput,
    ProcessAudioUseCase,
)
from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.entities.speaker import SpeakerTurn
from src.domain.services.transcript_merger import RawTranscriptSegment, TranscriptMerger


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"\x00\x00")
    return p


def test_process_audio_runs_full_pipeline(fake_audio: Path) -> None:
    meeting = Meeting(audio_path=fake_audio, title="Réu test")

    transcriber = MagicMock()
    transcriber.transcribe.return_value = [
        RawTranscriptSegment(0.0, 2.0, "Bonjour"),
        RawTranscriptSegment(2.0, 4.0, "Comment ça va"),
    ]

    diarizer = MagicMock()
    diarizer.diarize.return_value = [
        SpeakerTurn("SPEAKER_00", 0.0, 4.0),
    ]

    meetings = MagicMock()
    meetings.get.return_value = meeting

    transcripts = MagicMock()

    use_case = ProcessAudioUseCase(
        transcriber=transcriber,
        diarizer=diarizer,
        merger=TranscriptMerger(),
        meetings=meetings,
        transcripts=transcripts,
    )
    output = use_case.execute(ProcessAudioInput(meeting_id=meeting.id))

    transcriber.transcribe.assert_called_once()
    diarizer.diarize.assert_called_once()
    transcripts.add.assert_called_once()
    assert isinstance(output.transcript_id, UUID)
    assert meeting.transcript_id == output.transcript_id


def test_process_audio_marks_failed_on_error(fake_audio: Path) -> None:
    meeting = Meeting(audio_path=fake_audio, title="Réu test")

    transcriber = MagicMock()
    transcriber.transcribe.side_effect = RuntimeError("boom")
    diarizer = MagicMock()
    meetings = MagicMock()
    meetings.get.return_value = meeting

    use_case = ProcessAudioUseCase(
        transcriber=transcriber,
        diarizer=diarizer,
        merger=TranscriptMerger(),
        meetings=meetings,
        transcripts=MagicMock(),
    )

    with pytest.raises(RuntimeError):
        use_case.execute(ProcessAudioInput(meeting_id=meeting.id))
    assert meeting.status is MeetingStatus.FAILED
    assert meeting.error_message and "boom" in meeting.error_message
