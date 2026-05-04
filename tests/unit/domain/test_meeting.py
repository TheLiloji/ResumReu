from pathlib import Path
from uuid import uuid4

import pytest

from src.domain.entities.meeting import Meeting, MeetingStatus


def _make_meeting(tmp_path: Path) -> Meeting:
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake")
    return Meeting(audio_path=audio, title="Sprint review")


def test_meeting_starts_pending(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    assert meeting.status is MeetingStatus.PENDING
    assert meeting.transcript_id is None
    assert meeting.summary is None


def test_meeting_transitions_update_timestamp(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    initial = meeting.updated_at
    meeting.transition_to(MeetingStatus.TRANSCRIBING)
    assert meeting.status is MeetingStatus.TRANSCRIBING
    assert meeting.updated_at >= initial


def test_meeting_cannot_leave_completed(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    meeting.transition_to(MeetingStatus.COMPLETED)
    with pytest.raises(ValueError):
        meeting.transition_to(MeetingStatus.SUMMARIZING)


def test_meeting_can_fail_from_completed(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    meeting.transition_to(MeetingStatus.COMPLETED)
    meeting.mark_failed("post-mortem error")
    assert meeting.status is MeetingStatus.FAILED
    assert meeting.error_message == "post-mortem error"


def test_attach_summary_rejects_empty(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    with pytest.raises(ValueError):
        meeting.attach_summary("   ")


def test_attach_transcript_records_id(tmp_path: Path) -> None:
    meeting = _make_meeting(tmp_path)
    tid = uuid4()
    meeting.attach_transcript(tid)
    assert meeting.transcript_id == tid
