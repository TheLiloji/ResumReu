from uuid import uuid4

from src.domain.entities.speaker import SpeakerTurn
from src.domain.services.transcript_merger import (
    RawTranscriptSegment,
    TranscriptMerger,
)


def test_merger_attributes_speaker_with_max_overlap() -> None:
    merger = TranscriptMerger()
    raw = [RawTranscriptSegment(0.0, 4.0, "Bonjour à tous.")]
    turns = [
        SpeakerTurn("SPEAKER_00", 0.0, 1.0),
        SpeakerTurn("SPEAKER_01", 1.0, 4.5),  # bigger overlap
    ]
    transcript = merger.merge(uuid4(), raw, turns)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].speaker_id == "SPEAKER_01"


def test_merger_uses_unknown_when_no_overlap() -> None:
    merger = TranscriptMerger()
    raw = [RawTranscriptSegment(10.0, 11.0, "Hors plage")]
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 5.0)]
    transcript = merger.merge(uuid4(), raw, turns)
    assert transcript.segments[0].speaker_id == TranscriptMerger.UNKNOWN_SPEAKER


def test_merger_coalesces_consecutive_same_speaker() -> None:
    merger = TranscriptMerger()
    raw = [
        RawTranscriptSegment(0.0, 1.0, "Bonjour."),
        RawTranscriptSegment(1.2, 2.0, "Comment ça va ?"),
        RawTranscriptSegment(5.0, 6.0, "Bien et toi ?"),
    ]
    turns = [
        SpeakerTurn("SPEAKER_00", 0.0, 2.5),
        SpeakerTurn("SPEAKER_01", 4.5, 6.5),
    ]
    transcript = merger.merge(uuid4(), raw, turns)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Bonjour. Comment ça va ?"
    assert transcript.segments[0].speaker_id == "SPEAKER_00"
    assert transcript.segments[1].speaker_id == "SPEAKER_01"


def test_merger_skips_empty_text() -> None:
    merger = TranscriptMerger()
    raw = [
        RawTranscriptSegment(0.0, 1.0, "   "),
        RawTranscriptSegment(1.0, 2.0, "Hello"),
    ]
    turns = [SpeakerTurn("SPEAKER_00", 0.0, 2.0)]
    transcript = merger.merge(uuid4(), raw, turns)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello"
