"""Domain service: merges raw Whisper segments with pyannote speaker turns.

This is *pure* domain logic — no I/O, no model dependency.
The infrastructure adapters call this service after running their respective
inference steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.domain.entities.speaker import SpeakerTurn
from src.domain.entities.transcript import Transcript, TranscriptSegment


@dataclass(frozen=True, slots=True)
class RawTranscriptSegment:
    """ASR output — speaker is not yet known."""

    start_seconds: float
    end_seconds: float
    text: str


class TranscriptMerger:
    """Attribute each ASR segment to the speaker whose turn overlaps it the most."""

    UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"

    def merge(
        self,
        meeting_id: UUID,
        raw_segments: list[RawTranscriptSegment],
        speaker_turns: list[SpeakerTurn],
        language: str = "fr",
    ) -> Transcript:
        merged: list[TranscriptSegment] = []
        for raw in raw_segments:
            speaker_id = self._best_speaker(raw, speaker_turns)
            text = raw.text.strip()
            if not text:
                continue
            merged.append(
                TranscriptSegment(
                    speaker_id=speaker_id,
                    start_seconds=raw.start_seconds,
                    end_seconds=raw.end_seconds,
                    text=text,
                )
            )
        return Transcript(
            meeting_id=meeting_id,
            segments=self._coalesce_consecutive(merged),
            language=language,
        )

    def _best_speaker(
        self, raw: RawTranscriptSegment, turns: list[SpeakerTurn]
    ) -> str:
        best_id = self.UNKNOWN_SPEAKER
        best_overlap = 0.0
        for turn in turns:
            if not turn.overlaps(raw.start_seconds, raw.end_seconds):
                continue
            overlap = min(turn.end_seconds, raw.end_seconds) - max(
                turn.start_seconds, raw.start_seconds
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = turn.speaker_id
        return best_id

    @staticmethod
    def _coalesce_consecutive(
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Merge consecutive segments by the same speaker for cleaner output."""
        if not segments:
            return []
        coalesced: list[TranscriptSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = coalesced[-1]
            if seg.speaker_id == prev.speaker_id and (seg.start_seconds - prev.end_seconds) < 1.5:
                coalesced[-1] = TranscriptSegment(
                    speaker_id=prev.speaker_id,
                    start_seconds=prev.start_seconds,
                    end_seconds=seg.end_seconds,
                    text=f"{prev.text} {seg.text}",
                )
            else:
                coalesced.append(seg)
        return coalesced
