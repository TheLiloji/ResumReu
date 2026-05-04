"""Transcript entities — speaker-attributed segments of speech."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One contiguous span of speech, attributed to a single speaker."""

    speaker_id: str
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError("Segment end precedes start")
        if not self.text.strip():
            raise ValueError("TranscriptSegment.text must not be empty")

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(slots=True)
class Transcript:
    """The full diarized transcript of a meeting."""

    meeting_id: UUID
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "fr"
    id: UUID = field(default_factory=uuid4)

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end_seconds

    def speaker_ids(self) -> tuple[str, ...]:
        """Distinct speaker ids preserving first-appearance order."""
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker_id not in seen:
                seen.append(segment.speaker_id)
        return tuple(seen)

    def as_text(self, with_timestamps: bool = False) -> str:
        """Plain-text rendering — useful as LLM input."""
        lines: list[str] = []
        for seg in self.segments:
            if with_timestamps:
                ts = _format_timestamp(seg.start_seconds)
                lines.append(f"[{ts}] {seg.speaker_id}: {seg.text}")
            else:
                lines.append(f"{seg.speaker_id}: {seg.text}")
        return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
