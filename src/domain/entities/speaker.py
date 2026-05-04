"""Speaker entities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Speaker:
    """A meeting participant.

    `id` is the diarization-assigned label (e.g. "SPEAKER_00").
    `display_name` is the human-readable name once known.
    """

    id: str
    display_name: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.id


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A continuous speech segment attributed to one speaker."""

    speaker_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"SpeakerTurn end ({self.end_seconds}) precedes start ({self.start_seconds})"
            )

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def overlaps(self, start: float, end: float) -> bool:
        return self.start_seconds < end and self.end_seconds > start
