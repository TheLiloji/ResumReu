"""Port for speaker-diarization providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.speaker import SpeakerTurn
from src.domain.value_objects.audio_file import AudioFile


class DiarizerPort(ABC):
    """Speaker-diarization contract — implemented in `infrastructure/speech`."""

    @abstractmethod
    def diarize(self, audio: AudioFile) -> list[SpeakerTurn]:
        """Return speaker turns ordered by start time."""
