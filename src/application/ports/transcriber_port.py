"""Port for ASR (speech-to-text) providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.services.transcript_merger import RawTranscriptSegment
from src.domain.value_objects.audio_file import AudioFile


class TranscriberPort(ABC):
    """Speech-to-text contract — implemented in `infrastructure/speech`."""

    @abstractmethod
    def transcribe(
        self, audio: AudioFile, language: str | None = None
    ) -> list[RawTranscriptSegment]:
        """Return raw timestamped text segments (without speaker info)."""
