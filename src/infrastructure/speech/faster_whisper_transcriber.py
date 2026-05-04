"""TranscriberPort implementation backed by faster-whisper."""

from __future__ import annotations

import logging
from functools import lru_cache

# Preload CUDA 12 .so files BEFORE faster-whisper imports CTranslate2.
# Required because CTranslate2 4.7.x is built against CUDA 12 while PyTorch
# in this project ships CUDA 13 libs.
from src.infrastructure.speech import _cuda_bootstrap  # noqa: F401

from faster_whisper import WhisperModel

from src.application.ports.transcriber_port import TranscriberPort
from src.config.settings import WhisperSettings
from src.domain.services.transcript_merger import RawTranscriptSegment
from src.domain.value_objects.audio_file import AudioFile

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(TranscriberPort):
    """Lazy-loaded Whisper model. The model is shared across calls."""

    def __init__(self, settings: WhisperSettings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' on %s (compute_type=%s)",
                self._settings.model,
                self._settings.device,
                self._settings.compute_type,
            )
            self._model = _load_model(
                self._settings.model,
                self._settings.device,
                self._settings.compute_type,
            )
        return self._model

    def transcribe(
        self, audio: AudioFile, language: str | None = None
    ) -> list[RawTranscriptSegment]:
        model = self._ensure_model()
        target_language = language or self._settings.language
        segments_iter, info = model.transcribe(
            str(audio.path),
            language=target_language,
            vad_filter=True,
            beam_size=5,
            word_timestamps=False,
        )
        logger.info(
            "Whisper detected language=%s probability=%.2f duration=%.1fs",
            info.language,
            info.language_probability,
            info.duration,
        )
        return [
            RawTranscriptSegment(
                start_seconds=seg.start,
                end_seconds=seg.end,
                text=seg.text,
            )
            for seg in segments_iter
        ]


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)
