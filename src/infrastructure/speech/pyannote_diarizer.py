"""DiarizerPort implementation backed by pyannote.audio."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from pyannote.audio import Pipeline

from src.application.ports.diarizer_port import DiarizerPort
from src.config.settings import HuggingFaceSettings, PyannoteSettings
from src.domain.entities.speaker import SpeakerTurn
from src.domain.value_objects.audio_file import AudioFile

logger = logging.getLogger(__name__)


class PyannoteDiarizer(DiarizerPort):
    """Lazy-loaded pyannote diarization pipeline."""

    def __init__(
        self,
        settings: PyannoteSettings,
        huggingface_settings: HuggingFaceSettings,
    ) -> None:
        self._settings = settings
        self._hf = huggingface_settings
        self._pipeline: Pipeline | None = None

    def _ensure_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            if not self._hf.token:
                raise RuntimeError(
                    "HF_TOKEN is required to download the pyannote diarization pipeline. "
                    "Set HF_TOKEN in your environment."
                )
            logger.info(
                "Loading pyannote pipeline '%s' on %s",
                self._settings.pipeline,
                self._settings.device,
            )
            self._pipeline = _load_pipeline(
                self._settings.pipeline,
                self._hf.token,
                self._settings.device,
            )
        return self._pipeline

    def diarize(self, audio: AudioFile) -> list[SpeakerTurn]:
    pipeline = self._ensure_pipeline()

    output = pipeline(str(audio.path))

    # pyannote.audio < 4 retournait directement une Annotation.
    # pyannote.audio 4.x retourne un DiarizeOutput.
    #
    # Pour Whisper, on préfère exclusive_speaker_diarization si dispo :
    # les segments sont plus propres et non chevauchants.
    diarization = getattr(output, "exclusive_speaker_diarization", None)

    if diarization is None:
        diarization = getattr(output, "speaker_diarization", None)

    if diarization is None:
        # fallback ancienne API : output est déjà une Annotation
        diarization = output

    speaker_turns: list[SpeakerTurn] = []

    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append(
            SpeakerTurn(
                speaker_id=str(speaker),
                start_seconds=float(turn.start),
                end_seconds=float(turn.end),
            )
        )

    return speaker_turns


@lru_cache(maxsize=1)
def _load_pipeline(name: str, token: str, device: str) -> Pipeline:
    # Newer pyannote.audio (>= 3.4) renamed `use_auth_token` to `token`.
    # Try the new name first, fall back for older versions.
    try:
        pipeline = Pipeline.from_pretrained(name, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline
