"""Value object representing an audio file on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SUPPORTED_FORMATS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"})


@dataclass(frozen=True, slots=True)
class AudioFile:
    """An immutable reference to an audio file.

    Validation rules:
        * the file must exist
        * its extension must belong to the supported formats
    """

    path: Path
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.path}")
        if self.path.suffix.lower() not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{self.path.suffix}'. "
                f"Supported: {sorted(_SUPPORTED_FORMATS)}"
            )

    @property
    def format(self) -> str:
        return self.path.suffix.lower().lstrip(".")

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size
