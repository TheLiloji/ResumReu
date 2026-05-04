"""EmbedderPort implementation backed by sentence-transformers."""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.application.ports.embedder_port import EmbedderPort
from src.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbedderPort):
    """Lazy-loaded multilingual embedding model."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(
                "Loading embedder '%s' on %s",
                self._settings.model_id,
                self._settings.device,
            )
            self._model = _load(self._settings.model_id, self._settings.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return self._ensure_model().get_sentence_embedding_dimension()


@lru_cache(maxsize=2)
def _load(model_id: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_id, device=device)
