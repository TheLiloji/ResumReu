"""EmbedderPort implementation backed by HuggingFace transformers."""

from __future__ import annotations

import gc
import logging
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.application.ports.embedder_port import EmbedderPort
from src.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbedderPort):
    """Lazy-loaded multilingual embedding model.

    This intentionally avoids the `sentence-transformers` package. Its 5.x
    import path eagerly imports `torchcodec`, which is unnecessary for text
    embeddings and breaks worker startup in the CUDA runtime image.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            logger.info(
                "Loading embedder '%s' on %s",
                self._settings.model_id,
                self._settings.device,
            )
            self._tokenizer, self._model = _load(
                self._settings.model_id, self._settings.device
            )
        return self._tokenizer, self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenizer, model = self._ensure_model()
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        input_mask = input_mask.to(token_embeddings.dtype)
        pooled = (token_embeddings * input_mask).sum(dim=1)
        pooled = pooled / input_mask.sum(dim=1).clamp(min=1e-9)
        vectors = F.normalize(pooled, p=2, dim=1)
        return vectors.detach().cpu().tolist()

    @property
    def dimension(self) -> int:
        _tokenizer, model = self._ensure_model()
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            return len(self.embed([""])[0])
        return int(hidden_size)

    def unload(self) -> None:
        if self._model is None and self._tokenizer is None:
            return
        logger.info("Unloading embedder '%s' from VRAM", self._settings.model_id)
        self._model = None
        self._tokenizer = None
        _load.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=2)
def _load(model_id: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    target_device = _resolve_device(device)
    model.to(target_device)
    model.eval()
    return tokenizer, model


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for embedder but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)
