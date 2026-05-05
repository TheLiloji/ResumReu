"""Centralized application settings, loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Base(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class AppSettings(_Base):
    env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), alias="APP_DATA_DIR")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


class HuggingFaceSettings(_Base):
    token: str | None = Field(default=None, alias="HF_TOKEN")


class WhisperSettings(_Base):
    model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    device: str = Field(default="cuda", alias="WHISPER_DEVICE")
    # int8_float16 by default: cuBLAS 12.9 on Ada hits CUBLAS_STATUS_NOT_SUPPORTED
    # with pure float16, and we need to leave VRAM headroom for Gemma.
    compute_type: str = Field(default="int8_float16", alias="WHISPER_COMPUTE_TYPE")
    language: str = Field(default="fr", alias="WHISPER_LANGUAGE")


class PyannoteSettings(_Base):
    pipeline: str = Field(
        default="pyannote/speaker-diarization-community-1", alias="PYANNOTE_PIPELINE"
    )
    device: str = Field(default="cuda", alias="PYANNOTE_DEVICE")


class LlmSettings(_Base):
    model_id: str = Field(default="google/gemma-4-E4B-it", alias="LLM_MODEL_ID")
    load_in_4bit: bool = Field(default=True, alias="LLM_LOAD_IN_4BIT")
    device_map: str = Field(default="cuda", alias="LLM_DEVICE_MAP")
    cpu_offload: bool = Field(default=False, alias="LLM_CPU_OFFLOAD")
    max_gpu_memory: str | None = Field(default="6GiB", alias="LLM_MAX_GPU_MEMORY")
    max_cpu_memory: str | None = Field(default="24GiB", alias="LLM_MAX_CPU_MEMORY")
    offload_folder: Path = Field(
        default=Path("./data/models/llm-offload"), alias="LLM_OFFLOAD_FOLDER"
    )
    max_new_tokens: int = Field(default=512, alias="LLM_MAX_NEW_TOKENS")
    temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")


class EmbeddingSettings(_Base):
    model_id: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        alias="EMBEDDING_MODEL_ID",
    )
    device: str = Field(default="cuda", alias="EMBEDDING_DEVICE")


class VectorStoreSettings(_Base):
    persist_dir: Path = Field(
        default=Path("./data/vector_store"), alias="CHROMA_PERSIST_DIR"
    )
    collection_meetings: str = Field(
        default="meetings", alias="CHROMA_COLLECTION_MEETINGS"
    )
    collection_cegid_docs: str = Field(
        default="cegid_docs", alias="CHROMA_COLLECTION_CEGID_DOCS"
    )


class DatabaseSettings(_Base):
    url: str = Field(default="sqlite:///./data/resumreu.db", alias="DATABASE_URL")


class CelerySettings(_Base):
    broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    result_backend: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND"
    )


class ApiSettings(_Base):
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")


class Settings:
    """Aggregate settings — composed once and accessed via `get_settings()`."""

    def __init__(self) -> None:
        self.app = AppSettings()
        self.huggingface = HuggingFaceSettings()
        self.whisper = WhisperSettings()
        self.pyannote = PyannoteSettings()
        self.llm = LlmSettings()
        self.embedding = EmbeddingSettings()
        self.vector_store = VectorStoreSettings()
        self.database = DatabaseSettings()
        self.celery = CelerySettings()
        self.api = ApiSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
