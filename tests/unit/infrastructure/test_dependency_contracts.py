"""Dependency and default-configuration contracts for the ML stack."""

import tomllib
from pathlib import Path

from src.config.settings import EmbeddingSettings, LlmSettings, PyannoteSettings

ROOT = Path(__file__).resolve().parents[3]


def _dependencies() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    return set(project["dependencies"])


def test_torch_family_versions_are_pinned_together() -> None:
    deps = _dependencies()

    assert "torch>=2.11,<2.12" in deps
    assert "torchaudio>=2.11,<2.12" in deps
    assert "torchvision>=0.26,<0.27" in deps


def test_transformers_stack_supports_gemma_4() -> None:
    deps = _dependencies()

    assert "transformers>=5.5,<6" in deps
    assert "accelerate>=1.1" in deps
    assert "bitsandbytes>=0.49" in deps
    assert "sentencepiece>=0.2" in deps
    assert "outlines>=1.2.13" in deps


def test_docker_image_keeps_triton_build_toolchain() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "python3.11-dev" in dockerfile
    assert "build-essential" in dockerfile
    assert "TORCH_COMPILE_DISABLE=1" in dockerfile
    assert "TORCHDYNAMO_DISABLE=1" in dockerfile


def test_worker_uses_cuda_safe_celery_pool() -> None:
    entrypoint = (ROOT / "docker" / "entrypoint-worker.sh").read_text(
        encoding="utf-8"
    )

    assert "--pool=solo" in entrypoint


def test_removed_packages_do_not_reenter_dependency_set() -> None:
    deps = _dependencies()

    assert not any(dep.startswith("sentence-transformers") for dep in deps)
    assert not any(dep.startswith("numpy") for dep in deps)


def test_default_models_match_8gb_vram_profile() -> None:
    llm = LlmSettings(_env_file=None)
    pyannote = PyannoteSettings(_env_file=None)
    embedding = EmbeddingSettings(_env_file=None)

    assert llm.model_id == "google/gemma-4-E4B-it"
    assert llm.load_in_4bit is True
    assert llm.device_map == "cuda"
    assert llm.cpu_offload is False
    assert llm.max_gpu_memory == "6GiB"
    assert llm.max_cpu_memory == "24GiB"
    assert llm.max_new_tokens == 512
    assert pyannote.pipeline == "pyannote/speaker-diarization-community-1"
    assert embedding.model_id == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
