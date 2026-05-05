"""Gemma LLM adapter tests."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.application.ports.llm_port import ChatMessage
from src.config.settings import HuggingFaceSettings, LlmSettings
from src.infrastructure.llm import gemma_llm_service


def test_load_model_configures_4bit_cpu_offload(
    monkeypatch: Any, tmp_path: Path
) -> None:
    gemma_llm_service._load_model.cache_clear()
    captured: dict[str, Any] = {}
    fake_processor = MagicMock()
    fake_model = MagicMock()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id: str, token: str | None) -> MagicMock:
            captured["processor"] = {"model_id": model_id, "token": token}
            return fake_processor

    class FakeModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> MagicMock:
            captured["model"] = {"model_id": model_id, "kwargs": kwargs}
            return fake_model

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["quantization_config"] = kwargs

    monkeypatch.setattr(gemma_llm_service, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(gemma_llm_service, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(gemma_llm_service, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(gemma_llm_service.torch.cuda, "is_available", lambda: True)

    offload_folder = tmp_path / "offload"
    processor, model = gemma_llm_service._load_model(
        model_id="google/gemma-4-E4B-it",
        load_in_4bit=True,
        device_map="auto",
        cpu_offload=True,
        max_gpu_memory="6GiB",
        max_cpu_memory="24GiB",
        offload_folder=offload_folder,
        hf_token="hf_test",
    )

    assert processor is fake_processor
    assert model is fake_model
    assert offload_folder.exists()
    assert captured["quantization_config"]["load_in_4bit"] is True
    assert captured["quantization_config"]["llm_int8_enable_fp32_cpu_offload"] is True
    assert captured["model"]["kwargs"]["device_map"] == "auto"
    assert captured["model"]["kwargs"]["max_memory"] == {0: "6GiB", "cpu": "24GiB"}
    assert captured["model"]["kwargs"]["offload_folder"] == str(offload_folder)
    assert captured["model"]["kwargs"]["offload_state_dict"] is True


def test_load_model_can_force_single_gpu_without_offload(
    monkeypatch: Any, tmp_path: Path
) -> None:
    gemma_llm_service._load_model.cache_clear()
    captured: dict[str, Any] = {}
    fake_processor = MagicMock()
    fake_model = MagicMock()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id: str, token: str | None) -> MagicMock:
            return fake_processor

    class FakeModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> MagicMock:
            captured["model"] = {"model_id": model_id, "kwargs": kwargs}
            return fake_model

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["quantization_config"] = kwargs

    monkeypatch.setattr(gemma_llm_service, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(gemma_llm_service, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(gemma_llm_service, "BitsAndBytesConfig", FakeBitsAndBytesConfig)

    _processor, _model = gemma_llm_service._load_model(
        model_id="google/gemma-4-E4B-it",
        load_in_4bit=True,
        device_map="cuda",
        cpu_offload=True,
        max_gpu_memory="6GiB",
        max_cpu_memory="24GiB",
        offload_folder=tmp_path / "offload",
        hf_token=None,
    )

    assert captured["quantization_config"]["llm_int8_enable_fp32_cpu_offload"] is False
    assert captured["model"]["kwargs"]["device_map"] == {"": 0}
    assert "max_memory" not in captured["model"]["kwargs"]
    assert "offload_folder" not in captured["model"]["kwargs"]
    assert "offload_state_dict" not in captured["model"]["kwargs"]


def test_complete_uses_processor_chat_template(
    monkeypatch: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="src.infrastructure.llm.gemma_llm_service")
    calls: dict[str, Any] = {}

    class FakeInputs(dict):
        def to(self, device: torch.device) -> "FakeInputs":
            calls["inputs_device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, chat: list[dict[str, str]], **kwargs: Any) -> FakeInputs:
            calls["chat"] = chat
            calls["template_kwargs"] = kwargs
            return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
            calls["decode_skip_special_tokens"] = skip_special_tokens
            return "<start>ok</start>"

        def parse_response(self, response: str) -> str:
            calls["parsed_response"] = response
            return "ok"

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs: Any) -> torch.Tensor:
            calls["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 4, 5]])

    fake_processor = FakeProcessor()
    fake_model = FakeModel()
    monkeypatch.setattr(
        gemma_llm_service,
        "_load_model",
        MagicMock(return_value=(fake_processor, fake_model)),
    )
    settings = LlmSettings(
        LLM_MODEL_ID="google/gemma-4-E4B-it",
        LLM_MAX_NEW_TOKENS=64,
        LLM_OFFLOAD_FOLDER=tmp_path,
    )
    service = gemma_llm_service.GemmaLlmService(settings, HuggingFaceSettings())

    response = service.complete(
        [
            ChatMessage(role="system", content="Réponds en JSON."),
            ChatMessage(role="user", content="Résumé"),
        ],
        temperature=0.0,
    )

    assert response == "ok"
    assert calls["chat"] == [
        {"role": "system", "content": "Réponds en JSON."},
        {"role": "user", "content": "Résumé"},
    ]
    assert calls["template_kwargs"]["tokenize"] is True
    assert calls["template_kwargs"]["return_dict"] is True
    assert calls["template_kwargs"]["return_tensors"] == "pt"
    assert calls["template_kwargs"]["add_generation_prompt"] is True
    assert calls["generate_kwargs"]["max_new_tokens"] == 64
    assert calls["generate_kwargs"]["do_sample"] is False
    assert "temperature" not in calls["generate_kwargs"]
    assert calls["parsed_response"] == "<start>ok</start>"
    assert "Starting LLM generation" in caplog.text
    assert "Finished LLM generation" in caplog.text
