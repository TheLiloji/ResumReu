"""LlmPort implementation for Gemma 4 via HuggingFace transformers.

The default E4B model is still too large for an 8 GB GPU without careful
placement. The default profile keeps the 4-bit model on the GPU to avoid the
current Accelerate/bitsandbytes split-device Params4bit issue. CPU offload stays
available as an explicit opt-in for environments where that stack works.

Structured output uses Outlines (`outlines.from_transformers`) to constrain
generation to a Pydantic schema — this guarantees the JSON conforms to the
schema and removes the need for defensive parsing in callers.
"""

from __future__ import annotations

import gc
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import outlines
import torch
from outlines.inputs import Chat
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.config.settings import HuggingFaceSettings, LlmSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GemmaLlmService(LlmPort):
    """HuggingFace-backed LLM, lazy-loaded on first use."""

    def __init__(
        self, settings: LlmSettings, huggingface_settings: HuggingFaceSettings
    ) -> None:
        self._settings = settings
        self._hf = huggingface_settings
        self._model: AutoModelForCausalLM | None = None
        self._processor: Any | None = None
        self._outlines_model: Any | None = None

    def _ensure_loaded(self) -> tuple[AutoModelForCausalLM, Any]:
        if self._model is None or self._processor is None:
            logger.info(
                "Loading LLM '%s' (load_in_4bit=%s, device_map=%s)",
                self._settings.model_id,
                self._settings.load_in_4bit,
                self._settings.device_map,
            )
            self._processor, self._model = _load_model(
                model_id=self._settings.model_id,
                load_in_4bit=self._settings.load_in_4bit,
                device_map=self._settings.device_map,
                cpu_offload=self._settings.cpu_offload,
                max_gpu_memory=self._settings.max_gpu_memory,
                max_cpu_memory=self._settings.max_cpu_memory,
                offload_folder=self._settings.offload_folder,
                hf_token=self._hf.token,
            )
        return self._model, self._processor

    def _ensure_outlines_model(self) -> Any:
        model, processor = self._ensure_loaded()
        if self._outlines_model is None:
            tokenizer = getattr(processor, "tokenizer", processor)
            logger.info(
                "Wrapping LLM '%s' with Outlines for constrained decoding",
                self._settings.model_id,
            )
            self._outlines_model = outlines.from_transformers(model, tokenizer)
        return self._outlines_model

    def complete(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        model, processor = self._ensure_loaded()
        chat = [{"role": m.role, "content": m.content} for m in messages]
        inputs = processor.apply_chat_template(
            chat,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        effective_temperature = (
            self._settings.temperature if temperature is None else temperature
        )
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens or self._settings.max_new_tokens,
            "do_sample": effective_temperature > 0.0,
            "pad_token_id": _eos_token_id(processor),
        }
        if effective_temperature > 0.0:
            generation_kwargs["temperature"] = effective_temperature

        logger.info(
            "Starting LLM generation (model=%s, input_tokens=%s, max_new_tokens=%s, "
            "temperature=%.2f, do_sample=%s, device=%s)",
            self._settings.model_id,
            input_len,
            generation_kwargs["max_new_tokens"],
            effective_temperature,
            generation_kwargs["do_sample"],
            model.device,
        )
        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)
        elapsed_s = time.perf_counter() - started_at
        new_tokens = output_ids[0, input_len:]
        response = processor.decode(new_tokens, skip_special_tokens=False)
        parse_response = getattr(processor, "parse_response", None)
        logger.info(
            "Finished LLM generation (model=%s, duration_s=%.2f, output_tokens=%s, "
            "raw_chars=%s)",
            self._settings.model_id,
            elapsed_s,
            new_tokens.shape[-1],
            len(response),
        )
        logger.debug("LLM generation raw response preview: %r", _preview(response))
        if callable(parse_response):
            parsed_response = str(parse_response(response)).strip()
            logger.debug(
                "LLM parsed response preview: %r", _preview(parsed_response)
            )
            return parsed_response
        decoded = processor.decode(new_tokens, skip_special_tokens=True).strip()
        logger.debug("LLM decoded response preview: %r", _preview(decoded))
        return decoded

    def complete_structured(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> T:
        outlines_model = self._ensure_outlines_model()
        chat = Chat([{"role": m.role, "content": m.content} for m in messages])

        effective_temperature = (
            self._settings.temperature if temperature is None else temperature
        )
        effective_max_new_tokens = max_new_tokens or self._settings.max_new_tokens
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": effective_temperature > 0.0,
        }
        if effective_temperature > 0.0:
            generation_kwargs["temperature"] = effective_temperature

        logger.info(
            "Starting constrained LLM generation (model=%s, schema=%s, "
            "max_new_tokens=%s, temperature=%.2f, do_sample=%s)",
            self._settings.model_id,
            output_schema.__name__,
            effective_max_new_tokens,
            effective_temperature,
            generation_kwargs["do_sample"],
        )
        started_at = time.perf_counter()
        result = outlines_model(chat, output_type=output_schema, **generation_kwargs)
        elapsed_s = time.perf_counter() - started_at
        logger.info(
            "Finished constrained LLM generation (model=%s, schema=%s, duration_s=%.2f)",
            self._settings.model_id,
            output_schema.__name__,
            elapsed_s,
        )

        if isinstance(result, output_schema):
            return result
        # Some Outlines paths return a JSON string for BaseModel output_type;
        # validate it through Pydantic to honor `Field` semantic constraints.
        return output_schema.model_validate_json(result)

    def unload(self) -> None:
        if self._model is None and self._processor is None:
            return
        logger.info("Unloading LLM '%s' from VRAM", self._settings.model_id)
        self._model = None
        self._processor = None
        self._outlines_model = None
        _load_model.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def _load_model(
    model_id: str,
    load_in_4bit: bool,
    device_map: str,
    cpu_offload: bool,
    max_gpu_memory: str | None,
    max_cpu_memory: str | None,
    offload_folder: Path,
    hf_token: str | None,
) -> tuple[Any, AutoModelForCausalLM]:
    resolved_device_map = _device_map(device_map)
    use_cpu_offload = cpu_offload and not _is_single_gpu_map(resolved_device_map)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=use_cpu_offload,
        )
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model_kwargs: dict[str, Any] = {
        "quantization_config": quantization_config,
        "device_map": resolved_device_map,
        "dtype": torch.float16,
        "token": hf_token,
    }
    if use_cpu_offload:
        offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs.update(
            max_memory=_max_memory(max_gpu_memory, max_cpu_memory),
            offload_folder=str(offload_folder),
            offload_state_dict=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    logger.info(
        "Loaded LLM '%s' with device map: %s",
        model_id,
        getattr(model, "hf_device_map", None),
    )
    return processor, model


def _device_map(device_map: str) -> str | dict[str, int]:
    normalized = device_map.strip().lower()
    if normalized in {"cuda", "gpu", "all-gpu", "all_gpu", "0"}:
        return {"": 0}
    return device_map


def _is_single_gpu_map(device_map: str | dict[str, int]) -> bool:
    return isinstance(device_map, dict) and device_map == {"": 0}


def _max_memory(
    max_gpu_memory: str | None, max_cpu_memory: str | None
) -> dict[Any, str] | None:
    max_memory: dict[Any, str] = {}
    if max_gpu_memory and torch.cuda.is_available():
        max_memory[0] = max_gpu_memory
    if max_cpu_memory:
        max_memory["cpu"] = max_cpu_memory
    return max_memory or None


def _eos_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    return getattr(tokenizer, "eos_token_id", None)


def _preview(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
