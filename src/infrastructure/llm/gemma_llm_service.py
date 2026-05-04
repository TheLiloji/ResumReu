"""LlmPort implementation for Gemma 4 26B-A4B (MoE) via HuggingFace transformers.

The 26B-A4B model is a Mixture-of-Experts: 26B total parameters but only 4B
are active per token. With 4-bit quantization (bitsandbytes) the full weights
fit in ~12-13 GB; combined with `device_map="auto"`, transformers spreads
the layers between the 12 GB VRAM and host RAM as needed.
"""

from __future__ import annotations

import gc
import logging
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.application.ports.llm_port import ChatMessage, LlmPort
from src.config.settings import HuggingFaceSettings, LlmSettings

logger = logging.getLogger(__name__)


class GemmaLlmService(LlmPort):
    """HuggingFace-backed LLM, lazy-loaded on first use."""

    def __init__(
        self, settings: LlmSettings, huggingface_settings: HuggingFaceSettings
    ) -> None:
        self._settings = settings
        self._hf = huggingface_settings
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def _ensure_loaded(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self._model is None or self._tokenizer is None:
            logger.info(
                "Loading LLM '%s' (load_in_4bit=%s, device_map=%s)",
                self._settings.model_id,
                self._settings.load_in_4bit,
                self._settings.device_map,
            )
            self._tokenizer, self._model = _load_model(
                model_id=self._settings.model_id,
                load_in_4bit=self._settings.load_in_4bit,
                device_map=self._settings.device_map,
                hf_token=self._hf.token,
            )
        return self._model, self._tokenizer

    def complete(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        model, tokenizer = self._ensure_loaded()
        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self._settings.max_new_tokens,
                temperature=temperature or self._settings.temperature,
                do_sample=(temperature or self._settings.temperature) > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        if self._model is None and self._tokenizer is None:
            return
        logger.info("Unloading LLM '%s' from VRAM", self._settings.model_id)
        self._model = None
        self._tokenizer = None
        _load_model.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def _load_model(
    model_id: str, load_in_4bit: bool, device_map: str, hf_token: str | None
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model.eval()
    return tokenizer, model
