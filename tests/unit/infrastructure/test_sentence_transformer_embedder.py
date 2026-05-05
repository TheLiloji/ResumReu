"""Embedding adapter regression tests."""

from __future__ import annotations

import ast
import inspect

import torch

from src.infrastructure.llm import sentence_transformer_embedder


def test_embedder_does_not_import_sentence_transformers_package() -> None:
    source = inspect.getsource(sentence_transformer_embedder)
    tree = ast.parse(source)

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert "sentence_transformers" not in imported_modules


def test_cuda_embedding_device_falls_back_to_cpu_when_unavailable(monkeypatch: object) -> None:
    monkeypatch.setattr(sentence_transformer_embedder.torch.cuda, "is_available", lambda: False)

    device = sentence_transformer_embedder._resolve_device("cuda")

    assert device == torch.device("cpu")
