"""Preload the CUDA 12 shared libraries that CTranslate2 needs.

faster-whisper -> CTranslate2 4.7.x is compiled against CUDA 12 and looks for
libcublas.so.12 / libcudnn.so.9 at runtime. When PyTorch is built against
CUDA 13 (as in this project), only CUDA 13 versions live on LD_LIBRARY_PATH,
so dlopen("libcublas.so.12") fails with `Library libcublas.so.12 is not found`.

We install the CUDA 12 wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) into
the venv. Their .so files sit under `site-packages/nvidia/.../lib/`. Importing
this module loads them via `ctypes.CDLL(..., RTLD_GLOBAL)`, which makes them
available to any subsequent dlopen() — including the one CTranslate2 performs.

This is idempotent: a second call is a no-op because the loader keeps a refcount.
Failures are non-fatal so CPU-only environments still work.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LIBS_TO_PRELOAD: tuple[tuple[str, str], ...] = (
    # (package directory under site-packages/nvidia/, .so filename)
    #
    # CUDA 12 libs — required by CTranslate2 (faster-whisper backend),
    # which is built against CUDA 12 even when PyTorch ships CUDA 13.
    ("cublas", "libcublas.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cudnn", "libcudnn.so.9"),
    ("cudnn", "libcudnn_ops.so.9"),
    ("cudnn", "libcudnn_cnn.so.9"),
    ("cuda_nvrtc", "libnvrtc.so.12"),
    # CUDA 13 libs — `libnvrtc.so.13` (loaded by torch) lazily dlopens
    # `libnvrtc-builtins.so.13.0` by relative name when JIT-compiling kernels.
    # The file lives under `nvidia/cu13/lib/` which is not on LD_LIBRARY_PATH,
    # so we preload it here with RTLD_GLOBAL to make it visible to the loader.
    ("cu13", "libnvrtc-builtins.so.13.0"),
    ("cu13", "libnvrtc.so.13"),
)


def preload_cuda12_libs() -> None:
    """Load CUDA 12 .so files from the venv into the global symbol table."""
    base = _nvidia_site_packages_dir()
    if base is None:
        return
    for pkg, lib in _LIBS_TO_PRELOAD:
        path = base / pkg / "lib" / lib
        if not path.exists():
            logger.debug("CUDA bootstrap: %s not found, skipping", path)
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            logger.debug("CUDA bootstrap: loaded %s", path.name)
        except OSError as exc:
            logger.warning("CUDA bootstrap: failed to load %s: %s", path, exc)


def _nvidia_site_packages_dir() -> Path | None:
    """Locate the venv's `site-packages/nvidia/` directory.

    `nvidia` is a PEP 420 namespace package — it has no `__file__`, only
    a `__path__` containing one or more directory entries.
    """
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        return None
    paths = list(getattr(nvidia, "__path__", []) or [])
    for entry in paths:
        candidate = Path(entry)
        if candidate.is_dir():
            return candidate
    return None


# Run once on import.
preload_cuda12_libs()
