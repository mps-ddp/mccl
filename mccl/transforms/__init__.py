"""MCCL transforms — Metal-native DSP ops with CPU-reference parity."""
from __future__ import annotations

import os
import platform

from mccl.transforms.stft import istft, resolve_backend, stft

__all__ = ["stft", "istft", "resolve_backend", "is_available"]


def is_available() -> bool:
    """True when the native extension is built on Apple Silicon."""
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return False
    try:
        from mccl import _C  # noqa: F401

        return hasattr(_C, "_stft_forward")
    except Exception:
        return False


def default_backend() -> str:
    return resolve_backend(os.environ.get("MCCL_STFT_BACKEND", "auto"))
