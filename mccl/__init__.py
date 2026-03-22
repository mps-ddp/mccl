"""
MCCL — MPS-native ProcessGroup backend for PyTorch Distributed.

Usage::

    import mccl
    mccl.init(compression="fp16", fast_math=True)   # optional — sets config
    torch.distributed.init_process_group(backend="mccl", ...)
"""
from __future__ import annotations

from mccl.version import __version__, COMPATIBILITY_MATRIX
from mccl.config import MCCLConfig
from mccl.tuning import apply_thunderbolt_production_defaults

import platform
import warnings
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mccl._C import MetricsSummary

# ── Active state ─────────────────────────────────────────────────────

_active_config: Optional[MCCLConfig] = None
_backend_registered: bool = False


# ── Platform check ───────────────────────────────────────────────────

def _check_platform():
    if platform.system() != "Darwin":
        warnings.warn(
            "MCCL is designed for macOS on Apple Silicon. "
            "Import will succeed but the native extension cannot load.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        warnings.warn(
            "MCCL requires Apple Silicon (arm64). "
            "Detected: " + platform.machine(),
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


_platform_ok = _check_platform()

if _platform_ok:
    try:
        import torch.distributed  # registers c10d base types in pybind11
        from mccl._C import _register_backend
        _register_backend()
        _backend_registered = True
    except ImportError as e:
        warnings.warn(
            f"MCCL native extension not found: {e}. "
            "Build with: pip install -e . (on macOS/Apple Silicon)",
            RuntimeWarning,
            stacklevel=2,
        )


# ── Public API ───────────────────────────────────────────────────────

def init(
    config: MCCLConfig | Dict[str, Any] | None = None,
    **kwargs: Any,
) -> MCCLConfig:
    """Configure MCCL before ``dist.init_process_group(backend="mccl")``.

    Accepts an :class:`MCCLConfig`, a plain dict, or keyword arguments.
    Writes the resolved values into ``os.environ`` so the C++ layer picks
    them up at ProcessGroup creation time.

    Returns the active config object.

    Examples::

        # Keyword shorthand
        mccl.init(compression="fp16")

        # Full config object
        cfg = mccl.MCCLConfig(fast_math=False, gpu_threshold=8192)
        mccl.init(cfg)

        # Dict (e.g. from YAML)
        mccl.init({"compression": "topk", "topk_ratio": 0.05})
    """
    global _active_config

    if config is None and kwargs:
        config = MCCLConfig(**kwargs)
    elif isinstance(config, dict):
        merged = dict(config)
        merged.update(kwargs)
        config = MCCLConfig.from_dict(merged)
    elif config is None:
        config = MCCLConfig.from_env()
    elif kwargs:
        for k, v in kwargs.items():
            setattr(config, k, v)

    config.to_env()
    _active_config = config
    return config


def get_config() -> Optional[MCCLConfig]:
    """Return the active config, or ``None`` if :func:`init` was not called."""
    return _active_config


def get_metrics() -> Optional["MetricsSummary"]:
    """Return a snapshot of collective-level metrics from the C++ backend.

    Returns ``None`` if the native extension is not loaded or if no
    ProcessGroup has been created yet.
    """
    try:
        from mccl._C import _get_metrics_summary
        return _get_metrics_summary()
    except (ImportError, AttributeError, RuntimeError):
        return None


def log_metrics() -> None:
    """Dump the current metrics summary to stderr via the C++ logger."""
    try:
        from mccl._C import _log_metrics
        _log_metrics()
    except (ImportError, AttributeError, RuntimeError):
        pass


def reset_metrics() -> None:
    """Reset all counters in the C++ metrics collector."""
    try:
        from mccl._C import _reset_metrics
        _reset_metrics()
    except (ImportError, AttributeError, RuntimeError):
        pass


__all__ = [
    "__version__",
    "COMPATIBILITY_MATRIX",
    "MCCLConfig",
    "init",
    "get_config",
    "get_metrics",
    "log_metrics",
    "reset_metrics",
    "apply_thunderbolt_production_defaults",
]
