"""Helpers for safe parameter layouts on MPS (optimizer + MCCL)."""
from __future__ import annotations

import warnings

import torch.nn as nn


def ensure_contiguous_parameters(
    module: nn.Module,
    *,
    warn: bool = True,
) -> list[str]:
    """Re-layout ``param.data`` to contiguous storage when needed.

    Non-contiguous parameter buffers break ``torch.optim.Adam`` on MPS
    (``addcmul_`` / ``addcdiv_`` can silently skip updates) and force extra
    copies in MCCL collectives. Call once after model construction and
    **before** ``DistributedDataParallel`` / the optimizer.

    Returns names of parameters that were re-laid-out.
    """
    fixed: list[str] = []
    for name, param in module.named_parameters():
        if param.data.is_contiguous():
            continue
        if warn:
            warnings.warn(
                f"Parameter {name!r} is non-contiguous; calling .contiguous() "
                "(required for reliable Adam on MPS).",
                RuntimeWarning,
                stacklevel=2,
            )
        param.data = param.data.contiguous()
        fixed.append(name)
    return fixed


__all__ = ["ensure_contiguous_parameters"]
