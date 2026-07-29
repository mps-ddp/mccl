"""STFT / iSTFT with vDSP or Metal backends (parity-tested vs CPU torch.stft)."""
from __future__ import annotations

import os
from typing import Optional

import torch

_BACKENDS = frozenset({"vdsp", "metal", "auto"})


def resolve_backend(name: Optional[str]) -> str:
    raw = (name or os.environ.get("MCCL_STFT_BACKEND", "auto")).strip().lower()
    if raw not in _BACKENDS:
        raise ValueError(f"MCCL STFT backend must be one of {sorted(_BACKENDS)}, got {name!r}")
    return raw


class _StftFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        waveform: torch.Tensor,
        window: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool,
        normalized: bool,
        backend: str,
    ):
        from mccl import _C

        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        waveform = waveform.contiguous()
        window = window.contiguous().to(device=waveform.device, dtype=waveform.dtype)

        spec = _C._stft_forward(
            waveform,
            window,
            int(n_fft),
            int(hop_length),
            int(win_length),
            bool(center),
            bool(normalized),
            resolve_backend(backend),
        )
        ctx.save_for_backward(window)
        ctx.n_fft = int(n_fft)
        ctx.hop_length = int(hop_length)
        ctx.win_length = int(win_length)
        ctx.center = bool(center)
        ctx.normalized = bool(normalized)
        ctx.signal_length = int(waveform.shape[-1])
        ctx.backend = resolve_backend(backend)
        return spec

    @staticmethod
    def backward(ctx, grad_spec: torch.Tensor):
        from mccl import _C

        (window,) = ctx.saved_tensors
        grad_waveform = _C._stft_backward(
            grad_spec.contiguous(),
            window,
            ctx.n_fft,
            ctx.hop_length,
            ctx.win_length,
            ctx.center,
            ctx.normalized,
            ctx.signal_length,
            ctx.backend,
        )
        return grad_waveform, None, None, None, None, None, None, None


class _IstftFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        spec: torch.Tensor,
        window: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        length: int,
        center: bool,
        normalized: bool,
        backend: str,
    ):
        from mccl import _C

        window = window.contiguous().to(device=spec.device, dtype=spec.real.dtype)
        wav = _C._istft_forward(
            spec.contiguous(),
            window,
            int(n_fft),
            int(hop_length),
            int(win_length),
            bool(center),
            bool(normalized),
            int(length),
            resolve_backend(backend),
        )
        ctx.save_for_backward(window)
        ctx.n_fft = int(n_fft)
        ctx.hop_length = int(hop_length)
        ctx.win_length = int(win_length)
        ctx.length = int(length)
        ctx.center = bool(center)
        ctx.normalized = bool(normalized)
        ctx.n_frames = int(spec.shape[-1])
        ctx.backend = resolve_backend(backend)
        return wav.unsqueeze(1) if wav.dim() == 2 else wav

    @staticmethod
    def backward(ctx, grad_waveform: torch.Tensor):
        from mccl import _C

        (window,) = ctx.saved_tensors
        if grad_waveform.dim() == 3 and grad_waveform.size(1) == 1:
            grad_waveform = grad_waveform.squeeze(1)
        grad_spec = _C._istft_backward(
            grad_waveform.contiguous(),
            window,
            ctx.n_fft,
            ctx.hop_length,
            ctx.win_length,
            ctx.center,
            ctx.normalized,
            ctx.length,
            ctx.backend,
        )
        return grad_spec, None, None, None, None, None, None, None, None


def stft(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool = True,
    normalized: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Complex STFT ``[batch, freq, frames]`` on MPS via MCCL transforms."""
    return _StftFn.apply(
        waveform,
        window,
        n_fft,
        hop_length,
        win_length,
        center,
        normalized,
        backend,
    )


def istft(
    spec: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
    center: bool = True,
    normalized: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Inverse STFT returning waveform ``[batch, time]`` or ``[batch, 1, time]``."""
    return _IstftFn.apply(
        spec,
        window,
        n_fft,
        hop_length,
        win_length,
        length,
        center,
        normalized,
        backend,
    )
