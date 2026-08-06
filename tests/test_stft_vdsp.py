"""MCCL vDSP STFT parity + lightweight perf guard."""
from __future__ import annotations

import time

import pytest
import torch

from mccl.transforms.stft import istft, stft


def _hann(win: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(win, periodic=True, device=device)


def _grad_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().flatten().float()
    bf = b.detach().flatten().float()
    return float((af @ bf).abs() / (af.norm() * bf.norm() + 1e-12))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
@pytest.mark.parametrize("n_fft,hop,win", [(2048, 512, 2048), (1024, 256, 1024)])
def test_vdsp_mps_parity(n_fft: int, hop: int, win: int) -> None:
    device = torch.device("mps")
    batch, seconds = 2, 1
    sr = 48000
    t = seconds * sr
    w = _hann(win, device)
    x = torch.randn(batch, t, device=device, dtype=torch.float32)

    xref = x.detach().cpu()
    wref = w.detach().cpu()
    ref = torch.stft(
        xref,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=wref,
        center=True,
        return_complex=True,
    )
    got = stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=w,
        backend="vdsp",
    )
    assert torch.isfinite(ref).all(), "CPU STFT reference produced non-finite values"
    assert torch.isfinite(got).all(), "MCCL float32 STFT produced non-finite values"
    assert float((ref - got.detach().cpu()).abs().max()) < 1e-4

    xi = x.detach().clone().requires_grad_(True)
    stft(
        xi,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=w,
        backend="vdsp",
    ).abs().sum().backward()
    xref_g = xref.clone().requires_grad_(True)
    torch.stft(
        xref_g,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=wref,
        center=True,
        return_complex=True,
    ).abs().sum().backward()
    assert torch.isfinite(xref_g.grad).all(), (
        "CPU STFT reference produced non-finite input gradients"
    )
    assert torch.isfinite(xi.grad).all(), (
        "MCCL float32 STFT produced non-finite input gradients"
    )
    assert _grad_cosine(xref_g.grad, xi.grad.cpu()) > 0.75

    spec = stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=w,
        backend="vdsp",
    )
    wav = istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=w,
        length=t,
        backend="vdsp",
    )
    assert wav.shape[-1] == t
    assert torch.isfinite(wav).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_vdsp_mps_fwd_bwd_under_budget() -> None:
    """Smoke perf: full fwd+bwd should stay well under 1s for 1s audio @ 2048."""
    device = torch.device("mps")
    n_fft, hop, win = 2048, 512, 2048
    w = _hann(win, device)
    x = torch.randn(2, 48000, device=device, dtype=torch.float32)

    def step() -> None:
        xi = x.detach().clone().requires_grad_(True)
        stft(
            xi,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=w,
            backend="vdsp",
        ).abs().sum().backward()
        if device.type == "mps":
            torch.mps.synchronize()

    for _ in range(3):
        step()
    t0 = time.perf_counter()
    for _ in range(10):
        step()
    if device.type == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    assert ms < 1000.0, f"vDSP fwd+bwd too slow: {ms:.1f} ms"
