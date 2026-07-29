"""Prove PyTorch MPS DDP grads cannot use cpu_ptr wire paths for MCCL collectives.

PyTorch MPS tensors on torch 2.11 (cluster) use ``MTLStorageModePrivate``:
``tensor_cpu_accessible()`` is false and there is no valid ``cpu_ptr`` for wire I/O.

Even when a newer PyTorch uses shared storage, ``stage_for_send_collective`` still
blits every MPS tensor (overlap-safe fence), and recv into ``cpu_ptr`` without a
GPU blit broke ring allreduce parity (MCCL v5.3 CHANGELOG).

These tests call the real C++ helpers in ``MPSInterop.mm`` via ``mccl._C``.
"""

from __future__ import annotations

import platform

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MPS storage tests require macOS Apple Silicon",
)


@pytest.fixture(scope="module")
def mccl_introspect():
    import mccl  # noqa: F401 — register backend

    from mccl._C import (
        _collective_send_uses_blit,
        _ensure_shared_storage,
        _mps_storage_mode,
        _stage_for_send_uses_blit,
        _tensor_cpu_accessible,
        _unstage_from_recv,
    )

    return {
        "cpu_accessible": _tensor_cpu_accessible,
        "storage_mode": _mps_storage_mode,
        "collective_blit": _collective_send_uses_blit,
        "send_blit": _stage_for_send_uses_blit,
        "unstage": _unstage_from_recv,
        "ensure_shared": _ensure_shared_storage,
        "torch": torch.__version__,
    }


def _backward_grads(module: nn.Module, x: torch.Tensor | None = None) -> list[torch.Tensor]:
    module.zero_grad(set_to_none=True)
    if x is None:
        x = torch.randn(2, 3, 32, 32, device="mps")
    module(x).sum().backward()
    return [p.grad for p in module.parameters() if p.grad is not None]


class TestMPSStorageLayout:
    """Document actual Metal storage for tensors MCCL reduces in DDP."""

    def test_conv_grad_collective_policy(self, mccl_introspect):
        conv = nn.Conv2d(3, 48, 3, padding=1).to("mps")
        grad = _backward_grads(conv)[0]
        mode = mccl_introspect["storage_mode"](grad)

        # torch 2.11 (cluster): private. torch 2.12+: may be shared.
        assert mode in ("private", "shared")
        assert mccl_introspect["collective_blit"](grad)

        if mode == "private":
            assert not mccl_introspect["cpu_accessible"](grad)
        else:
            assert mccl_introspect["cpu_accessible"](grad)

    def test_linear_grad_collective_policy(self, mccl_introspect):
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to("mps")
        x = torch.randn(4, 256, device="mps")
        grad = _backward_grads(model, x)[0]
        assert mccl_introspect["collective_blit"](grad)


class TestCollectiveStagingPolicy:
    """MCCL send policy for tensors DDP actually hands to allreduce."""

    def test_mps_grad_always_blits_for_collective_send(self, mccl_introspect):
        linear = nn.Linear(128, 64).to("mps")
        x = torch.randn(8, 128, device="mps")
        grad = _backward_grads(linear, x)[0]
        assert mccl_introspect["collective_blit"](grad)

    def test_collective_blit_even_when_cpu_accessible(self, mccl_introspect):
        """stage_for_send_collective blits all MPS tensors (not stage_for_send)."""
        t = torch.randn(64, device="mps")
        if mccl_introspect["storage_mode"](t) == "shared":
            assert mccl_introspect["cpu_accessible"](t)
            assert mccl_introspect["collective_blit"](t)
            assert not mccl_introspect["send_blit"](t)
        else:
            assert mccl_introspect["collective_blit"](t)
            assert mccl_introspect["send_blit"](t)

    def test_cpu_metadata_direct_path(self, mccl_introspect):
        meta = torch.zeros(8, dtype=torch.int64, device="cpu")
        assert mccl_introspect["storage_mode"](meta) == "cpu"
        assert mccl_introspect["cpu_accessible"](meta)
        assert not mccl_introspect["collective_blit"](meta)
        assert not mccl_introspect["send_blit"](meta)


class TestRecvBlitRequired:
    """Wire recv must populate GPU-visible bytes (blit path on private storage)."""

    def test_unstage_blit_updates_gpu(self, mccl_introspect):
        from mccl._C import _metal_accumulate_chunk

        dst = torch.ones(256, device="mps")
        torch.mps.synchronize()
        wire = torch.full((256,), 7.0, device="cpu", dtype=torch.float32)
        mccl_introspect["unstage"](dst, wire, cpu_unified_stage=False)

        acc = torch.zeros(256, device="mps")
        _metal_accumulate_chunk(acc, dst)
        assert float(acc.sum().cpu()) == pytest.approx(256 * 7.0)

    def test_private_storage_unstage_uses_blit_not_cpu_ptr(self, mccl_introspect):
        """cpu_unified_stage is ignored when storage is private; blit still updates GPU."""
        dst = torch.ones(128, device="mps")
        if mccl_introspect["storage_mode"](dst) != "private":
            pytest.skip(f"torch {mccl_introspect['torch']}: MPS storage is shared, not private")
        assert not mccl_introspect["cpu_accessible"](dst)
        torch.mps.synchronize()
        wire = torch.full((128,), 9.0, device="cpu", dtype=torch.float32)
        mccl_introspect["unstage"](dst, wire, cpu_unified_stage=True)
        torch.mps.synchronize()
        assert float(dst.mean().cpu()) == pytest.approx(9.0)


class TestEnsureSharedIsOptIn:
    """Private GPU storage needs an explicit blit to make a CPU/shared view."""

    def test_ensure_shared_storage_blits_private_to_cpu(self, mccl_introspect):
        src = torch.arange(8, device="mps", dtype=torch.float32)
        if mccl_introspect["storage_mode"](src) != "private":
            pytest.skip(f"torch {mccl_introspect['torch']}: already shared MPS storage")
        shared = mccl_introspect["ensure_shared"](src)
        assert shared.device.type == "cpu"
        assert mccl_introspect["cpu_accessible"](shared)
        assert torch.allclose(shared, src.cpu())

    def test_ensure_shared_noop_when_already_accessible(self, mccl_introspect):
        src = torch.arange(8, device="mps", dtype=torch.float32)
        if not mccl_introspect["cpu_accessible"](src):
            pytest.skip(f"torch {mccl_introspect['torch']}: private storage only")
        out = mccl_introspect["ensure_shared"](src)
        assert out.data_ptr() == src.data_ptr()
