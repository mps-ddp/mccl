"""Stress + parity tests for MCCL_UNIFIED_COLLECTIVE (shared-storage fast path).

Unified collective is default-on; this matrix pins ``MCCL_UNIFIED_COLLECTIVE=1``
and ``=0`` explicitly.  Runs the same race/DDP workloads as the Metal+blit path
and verifies bit-exact or rtol parity.  Fails loudly if unified cpu_ptr staging
corrupts gradients under overlap.
"""

from __future__ import annotations

import platform

import pytest
import torch

from mccl_test_utils import _submit_job_mccl_env, run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="unified collective tests require macOS Apple Silicon",
)

_SUBMIT = _submit_job_mccl_env(25)

_UNIFIED_ENV = {
    **_SUBMIT,
    "MCCL_UNIFIED_COLLECTIVE": "1",
    "MCCL_FP32_CPU_REDUCE": "0",
}

_BLIT_ENV = {
    **_SUBMIT,
    "MCCL_UNIFIED_COLLECTIVE": "0",
    "MCCL_FP32_CPU_REDUCE": "0",
}


def _require_shared_mps_storage():
    t = torch.randn(8, device="mps")
    from mccl._C import _mps_storage_mode, _tensor_cpu_accessible

    if _mps_storage_mode(t) != "shared" or not _tensor_cpu_accessible(t):
        pytest.skip(
            f"torch {torch.__version__}: MPS storage is not shared "
            "(need torch 2.12+ for unified collective fast path)"
        )


def _fp32_ring_race_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    n = 150_001
    iters = 50
    a = torch.randn(256, 256, device="mps")
    for it in range(iters):
        val = float(it + 1)
        t = torch.full((n,), val * (rank + 1), dtype=torch.float32, device="mps")
        _ = (a @ a).sum()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = val * sum(r + 1 for r in range(world_size))
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"iter {it}: {bad}/{n} bad elems"


def _async_buckets_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    sizes = [70_001, 131_072, 99_999, 262_144, 12_345] * 6
    tensors, works = [], []
    a = torch.randn(128, 128, device="mps")
    for i, n in enumerate(sizes):
        t = torch.full((n,), float(i + 1) * (rank + 1), dtype=torch.float32, device="mps")
        _ = a @ a
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))
    for w in works:
        w.wait()
    total = sum(r + 1 for r in range(world_size))
    for i, (n, t) in enumerate(zip(sizes, tensors)):
        bad = (t.cpu() != float(i + 1) * total).sum().item()
        assert bad == 0, f"bucket {i}: {bad}/{n} bad"


def _ddp_conv_overlap_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    iters = int(os.environ.get("MCCL_TEST_ITERS", "20"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "1"))
    ch, sp, bs = 48, 128, 4
    torch.manual_seed(42)

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
        )

    model = make_model().to("mps")
    ddp = DDP(model, bucket_cap_mb=bucket_mb)

    def batch_for(r):
        g = torch.Generator().manual_seed(2000 + r)
        return (
            torch.randn(bs, 3, sp, sp, generator=g),
            torch.randn(bs, 3, sp, sp, generator=g),
        )

    for it in range(iters):
        x, y = batch_for((rank + it) % world_size)
        ddp.zero_grad(set_to_none=True)
        loss = F.mse_loss(ddp(x.to("mps")), y.to("mps"))
        loss.backward()

    torch.manual_seed(42)
    ref = make_model().to("mps")
    ref.zero_grad(set_to_none=True)
    last = iters - 1
    losses = []
    for r in range(world_size):
        xr, yr = batch_for((r + last) % world_size)
        losses.append(F.mse_loss(ref(xr.to("mps")), yr.to("mps")))
    (sum(losses) / world_size).backward()

    rtol = 2e-3
    for (_, p_ddp), (_, p_ref) in zip(ddp.module.named_parameters(), ref.named_parameters()):
        if p_ddp.grad is None:
            continue
        rel = (p_ddp.grad.cpu() - p_ref.grad.cpu()).abs().max().item() / (
            p_ref.grad.cpu().abs().max().item() + 1e-8
        )
        assert rel < rtol, f"grad rel {rel:.2e}"


class TestUnifiedCollectiveStress:
    @pytest.fixture(autouse=True)
    def _shared_only(self):
        _require_shared_mps_storage()

    @pytest.mark.parametrize("world_size", [2, 3, 4])
    def test_fp32_ring_under_gpu_load(self, world_size):
        run_workers(_fp32_ring_race_fn, world_size=world_size, env=_UNIFIED_ENV, timeout=600)

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_async_bucket_stream(self, world_size):
        run_workers(_async_buckets_fn, world_size=world_size, env=_UNIFIED_ENV, timeout=600)

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    def test_ddp_conv_overlap_parity(self, world_size):
        run_workers(_ddp_conv_overlap_fn, world_size=world_size, env=_UNIFIED_ENV, timeout=900)

    def test_unified_matches_blit_under_submit_env(self):
        """Regression: unified Metal path must match blit+Metal (not CPU reduce)."""
        _require_shared_mps_storage()

        def once_fn(rank, world_size):
            import os

            import torch
            import torch.distributed as dist

            n = int(os.environ.get("MCCL_TEST_N", "262144"))
            torch.manual_seed(99 + rank)
            t = torch.randn(n, device="mps", dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out = os.environ["MCCL_TEST_OUT"]
            torch.save(t.cpu(), f"{out}/r{rank}.pt")

        import tempfile
        from pathlib import Path

        out = Path(tempfile.gettempdir()) / "mccl_unified_metal_parity"
        out.mkdir(parents=True, exist_ok=True)
        base = {**_SUBMIT, "MCCL_TEST_N": "262144", "MCCL_TEST_OUT": str(out)}
        for tag, env in (("blit", _BLIT_ENV), ("uni", _UNIFIED_ENV)):
            tag_out = out / tag
            tag_out.mkdir(parents=True, exist_ok=True)
            run_workers(
                once_fn,
                world_size=4,
                env={**base, "MCCL_TEST_OUT": str(tag_out), **env},
                timeout=300,
            )
        for r in range(4):
            blit = torch.load(out / "blit" / f"r{r}.pt", weights_only=True)
            uni = torch.load(out / "uni" / f"r{r}.pt", weights_only=True)
            assert torch.allclose(blit, uni, rtol=0, atol=0), f"rank {r} mismatch"


class TestUnifiedVsBlitParity:
    """Same seeds: unified path must match default Metal+blit allreduce."""

    def test_ws2_allreduce_matches_default(self):
        _require_shared_mps_storage()
        import tempfile
        from pathlib import Path

        out = Path(tempfile.gettempdir()) / "mccl_unified_parity"

        def save_fn(rank, world_size):
            import os
            from pathlib import Path

            import torch
            import torch.distributed as dist

            tag = os.environ["MCCL_TEST_TAG"]
            sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
            out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])
            for n in sizes:
                torch.manual_seed(4242 + n + rank)
                t = torch.randn(n, device="mps", dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(t.cpu(), out_dir / f"{tag}_n{n}_r{rank}.pt")

        def compare_fn(rank, world_size):
            import os
            from pathlib import Path

            import torch

            sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
            out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])
            for n in sizes:
                blit = torch.load(out_dir / f"blit_n{n}_r{rank}.pt", weights_only=True)
                uni = torch.load(out_dir / f"unified_n{n}_r{rank}.pt", weights_only=True)
                if not torch.allclose(blit, uni, rtol=0, atol=0):
                    err = (blit - uni).abs().max().item()
                    raise AssertionError(f"n={n} rank={rank} max_err={err}")

        base = {
            "MCCL_TEST_SIZES": "4096,65537,1048576",
            "MCCL_TEST_OUT_DIR": str(out),
            "MCCL_FP32_CPU_REDUCE": "0",
            **_SUBMIT,
        }
        run_workers(save_fn, world_size=2, env={**base, "MCCL_TEST_TAG": "blit", "MCCL_UNIFIED_COLLECTIVE": "0"}, timeout=300)
        run_workers(
            save_fn,
            world_size=2,
            env={**base, "MCCL_TEST_TAG": "unified", "MCCL_UNIFIED_COLLECTIVE": "1"},
            timeout=300,
        )
        run_workers(compare_fn, world_size=2, env=base, timeout=120)
