"""v6.0 collective concurrency: bucket-aware cap (not blind ws>=5 -> 1).

``MCCL_COLLECTIVE_CONCURRENCY=2`` with large DDP buckets at ws>=5 flooded TCP
demux / socket buffers (ENOBUFS).  v6.0 caps concurrency using
``MCCL_DEMUX_MAX_COLLECTIVE_BYTES`` (wire to ``DDP_BUCKET_MB``):

- ws>=8, bucket > 16 MiB  -> effective concurrency 1
- ws>=5, bucket > 25 MiB  -> effective concurrency 1
- smaller buckets         -> concurrency 2 allowed (ws>=5)

Larger *batch size* increases compute per step; it does **not** require higher
collective concurrency — that knob overlaps **multiple allreduces on the wire**,
which gets worse with larger buckets.
"""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="requires macOS Apple Silicon + MCCL",
)

# 16 MiB fp32
_NUMEL_16MB = 4 * 1024 * 1024
# 64 MiB fp32
_NUMEL_64MB = 16 * 1024 * 1024


def _dual_async_allreduce_fn(rank, world_size):
    """Two large async allreduces — exercises collective pool concurrency."""
    import os

    import torch
    import torch.distributed as dist

    n = int(os.environ["MCCL_TEST_NUMEL"])
    torch.manual_seed(900 + rank)
    t0 = torch.randn(n, device="mps", dtype=torch.float32) + float(rank)
    t1 = torch.randn(n, device="mps", dtype=torch.float32) + float(rank) * 0.5
    w0 = dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True)
    w1 = dist.all_reduce(t1, op=dist.ReduceOp.SUM, async_op=True)
    w0.wait()
    w1.wait()
    if rank == 0:
        print(
            f"[dual-async-ar] ws={world_size} numel={n} "
            f"bucket_mb={os.environ.get('MCCL_DEMUX_MAX_COLLECTIVE_BYTES', '?')}",
            flush=True,
        )


def _ddp_multibucket_steps_fn(rank, world_size):
    """DDP backward with many buckets — overlap comm with backward."""
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    steps = int(os.environ.get("MCCL_TEST_STEPS", "8"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "16"))
    ch = 96

    def stack():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
        )

    model = DDP(stack().to("mps"), bucket_cap_mb=bucket_mb)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(steps):
        g = torch.Generator().manual_seed(7000 + rank * 997 + step)
        x = torch.randn(4, 3, 96, 96, generator=g).to("mps")
        y = torch.randn(4, 3, 96, 96, generator=g).to("mps")
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        opt.step()
    if rank == 0:
        print(
            f"[ddp-multibucket] ws={world_size} steps={steps} bucket_mb={bucket_mb}",
            flush=True,
        )


_BASE = {
    "MCCL_OVERLAP_COMM": "1",
    "MCCL_RING_ALGO": "ring_chunked",
    "MCCL_RING_PIPELINE": "1",
    "MCCL_COLLECTIVE_CONCURRENCY": "2",
    "MCCL_PIPELINE_DEPTH": "2",
}


@pytest.mark.parametrize("world_size", [5, 8])
def test_conc2_small_bucket_dual_async_allreduce(world_size):
    """v6.0: concurrency=2 allowed when demux bucket cap <= 16 MiB at ws=8."""
    run_workers(
        _dual_async_allreduce_fn,
        world_size=world_size,
        env={
            **_BASE,
            "MCCL_TEST_NUMEL": str(_NUMEL_16MB),
            "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(16 * 1024 * 1024),
        },
        timeout=900,
    )


@pytest.mark.parametrize("world_size", [5, 8])
def test_conc1_large_bucket_dual_async_allreduce(world_size):
    """64 MiB buckets: effective concurrency must be 1 (serialized dual AR)."""
    run_workers(
        _dual_async_allreduce_fn,
        world_size=world_size,
        env={
            **_BASE,
            "MCCL_TEST_NUMEL": str(_NUMEL_64MB),
            "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(64 * 1024 * 1024),
        },
        timeout=1200,
    )


@pytest.mark.parametrize("world_size", [8])
def test_conc2_small_bucket_ddp_multibucket(world_size):
    """DDP 16 MB buckets + concurrency=2 request at ws=8."""
    run_workers(
        _ddp_multibucket_steps_fn,
        world_size=world_size,
        env={
            **_BASE,
            "MCCL_TEST_BUCKET_MB": "16",
            "MCCL_TEST_STEPS": "10",
            "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(16 * 1024 * 1024),
        },
        timeout=1200,
    )


@pytest.mark.parametrize("world_size", [8])
def test_conc4_small_bucket_ddp_multibucket(world_size):
    """8 MiB buckets: concurrency=4 request at ws=8 (budget allows 4+)."""
    run_workers(
        _ddp_multibucket_steps_fn,
        world_size=world_size,
        env={
            **_BASE,
            "MCCL_COLLECTIVE_CONCURRENCY": "4",
            "MCCL_TEST_BUCKET_MB": "8",
            "MCCL_TEST_STEPS": "8",
            "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(8 * 1024 * 1024),
        },
        timeout=1200,
    )


@pytest.mark.parametrize("world_size", [8])
def test_conc1_large_bucket_ddp_multibucket(world_size):
    """Prod-like 64 MB buckets: concurrency=2 env capped to 1 — must not ENOBUFS."""
    run_workers(
        _ddp_multibucket_steps_fn,
        world_size=world_size,
        env={
            **_BASE,
            "MCCL_TEST_BUCKET_MB": "64",
            "MCCL_TEST_STEPS": "6",
            "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(64 * 1024 * 1024),
        },
        timeout=1500,
    )
