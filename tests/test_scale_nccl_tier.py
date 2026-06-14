"""NCCL-tier scale tests: large buckets, high concurrency, pipeline depth,
transport metrics, and multi-rank (ws=8) smoke correctness."""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _large_bucket_allreduce_fn(rank, world_size):
    """Single collective at ~25 MB — exercises credit flow and demux at scale."""
    import torch
    import torch.distributed as dist

    n = 6_500_000  # ~26 MB fp32
    t = torch.full((n,), float(rank + 1), dtype=torch.float32, device="mps")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(r + 1 for r in range(world_size))
    bad = (t.cpu() != expected).sum().item()
    assert bad == 0, f"{bad} corrupted elements in large bucket allreduce"


def _pipeline_depth_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    sizes = [70_001, 400_000, 1_200_000]
    for n in sizes:
        t = torch.full((n,), float(n % 17 + 1) * (rank + 1),
                       dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(n % 17 + 1) * sum(r + 1 for r in range(world_size))
        assert (t.cpu() == expected).all(), f"size {n} corrupted"


def _concurrency_stream_fn(rank, world_size):
    """Many async allreduces with elevated collective concurrency."""
    import torch
    import torch.distributed as dist

    sizes = [500_000, 800_001, 1_000_000] * 4
    tensors, works = [], []
    for i, n in enumerate(sizes):
        t = torch.full((n,), float(i + 1) * (rank + 1), dtype=torch.float32, device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))
    for w in works:
        w.wait()
    total = sum(r + 1 for r in range(world_size))
    for i, t in enumerate(tensors):
        assert (t.cpu() == float(i + 1) * total).all(), f"bucket {i} corrupted"


def _transport_metrics_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    import mccl

    for _ in range(3):
        t = torch.full((800_000,), float(rank + 1), dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    m = mccl.get_metrics()
    assert m is not None
    assert m.total_ops >= 3
    assert m.demux_zerocopy_hits + m.demux_park_hits > 0, (
        "demux metrics did not record any routed payloads"
    )
    assert m.avg_sync_ms >= 0.0
    if rank == 0:
        print(
            f"\n[transport metrics] zerocopy={m.demux_zerocopy_hits} "
            f"park={m.demux_park_hits} peak_parked={m.demux_parked_bytes_peak} "
            f"credit_wait_ms={m.total_credit_wait_ms:.3f}",
            flush=True,
        )
    dist.barrier()


def _broadcast_ws3_fn(rank, world_size):
    """ws=3 uses tree/ring broadcast (not root fan-out) since NCCL-tier change."""
    import torch
    import torch.distributed as dist

    assert world_size == 3
    n = 500_000  # large enough for ring broadcast path
    b = (
        torch.arange(n, dtype=torch.float32, device="mps")
        if rank == 0
        else torch.zeros(n, dtype=torch.float32, device="mps")
    )
    dist.broadcast(b, src=0)
    assert (b.cpu() == torch.arange(n, dtype=torch.float32)).all()


def _reduce_scatter_input_preserved_fn(rank, world_size):
    """reduce_scatter must not mutate caller input tensors."""
    import torch
    import torch.distributed as dist

    n = 50_000
    inputs = [
        torch.full((n,), float(r + 1) * 10.0, dtype=torch.float32, device="mps")
        for r in range(world_size)
    ]
    snapshots = [t.clone() for t in inputs]
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM)
    for r, (t, snap) in enumerate(zip(inputs, snapshots)):
        assert (t.cpu() == snap.cpu()).all(), f"input slot {r} was mutated"

    avg_out = torch.zeros(n, dtype=torch.float32, device="mps")
    dist.reduce_scatter(avg_out, inputs, op=dist.ReduceOp.AVG)
    expected = sum(float(r + 1) * 10.0 for r in range(world_size)) / world_size
    assert (avg_out.cpu() == expected).all(), "reduce_scatter AVG not scaled"


def _peer_death_ws4_fn(rank, world_size):
    import os
    import sys
    import time
    import torch
    import torch.distributed as dist

    t = torch.ones(4096, device="mps")
    dist.all_reduce(t)
    if rank == 2:
        os._exit(0)
    time.sleep(1.5)
    t2 = torch.ones(200_000, device="mps")
    try:
        work = dist.all_reduce(t2, async_op=True)
        work.wait()
    except Exception as e:
        print(f"rank {rank}: expected error: {e}", file=sys.stderr)
        os._exit(0)
    try:
        dist.barrier()
    except Exception:
        os._exit(0)
    raise AssertionError("collectives succeeded after peer death at ws=4")


class TestLargeBuckets:
    @pytest.mark.parametrize("world_size", [3, 4])
    def test_large_allreduce(self, world_size):
        run_workers(_large_bucket_allreduce_fn, world_size=world_size, timeout=420)


class TestPipelineDepth:
    @pytest.mark.parametrize("depth", ["1", "3", "4"])
    @pytest.mark.parametrize("world_size", [4])
    def test_depth_sweep(self, world_size, depth):
        run_workers(
            _pipeline_depth_fn,
            world_size=world_size,
            env={"MCCL_PIPELINE_DEPTH": depth},
            timeout=420,
        )


class TestHighConcurrency:
    @pytest.mark.parametrize("concurrency", ["3", "4"])
    @pytest.mark.parametrize("world_size", [4])
    def test_concurrency_sweep(self, world_size, concurrency):
        run_workers(
            _concurrency_stream_fn,
            world_size=world_size,
            env={"MCCL_COLLECTIVE_CONCURRENCY": concurrency},
            timeout=420,
        )


class TestTransportMetrics:
    @pytest.mark.parametrize("world_size", [4])
    def test_demux_metrics_populated(self, world_size):
        run_workers(_transport_metrics_fn, world_size=world_size, timeout=300)


class TestBroadcastScale:
    def test_broadcast_ring_at_ws3(self):
        run_workers(_broadcast_ws3_fn, world_size=3, timeout=300)


class TestReduceScatterContract:
    @pytest.mark.parametrize("world_size", [3, 4])
    def test_inputs_not_mutated(self, world_size):
        run_workers(_reduce_scatter_input_preserved_fn, world_size=world_size, timeout=300)


class TestPeerDeathMultiRank:
    @pytest.mark.parametrize("world_size", [4])
    def test_peer_death_clean_abort(self, world_size):
        run_workers(_peer_death_ws4_fn, world_size=world_size, timeout=180)


class TestWorldSize8Smoke:
    @pytest.mark.timeout(600)
    def test_allreduce_smoke(self):
        def fn(rank, world_size):
            import torch
            import torch.distributed as dist

            assert world_size == 8
            for it in range(3):
                t = torch.full((50_000,), float(it + 1) * (rank + 1),
                               dtype=torch.float32, device="mps")
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                expected = float(it + 1) * sum(r + 1 for r in range(world_size))
                assert (t.cpu() == expected).all()

        run_workers(fn, world_size=8, timeout=540)
