"""Pipelined-ring correctness at ws=4/6 (the NCCL-grade ws>=3 hot paths).

Covers the streaming TX/RX pipeline (default), the lock-step fallback
(MCCL_RING_PIPELINE=0) for differential debugging, concurrent collectives
(MCCL_COLLECTIVE_CONCURRENCY=2), and the demux transport's ability to
interleave p2p traffic with a collective on the same link.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)

# Sizes spanning small (tree), ring with even chunks, odd numel, and
# chunk-boundary (+/-1 around tensor/(2*ws) splits).
RING_SIZES = "70001,131072,262147,393216,524289"


def _allreduce_pipeline_fn(rank, world_size):
    import os
    import torch
    import torch.distributed as dist

    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    dtype = getattr(torch, os.environ.get("MCCL_TEST_DTYPE", "float32"))

    for n in sizes:
        base = torch.arange(n, dtype=torch.float64) % 11 - 5.0
        mine = (base + float(rank + 1)).to(dtype).to("mps")

        expected = sum(
            (base + float(r + 1)).to(dtype).to(torch.float64)
            for r in range(world_size)
        )

        dist.all_reduce(mine, op=dist.ReduceOp.SUM)
        got = mine.cpu().to(torch.float64)
        tol = {torch.float32: 1e-5, torch.float16: 1e-2}[dtype]
        max_err = (got - expected).abs().max().item()
        assert max_err <= tol, f"n={n} dtype={dtype}: max_err={max_err}"


def _determinism_pipeline_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    n = 300_007
    torch.manual_seed(99 + rank)
    src = torch.randn(n, dtype=torch.float32)

    results = []
    for _ in range(3):
        t = src.clone().to("mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        results.append(t.cpu())

    assert torch.equal(results[0], results[1]) and torch.equal(results[1], results[2]), \
        "pipelined ring allreduce is nondeterministic"


def _rs_ag_pipeline_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    n = 131_072
    # reduce_scatter
    ins = [
        torch.full((n,), float(rank + 1) * (i + 1), dtype=torch.float32, device="mps")
        for i in range(world_size)
    ]
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    dist.reduce_scatter(out, ins, op=dist.ReduceOp.SUM)
    expected = sum(float(r + 1) * (rank + 1) for r in range(world_size))
    assert (out.cpu() == expected).all(), f"reduce_scatter rank {rank}"

    # allgather (fp32 zero-copy receive path)
    mine = (torch.arange(n, dtype=torch.float32) + 1000.0 * (rank + 1)).to("mps")
    outs = [torch.zeros(n, dtype=torch.float32, device="mps") for _ in range(world_size)]
    dist.all_gather(outs, mine)
    for r in range(world_size):
        exp = torch.arange(n, dtype=torch.float32) + 1000.0 * (r + 1)
        assert torch.equal(outs[r].cpu(), exp), f"allgather slot {r}"

    # allgather f16 (COPY path without the cpu-reduce flag)
    mine16 = torch.full((n,), float(rank + 2), dtype=torch.float16, device="mps")
    outs16 = [torch.zeros(n, dtype=torch.float16, device="mps") for _ in range(world_size)]
    dist.all_gather(outs16, mine16)
    for r in range(world_size):
        assert (outs16[r].cpu() == float(r + 2)).all(), f"f16 allgather slot {r}"


def _concurrent_large_buckets_fn(rank, world_size):
    """DDP-scale async buckets: ~33 MB ring chunks at ws=5, concurrency=2.

    Pre-transport_collective_mu_ this interleaved on the wire and deadlocked.
    """
    import torch
    import torch.distributed as dist

    # 2*ws chunks; ~8.43M elems/chunk fp32 ≈ 33.7 MB (matches cluster DDP bucket scale).
    n = 84_286_720
    sizes = [n] * 12
    tensors = []
    works = []
    for i, n_i in enumerate(sizes):
        t = torch.full((n_i,), float(i % 7 + 1) * (rank + 1),
                       dtype=torch.float32, device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))
    for w in works:
        w.wait()

    total = sum(r + 1 for r in range(world_size))
    for i, t in enumerate(tensors):
        expected = float(i % 7 + 1) * total
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"bucket {i}: {bad} corrupted elements"


def _concurrent_buckets_fn(rank, world_size):
    """Many async allreduces in flight: with MCCL_COLLECTIVE_CONCURRENCY=2
    bucket N+1's pipeline overlaps bucket N's on the wire; every result is
    verified bit-exactly."""
    import torch
    import torch.distributed as dist

    sizes = [70_001, 131_072, 99_999, 262_144, 150_001] * 6
    tensors = []
    works = []
    for i, n in enumerate(sizes):
        t = torch.full((n,), float(i % 40 + 1) * (rank + 1),
                       dtype=torch.float32, device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))
    for w in works:
        w.wait()

    total = sum(r + 1 for r in range(world_size))
    for i, t in enumerate(tensors):
        expected = float(i % 40 + 1) * total
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"bucket {i}: {bad} corrupted elements"


def _skewed_start_flood_fn(rank, world_size):
    """Credit flow-control regression: rank (ws-1) starts each collective
    LATE (simulating a slower GPU / later bucket start on a mixed cluster).
    Its upstream neighbor could otherwise legally stream nearly the whole
    bucket unsolicited; with a deliberately tiny park limit (8 MB) that
    would overflow and abort.  Credits cap the sender's lead at depth+2
    chunks, so this must pass."""
    import time
    import torch
    import torch.distributed as dist

    n = 12_000_000  # 48 MB fp32 -> chunks ~6 MB at ws=4 (2*ws chunking)
    for it in range(3):
        if rank == world_size - 1:
            time.sleep(1.0)  # late starter
        t = torch.full((n,), float(it + 1) * (rank + 1),
                       dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(it + 1) * sum(r + 1 for r in range(world_size))
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"iter {it}: {bad} corrupted elements"


def _p2p_during_collective_fn(rank, world_size):
    """Demux regression: p2p send/recv interleaved with a large collective on
    the same links.  Pre-demux this was a documented interleaving hazard;
    routing by (seq, tid) makes it a supported case."""
    import torch
    import torch.distributed as dist

    big = torch.full((400_000,), float(rank + 1), dtype=torch.float32, device="mps")
    w = dist.all_reduce(big, op=dist.ReduceOp.SUM, async_op=True)

    # Neighbor exchange while the collective is in flight.
    nxt = (rank + 1) % world_size
    prv = (rank - 1 + world_size) % world_size
    payload = torch.full((4096,), float(rank + 100), dtype=torch.float32, device="mps")
    inbox = torch.zeros(4096, dtype=torch.float32, device="mps")
    if rank % 2 == 0:
        dist.send(payload, dst=nxt, tag=7)
        dist.recv(inbox, src=prv, tag=7)
    else:
        dist.recv(inbox, src=prv, tag=7)
        dist.send(payload, dst=nxt, tag=7)

    w.wait()

    assert (inbox.cpu() == float(prv + 100)).all(), "p2p payload corrupted"
    total = sum(r + 1 for r in range(world_size))
    assert (big.cpu() == total).all(), "collective corrupted by p2p interleave"


class TestPipelinedRing:
    @pytest.mark.parametrize("world_size", [4, 6])
    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_allreduce_chunked_default(self, world_size, dtype):
        run_workers(
            _allreduce_pipeline_fn, world_size=world_size,
            env={"MCCL_TEST_SIZES": RING_SIZES, "MCCL_TEST_DTYPE": dtype},
            timeout=420,
        )

    @pytest.mark.parametrize("world_size", [4, 6])
    def test_allreduce_plain_ring(self, world_size):
        run_workers(
            _allreduce_pipeline_fn, world_size=world_size,
            env={"MCCL_TEST_SIZES": RING_SIZES, "MCCL_TEST_DTYPE": "float32",
                 "MCCL_RING_ALGO": "basic"},
            timeout=420,
        )

    @pytest.mark.parametrize("world_size", [4])
    def test_allreduce_lockstep_fallback(self, world_size):
        """MCCL_RING_PIPELINE=0 must produce identical results (differential
        oracle for pipeline bugs)."""
        run_workers(
            _allreduce_pipeline_fn, world_size=world_size,
            env={"MCCL_TEST_SIZES": RING_SIZES, "MCCL_TEST_DTYPE": "float32",
                 "MCCL_RING_PIPELINE": "0"},
            timeout=420,
        )

    @pytest.mark.parametrize("world_size", [4, 6])
    def test_determinism(self, world_size):
        run_workers(_determinism_pipeline_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [4, 6])
    def test_reduce_scatter_allgather(self, world_size):
        run_workers(_rs_ag_pipeline_fn, world_size=world_size, timeout=420)


class TestConcurrentCollectives:
    @pytest.mark.parametrize("world_size", [4])
    @pytest.mark.parametrize("concurrency", ["1", "2"])
    def test_async_bucket_stream(self, world_size, concurrency):
        run_workers(
            _concurrent_buckets_fn, world_size=world_size,
            env={"MCCL_COLLECTIVE_CONCURRENCY": concurrency},
            timeout=420,
        )

    @pytest.mark.parametrize("world_size", [5])
    def test_async_large_buckets_ws5(self, world_size):
        """Regression: cluster deadlock (seq=21, ~33 MB chunks, concurrency=2)."""
        run_workers(
            _concurrent_large_buckets_fn, world_size=world_size,
            env={"MCCL_COLLECTIVE_CONCURRENCY": "2"},
            timeout=900,
        )


class TestDemuxInterleave:
    @pytest.mark.parametrize("world_size", [4])
    def test_p2p_during_collective(self, world_size):
        run_workers(_p2p_during_collective_fn, world_size=world_size, timeout=300)


class TestCreditFlowControl:
    @pytest.mark.parametrize("world_size", [4, 7, 8])
    def test_skewed_start_bounded_by_credits(self, world_size):
        run_workers(
            _skewed_start_flood_fn, world_size=world_size,
            env={
                # Small enough that an uncredited sender flooding a late rank
                # (~bucket size) would overflow and abort; credits keep the
                # lead at (depth+2) x chunk ~ 24 MB.
                "MCCL_DEMUX_PARK_BYTES": str(48 * 1024 * 1024),
                "MCCL_CREDIT_MIN_CHUNK": str(1024 * 1024),
            },
            timeout=420,
        )
