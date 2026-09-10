"""Collectives without the per-collective TCPStore barrier.

``rendezvous_collective_enter`` is off by default (``MCCL_COLLECTIVE_STORE_BARRIER``
unset).  A fast rank may therefore start sending a collective before a slow
rank has posted its receives; the demux must park that data (bounded by the
credit window on the ring, by one message per peer on star paths) and deliver
it once the receive is posted.  These tests inject a slow rank and check that
results stay exact through a mixed sequence of ring / star collectives, for
fp32 and for the f16/bf16 unified-CPU ring path.
"""

import platform

import pytest
import torch

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS Apple Silicon",
)

SLOW_RANK_ENV = {
    "MCCL_COLLECTIVE_CONCURRENCY": "1",
    "MCCL_LOG_LEVEL": "WARN",
}


def _slow_rank_mixed_collectives(rank, world_size):
    import time

    dev = torch.device("mps:0")
    slow = 3 % world_size
    n_big = 2 * 1024 * 1024  # 8 MB fp32: ring path, credits engaged (chunk >= 1 MB)

    for it in range(4):
        if rank == slow:
            time.sleep(0.4)  # neighbours run ahead and must park at this rank
        # Ring allreduce, SUM
        t = torch.full((n_big,), float(rank + 1 + it), device=dev, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(sum(r + 1 + it for r in range(world_size)))
        got = t.cpu()
        assert torch.equal(got, torch.full_like(got, expected)), (
            f"iter {it} SUM: {got[0].item()} != {expected}, max abs err "
            f"{(got - expected).abs().max().item()}"
        )
        # Ring allreduce, AVG
        a = torch.full((n_big,), float(rank * 2 + it), device=dev, dtype=torch.float32)
        dist.all_reduce(a, op=dist.ReduceOp.AVG)
        exp_avg = sum(r * 2 + it for r in range(world_size)) / world_size
        got = a.cpu()
        assert torch.allclose(got, torch.full_like(got, exp_avg), atol=1e-5), (
            f"iter {it} AVG: {got[0].item()} != {exp_avg}"
        )
        # Star broadcast (small) from a rotating root
        root = it % world_size
        b = torch.full((1024,), float(rank + 100 * it), device=dev, dtype=torch.float32)
        dist.broadcast(b, src=root)
        assert torch.equal(b.cpu(), torch.full((1024,), float(root + 100 * it)))
        # Small allreduce (star / tree path)
        s = torch.full((64,), float(rank), device=dev, dtype=torch.float32)
        dist.all_reduce(s, op=dist.ReduceOp.MAX)
        assert torch.equal(s.cpu(), torch.full((64,), float(world_size - 1)))
        # Small allgather (DDP init_sync path)
        outs = [torch.zeros(4, device=dev, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(outs, torch.full((4,), rank + 10 * it, device=dev, dtype=torch.int64))
        for r, o in enumerate(outs):
            assert torch.equal(o.cpu(), torch.full((4,), r + 10 * it, dtype=torch.int64))
    dist.barrier()


@pytest.mark.timeout(240)
def test_slow_rank_ws6_mixed_collectives_no_store_barrier():
    run_workers(_slow_rank_mixed_collectives, world_size=6, env=SLOW_RANK_ENV, timeout=240)


@pytest.mark.timeout(240)
def test_slow_rank_ws6_with_store_barrier_env():
    """The legacy per-collective barrier still works when explicitly re-enabled."""
    env = {**SLOW_RANK_ENV, "MCCL_COLLECTIVE_STORE_BARRIER": "1"}
    run_workers(_slow_rank_mixed_collectives, world_size=6, env=env, timeout=240)


def _half_dtypes_ring(rank, world_size):
    import time

    dev = torch.device("mps:0")
    slow = 1 % world_size
    n = 1024 * 1024  # 2 MB in f16/bf16: ring path
    for dtype, atol in ((torch.float16, 0.0), (torch.bfloat16, 0.0)):
        if rank == slow:
            time.sleep(0.3)
        # Values exactly representable in both formats: results must be exact.
        t = torch.full((n,), float(rank + 1), device=dev, dtype=dtype)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(sum(r + 1 for r in range(world_size)))
        got = t.float().cpu()
        assert torch.allclose(got, torch.full_like(got, expected), atol=atol), (
            f"{dtype} SUM: {got[0].item()} != {expected}"
        )
        m = torch.full((n,), float(rank), device=dev, dtype=dtype)
        dist.all_reduce(m, op=dist.ReduceOp.MAX)
        got = m.float().cpu()
        assert torch.equal(got, torch.full_like(got, float(world_size - 1)))
        # AVG of {0,2,4,...} over ws=4 is exact in bf16 (3.0).
        a = torch.full((n,), float(rank * 2), device=dev, dtype=dtype)
        dist.all_reduce(a, op=dist.ReduceOp.AVG)
        exp_avg = sum(r * 2 for r in range(world_size)) / world_size
        got = a.float().cpu()
        assert torch.allclose(got, torch.full_like(got, exp_avg), atol=1e-2), (
            f"{dtype} AVG: {got[0].item()} != {exp_avg}"
        )
    dist.barrier()


@pytest.mark.timeout(240)
def test_f16_bf16_ring_unified_cpu_reduce_ws4():
    """Default path: shared-storage f16/bf16 ring reduce on the CPU (widen to fp32 per hop)."""
    run_workers(_half_dtypes_ring, world_size=4, env=SLOW_RANK_ENV, timeout=240)


@pytest.mark.timeout(240)
def test_f16_bf16_ring_metal_reduce_ws4():
    """MCCL_UNIFIED_CPU_REDUCE=0 restores the Metal-kernel reduce; same results."""
    env = {**SLOW_RANK_ENV, "MCCL_UNIFIED_CPU_REDUCE": "0"}
    run_workers(_half_dtypes_ring, world_size=4, env=env, timeout=240)


def _fp32_paths_bit_identical(rank, world_size):
    """Unified CPU reduce vs Metal reduce vs legacy CPU: same bytes for the same inputs."""
    import os

    dev = torch.device("mps:0")
    g = torch.Generator().manual_seed(1234 + rank)
    x = torch.randn(3 * 1024 * 1024, generator=g, dtype=torch.float32).to(dev)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    tag = os.environ["MCCL_TEST_TAG"]
    torch.save(x.cpu(), f"/tmp/mccl_slowrank_{tag}_{rank}.pt")
    dist.barrier()


@pytest.mark.timeout(300)
def test_fp32_ring_paths_parity_ws4():
    """Unified CPU reduce (new default) reduces along the same ring path as the
    Metal kernel path, so fp32 SUM results must match to fp32 rounding; the
    legacy MCCL_FP32_CPU_REDUCE=1 path is the same vDSP code and must be
    byte-identical to the new default."""
    base = {**SLOW_RANK_ENV}
    run_workers(_fp32_paths_bit_identical, world_size=4,
                env={**base, "MCCL_TEST_TAG": "cpu_default"}, timeout=300)
    run_workers(_fp32_paths_bit_identical, world_size=4,
                env={**base, "MCCL_TEST_TAG": "cpu_legacy", "MCCL_FP32_CPU_REDUCE": "1"}, timeout=300)
    run_workers(_fp32_paths_bit_identical, world_size=4,
                env={**base, "MCCL_TEST_TAG": "metal", "MCCL_UNIFIED_CPU_REDUCE": "0"}, timeout=300)
    for r in range(4):
        d = torch.load(f"/tmp/mccl_slowrank_cpu_default_{r}.pt")
        l = torch.load(f"/tmp/mccl_slowrank_cpu_legacy_{r}.pt")
        m = torch.load(f"/tmp/mccl_slowrank_metal_{r}.pt")
        assert torch.equal(d, l), f"rank {r}: unified-CPU vs legacy-CPU differ"
        assert torch.allclose(d, m, rtol=1e-6, atol=1e-6), (
            f"rank {r}: unified-CPU vs Metal max abs diff {(d - m).abs().max().item()}"
        )
    # All ranks hold identical bytes (ring result is copied, not recomputed).
    d0 = torch.load("/tmp/mccl_slowrank_cpu_default_0.pt")
    for r in range(1, 4):
        assert torch.equal(d0, torch.load(f"/tmp/mccl_slowrank_cpu_default_{r}.pt"))
