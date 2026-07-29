"""Stress tests targeting GPU/network ordering races.

The fp32/f16 Metal ring path commits reduce kernels asynchronously; a chunk
must not be staged for a network send (or overwritten by an incoming
allgather chunk) until its kernel completes.  These tests run many
back-to-back collectives with GPU compute interleaved between steps so an
unfenced kernel has a wide window to corrupt wire traffic.  All results are
verified against exact expected values; a single mismatched element fails.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _ring_kernel_race_fn(rank, world_size):
    """f16 ring with interleaved MPS matmuls: exercises the Metal reduce
    path (use_cpu = False) where the kernel-fence protects staged sends."""
    import torch
    import torch.distributed as dist

    n = 200_001  # > small_msg_threshold for f16, odd numel
    iters = 30
    # Keep the GPU busy so MCCL's reduce kernels queue behind real work.
    a = torch.randn(512, 512, device="mps")

    for it in range(iters):
        val = float((it % 7) + 1)
        t = torch.full((n,), val * (rank + 1), dtype=torch.float16, device="mps")
        _ = a @ a  # in-flight compute on PyTorch's MPS stream
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = val * sum(r + 1 for r in range(world_size))
        got = t.cpu()
        bad = (got != expected).sum().item()
        assert bad == 0, f"iter {it}: {bad}/{n} corrupted elements (expected {expected})"


def _fp32_default_ring_race_fn(rank, world_size):
    """Default fp32 path, many quick iterations with distinct values per
    iteration so stale/pre-reduction bytes are detectable."""
    import torch
    import torch.distributed as dist

    n = 150_001
    iters = 40
    a = torch.randn(256, 256, device="mps")

    for it in range(iters):
        val = float(it + 1)
        t = torch.full((n,), val * (rank + 1), dtype=torch.float32, device="mps")
        _ = (a @ a).sum()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = val * sum(r + 1 for r in range(world_size))
        got = t.cpu()
        bad = (got != expected).sum().item()
        assert bad == 0, f"iter {it}: {bad}/{n} corrupted elements"


def _back_to_back_buckets_fn(rank, world_size):
    """Simulate DDP bucket streams: many allreduces of varying sizes issued
    without waiting in between (async work objects), then verified."""
    import torch
    import torch.distributed as dist

    sizes = [70_001, 131_072, 99_999, 262_144, 12_345] * 4
    tensors = []
    works = []
    for i, n in enumerate(sizes):
        t = torch.full((n,), float(i + 1) * (rank + 1), dtype=torch.float32, device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))

    for w in works:
        w.wait()

    total = sum(r + 1 for r in range(world_size))
    for i, (n, t) in enumerate(zip(sizes, tensors)):
        expected = float(i + 1) * total
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"bucket {i} (n={n}): {bad} corrupted elements"


def _mixed_op_stream_fn(rank, world_size):
    """Interleave allreduce / broadcast / allgather / barrier on one group —
    catches cross-op sequencing problems (tids/seqs now validated on wire)."""
    import torch
    import torch.distributed as dist

    for it in range(10):
        t = torch.full((70_001,), float(rank + 1 + it), dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        assert (t.cpu() == sum(r + 1 + it for r in range(world_size))).all()

        b = (
            torch.full((1024,), float(it), device="mps")
            if rank == 0
            else torch.zeros(1024, device="mps")
        )
        dist.broadcast(b, src=0)
        assert (b.cpu() == float(it)).all()

        outs = [torch.zeros(513, device="mps") for _ in range(world_size)]
        mine = torch.full((513,), float(rank + it), device="mps")
        dist.all_gather(outs, mine)
        for r in range(world_size):
            assert (outs[r].cpu() == float(r + it)).all()

        dist.barrier()


class TestRingKernelRace:
    @pytest.mark.parametrize("world_size", [3])
    def test_f16_metal_ring_under_gpu_load(self, world_size):
        run_workers(_ring_kernel_race_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_fp32_default_ring_under_gpu_load(self, world_size):
        run_workers(_fp32_default_ring_race_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [3])
    def test_f16_chunked_ring_under_gpu_load(self, world_size):
        run_workers(
            _ring_kernel_race_fn,
            world_size=world_size,
            env={"MCCL_RING_ALGO": "chunked"},
            timeout=420,
        )


class TestPipelineStress:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_async_bucket_stream(self, world_size):
        run_workers(_back_to_back_buckets_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_mixed_op_stream(self, world_size):
        run_workers(_mixed_op_stream_fn, world_size=world_size, timeout=420)

    def test_repeated_barrier(self):
        """Regression: the store barrier must be reusable (epoch-keyed); the
        old one-shot keys made every barrier after the first a no-op."""

        def fn(rank, world_size):
            import time
            import torch.distributed as dist

            for i in range(5):
                # Skew ranks so a broken (instant) barrier is observable as
                # one rank racing far ahead.
                if rank == 0:
                    time.sleep(0.2)
                t0 = time.monotonic()
                dist.barrier()
                # Real barrier: non-delayed ranks must have waited for rank 0
                if rank != 0:
                    assert time.monotonic() - t0 > 0.05 or i == 0, (
                        f"barrier {i} returned instantly on rank {rank}"
                    )

        run_workers(fn, world_size=2, timeout=180)
