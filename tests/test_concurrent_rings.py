"""F3: MCCL_CONCURRENT_RINGS=1 lets unified-CPU ring allreduces share the wire
(bucket k+1 starts while bucket k drains) when MCCL_COLLECTIVE_CONCURRENCY>=2.

Every result is checked bit-exactly; mixed small (star/tree) + ring streams
exercise the "serialized collective waits for in-flight rings" path.
"""

import platform

import pytest

from mccl_test_utils import run_workers
from test_ring_pipeline import _concurrent_buckets_fn, _concurrent_large_buckets_fn
from test_ddp_multibucket import _ddp_multibucket_parity_fn

pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
        reason="MCCL tests require macOS on Apple Silicon",
    ),
    pytest.mark.slow,
]

F3_ENV = {"MCCL_CONCURRENT_RINGS": "1", "MCCL_COLLECTIVE_CONCURRENCY": "2"}


def _mixed_small_and_ring_fn(rank, world_size):
    """Interleave sub-threshold (star/tree) and ring allreduces plus a broadcast
    while rings are in flight: non-ring collectives must wait for and not
    corrupt concurrent rings."""
    import torch
    import torch.distributed as dist

    sizes = [1024, 1_000_000, 2048, 3_000_000, 512, 1_500_000] * 5
    tensors, works = [], []
    for i, n in enumerate(sizes):
        t = torch.full((n,), float(i % 13 + 1) * (rank + 1), dtype=torch.float32, device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True))
        if i % 7 == 3:
            b = torch.full((70_000,), float(i), dtype=torch.float32, device="mps") if rank == 0 \
                else torch.zeros(70_000, dtype=torch.float32, device="mps")
            works.append(dist.broadcast(b, src=0, async_op=True))
            tensors.append(("bcast", b, float(i)))
    for w in works:
        w.wait()

    total = sum(r + 1 for r in range(world_size))
    j = 0
    for item in tensors:
        if isinstance(item, tuple):
            _, b, val = item
            bad = (b.cpu() != val).sum().item()
            assert bad == 0, f"broadcast {val}: {bad} corrupted elements"
            continue
        expected = float(j % 13 + 1) * total
        bad = (item.cpu() != expected).sum().item()
        assert bad == 0, f"bucket {j}: {bad} corrupted elements"
        j += 1


class TestConcurrentRings:
    @pytest.mark.parametrize("world_size", [4, 6])
    def test_async_bucket_stream(self, world_size):
        run_workers(_concurrent_buckets_fn, world_size=world_size, env=F3_ENV, timeout=420)

    @pytest.mark.parametrize("world_size", [4, 6])
    def test_mixed_small_ring_broadcast(self, world_size):
        run_workers(_mixed_small_and_ring_fn, world_size=world_size, env=F3_ENV, timeout=420)

    def test_large_buckets_ws8(self):
        run_workers(
            _concurrent_large_buckets_fn, world_size=8,
            env={**F3_ENV, "MCCL_TEST_LARGE_N": "6553600"},
            timeout=1200,
        )

    @pytest.mark.parametrize("world_size", [4])
    def test_ddp_multibucket_parity(self, world_size):
        run_workers(_ddp_multibucket_parity_fn, world_size=world_size, env=F3_ENV, timeout=420)
