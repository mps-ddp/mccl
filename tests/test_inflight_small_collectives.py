"""Several collectives in flight at once, every result checked exactly.

Non-root ranks of the small-message star allreduce and of the small broadcasts used to
receive into a temporary and queue ``tensor.copy_()`` on PyTorch's stream.  The temporary
was freed with the copy still pending, the allocator handed its buffer to the next
same-size temporary, and that receive (a CPU write) landed first: rank 1 of a burst of
1024-element allreduces held the result of the seventh one.  Equal sizes in one burst are
what make the allocator reuse the buffer, and the bursts are polled to completion before
anything commits PyTorch's stream, so on the old code every queued copy read a recycled
buffer.

The two-rank allreduce exchanges both directions at once and keeps its receive
temporary alive until the reduce has read it; the large-burst test covers that path.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)

F3_ENV = {"MCCL_CONCURRENT_RINGS": "1", "MCCL_COLLECTIVE_CONCURRENCY": "2"}


def _allreduce_burst_fn(rank, world_size):
    import time

    import torch
    import torch.distributed as dist

    total = sum(r + 1 for r in range(world_size))
    for rnd in range(4):
        head = torch.full((2_000_000,), float(rank + 1), dtype=torch.float32, device="mps")
        tensors, works = [(head, float(total))], [dist.all_reduce(head, async_op=True)]
        for i in range(24):
            n = 1024 if i % 2 == 0 else 2048
            v = float(rnd * 24 + i + 1)
            t = torch.full((n,), v * (rank + 1), dtype=torch.float32, device="mps")
            tensors.append((t, v * total))
            works.append(dist.all_reduce(t, async_op=True))
        # Poll without touching PyTorch's stream, so nothing commits it until every
        # collective of the burst has finished on the engine side.
        while not all(w.is_completed() for w in works):
            time.sleep(0.001)
        for w in works:
            w.wait()
        for i, (t, expected) in enumerate(tensors):
            bad = (t.cpu() != expected).sum().item()
            assert bad == 0, f"round {rnd} allreduce {i}: {bad} wrong elements"


def _broadcast_burst_fn(rank, world_size):
    import time

    import torch
    import torch.distributed as dist

    for rnd in range(4):
        head = torch.full((2_000_000,), 7.0, dtype=torch.float32, device="mps") if rank == 0 \
            else torch.zeros(2_000_000, dtype=torch.float32, device="mps")
        tensors, works = [(head, 7.0)], [dist.broadcast(head, src=0, async_op=True)]
        for i in range(24):
            n = 1024 if i % 2 == 0 else 2048
            v = float(rnd * 24 + i + 1)
            t = torch.full((n,), v, dtype=torch.float32, device="mps") if rank == 0 \
                else torch.zeros(n, dtype=torch.float32, device="mps")
            tensors.append((t, v))
            works.append(dist.broadcast(t, src=0, async_op=True))
        # Poll without touching PyTorch's stream, so nothing commits it until every
        # collective of the burst has finished on the engine side.
        while not all(w.is_completed() for w in works):
            time.sleep(0.001)
        for w in works:
            w.wait()
        for i, (t, expected) in enumerate(tensors):
            bad = (t.cpu() != expected).sum().item()
            assert bad == 0, f"round {rnd} broadcast {i}: {bad} wrong elements"


def _two_rank_large_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    # Patterns stay exact in every dtype: integers below 256 for bf16.
    cases = [(torch.float32, 1000), (torch.float16, 500), (torch.bfloat16, 64)]
    for dtype, mod in cases:
        for n in (300_000, 2_000_001, 6_000_000):
            for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                base = (torch.arange(n) % mod).to(dtype)
                works, tensors = [], []
                for k in range(3):
                    t = (base * (rank + 1 + k)).to("mps")
                    tensors.append((t, k))
                    works.append(dist.all_reduce(t, op=op, async_op=True))
                for w in works:
                    w.wait()
                for t, k in tensors:
                    factor = sum(r + 1 + k for r in range(world_size))
                    expected = base.float() * factor
                    if op == dist.ReduceOp.AVG:
                        expected = expected / world_size
                    got = t.cpu().float()
                    bad = (got != expected.to(dtype).float()).sum().item()
                    assert bad == 0, f"{dtype} n={n} {op} k={k}: {bad} wrong elements"


class TestSmallCollectivesInFlight:
    @pytest.mark.parametrize("world_size", [3, 4])
    @pytest.mark.parametrize("env", [{}, {"MCCL_SMALL_ALGO": "star"}, F3_ENV],
                             ids=["tree", "star", "f3"])
    def test_allreduce_burst(self, world_size, env):
        run_workers(_allreduce_burst_fn, world_size=world_size, env=env, timeout=300)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_broadcast_burst(self, world_size):
        run_workers(_broadcast_burst_fn, world_size=world_size, timeout=300)


class TestTwoRankAllreduce:
    def test_small_burst(self):
        run_workers(_allreduce_burst_fn, world_size=2, timeout=300)

    def test_large_inflight(self):
        run_workers(_two_rank_large_fn, world_size=2, timeout=600)
