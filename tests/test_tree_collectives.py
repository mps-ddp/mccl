"""Scale-aware algorithm tests: recursive-doubling small allreduce
(non-power-of-2 folding) and tree/ring broadcast at ws >= 4.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _tree_small_allreduce_fn(rank, world_size):
    """Small (below MCCL_SMALL_MSG_THRESHOLD) allreduces take the
    recursive-doubling tree.  Exercise SUM/AVG/MAX across dtypes; values are
    exact in fp16 so checks are equality-tight."""
    import torch
    import torch.distributed as dist

    for n in (1, 17, 1024, 4099, 30_000):
        for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 1e-2)):
            t = torch.full((n,), float(rank + 1), dtype=dtype, device="mps")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            expected = float(sum(r + 1 for r in range(world_size)))
            assert (t.cpu().float() - expected).abs().max().item() <= tol, \
                f"SUM n={n} dtype={dtype}"

        t = torch.full((n,), float(rank + 1), dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        assert (t.cpu() == float(world_size)).all(), f"MAX n={n}"

        t = torch.full((n,), float(rank + 1), dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        expected = sum(r + 1 for r in range(world_size)) / world_size
        assert (t.cpu() - expected).abs().max().item() <= 1e-5, f"AVG n={n}"


def _tree_bitwise_identical_fn(rank, world_size):
    """All ranks must produce BIT-IDENTICAL results (recursive doubling
    computes the same parenthesization everywhere modulo fp commutativity)."""
    import torch
    import torch.distributed as dist

    torch.manual_seed(7 + rank)
    t = torch.randn(20_000, dtype=torch.float32).to("mps")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    digest = t.cpu()
    outs = [torch.zeros_like(digest, device="mps") for _ in range(world_size)]
    dist.all_gather(outs, digest.to("mps"))
    for r in range(world_size):
        assert torch.equal(outs[r].cpu(), digest), \
            f"rank {r} result differs bitwise from rank {rank}"


def _broadcast_algos_fn(rank, world_size):
    """Small payloads -> binomial tree; large payloads -> pipelined ring
    (multi-slice).  Cover both, every root, with verifiable patterns."""
    import torch
    import torch.distributed as dist

    for root in range(world_size):
        # Tree path (small)
        n = 2048
        if rank == root:
            t = (torch.arange(n, dtype=torch.float32) + 31.0 * root).to("mps")
        else:
            t = torch.full((n,), -1.0, dtype=torch.float32, device="mps")
        dist.broadcast(t, src=root)
        exp = torch.arange(n, dtype=torch.float32) + 31.0 * root
        assert torch.equal(t.cpu(), exp), f"tree broadcast root={root}"

    # Ring path (large, multiple slices), root 0 and a middle root
    for root in (0, world_size // 2):
        n = 3_000_001  # 12 MB fp32, odd numel -> uneven final slice
        if rank == root:
            t = (torch.arange(n, dtype=torch.float32) % 1009).to("mps")
        else:
            t = torch.zeros(n, dtype=torch.float32, device="mps")
        dist.broadcast(t, src=root)
        exp = torch.arange(n, dtype=torch.float32) % 1009
        assert torch.equal(t.cpu(), exp), f"ring broadcast root={root}"


class TestTreeSmallAllreduce:
    # 5 and 6 are non-powers-of-two: exercises the fold-in/unfold rounds.
    @pytest.mark.parametrize("world_size", [3, 4, 5, 6])
    def test_values(self, world_size):
        run_workers(_tree_small_allreduce_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [5])
    def test_bitwise_identical_across_ranks(self, world_size):
        run_workers(_tree_bitwise_identical_fn, world_size=world_size, timeout=300)


class TestBroadcastAlgos:
    @pytest.mark.parametrize("world_size", [4, 6])
    def test_tree_and_ring(self, world_size):
        run_workers(_broadcast_algos_fn, world_size=world_size, timeout=420)
