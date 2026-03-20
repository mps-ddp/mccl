"""Phase D: Local process group correctness tests.

Spawns two processes on the same host to test allreduce, broadcast, barrier.
Requires macOS Apple Silicon with MCCL built.
"""

import platform
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Process group tests require macOS on Apple Silicon",
)


def _worker(rank, world_size, fn, port):
    """Process entry point for distributed tests."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "DEBUG"

    import mccl  # noqa: F401

    dist.init_process_group(
        backend="mccl",
        rank=rank,
        world_size=world_size,
    )

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_distributed(fn, world_size=2, port=29500):
    """Spawn world_size processes running fn(rank, world_size)."""
    mp.spawn(_worker, args=(world_size, fn, port), nprocs=world_size, join=True)


class TestAllreduce:
    def test_two_rank_f32_sum(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            # SUM returns raw sum: rank0 had 1, rank1 had 2 → result = 3
            expected = torch.ones(100, device="mps") * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), \
                f"Rank {rank}: expected {expected[0].item()}, got {tensor[0].item()}"

        _run_distributed(fn, world_size=2, port=29600)

    def test_two_rank_large(self):
        def fn(rank, world_size):
            torch.manual_seed(42 + rank)
            tensor = torch.randn(1_000_000, device="mps")
            local_copy = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            # Just verify it didn't crash and values changed
            assert not torch.allclose(tensor, local_copy)

        _run_distributed(fn, world_size=2, port=29700)


class TestBroadcast:
    def test_from_rank0(self):
        def fn(rank, world_size):
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0, 3.0], device="mps")
            else:
                tensor = torch.zeros(3, device="mps")
            dist.broadcast(tensor, src=0)
            expected = torch.tensor([1.0, 2.0, 3.0], device="mps")
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=29800)


class TestBarrier:
    def test_basic_barrier(self):
        def fn(rank, world_size):
            dist.barrier()
            # If we get here, barrier worked

        _run_distributed(fn, world_size=2, port=29900)


class TestAllreduceOps:
    """Cover all ReduceOp variants and the AVG scaling path."""

    def test_avg(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.ones(100, device="mps") * 1.5  # (1+2)/2
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=2, port=32000)

    def test_min(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            expected = torch.ones(100, device="mps") * 1.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32100)

    def test_max(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            expected = torch.ones(100, device="mps") * 2.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32200)

    def test_product(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 2)
            dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT)
            expected = torch.ones(100, device="mps") * 6.0  # 2*3
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=2, port=32300)


class TestAllreduceAlignment:
    """Non-4-aligned sizes exercise the scalar tail path in Metal kernels."""

    def test_size_1(self):
        def fn(rank, world_size):
            tensor = torch.ones(1, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert torch.allclose(tensor, torch.tensor([3.0], device="mps"))

        _run_distributed(fn, world_size=2, port=32400)

    def test_size_3(self):
        def fn(rank, world_size):
            tensor = torch.ones(3, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(3, device="mps") * 3.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32500)

    def test_size_7(self):
        def fn(rank, world_size):
            tensor = torch.ones(7, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(7, device="mps") * 3.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32600)

    def test_size_33(self):
        def fn(rank, world_size):
            tensor = torch.ones(33, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(33, device="mps") * 3.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32700)

    def test_prime_size(self):
        def fn(rank, world_size):
            tensor = torch.ones(997, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(997, device="mps") * 3.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=32800)


class TestAllreduceF16:
    """Half-precision tensors exercise the Metal kernel path (not Accelerate)."""

    def test_f16_sum(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps", dtype=torch.float16) * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=32900)

    def test_f16_avg(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.ones(100, device="mps", dtype=torch.float16) * 1.5
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=33000)

    def test_f16_non_aligned_size(self):
        def fn(rank, world_size):
            tensor = torch.ones(13, device="mps", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(13, device="mps", dtype=torch.float16) * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=33100)

    def test_f16_large(self):
        def fn(rank, world_size):
            tensor = torch.randn(500_000, device="mps", dtype=torch.float16)
            local_copy = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert not torch.allclose(tensor, local_copy)

        _run_distributed(fn, world_size=2, port=33200)


class TestThreeRankAllreduce:
    """3-rank tests activate the ring algorithm (not the 2-rank fast path)."""

    def test_three_rank_sum(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps") * 6.0  # 1+2+3
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), \
                f"Rank {rank}: expected 6.0, got {tensor[0].item()}"

        _run_distributed(fn, world_size=3, port=33300)

    def test_three_rank_avg(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.ones(100, device="mps") * 2.0  # (1+2+3)/3
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=3, port=33400)

    def test_three_rank_min(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            expected = torch.ones(100, device="mps") * 1.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=3, port=33500)

    def test_three_rank_max(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            expected = torch.ones(100, device="mps") * 3.0
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=3, port=33600)

    def test_three_rank_non_aligned(self):
        def fn(rank, world_size):
            tensor = torch.ones(13, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(13, device="mps") * 6.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=3, port=33700)

    def test_three_rank_large_ring(self):
        def fn(rank, world_size):
            tensor = torch.randn(500_000, device="mps")
            local_copy = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert not torch.allclose(tensor, local_copy)

        _run_distributed(fn, world_size=3, port=33800)

    def test_three_rank_f16(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps", dtype=torch.float16) * 6.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=3, port=33900)


class TestSequenceOrdering:
    def test_multiple_allreduce_in_order(self):
        def fn(rank, world_size):
            for i in range(10):
                tensor = torch.ones(64, device="mps") * (rank + 1 + i)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=2, port=30000)
