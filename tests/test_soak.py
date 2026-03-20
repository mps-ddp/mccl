"""Phase F: Soak / reliability tests.

Long-running stress tests for production hardening.
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
    reason="Soak tests require macOS on Apple Silicon",
)


def _worker(rank, world_size, fn, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "WARN"

    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_distributed(fn, world_size=2, port=29500):
    mp.spawn(_worker, args=(world_size, fn, port), nprocs=world_size, join=True)


class TestSoakAllreduce:
    @pytest.mark.timeout(600)
    def test_10k_allreduce_loop(self):
        def fn(rank, world_size):
            for i in range(10_000):
                tensor = torch.randn(1024, device="mps")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=2, port=31000)

    @pytest.mark.timeout(300)
    def test_mixed_sizes(self):
        def fn(rank, world_size):
            sizes = [64, 1024, 4096, 65536, 262144, 1048576]
            for i in range(100):
                size = sizes[i % len(sizes)]
                tensor = torch.randn(size, device="mps")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=2, port=31100)

    @pytest.mark.timeout(300)
    def test_large_bucket_stress(self):
        def fn(rank, world_size):
            for _ in range(50):
                tensor = torch.randn(25_000_000, device="mps")  # ~100 MB
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=2, port=31200)


class TestSoakMixed:
    @pytest.mark.timeout(300)
    def test_interleaved_collectives(self):
        def fn(rank, world_size):
            for i in range(500):
                # Allreduce
                tensor = torch.randn(4096, device="mps")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

                # Broadcast
                if i % 10 == 0:
                    bcast = torch.randn(1024, device="mps") if rank == 0 else torch.zeros(1024, device="mps")
                    dist.broadcast(bcast, src=0)

                # Barrier
                if i % 50 == 0:
                    dist.barrier()

        _run_distributed(fn, world_size=2, port=31300)


class TestSoakThreeRank:
    """Ring algorithm stress tests with 3 ranks."""

    @pytest.mark.timeout(600)
    def test_three_rank_allreduce_loop(self):
        def fn(rank, world_size):
            for i in range(2000):
                tensor = torch.randn(1024, device="mps")
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=3, port=35000)

    @pytest.mark.timeout(300)
    def test_three_rank_mixed_ops(self):
        def fn(rank, world_size):
            ops = [dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX,
                   dist.ReduceOp.PRODUCT, dist.ReduceOp.AVG]
            for i in range(500):
                op = ops[i % len(ops)]
                tensor = torch.ones(256, device="mps") * (rank + 1)
                dist.all_reduce(tensor, op=op)

        _run_distributed(fn, world_size=3, port=35100)


class TestSoakAlignment:
    """Stress test non-4-aligned sizes to catch vectorization tail bugs."""

    @pytest.mark.timeout(300)
    def test_non_aligned_sizes_loop(self):
        def fn(rank, world_size):
            sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 31, 33, 63, 65, 127,
                     129, 255, 257, 511, 997, 1023, 4097]
            for i in range(200):
                size = sizes[i % len(sizes)]
                tensor = torch.ones(size, device="mps") * (rank + 1)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                expected_val = sum(r + 1 for r in range(world_size))
                expected = torch.ones(size, device="mps") * expected_val
                assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), \
                    f"Rank {rank}, size {size}, iter {i}: mismatch"

        _run_distributed(fn, world_size=2, port=35200)

    @pytest.mark.timeout(300)
    def test_non_aligned_f16(self):
        def fn(rank, world_size):
            sizes = [1, 3, 7, 13, 33, 65, 127, 997]
            for i in range(100):
                size = sizes[i % len(sizes)]
                tensor = torch.ones(size, device="mps", dtype=torch.float16) * (rank + 1)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                expected = torch.ones(size, device="mps", dtype=torch.float16) * 3.0
                assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2), \
                    f"Rank {rank}, size {size}, iter {i}: mismatch"

        _run_distributed(fn, world_size=2, port=35300)

    @pytest.mark.timeout(300)
    def test_three_rank_non_aligned(self):
        def fn(rank, world_size):
            sizes = [1, 5, 11, 31, 97, 257, 997]
            for i in range(100):
                size = sizes[i % len(sizes)]
                tensor = torch.ones(size, device="mps") * (rank + 1)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                expected = torch.ones(size, device="mps") * 6.0  # 1+2+3
                assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=3, port=35400)
