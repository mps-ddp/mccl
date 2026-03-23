#!/usr/bin/env python3
"""
Quick correctness tests for MCCL transport optimizations.

Tests every new code path:
  - Two-rank split (large tensor, ws=2) -- bucket overlap path
  - Small message (small tensor, ws=2) -- net_engine small path
  - Multiple sequential allreduces -- ordering under multi-engine
  - Large buffer (>16MB) -- exercises 16MB chunk_bytes

Usage:
    python test_quick.py
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _init(rank, world_size, port=29500):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "WARN"
    import mccl  # noqa: F401
    dist.init_process_group("mccl", rank=rank, world_size=world_size)


def test_two_rank_split(rank, world_size):
    """Large tensor (4MB) on 2 ranks -- exercises the net/reduce split path."""
    _init(rank, world_size, port=29500)
    try:
        x = torch.ones(1_000_000, device="mps") * (rank + 1)  # 4MB float32
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = torch.ones(1_000_000, device="mps") * 3.0
        ok = torch.allclose(x, expected, rtol=1e-4, atol=1e-4)
        print(f"  [rank {rank}] two_rank_split (4MB, SUM): {'PASS' if ok else 'FAIL'}")
        assert ok, f"Expected 3.0, got {x[0].item()}"
    finally:
        dist.destroy_process_group()


def test_two_rank_split_avg(rank, world_size):
    """Large tensor AVG -- tests the reduce phase scaling."""
    _init(rank, world_size, port=29510)
    try:
        x = torch.ones(500_000, device="mps") * (rank + 1)  # 2MB
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
        expected = torch.ones(500_000, device="mps") * 1.5
        ok = torch.allclose(x, expected, rtol=1e-4, atol=1e-4)
        print(f"  [rank {rank}] two_rank_split (2MB, AVG): {'PASS' if ok else 'FAIL'}")
        assert ok, f"Expected 1.5, got {x[0].item()}"
    finally:
        dist.destroy_process_group()


def test_small_message(rank, world_size):
    """Small tensor (<64KB) on 2 ranks -- exercises small message path on net_engine."""
    _init(rank, world_size, port=29520)
    try:
        x = torch.ones(100, device="mps") * (rank + 1)  # 400 bytes
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = torch.ones(100, device="mps") * 3.0
        ok = torch.allclose(x, expected, rtol=1e-4, atol=1e-4)
        print(f"  [rank {rank}] small_message (400B, SUM): {'PASS' if ok else 'FAIL'}")
        assert ok
    finally:
        dist.destroy_process_group()


def test_large_buffer(rank, world_size):
    """Very large tensor (32MB) -- exercises 16MB chunk_bytes and 32MB socket buffers."""
    _init(rank, world_size, port=29530)
    try:
        x = torch.ones(8_000_000, device="mps") * (rank + 1)  # 32MB float32
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = torch.ones(8_000_000, device="mps") * 3.0
        ok = torch.allclose(x, expected, rtol=1e-4, atol=1e-4)
        print(f"  [rank {rank}] large_buffer (32MB, SUM): {'PASS' if ok else 'FAIL'}")
        assert ok, f"Expected 3.0, got {x[0].item()}"
    finally:
        dist.destroy_process_group()


def test_sequential_ordering(rank, world_size):
    """Multiple allreduces in sequence -- tests FIFO ordering across engines."""
    _init(rank, world_size, port=29540)
    try:
        all_ok = True
        for i in range(20):
            size = 100 if i % 2 == 0 else 500_000  # alternate small/large
            x = torch.ones(size, device="mps") * (rank + 1 + i)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            expected_val = sum(r + 1 + i for r in range(world_size))
            expected = torch.ones(size, device="mps") * expected_val
            if not torch.allclose(x, expected, rtol=1e-3, atol=1e-3):
                print(f"  [rank {rank}] ordering step {i}: FAIL (expected {expected_val}, got {x[0].item()})")
                all_ok = False
        print(f"  [rank {rank}] sequential_ordering (20 ops): {'PASS' if all_ok else 'FAIL'}")
        assert all_ok
    finally:
        dist.destroy_process_group()


def test_broadcast_large(rank, world_size):
    """Large broadcast -- exercises the fan-out path with atomic completion."""
    _init(rank, world_size, port=29550)
    try:
        if rank == 0:
            x = torch.arange(500_000, device="mps", dtype=torch.float32)
        else:
            x = torch.zeros(500_000, device="mps")
        dist.broadcast(x, src=0)
        expected = torch.arange(500_000, device="mps", dtype=torch.float32)
        ok = torch.allclose(x, expected, rtol=1e-4, atol=1e-4)
        print(f"  [rank {rank}] broadcast_large (2MB): {'PASS' if ok else 'FAIL'}")
        assert ok
    finally:
        dist.destroy_process_group()


TESTS = [
    ("two_rank_split (4MB SUM)", test_two_rank_split, 2),
    ("two_rank_split (2MB AVG)", test_two_rank_split_avg, 2),
    ("small_message (400B SUM)", test_small_message, 2),
    ("large_buffer (32MB SUM)", test_large_buffer, 2),
    ("sequential_ordering (20 ops)", test_sequential_ordering, 2),
    ("broadcast_large (2MB)", test_broadcast_large, 2),
]


if __name__ == "__main__":
    print("MCCL Transport Optimization Tests")
    print("=" * 50)
    passed = 0
    failed = 0
    for name, fn, ws in TESTS:
        print(f"\n[TEST] {name} (ws={ws})")
        try:
            mp.spawn(fn, args=(ws,), nprocs=ws, join=True)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    if failed == 0:
        print("All transport optimization tests passed.")
    else:
        print("Some tests failed -- check output above.")
