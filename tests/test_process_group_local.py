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


def _run_distributed(fn, world_size=2, port=29500, extra_mccl_env=None):
    """Spawn world_size processes running fn(rank, world_size).

    extra_mccl_env: optional dict of MCCL_* env vars injected before init_process_group
    (e.g. {\"MCCL_RING_ALGO\": \"basic\"}).
    """
    import subprocess, sys, textwrap, inspect
    src = textwrap.dedent(inspect.getsource(fn))
    extra_lines = ""
    if extra_mccl_env:
        for k, v in extra_mccl_env.items():
            extra_lines += f"os.environ[{k!r}] = {v!r}\n"
    script = (
        "import os, sys, torch, torch.distributed as dist\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        f"os.environ['MCCL_LOG_LEVEL'] = 'DEBUG'\n"
        f"{extra_lines}"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        "import mccl\n"
        "dist.init_process_group(backend='mccl', rank=rank, world_size=world_size, device_id=torch.device('mps:0'))\n"
            "try:\n"
            f"{textwrap.indent(src, '    ')}"
            "    fn(rank, world_size)\n"
            "finally:\n"
            "    dist.destroy_process_group()\n"
            "    os._exit(0)\n"
    )
    import time
    procs = []
    for r in range(world_size):
        p = subprocess.Popen([sys.executable, "-c", script, str(r), str(world_size)])
        procs.append(p)
        time.sleep(0.5)
    for p in procs:
        rc = p.wait()
        assert rc == 0, f"Worker exited with code {rc}"


# Default MCCL_SMALL_MSG_THRESHOLD is 262144 bytes → need >65536 float32 elems for ring path.
THREE_RANK_PLAIN_RING_NUMEL = 70_000  # 280000 bytes
THREE_RANK_PLAIN_RING_ENV = {"MCCL_RING_ALGO": "basic"}
# f16: 262144 / 2 = 131072 elems minimum
THREE_RANK_PLAIN_RING_NUMEL_F16 = 140_000
# bf16: same 2 bytes/elem as f16 for ring threshold
THREE_RANK_PLAIN_RING_NUMEL_BF16 = 140_000


def _run_three_rank_parity_workers(
    port: int, outfile: str, *, use_mccl: bool, use_avg: bool
) -> None:
    """Run a 3-rank all_reduce; rank 0 writes the result tensor to ``outfile`` (CPU)."""
    import subprocess
    import sys
    import time

    n = THREE_RANK_PLAIN_RING_NUMEL
    rop = "dist.ReduceOp.AVG" if use_avg else "dist.ReduceOp.SUM"
    env_block = ""
    if use_mccl:
        for k, v in THREE_RANK_PLAIN_RING_ENV.items():
            env_block += f"os.environ[{k!r}] = {v!r}\n"
    mps_arg = ', device="mps"' if use_mccl else ""
    if use_mccl:
        init_pg = (
            "import mccl\n"
            "dist.init_process_group(backend='mccl', rank=rank, world_size=world_size, "
            "device_id=torch.device('mps:0'))\n"
        )
    else:
        init_pg = (
            "dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)\n"
        )
    mccl_listen = ""
    if use_mccl:
        mccl_listen = (
            f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
            f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        )
    script = (
        "import os, sys, torch, torch.distributed as dist\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        f"{mccl_listen}"
        f"os.environ['PARITY_OUTFILE'] = {outfile!r}\n"
        f"{env_block}"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        f"{init_pg}"
        "try:\n"
        "    out = os.environ['PARITY_OUTFILE']\n"
        f"    n = {n}\n"
        f"    t = torch.full((n,), float(rank + 1), dtype=torch.float32{mps_arg})\n"
        f"    dist.all_reduce(t, op={rop})\n"
        "    if rank == 0:\n"
        "        torch.save(t.detach().cpu(), out)\n"
        "finally:\n"
        "    dist.destroy_process_group()\n"
        "    os._exit(0)\n"
    )
    procs = []
    for r in range(3):
        p = subprocess.Popen([sys.executable, "-c", script, str(r), str(3)])
        procs.append(p)
        time.sleep(0.5)
    for p in procs:
        rc = p.wait()
        assert rc == 0, f"Parity worker exited with code {rc}"


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

    def test_two_rank_f32_sum_no_cpu_reduce(self):
        """Explicit MCCL_FP32_CPU_REDUCE=0: Metal/staging (same as MCCL default since fp32 CPU is opt-in)."""

        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps") * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), (
                f"Rank {rank}: expected {expected[0].item()}, got {tensor[0].item()}"
            )

        _run_distributed(
            fn,
            world_size=2,
            port=29620,
            extra_mccl_env={"MCCL_FP32_CPU_REDUCE": "0"},
        )

    def test_two_rank_f32_sum_cpu_reduce_enabled(self):
        """MCCL_FP32_CPU_REDUCE=1 exercises CPU unified-buffer float32 path (two-rank)."""

        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps") * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), (
                f"Rank {rank}: expected {expected[0].item()}, got {tensor[0].item()}"
            )

        _run_distributed(
            fn,
            world_size=2,
            port=29621,
            extra_mccl_env={"MCCL_FP32_CPU_REDUCE": "1"},
        )


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


class TestAllreduceBF16:
    """BFloat16 tensors use the Metal / compressed reduce path (not the fp32 CPU ring)."""

    def test_bf16_sum(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps", dtype=torch.bfloat16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps", dtype=torch.bfloat16) * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=33210)

    def test_bf16_avg(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps", dtype=torch.bfloat16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.ones(100, device="mps", dtype=torch.bfloat16) * 1.5
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=33220)

    def test_bf16_non_aligned_size(self):
        def fn(rank, world_size):
            tensor = torch.ones(13, device="mps", dtype=torch.bfloat16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(13, device="mps", dtype=torch.bfloat16) * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=33230)

    def test_bf16_large(self):
        def fn(rank, world_size):
            tensor = torch.randn(500_000, device="mps", dtype=torch.bfloat16)
            local_copy = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert not torch.allclose(tensor, local_copy)

        _run_distributed(fn, world_size=2, port=33240)


class TestThreeRankAllreduce:
    """3-rank allreduce (not the 2-rank fast path).

    Float tests use nbytes > default MCCL_SMALL_MSG_THRESHOLD. Most set
    ``MCCL_RING_ALGO=basic`` to pin plain ring; ``test_three_rank_sum_large_unset_ring_algo_env``
    omits ``MCCL_RING_ALGO`` so the default (plain ring, not ``ring_chunked``) is covered.
    """

    def test_three_rank_sum(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0), \
                f"Rank {rank}: max err {(tensor - expected).abs().max().item()}"

        _run_distributed(
            fn, world_size=3, port=33300, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_sum_large_unset_ring_algo_env(self):
        """Large 3-rank SUM with ``MCCL_RING_ALGO`` unset: default must stay plain ring."""

        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0), (
                f"Rank {rank}: max err {(tensor - expected).abs().max().item()}"
            )

        _run_distributed(fn, world_size=3, port=33307)

    def test_three_rank_sum_no_cpu_reduce(self):
        """Explicit MCCL_FP32_CPU_REDUCE=0 on plain ring (matches default Metal fp32 path)."""

        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0), (
                f"Rank {rank}: max err {(tensor - expected).abs().max().item()}"
            )

        env = {**THREE_RANK_PLAIN_RING_ENV, "MCCL_FP32_CPU_REDUCE": "0"}
        _run_distributed(fn, world_size=3, port=33310, extra_mccl_env=env)

    def test_three_rank_sum_cpu_reduce_enabled(self):
        """Plain ring with MCCL_FP32_CPU_REDUCE=1 (CPU float32 reduce on unified buffer)."""

        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0), (
                f"Rank {rank}: max err {(tensor - expected).abs().max().item()}"
            )

        env = {**THREE_RANK_PLAIN_RING_ENV, "MCCL_FP32_CPU_REDUCE": "1"}
        _run_distributed(fn, world_size=3, port=33311, extra_mccl_env=env)

    def test_three_rank_avg(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.full((n,), 2.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0)

        _run_distributed(
            fn, world_size=3, port=33400, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_min(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            expected = torch.ones(n, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0)

        _run_distributed(
            fn, world_size=3, port=33500, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_max(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            expected = torch.full((n,), 3.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0)

        _run_distributed(
            fn, world_size=3, port=33600, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_non_aligned(self):
        def fn(rank, world_size):
            # Large + not divisible by world_size (chunk padding edge case)
            n = THREE_RANK_PLAIN_RING_NUMEL + 1
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            assert torch.allclose(tensor, expected, rtol=0.0, atol=0.0)

        _run_distributed(
            fn, world_size=3, port=33700, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_large_ring(self):
        def fn(rank, world_size):
            tensor = torch.randn(500_000, device="mps")
            local_copy = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert not torch.allclose(tensor, local_copy)

        _run_distributed(
            fn, world_size=3, port=33800, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_f16(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL_F16
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float16)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float16)
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(
            fn, world_size=3, port=33900, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_bf16(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL_BF16
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.bfloat16)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.bfloat16)
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(
            fn, world_size=3, port=33910, extra_mccl_env=THREE_RANK_PLAIN_RING_ENV
        )

    def test_three_rank_sum_star_small(self):
        """Below small_msg_threshold: star hub path (regression for small 3-rank tensors)."""

        def fn(rank, world_size):
            tensor = torch.ones(100, device="mps") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="mps") * 6.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=3, port=33950)


class TestThreeRankChunkedRingNumeric:
    """Same size as plain-ring tests but MCCL_RING_ALGO=ring_chunked (allreduce_ring_chunked)."""

    def test_chunked_ring_sum_exact(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.float32)
            err = (tensor - expected).abs().max().item()
            assert err == 0.0, f"Rank {rank}: max abs err {err}"

        _run_distributed(
            fn, world_size=3, port=34051, extra_mccl_env={"MCCL_RING_ALGO": "ring_chunked"}
        )

    def test_chunked_ring_bf16_sum(self):
        def fn(rank, world_size):
            n = THREE_RANK_PLAIN_RING_NUMEL_BF16
            tensor = torch.full((n,), float(rank + 1), device="mps", dtype=torch.bfloat16)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.full((n,), 6.0, device="mps", dtype=torch.bfloat16)
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2), (
                f"Rank {rank}: max err {(tensor - expected).abs().max().item()}"
            )

        _run_distributed(
            fn, world_size=3, port=34061, extra_mccl_env={"MCCL_RING_ALGO": "ring_chunked"}
        )


class TestSequenceOrdering:
    def test_multiple_allreduce_in_order(self):
        def fn(rank, world_size):
            for i in range(10):
                tensor = torch.ones(64, device="mps") * (rank + 1 + i)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        _run_distributed(fn, world_size=2, port=30000)


@pytest.mark.timeout(900)
class TestMetalRingStress:
    """Many all_reduces on 3-rank plain ring + large tensors (Metal scale path).

    Override iteration count for local soak: ``MCCL_STRESS_ITERS=50000``.
    """

    def test_metal_stress_loop(self):
        n_iters = int(os.environ.get("MCCL_STRESS_ITERS", "2500"))
        sizes = (
            1024,
            THREE_RANK_PLAIN_RING_NUMEL,
            4096,
            THREE_RANK_PLAIN_RING_NUMEL // 2,
        )

        def fn(rank, world_size):
            for i in range(n_iters):
                s = sizes[i % len(sizes)]
                tensor = torch.randn(s, device="mps", dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                if i % 200 == 0 and i > 0:
                    torch.mps.synchronize()

        _run_distributed(
            fn,
            world_size=3,
            port=34100,
            extra_mccl_env=THREE_RANK_PLAIN_RING_ENV,
        )


class TestMcclVsGlooParity:
    """MCCL (MPS, ring) vs gloo CPU: same shapes and ops must match numerically."""

    def test_ring_bucket_sum_matches_gloo(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gloo_path = os.path.join(tmp, "ref_sum.pt")
            mccl_path = os.path.join(tmp, "mccl_sum.pt")
            _run_three_rank_parity_workers(34200, gloo_path, use_mccl=False, use_avg=False)
            _run_three_rank_parity_workers(34250, mccl_path, use_mccl=True, use_avg=False)
            g = torch.load(gloo_path, weights_only=True)
            m = torch.load(mccl_path, weights_only=True)
            torch.testing.assert_close(m, g, rtol=0.0, atol=0.0)

    def test_ring_bucket_avg_matches_gloo(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gloo_path = os.path.join(tmp, "ref_avg.pt")
            mccl_path = os.path.join(tmp, "mccl_avg.pt")
            _run_three_rank_parity_workers(34300, gloo_path, use_mccl=False, use_avg=True)
            _run_three_rank_parity_workers(34350, mccl_path, use_mccl=True, use_avg=True)
            g = torch.load(gloo_path, weights_only=True)
            m = torch.load(mccl_path, weights_only=True)
            torch.testing.assert_close(m, g, rtol=0.0, atol=0.0)
