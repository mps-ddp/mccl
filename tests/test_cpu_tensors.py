"""CPU tensor support tests via unified memory.

Tests that MCCL can handle CPU tensors on Apple Silicon by wrapping them
as MTLStorageModeShared buffers, leveraging unified memory.
"""

import platform
import os
import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="CPU tensor tests require macOS on Apple Silicon",
)


def _run_distributed(fn, world_size=2, port=29500):
    """Spawn world_size processes running fn(rank, world_size)."""
    import subprocess, sys, textwrap, inspect
    src = textwrap.dedent(inspect.getsource(fn))
    script = (
        "import os, sys, torch, torch.distributed as dist\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        f"os.environ['MCCL_LOG_LEVEL'] = 'DEBUG'\n"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        "import mccl\n"
        "dist.init_process_group(backend='mccl', rank=rank, world_size=world_size)\n"
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


class TestCPUAllreduce:
    def test_cpu_f32_sum(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="cpu") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="cpu") * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4), \
                f"Rank {rank}: expected {expected[0].item()}, got {tensor[0].item()}"

        _run_distributed(fn, world_size=2, port=40000)

    def test_cpu_f32_avg(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="cpu") * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            expected = torch.ones(100, device="cpu") * 1.5
            assert torch.allclose(tensor, expected, rtol=1e-4, atol=1e-4)

        _run_distributed(fn, world_size=2, port=40100)

    def test_cpu_f16(self):
        def fn(rank, world_size):
            tensor = torch.ones(100, device="cpu", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(100, device="cpu", dtype=torch.float16) * 3.0
            assert torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=40200)


class TestCPUBroadcast:
    def test_cpu_broadcast(self):
        def fn(rank, world_size):
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
            else:
                tensor = torch.zeros(3, device="cpu")
            dist.broadcast(tensor, src=0)
            expected = torch.tensor([1.0, 2.0, 3.0], device="cpu")
            assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=40300)


class TestMixedDevices:
    def test_cpu_and_mps_separate_collectives(self):
        """Verify CPU and MPS collectives can coexist in same process group."""
        def fn(rank, world_size):
            # CPU allreduce
            cpu_tensor = torch.ones(50, device="cpu") * (rank + 1)
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM)
            assert torch.allclose(cpu_tensor, torch.ones(50) * 3.0, rtol=1e-4)

            # MPS allreduce
            mps_tensor = torch.ones(50, device="mps") * (rank + 2)
            dist.all_reduce(mps_tensor, op=dist.ReduceOp.SUM)
            expected_mps = torch.ones(50, device="mps") * 5.0
            assert torch.allclose(mps_tensor, expected_mps, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=40400)
