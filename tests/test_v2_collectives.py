"""Phase D+: v2 collective correctness tests (allgather, reduce_scatter, send/recv).

Spawns processes on the same host. Requires macOS Apple Silicon with MCCL built.
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
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "DEBUG"

    import mccl  # noqa: F401

    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_distributed(fn, world_size=2, port=29500):
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


class TestAllgather:
    def test_basic(self):
        def fn(rank, world_size):
            input_tensor = torch.ones(10, device="mps") * (rank + 1)
            output_list = [torch.zeros(10, device="mps") for _ in range(world_size)]
            dist.all_gather(output_list, input_tensor)

            for r in range(world_size):
                expected = torch.ones(10, device="mps") * (r + 1)
                assert torch.allclose(output_list[r], expected), \
                    f"Rank {rank}: output[{r}] expected {r+1}, got {output_list[r][0].item()}"

        _run_distributed(fn, world_size=2, port=30100)

    def test_large(self):
        def fn(rank, world_size):
            input_tensor = torch.randn(100_000, device="mps") + rank
            output_list = [torch.zeros(100_000, device="mps") for _ in range(world_size)]
            dist.all_gather(output_list, input_tensor)

            assert output_list[rank].allclose(input_tensor, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=30200)


class TestReduceScatter:
    def test_basic(self):
        def fn(rank, world_size):
            input_list = [torch.ones(10, device="mps") * (i + 1) for i in range(world_size)]
            output = torch.zeros(10, device="mps")
            dist.reduce_scatter(output, input_list)

            # After reduce_scatter with SUM: each rank gets sum of all input_list[rank]
            # For 2 ranks: rank 0 gets input_list[0] of both ranks summed,
            #              rank 1 gets input_list[1] of both ranks summed.
            expected_val = (rank + 1) * world_size
            expected = torch.ones(10, device="mps") * expected_val
            assert torch.allclose(output, expected, rtol=1e-4), \
                f"Rank {rank}: expected {expected_val}, got {output[0].item()}"

        _run_distributed(fn, world_size=2, port=30300)


class TestSendRecv:
    def test_basic(self):
        def fn(rank, world_size):
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0, 3.0], device="mps")
                dist.send(tensor, dst=1)
            else:
                tensor = torch.zeros(3, device="mps")
                dist.recv(tensor, src=0)
                expected = torch.tensor([1.0, 2.0, 3.0], device="mps")
                assert torch.allclose(tensor, expected)

        _run_distributed(fn, world_size=2, port=30400)

    def test_bidirectional(self):
        def fn(rank, world_size):
            peer = 1 - rank
            send_tensor = torch.ones(100, device="mps") * (rank + 1)
            recv_tensor = torch.zeros(100, device="mps")

            if rank == 0:
                dist.send(send_tensor, dst=peer)
                dist.recv(recv_tensor, src=peer)
            else:
                dist.recv(recv_tensor, src=peer)
                dist.send(send_tensor, dst=peer)

            expected = torch.ones(100, device="mps") * (peer + 1)
            assert torch.allclose(recv_tensor, expected)

        _run_distributed(fn, world_size=2, port=30500)


class TestThreeRankAllgather:
    """3-rank allgather exercises the ring path."""

    def test_basic(self):
        def fn(rank, world_size):
            input_tensor = torch.ones(10, device="mps") * (rank + 1)
            output_list = [torch.zeros(10, device="mps") for _ in range(world_size)]
            dist.all_gather(output_list, input_tensor)

            for r in range(world_size):
                expected = torch.ones(10, device="mps") * (r + 1)
                assert torch.allclose(output_list[r], expected), \
                    f"Rank {rank}: output[{r}] expected {r+1}, got {output_list[r][0].item()}"

        _run_distributed(fn, world_size=3, port=34000)

    def test_non_aligned_size(self):
        def fn(rank, world_size):
            input_tensor = torch.ones(7, device="mps") * (rank + 1)
            output_list = [torch.zeros(7, device="mps") for _ in range(world_size)]
            dist.all_gather(output_list, input_tensor)

            for r in range(world_size):
                expected = torch.ones(7, device="mps") * (r + 1)
                assert torch.allclose(output_list[r], expected)

        _run_distributed(fn, world_size=3, port=34100)


class TestThreeRankReduceScatter:
    """3-rank reduce_scatter exercises the ring path."""

    def test_basic(self):
        def fn(rank, world_size):
            input_list = [torch.ones(10, device="mps") * (i + 1) for i in range(world_size)]
            output = torch.zeros(10, device="mps")
            dist.reduce_scatter(output, input_list)

            expected_val = (rank + 1) * world_size
            expected = torch.ones(10, device="mps") * expected_val
            assert torch.allclose(output, expected, rtol=1e-4), \
                f"Rank {rank}: expected {expected_val}, got {output[0].item()}"

        _run_distributed(fn, world_size=3, port=34200)

    def test_non_aligned_size(self):
        def fn(rank, world_size):
            input_list = [torch.ones(11, device="mps") * (i + 1) for i in range(world_size)]
            output = torch.zeros(11, device="mps")
            dist.reduce_scatter(output, input_list)

            expected_val = (rank + 1) * world_size
            expected = torch.ones(11, device="mps") * expected_val
            assert torch.allclose(output, expected, rtol=1e-4)

        _run_distributed(fn, world_size=3, port=34300)


class TestF16Collectives:
    """Half-precision coverage for v2 collectives."""

    def test_allgather_f16(self):
        def fn(rank, world_size):
            input_tensor = torch.ones(20, device="mps", dtype=torch.float16) * (rank + 1)
            output_list = [torch.zeros(20, device="mps", dtype=torch.float16)
                           for _ in range(world_size)]
            dist.all_gather(output_list, input_tensor)

            for r in range(world_size):
                expected = torch.ones(20, device="mps", dtype=torch.float16) * (r + 1)
                assert torch.allclose(output_list[r], expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=34400)

    def test_reduce_scatter_f16(self):
        def fn(rank, world_size):
            input_list = [torch.ones(10, device="mps", dtype=torch.float16) * (i + 1)
                          for i in range(world_size)]
            output = torch.zeros(10, device="mps", dtype=torch.float16)
            dist.reduce_scatter(output, input_list)

            expected_val = (rank + 1) * world_size
            expected = torch.ones(10, device="mps", dtype=torch.float16) * expected_val
            assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=34500)

    def test_send_recv_f16(self):
        def fn(rank, world_size):
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
                dist.send(tensor, dst=1)
            else:
                tensor = torch.zeros(3, device="mps", dtype=torch.float16)
                dist.recv(tensor, src=0)
                expected = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
                assert torch.allclose(tensor, expected, rtol=1e-2)

        _run_distributed(fn, world_size=2, port=34600)

    def test_broadcast_f16(self):
        def fn(rank, world_size):
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
            else:
                tensor = torch.zeros(3, device="mps", dtype=torch.float16)
            dist.broadcast(tensor, src=0)
            expected = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
            assert torch.allclose(tensor, expected, rtol=1e-2)

        _run_distributed(fn, world_size=2, port=34700)
