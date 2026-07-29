"""Ring allreduce smoke at world_size=8 (lab profile: basic ring, concurrency=1)."""

import os
import platform
import socket
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL ring ws8 test requires macOS Apple Silicon",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_allreduce(world_size: int) -> None:
    master_port = _free_port()
    mccl_port_base = _free_port()
    while mccl_port_base <= master_port + 500:
        mccl_port_base = _free_port()

    body = textwrap.dedent(
        """
        import torch
        import torch.distributed as dist

        rank = int(__import__('sys').argv[1])
        world_size = int(__import__('sys').argv[2])
        dev = torch.device('mps:0')
        dist.init_process_group(
            backend='mccl', rank=rank, world_size=world_size, device_id=dev,
        )
        t = torch.ones(256 * 1024, device=dev, dtype=torch.float32) * (rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size + 1) // 2)
        assert torch.allclose(t, torch.full_like(t, expected)), (t[0].item(), expected)
        dist.destroy_process_group()
        """
    )
    script = (
        "import os, sys\n"
        f"os.environ['MASTER_ADDR']='127.0.0.1'\n"
        f"os.environ['MASTER_PORT']='{master_port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR']='127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE']='{mccl_port_base}'\n"
        "os.environ['MCCL_RING_ALGO']='basic'\n"
        "os.environ['MCCL_COLLECTIVE_CONCURRENCY']='1'\n"
        "os.environ['MCCL_LOG_LEVEL']='WARN'\n"
        f"{body}"
    )
    procs = []
    for r in range(world_size):
        p = subprocess.Popen([sys.executable, "-c", script, str(r), str(world_size)])
        procs.append(p)
    for p in procs:
        rc = p.wait()
        assert rc == 0, f"rank worker exited {rc}"


@pytest.mark.timeout(180)
def test_allreduce_ring_ws8():
    _spawn_allreduce(8)
