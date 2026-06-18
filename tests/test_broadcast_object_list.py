"""broadcast_object_list repro — matches Lightning log_dir setup failure."""

import os
import platform
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="requires macOS Apple Silicon + MCCL",
)


def _run(body: str, world_size: int, port: int, extra_env: dict | None = None) -> None:
    extra_lines = ""
    if extra_env:
        for k, v in extra_env.items():
            extra_lines += f"os.environ[{k!r}] = {v!r}\n"
    script = (
        "import os, sys, torch, torch.distributed as dist\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        "os.environ['MCCL_OVERLAP_COMM'] = '1'\n"
        "os.environ['MCCL_COLLECTIVE_CONCURRENCY'] = '1'\n"
        f"{extra_lines}"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        "import mccl\n"
        "dist.init_process_group(backend='mccl', rank=rank, world_size=world_size, "
        "device_id=torch.device('mps:0'))\n"
        "try:\n"
        f"{textwrap.indent(body.strip(), '    ')}\n"
        "finally:\n"
        "    dist.destroy_process_group()\n"
        "    os._exit(0)\n"
    )
    procs = []
    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, "-c", script, str(r), str(world_size)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(p)
        time.sleep(0.3)
    for r, p in enumerate(procs):
        out, _ = p.communicate(timeout=120)
        assert p.returncode == 0, f"rank {r} failed:\n{out}"


class TestBroadcastObjectList:
    def test_mps_broadcast_ws5(self):
        _run(
            """
if rank == 0:
    t = torch.tensor([1.0, 2.0, 3.0], device="mps")
else:
    t = torch.zeros(3, device="mps")
dist.broadcast(t, src=0)
assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0], device="mps"))
""",
            world_size=5,
            port=36500,
        )

    def test_broadcast_object_list_ws5(self):
        _run(
            """
obj = ["/tmp/lightning_logs/version_0"] if rank == 0 else [None]
dist.broadcast_object_list(obj, src=0)
assert obj[0] == "/tmp/lightning_logs/version_0", repr(obj)
""",
            world_size=5,
            port=36600,
        )

    def test_broadcast_object_list_default_mps_device_ws5(self):
        _run(
            """
torch.set_default_device("mps")
obj = ["/tmp/lightning_logs/version_0"] if rank == 0 else [None]
dist.broadcast_object_list(obj, src=0)
assert obj[0] == "/tmp/lightning_logs/version_0", repr(obj)
""",
            world_size=5,
            port=36700,
        )

    def test_allgather_int64_ws6(self):
        """DDP param verify exchanges small int64 metadata via all_gather."""
        _run(
            """
n_trainable = 111 + rank
inp = torch.tensor([n_trainable], dtype=torch.long, device="cpu")
outs = [torch.zeros(1, dtype=torch.long, device="cpu") for _ in range(world_size)]
dist.all_gather(outs, inp)
expected = [111 + r for r in range(world_size)]
got = [int(t.item()) for t in outs]
assert got == expected, f"rank {rank}: got {got} expected {expected}"
""",
            world_size=6,
            port=36800,
        )

    def test_allgather_int64_ws8_ring_pipeline(self):
        """DDP init_sync verify: int64 allgather with MCCL_RING_PIPELINE=1 (prod submit defaults)."""
        _run(
            """
n_trainable = 111
inp = torch.tensor([n_trainable], dtype=torch.long, device="cpu")
outs = [torch.zeros(1, dtype=torch.long, device="cpu") for _ in range(world_size)]
dist.all_gather(outs, inp)
expected = [111 for _ in range(world_size)]
got = [int(t.item()) for t in outs]
assert got == expected, f"rank {rank}: got {got} expected {expected}"
""",
            world_size=8,
            port=37000,
            extra_env={"MCCL_RING_PIPELINE": "1", "MCCL_COLLECTIVE_CONCURRENCY": "2"},
        )

    def test_ddp_init_sync_ws8_ring_pipeline(self):
        """Full DDP(init_sync=True) wrap — reproduces Lightning MPSMCCLDDPStrategy path."""
        _run(
            """
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to("mps")
n = sum(1 for p in model.parameters() if p.requires_grad)
ddp = DDP(model, init_sync=True)
assert n > 0, f"rank {rank}: empty model"
""",
            world_size=8,
            port=37100,
            extra_env={"MCCL_RING_PIPELINE": "1", "MCCL_COLLECTIVE_CONCURRENCY": "2"},
        )

    def test_broadcast_object_list_ws8(self):
        _run(
            """
obj = ["/tmp/lightning_logs/version_0"] if rank == 0 else [None]
dist.broadcast_object_list(obj, src=0)
assert obj[0] == "/tmp/lightning_logs/version_0", repr(obj)
""",
            world_size=8,
            port=36900,
        )
