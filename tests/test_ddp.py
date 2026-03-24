"""DDP end-to-end tests over the MCCL backend.

Spawns processes on the same host to verify that
torch.nn.parallel.DistributedDataParallel works with the MCCL backend
on MPS tensors.  Requires macOS Apple Silicon with MCCL built.
"""

import platform
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="DDP tests require macOS on Apple Silicon",
)


def _run_distributed(fn, world_size=2, port=None):
    import subprocess, sys, textwrap, inspect, socket
    
    # Dynamically allocate free ports to avoid collisions
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]
    
    # If port not specified, get a free one; otherwise use provided port (for compatibility)
    master_port = port if port is not None else get_free_port()
    
    # Allocate MCCL_PORT_BASE with sufficient spacing to avoid overlap
    # Use a separate free port starting from master_port + 1000 to ensure no collision
    mccl_port_base = get_free_port()
    while mccl_port_base <= master_port + 500:  # Ensure sufficient spacing
        mccl_port_base = get_free_port()
    
    src = textwrap.dedent(inspect.getsource(fn))
    script = (
        "import os, sys, torch, torch.nn as nn, torch.distributed as dist\n"
        "from torch.nn.parallel import DistributedDataParallel as DDP\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{master_port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{mccl_port_base}'\n"
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


class TestDDPBasicTraining:
    """Wrap a Linear layer in DDP, run a few training steps, verify convergence."""

    def test_ddp_basic_training(self):
        def fn(rank, world_size):
            torch.manual_seed(42)
            model = nn.Linear(16, 1, bias=False).to("mps")
            ddp_model = DDP(model)

            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            losses = []
            for step in range(5):
                torch.manual_seed(1000 + step * world_size + rank)
                x = torch.randn(8, 16, device="mps")
                y = torch.ones(8, 1, device="mps")

                optimizer.zero_grad()
                out = ddp_model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            assert losses[-1] < losses[0], \
                f"Rank {rank}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"

            # Verify parameters are identical across ranks
            for p in model.parameters():
                ref = p.data.clone()
                dist.broadcast(ref, src=0)
                assert torch.allclose(p.data, ref, rtol=1e-4, atol=1e-4), \
                    f"Rank {rank}: params diverged after training"

        _run_distributed(fn, world_size=2)


class TestDDPGradientSync:
    """Feed different inputs per rank, verify gradients are identical after backward."""

    def test_ddp_gradient_sync(self):
        def fn(rank, world_size):
            torch.manual_seed(42)
            model = nn.Linear(8, 4, bias=True).to("mps")
            ddp_model = DDP(model)
            loss_fn = nn.MSELoss()

            torch.manual_seed(100 + rank)
            x = torch.randn(4, 8, device="mps")
            y = torch.randn(4, 4, device="mps")

            ddp_model.zero_grad()
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()

            for name, p in model.named_parameters():
                assert p.grad is not None, f"Rank {rank}: no grad for {name}"
                grad_ref = p.grad.data.clone()
                dist.broadcast(grad_ref, src=0)
                assert torch.allclose(p.grad.data, grad_ref, rtol=1e-4, atol=1e-4), \
                    f"Rank {rank}: gradient mismatch for {name}"

        _run_distributed(fn, world_size=2)


class TestDDPMultiLayer:
    """Multi-layer MLP verifies DDP handles multiple parameter groups."""

    def test_ddp_multi_layer(self):
        def fn(rank, world_size):
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ).to("mps")
            ddp_model = DDP(model)

            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            losses = []
            for step in range(5):
                torch.manual_seed(2000 + step * world_size + rank)
                x = torch.randn(8, 16, device="mps")
                y = torch.ones(8, 1, device="mps")

                optimizer.zero_grad()
                out = ddp_model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            assert losses[-1] < losses[0], \
                f"Rank {rank}: MLP loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"

            for name, p in model.named_parameters():
                ref = p.data.clone()
                dist.broadcast(ref, src=0)
                assert torch.allclose(p.data, ref, rtol=1e-4, atol=1e-4), \
                    f"Rank {rank}: param {name} diverged"

        _run_distributed(fn, world_size=2)


class TestDDPThreeRank:
    """3-rank DDP exercises the ring allreduce path."""

    def test_ddp_three_rank(self):
        def fn(rank, world_size):
            torch.manual_seed(42)
            model = nn.Linear(16, 1, bias=False).to("mps")
            ddp_model = DDP(model)

            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            losses = []
            for step in range(5):
                torch.manual_seed(3000 + step * world_size + rank)
                x = torch.randn(8, 16, device="mps")
                y = torch.ones(8, 1, device="mps")

                optimizer.zero_grad()
                out = ddp_model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            assert losses[-1] < losses[0], \
                f"Rank {rank}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"

            for p in model.parameters():
                ref = p.data.clone()
                dist.broadcast(ref, src=0)
                assert torch.allclose(p.data, ref, rtol=1e-4, atol=1e-4), \
                    f"Rank {rank}: params diverged after 3-rank training"

        _run_distributed(fn, world_size=3)
