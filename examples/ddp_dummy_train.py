#!/usr/bin/env python3
"""
Minimal DDP training on dummy data using MCCL + MPS.

Run with torch.distributed.run (``torchrun``). MCCL must be importable on every node.

**Two Macs (1 process per machine)** — pick an IP on machine A that machine B can reach
(Thunderbolt bridge ``169.254.x.x`` or LAN). Open firewall for ``MASTER_PORT`` and
``MCCL_PORT_BASE .. MCCL_PORT_BASE + world_size - 1`` (defaults: master_port + 100).

Machine A (rank 0 / master)::

    export MASTER_ADDR=192.168.1.10   # or 169.254.x.x on TB bridge
    export MASTER_PORT=29500
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        examples/ddp_dummy_train.py

Machine B::

    export MASTER_ADDR=192.168.1.10   # same as machine A
    export MASTER_PORT=29500
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        examples/ddp_dummy_train.py

**Single Mac (smoke test, 2 processes, shares one GPU)**::

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/ddp_dummy_train.py

Optional env (see MCCL docs): ``MCCL_LISTEN_ADDR``, ``MCCL_PORT_BASE``, ``MCCL_TRANSPORT``.
"""
from __future__ import annotations

import os
import sys

# Register MCCL with torch.distributed before touching the process group.
import mccl  # noqa: F401

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _setup_mccl_env() -> None:
    """Avoid MCCL listen port colliding with PyTorch's TCP store on MASTER_PORT."""
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        # Reserve MASTER_PORT for c10d store; MCCL uses port_base + rank.
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)

    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


def main() -> None:
    _setup_mccl_env()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size < 2:
        print(
            "Need world_size >= 2 (MCCL transport). Examples:\n"
            "  torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 ...\n"
            "  torchrun --nproc_per_node=1 --nnodes=2 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    if not torch.backends.mps.is_available():
        print("MPS is not available; this example expects Apple Silicon + MPS.", file=sys.stderr)
        sys.exit(1)

    # Each Mac in this setup has a single MPS device (index 0). LOCAL_RANK is still
    # set by torchrun and is usually 0 when --nproc_per_node=1.
    device = torch.device("mps:0")

    dist.init_process_group(
        backend="mccl",
        device_id=device,
    )

    torch.manual_seed(42 + rank)

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    ddp = DDP(model)

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    steps = int(os.environ.get("TRAIN_STEPS", "20"))
    batch = 16
    dim = 32

    if rank == 0:
        print(
            f"DDP dummy train | world_size={world_size} rank={rank} "
            f"local_rank={local_rank} device={device} steps={steps}",
            flush=True,
        )

    for step in range(steps):
        # Different noise per rank (simulates sharded batches); gradients still align via DDP.
        x = torch.randn(batch, dim, device=device)
        y = torch.randn(batch, 1, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(ddp(x), y)
        loss.backward()
        optimizer.step()

        if rank == 0 and (step % 5 == 0 or step == steps - 1):
            print(f"  step {step:4d}  loss={loss.item():.6f}", flush=True)

    # Sanity: first few weights match across ranks (MCCL broadcast requires MPS tensors).
    head = next(model.parameters()).detach().flatten()[:8].to(device)
    ref = head.clone()
    dist.broadcast(ref, src=0)
    if not torch.allclose(head, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Parameter mismatch across ranks after DDP — check MCCL / network.")

    if rank == 0:
        print("Done. Parameters are in sync across ranks.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
