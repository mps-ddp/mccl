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

**Single GPU baseline (for performance comparison)**::

    SINGLE_GPU=1 TRAIN_STEPS=100 BATCH_SIZE=64 python examples/ddp_dummy_train.py

Optional env (see MCCL docs): ``MCCL_LISTEN_ADDR``, ``MCCL_PORT_BASE``, ``MCCL_TRANSPORT``.
Training env: ``TRAIN_STEPS`` (default 100), ``BATCH_SIZE`` (default 64 per rank).

**Why it looks like a "hang"**

1. **Torch elastic** — With ``--nnodes=2``, the launcher often **waits until both nodes join**
   before worker processes start. Starting only machine A can show **no script output** for a long
   time; that is normal until machine B runs the same command.

2. **PyTorch store** — Every rank must reach ``init_process_group``. If machine B cannot open
   TCP to ``MASTER_ADDR:MASTER_PORT`` (default 29500), it retries forever. **Open inbound TCP
   on the master** for that port (macOS Firewall / Security).

3. **MCCL** — After the store works, ranks exchange ``IP:port`` for MCCL's own sockets. If the
   published address is wrong (e.g. loopback or another interface), peers **hang in
   ``connect_all``**. On each Mac set ``MCCL_LISTEN_ADDR`` to that machine's **reachable**
   unicast IP on the link between the two (not ``.255`` broadcast, not ``127.0.0.1``).

4. **Ports to allow** (world_size=2, default): ``MASTER_PORT`` (e.g. 29500) and
   ``MCCL_PORT_BASE`` … ``MCCL_PORT_BASE+1`` (e.g. 29600–29601 if ``MCCL_PORT_BASE=29600``).

Debug: ``export TORCH_DISTRIBUTED_DEBUG=DETAIL`` and ``export PYTHONUNBUFFERED=1``.
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


def single_gpu_baseline() -> None:
    """Run single GPU training for comparison."""
    import time
    
    device = torch.device("mps")
    torch.manual_seed(42)
    
    # Same simple model as DDP version
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    steps = int(os.environ.get("TRAIN_STEPS", "30"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Single GPU baseline | device={device}\n"
        f"  Model: {total_params:,} params\n"
        f"  Batch: {batch_size}\n"
        f"  Steps: {steps}",
        flush=True,
    )
    
    warmup_steps = 5
    step_times = []
    losses = []
    
    for step in range(warmup_steps + steps):
        torch.manual_seed(1000 + step)
        x = torch.randn(batch_size, 256, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        start_time = time.perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        step_time = time.perf_counter() - start_time
        
        if step >= warmup_steps:
            step_times.append(step_time)
            losses.append(loss.item())
            
        if step % 5 == 0 or step == warmup_steps + steps - 1:
            status = "warmup" if step < warmup_steps else "train"
            print(f"  {status} step {step:4d}  loss={loss.item():.6f}  time={step_time:.3f}s", flush=True)
    
    if step_times:
        avg_time = sum(step_times) / len(step_times)
        print(
            f"\n=== Single GPU Stats ===\n"
            f"  Steps completed: {len(step_times)}\n"
            f"  Avg step time: {avg_time:.3f}s ({1/avg_time:.1f} steps/sec)\n"
            f"  Final loss: {losses[-1]:.6f}\n"
            f"  Throughput: {batch_size / avg_time:.1f} samples/sec",
            flush=True,
        )


def main() -> None:
    # Single GPU mode for comparison
    if os.environ.get("SINGLE_GPU"):
        single_gpu_baseline()
        return
        
    # Print before any distributed init so you see progress if something blocks later.
    print(
        "[ddp_dummy_train] starting "
        f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
        flush=True,
    )

    _setup_mccl_env()
    print(
        f"[ddp_dummy_train] MCCL_PORT_BASE={os.environ.get('MCCL_PORT_BASE')} "
        f"MCCL_LISTEN_ADDR={os.environ.get('MCCL_LISTEN_ADDR', '(unset)')}",
        flush=True,
    )

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

    print(f"[ddp_dummy_train] rank {rank}: calling init_process_group(mccl)...", flush=True)
    dist.init_process_group(
        backend="mccl",
        device_id=device,
    )
    print(f"[ddp_dummy_train] rank {rank}: init_process_group done", flush=True)

    torch.manual_seed(42 + rank)

    # Simple model that won't crash Metal during gradient sync
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    ddp = DDP(model, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(ddp.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    steps = int(os.environ.get("TRAIN_STEPS", "30"))  # Reasonable for testing
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))  # Smaller batch to be safe
    input_dim = 256

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(
            f"DDP training | world_size={world_size} device={device}\n"
            f"  Model: {total_params:,} params ({trainable_params:,} trainable)\n"
            f"  Batch: {batch_size} per rank ({batch_size * world_size} global)\n"
            f"  Steps: {steps}",
            flush=True,
        )

    # Warmup to get stable timing
    warmup_steps = 5
    import time

    step_times = []
    losses = []
    
    for step in range(warmup_steps + steps):
        # Different data per rank (simulates data sharding)
        torch.manual_seed(1000 + step * world_size + rank)
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)  # 10-class classification

        start_time = time.perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = loss_fn(logits, y)
        loss.backward()  # This triggers gradient allreduce via MCCL
        optimizer.step()
        
        step_time = time.perf_counter() - start_time
        
        # Skip warmup for timing stats
        if step >= warmup_steps:
            step_times.append(step_time)
            losses.append(loss.item())

        if rank == 0 and (step % 5 == 0 or step == warmup_steps + steps - 1):
            status = "warmup" if step < warmup_steps else "train"
            print(f"  {status} step {step:4d}  loss={loss.item():.6f}  time={step_time:.3f}s", flush=True)

    # Performance stats
    if step_times:
        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)
        
        # Collect MCCL metrics if available
        try:
            metrics = mccl.get_metrics()
            if metrics:
                mccl_info = (f"MCCL: {metrics.total_ops} ops, "
                           f"{metrics.total_bytes_sent/1e6:.1f}MB sent, "
                           f"{metrics.total_bytes_recv/1e6:.1f}MB recv, "
                           f"avg_lat={metrics.avg_latency_ms:.2f}ms")
            else:
                mccl_info = "MCCL metrics unavailable"
        except:
            mccl_info = "MCCL metrics error"
        
        if rank == 0:
            print(
                f"\n=== Training Stats (rank {rank}) ===\n"
                f"  Steps completed: {len(step_times)}\n"
                f"  Avg step time: {avg_time:.3f}s ({1/avg_time:.1f} steps/sec)\n"
                f"  Min/Max time: {min_time:.3f}s / {max_time:.3f}s\n"
                f"  Final loss: {losses[-1]:.6f} (started: {losses[0]:.6f})\n"
                f"  {mccl_info}\n"
                f"  Global batch size: {batch_size * world_size}\n"
                f"  Throughput: {batch_size * world_size / avg_time:.1f} samples/sec",
                flush=True,
            )

    # Sanity: parameters stay synced
    head = next(model.parameters()).detach().flatten()[:8].to(device)
    ref = head.clone()
    dist.broadcast(ref, src=0)
    if not torch.allclose(head, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Parameter mismatch across ranks after DDP — check MCCL / network.")

    if rank == 0:
        print("✓ Parameters are in sync across ranks.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
