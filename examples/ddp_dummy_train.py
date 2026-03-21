#!/usr/bin/env python3
"""
Minimal DDP training on dummy data using MCCL + MPS.

Run with torch.distributed.run (``torchrun``). MCCL must be importable on every node.

**Single-GPU baseline (same model/hparams as DDP, no torchrun)**::

    python examples/ddp_dummy_train.py --baseline

Or: ``SINGLE_GPU=1 python examples/ddp_dummy_train.py`` (same behavior).

**Single Mac (2 processes, one GPU)**::

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/ddp_dummy_train.py

**Compare single vs DDP on one machine**::

    ./examples/compare_single_vs_ddp.sh

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

**Model size** — defaults target **~100M+ parameters** so each step does enough matmul that
communication is not the whole story (multi-node / multi-GPU DDP is more meaningful at this scale).
Override with env:

- ``INPUT_DIM`` (default 2048)
- ``NUM_CLASSES`` (default 128)
- ``MODEL_HIDDEN`` (default 4096) — width of hidden blocks
- ``MODEL_DEPTH`` (default 8) — number of ``Linear+ReLU`` hidden blocks

If MPS runs out of memory, lower ``BATCH_SIZE`` (e.g. 8–16) or reduce ``MODEL_HIDDEN`` / ``MODEL_DEPTH``.

Optional env (see MCCL docs): ``MCCL_LISTEN_ADDR``, ``MCCL_PORT_BASE``, ``MCCL_TRANSPORT``.
Training env: ``TRAIN_STEPS`` (default 30), ``BATCH_SIZE`` (default 16 per rank).

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

import argparse
import os
import sys

# Register MCCL with torch.distributed before touching the process group.
import mccl  # noqa: F401

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _model_dims_from_env() -> tuple[int, int, int, int]:
    # Defaults ~126M params — heavy enough that DDP/comm overhead is a smaller slice of step time.
    input_dim = int(os.environ.get("INPUT_DIM", "2048"))
    num_classes = int(os.environ.get("NUM_CLASSES", "128"))
    hidden = int(os.environ.get("MODEL_HIDDEN", "4096"))
    depth = int(os.environ.get("MODEL_DEPTH", "8"))
    return input_dim, num_classes, hidden, depth


def build_dummy_classifier() -> nn.Sequential:
    """Wide/deep MLP for MCCL stress; dims from env (defaults favor compute over comm)."""
    input_dim, num_classes, hidden, depth = _model_dims_from_env()
    if depth < 1:
        raise ValueError("MODEL_DEPTH must be >= 1")
    layers: list[nn.Module] = []
    in_f = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_f, hidden))
        layers.append(nn.ReLU())
        in_f = hidden
    layers.append(nn.Linear(in_f, num_classes))
    return nn.Sequential(*layers)


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
    """Run single GPU training for comparison (same model + hparams as DDP path)."""
    import time

    device = torch.device("mps")
    torch.manual_seed(42)

    model = build_dummy_classifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    steps = int(os.environ.get("TRAIN_STEPS", "30"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    input_dim, num_classes, _, _ = _model_dims_from_env()

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Single GPU baseline | device={device}\n"
        f"  Model: {total_params:,} params\n"
        f"  Batch: {batch_size}\n"
        f"  Steps: {steps}\n"
        f"  INPUT_DIM={input_dim} NUM_CLASSES={num_classes} "
        f"(MODEL_HIDDEN/MODEL_DEPTH from env)",
        flush=True,
    )

    warmup_steps = 5
    step_times = []
    losses = []

    for step in range(warmup_steps + steps):
        torch.manual_seed(1000 + step)
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

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
            print(
                f"  {status} step {step:4d}  loss={loss.item():.6f}  time={step_time:.3f}s",
                flush=True,
            )

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
    parser = argparse.ArgumentParser(
        description="DDP dummy train (MCCL+MPS) or single-GPU baseline for comparison.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Single-process MPS run (no torchrun); same model/env as DDP for fair comparison.",
    )
    args = parser.parse_args()

    if args.baseline or os.environ.get("SINGLE_GPU"):
        if not torch.backends.mps.is_available():
            print("MPS is not available; this example expects Apple Silicon + MPS.", file=sys.stderr)
            sys.exit(1)
        single_gpu_baseline()
        return

    if "RANK" not in os.environ:
        print(
            "Distributed mode requires torchrun (RANK/WORLD_SIZE in env).\n"
            "  Single-GPU baseline:  python examples/ddp_dummy_train.py --baseline\n"
            "  DDP (2 ranks local): torchrun --nproc_per_node=2 --nnodes=1 "
            "--master_addr=127.0.0.1 --master_port=29500 examples/ddp_dummy_train.py",
            file=sys.stderr,
        )
        sys.exit(2)

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
    master_addr = os.environ.get("MASTER_ADDR", "")
    if master_addr and master_addr not in ("127.0.0.1", "localhost", "::1"):
        if "MCCL_LISTEN_ADDR" not in os.environ:
            print(
                "[ddp_dummy_train] hint: multi-node? set MCCL_LISTEN_ADDR on **each** machine to "
                "that machine's own IP on the peer link (not MASTER_ADDR). "
                "Open firewall for MCCL_PORT_BASE..+world_size-1.",
                flush=True,
            )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Try to avoid Metal command buffer issues during gradient sync
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    if world_size < 2:
        print(
            "Need world_size >= 2 (MCCL transport). Examples:\n"
            "  torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 ...\n"
            "  torchrun --nproc_per_node=1 --nnodes=2 ...\n"
            "Or use: python examples/ddp_dummy_train.py --baseline",
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

    # DDP() runs cross-rank collectives; if one rank is slow/OOM/stuck, others block here.
    print(f"[ddp_dummy_train] rank {rank}: allocating model on {device}...", flush=True)
    model = build_dummy_classifier().to(device)
    print(f"[ddp_dummy_train] rank {rank}: wrapping DDP (syncs with other ranks)...", flush=True)
    ddp = DDP(model, find_unused_parameters=False)
    print(f"[ddp_dummy_train] rank {rank}: DDP wrapper ready", flush=True)

    optimizer = torch.optim.AdamW(ddp.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    steps = int(os.environ.get("TRAIN_STEPS", "30"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    input_dim, num_classes, _, _ = _model_dims_from_env()

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        print(
            f"DDP training | world_size={world_size} device={device}\n"
            f"  Model: {total_params:,} params ({trainable_params:,} trainable)\n"
            f"  Batch: {batch_size} per rank ({batch_size * world_size} global)\n"
            f"  Steps: {steps}\n"
            f"  INPUT_DIM={input_dim} NUM_CLASSES={num_classes}",
            flush=True,
        )

    # Warmup to get stable timing
    warmup_steps = 5
    import time

    step_times = []
    losses = []

    verbose = os.environ.get("DDP_DUMMY_VERBOSE", "")

    for step in range(warmup_steps + steps):
        # Different data per rank (simulates data sharding)
        torch.manual_seed(1000 + step * world_size + rank)
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: forward pass...", flush=True)

        # Use no_sync() so DDP does NOT fire allreduce hooks inside backward().
        # DDP's hooks call our MCCL allreduce, which calls torch::mps::synchronize().
        # On MPS, PyTorch may have an open command encoder while backward is running;
        # calling synchronize() on that encoder triggers a Metal assertion crash.
        # By deferring allreduce until after backward() returns (encoder is closed),
        # we guarantee synchronize() is called at a safe point.
        with ddp.no_sync():
            logits = ddp(x)
            loss = loss_fn(logits, y)

            if verbose and rank == 0 and step < 3:
                print(f"    step {step}: backward pass...", flush=True)

            loss.backward()

        # backward() has returned — MPS encoder is committed and closed.
        # Safe to allreduce: torch::mps::synchronize() inside MCCL won't conflict.
        #
        # Flatten all gradients into one contiguous buffer and allreduce in a single
        # collective op. This avoids one TCP round-trip per parameter tensor, which
        # at 30ms+ latency per op would dominate step time for large models.
        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: allreduce (flat)...", flush=True)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            for grad, updated in zip(
                grads, torch._utils._unflatten_dense_tensors(flat, grads)
            ):
                grad.copy_(updated)

        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: optimizer step...", flush=True)
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
                mccl_info = (
                    f"MCCL: {metrics.total_ops} ops, "
                    f"{metrics.total_bytes_sent/1e6:.1f}MB sent, "
                    f"{metrics.total_bytes_recv/1e6:.1f}MB recv, "
                    f"avg_lat={metrics.avg_latency_ms:.2f}ms"
                )
            else:
                mccl_info = "MCCL metrics unavailable"
        except Exception:
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
