#!/usr/bin/env python3
"""
DDP training benchmark on a synthetic task. Supports MCCL and Gloo backends.

Examples:

    # Single-GPU baseline (no torchrun needed)
    python examples/ddp_dummy_train.py --baseline

    # 2-rank DDP on one Mac with MCCL (default)
    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
        examples/ddp_dummy_train.py

    # Same thing with Gloo for comparison
    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
        examples/ddp_dummy_train.py --backend gloo

    # 2-node MCCL over Thunderbolt
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=169.254.238.250 --master_port=29500 \
        examples/ddp_dummy_train.py --steps 100 --batch-size 128

    # 2-node Gloo for comparison (CPU tensors)
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=169.254.238.250 --master_port=29500 \
        examples/ddp_dummy_train.py --backend gloo --steps 100 --batch-size 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_utils import SyntheticDataset, build_model, write_training_stats


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDP training benchmark (MCCL / Gloo)")
    p.add_argument("--backend", choices=["mccl", "gloo"], default="mccl",
                   help="Distributed backend (default: mccl)")
    p.add_argument("--baseline", action="store_true",
                   help="Single-GPU run (no torchrun), same model for fair comparison")
    p.add_argument("--steps", type=int, default=500,
                   help="Training steps after warmup (default: 500)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Per-rank batch size (default: 128)")
    p.add_argument("--bucket-mb", type=int, default=25,
                   help="DDP gradient bucket size in MB (default: 25)")
    p.add_argument("--input-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--stress-model", action="store_true",
                   help="Use large model defaults (2048/128/8192/16)")
    p.add_argument("--fp16", action="store_true",
                   help="Enable torch.autocast fp16 for forward + loss")
    p.add_argument("--save-stats", metavar="PATH", default=None,
                   help="Write timing JSON (rank 0 only)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup steps before timing (default: 5)")
    return p.parse_args()


def get_model_dims(args) -> tuple[int, int, int, int]:
    if args.stress_model:
        return (
            args.input_dim if args.input_dim != 512 else 2048,
            args.num_classes if args.num_classes != 64 else 128,
            args.hidden if args.hidden != 1024 else 8192,
            args.depth if args.depth != 4 else 16,
        )
    return args.input_dim, args.num_classes, args.hidden, args.depth


# ── Single-GPU baseline ─────────────────────────────────────────────

def run_baseline(args) -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("mps")
    torch.manual_seed(42)

    dims = get_model_dims(args)
    model = build_model(*dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    global_batch = args.batch_size * 2
    dataset = SyntheticDataset(dims[0], dims[1])
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Single GPU baseline | device={device}\n"
          f"  Model: {total_params:,} params\n"
          f"  Batch: {global_batch} | Steps: {args.steps}", flush=True)

    step_times: list[float] = []
    losses: list[float] = []

    for step in range(args.warmup + args.steps):
        x, y = dataset.get_batch(global_batch, device, step, rank=0, world_size=1)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        dt = time.perf_counter() - t0

        if step >= args.warmup:
            step_times.append(dt)
            losses.append(loss.item())
        if step % 5 == 0:
            tag = "warmup" if step < args.warmup else "train"
            print(f"  {tag} {step:4d}  loss={loss.item():.6f}  {dt:.3f}s", flush=True)

    if step_times:
        avg = sum(step_times) / len(step_times)
        print(f"\n=== Baseline ===\n"
              f"  Avg: {avg:.3f}s  Throughput: {global_batch/avg:.0f} samples/s\n"
              f"  Final loss: {losses[-1]:.6f}", flush=True)
    if args.save_stats and step_times:
        write_training_stats(args.save_stats, "baseline", step_times, losses,
                             global_batch, 1, total_params)


# ── DDP training ─────────────────────────────────────────────────────

def run_ddp(args) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size < 2:
        print("Need world_size >= 2. Use --baseline for single-GPU.", file=sys.stderr)
        sys.exit(1)

    # Backend setup
    if args.backend == "mccl":
        import mccl  # noqa: F401
        _setup_mccl_env()
        if not torch.backends.mps.is_available():
            print("MPS not available (required for mccl backend).", file=sys.stderr)
            sys.exit(1)
        device = torch.device("mps:0")
        dist.init_process_group(backend="mccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    if rank == 0:
        print(f"[{args.backend}] rank={rank} world_size={world_size} device={device}", flush=True)

    torch.manual_seed(42)
    dims = get_model_dims(args)
    model = build_model(*dims).to(device)
    torch.manual_seed(42 + rank)

    ddp_model = DDP(model, find_unused_parameters=False, bucket_cap_mb=args.bucket_mb)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dataset = SyntheticDataset(dims[0], dims[1])
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"DDP training | backend={args.backend} world_size={world_size}\n"
              f"  Model: {total_params:,} params\n"
              f"  Batch: {args.batch_size}/rank ({args.batch_size * world_size} global)\n"
              f"  Steps: {args.steps}  bucket_mb={args.bucket_mb}"
              f"{'  fp16' if args.fp16 else ''}", flush=True)

    step_times: list[float] = []
    losses: list[float] = []

    for step in range(args.warmup + args.steps):
        x, y = dataset.get_batch(args.batch_size, device, step, rank, world_size)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        if args.fp16 and args.backend == "mccl":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                loss = loss_fn(ddp_model(x), y)
        else:
            loss = loss_fn(ddp_model(x), y)

        loss.backward()
        optimizer.step()
        dt = time.perf_counter() - t0

        if step >= args.warmup:
            step_times.append(dt)
            losses.append(loss.item())
        if rank == 0 and (step % 5 == 0 or step == args.warmup + args.steps - 1):
            tag = "warmup" if step < args.warmup else "train"
            print(f"  {tag} {step:4d}  loss={loss.item():.6f}  {dt:.3f}s", flush=True)

    # Results
    if step_times and rank == 0:
        avg = sum(step_times) / len(step_times)
        gbs = args.batch_size * world_size

        mccl_info = ""
        if args.backend == "mccl":
            try:
                import mccl as _m
                metrics = _m.get_metrics()
                if metrics:
                    mccl_info = (f"\n  MCCL: {metrics.total_ops} ops, "
                                 f"avg_lat={metrics.avg_latency_ms:.2f}ms")
                    for attr in ("avg_sync_ms", "avg_network_ms", "avg_reduce_ms"):
                        val = getattr(metrics, attr, None)
                        if val is not None:
                            mccl_info += f"  {attr}={val:.2f}ms"
            except Exception:
                pass

        print(f"\n=== {args.backend.upper()} DDP Stats ===\n"
              f"  Avg: {avg:.3f}s ({1/avg:.1f} steps/s)\n"
              f"  Min/Max: {min(step_times):.3f}s / {max(step_times):.3f}s\n"
              f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}\n"
              f"  Throughput: {gbs/avg:.0f} samples/s{mccl_info}", flush=True)

    if args.save_stats and rank == 0 and step_times:
        write_training_stats(args.save_stats, f"ddp_{args.backend}", step_times, losses,
                             args.batch_size, world_size, total_params)
        print(f"Wrote stats to {args.save_stats}", flush=True)

    # Flush all pending GPU work and wait for all ranks to finish before
    # reading parameters — on slower transports (TCP) the last DDP allreduce
    # bucket may still be in-flight when the training loop exits.
    if args.backend == "mccl":
        torch.mps.synchronize()
    dist.barrier()

    # Sanity check: params in sync
    head = next(model.parameters()).detach().flatten()[:8].to(device)
    ref = head.clone()
    dist.broadcast(ref, src=0)
    if not torch.allclose(head, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Parameter mismatch across ranks!")
    if rank == 0:
        print("Parameters in sync across ranks.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def _setup_mccl_env() -> None:
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)
    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.baseline:
        run_baseline(args)
    elif "RANK" in os.environ:
        run_ddp(args)
    else:
        print("Use --baseline for single-GPU, or launch with torchrun for DDP.\n"
              "  torchrun --nproc_per_node=2 examples/ddp_dummy_train.py\n"
              "  torchrun --nproc_per_node=2 examples/ddp_dummy_train.py --backend gloo",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
