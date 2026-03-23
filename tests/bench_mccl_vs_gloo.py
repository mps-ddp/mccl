"""
Head-to-head benchmark: MCCL (MPS) vs Gloo (CPU) on Apple Silicon.

Uses the same HeavyDummyModel (conv + attention + MLP, ~96M params) as the
DDP training script for representative gradient sizes and compute patterns.

Usage (single host, two processes):
    python tests/bench_mccl_vs_gloo.py
    python tests/bench_mccl_vs_gloo.py --steps 50 --batch-size 32
"""

import argparse
import os
import sys
import time
import platform

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from ddp_utils import build_model


def parse_args():
    p = argparse.ArgumentParser(description="MCCL vs Gloo benchmark")
    p.add_argument("--steps", type=int, default=50, help="Timed steps (default: 50)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup steps (default: 10)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-rank batch (default: 64)")
    p.add_argument("--input-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--bucket-mb", type=int, default=25, help="DDP bucket size MB")
    return p.parse_args()


def _worker_mccl(rank, world_size, port, results, cfg):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "WARN"

    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

    device = torch.device("mps")
    torch.manual_seed(42)
    model = build_model(cfg["input_dim"], cfg["num_classes"],
                        cfg["hidden"], cfg["depth"]).to(device)
    ddp_model = DDP(model, bucket_cap_mb=cfg["bucket_mb"])
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    bs = cfg["batch_size"]

    for _ in range(cfg["warmup"]):
        x = torch.randn(bs, cfg["input_dim"], device=device)
        y = torch.randint(0, cfg["num_classes"], (bs,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()

    torch.mps.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(cfg["steps"]):
        x = torch.randn(bs, cfg["input_dim"], device=device)
        y = torch.randint(0, cfg["num_classes"], (bs,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()
    torch.mps.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        results["mccl_ms_per_step"] = (elapsed * 1000) / cfg["steps"]
        results["mccl_samples_per_sec"] = (bs * world_size * cfg["steps"]) / elapsed

    dist.destroy_process_group()


def _worker_gloo(rank, world_size, port, results, cfg):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device("cpu")
    torch.manual_seed(42)
    model = build_model(cfg["input_dim"], cfg["num_classes"],
                        cfg["hidden"], cfg["depth"]).to(device)
    ddp_model = DDP(model, bucket_cap_mb=cfg["bucket_mb"])
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    bs = cfg["batch_size"]

    for _ in range(cfg["warmup"]):
        x = torch.randn(bs, cfg["input_dim"], device=device)
        y = torch.randint(0, cfg["num_classes"], (bs,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()

    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(cfg["steps"]):
        x = torch.randn(bs, cfg["input_dim"], device=device)
        y = torch.randint(0, cfg["num_classes"], (bs,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        results["gloo_ms_per_step"] = (elapsed * 1000) / cfg["steps"]
        results["gloo_samples_per_sec"] = (bs * world_size * cfg["steps"]) / elapsed

    dist.destroy_process_group()


def main():
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        print("This benchmark requires macOS on Apple Silicon.")
        sys.exit(1)
    if not torch.backends.mps.is_available():
        print("MPS not available.")
        sys.exit(1)

    args = parse_args()
    cfg = {
        "steps": args.steps, "warmup": args.warmup, "batch_size": args.batch_size,
        "input_dim": args.input_dim, "num_classes": args.num_classes,
        "hidden": args.hidden, "depth": args.depth, "bucket_mb": args.bucket_mb,
    }

    results = mp.Manager().dict()
    world_size = 2

    model = build_model(args.input_dim, args.num_classes, args.hidden, args.depth)
    total_params = sum(p.numel() for p in model.parameters())
    del model

    print("=" * 60)
    print(" MCCL vs Gloo Benchmark")
    print(f" Machine: {platform.machine()} / macOS {platform.mac_ver()[0]}")
    print(f" PyTorch: {torch.__version__}")
    print(f" Model: HeavyDummyModel ({total_params:,} params)")
    print(f" Batch: {args.batch_size}/rank  Steps: {args.steps}  Bucket: {args.bucket_mb}MB")
    print("=" * 60)

    print("\n[1/2] Running MCCL (MPS)...")
    mp.spawn(_worker_mccl, args=(world_size, 29500, results, cfg),
             nprocs=world_size, join=True)

    print("[2/2] Running Gloo (CPU)...")
    mp.spawn(_worker_gloo, args=(world_size, 29700, results, cfg),
             nprocs=world_size, join=True)

    mccl_ms = results.get("mccl_ms_per_step", float("inf"))
    gloo_ms = results.get("gloo_ms_per_step", float("inf"))
    mccl_sps = results.get("mccl_samples_per_sec", 0)
    gloo_sps = results.get("gloo_samples_per_sec", 0)
    speedup = gloo_ms / mccl_ms if mccl_ms > 0 else 0

    print("\n" + "=" * 60)
    print(" Results")
    print("=" * 60)
    print(f"  {'':30s} {'MCCL (MPS)':>15s} {'Gloo (CPU)':>15s}")
    print(f"  {'ms/step':30s} {mccl_ms:>15.2f} {gloo_ms:>15.2f}")
    print(f"  {'samples/sec':30s} {mccl_sps:>15.1f} {gloo_sps:>15.1f}")
    print(f"  {'':30s}")
    print(f"  {'Speedup (MCCL vs Gloo)':30s} {speedup:>15.2f}x")
    print("=" * 60)

    if speedup > 1.0:
        print(f"\n  MCCL is {speedup:.1f}x faster than Gloo on this machine.")
    elif speedup > 0:
        print(f"\n  Gloo is faster (speedup={speedup:.2f}x).")
        print("  This may happen on very small models where overhead dominates.")
    print()


if __name__ == "__main__":
    main()
