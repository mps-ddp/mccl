"""
Head-to-head benchmark: MCCL vs Gloo CPU fallback on Apple Silicon.

Proves the speedup thesis: MCCL avoids GPU↔CPU copies that Gloo requires.

Usage (single host, two processes):
    python tests/bench_mccl_vs_gloo.py

This script spawns two processes and runs both backends sequentially,
printing a comparison table at the end.
"""

import os
import time
import platform
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def _check_platform():
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        print("This benchmark requires macOS on Apple Silicon.")
        sys.exit(1)
    if not torch.backends.mps.is_available():
        print("MPS not available. Install PyTorch with MPS support.")
        sys.exit(1)


class BenchModel(nn.Module):
    def __init__(self, hidden=512, layers=4):
        super().__init__()
        mods = []
        for i in range(layers):
            ind = 128 if i == 0 else hidden
            outd = 10 if i == layers - 1 else hidden
            mods.append(nn.Linear(ind, outd))
            if i < layers - 1:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def _worker_mccl(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "WARN"

    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

    device = torch.device("mps")
    torch.manual_seed(42)
    model = BenchModel(hidden=512, layers=4).to(device)
    ddp_model = DDP(model)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    warmup, steps = 10, 100

    for _ in range(warmup):
        x = torch.randn(64, 128, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()

    torch.mps.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(steps):
        x = torch.randn(64, 128, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()
    torch.mps.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        results["mccl_ms_per_step"] = (elapsed * 1000) / steps
        results["mccl_samples_per_sec"] = (64 * world_size * steps) / elapsed

    dist.destroy_process_group()


def _worker_gloo(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Gloo requires CPU tensors
    device = torch.device("cpu")
    torch.manual_seed(42)
    model = BenchModel(hidden=512, layers=4).to(device)
    ddp_model = DDP(model)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    warmup, steps = 10, 100

    for _ in range(warmup):
        x = torch.randn(64, 128, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()

    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(steps):
        x = torch.randn(64, 128, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        opt.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        opt.step()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        results["gloo_ms_per_step"] = (elapsed * 1000) / steps
        results["gloo_samples_per_sec"] = (64 * world_size * steps) / elapsed

    dist.destroy_process_group()


def main():
    _check_platform()

    results = mp.Manager().dict()
    world_size = 2

    print("=" * 60)
    print(" MCCL vs Gloo Benchmark")
    print(f" Machine: {platform.machine()} / macOS {platform.mac_ver()[0]}")
    print(f" PyTorch: {torch.__version__}")
    print(f" Model: BenchModel(512, 4) — {sum(p.numel() for p in BenchModel().parameters()):,} params")
    print("=" * 60)

    print("\n[1/2] Running MCCL (MPS)...")
    os.environ["MCCL_SHADER_PATH"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "csrc", "metal", "shaders.metal")
    mp.spawn(_worker_mccl, args=(world_size, 29500, results),
             nprocs=world_size, join=True)

    print("[2/2] Running Gloo (CPU)...")
    mp.spawn(_worker_gloo, args=(world_size, 29600, results),
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
        print(f"\n  Gloo is faster on this config (speedup={speedup:.2f}x).")
        print("  This may happen on very small models where overhead dominates.")
    print()


if __name__ == "__main__":
    main()
