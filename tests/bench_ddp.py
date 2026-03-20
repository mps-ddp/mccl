"""DDP benchmark: measure throughput and latency with MCCL vs Gloo."""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class BenchModel(nn.Module):
    """Configurable model for benchmarking."""

    def __init__(self, hidden=512, layers=4):
        super().__init__()
        modules = []
        for i in range(layers):
            in_dim = 128 if i == 0 else hidden
            out_dim = 10 if i == layers - 1 else hidden
            modules.append(nn.Linear(in_dim, out_dim))
            if i < layers - 1:
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def bench(backend, num_steps=100, warmup=10, batch_size=64, hidden=512, layers=4):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if backend == "mccl":
        import mccl  # noqa: F401

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("mps") if backend == "mccl" else torch.device("cpu")

    torch.manual_seed(42)
    model = BenchModel(hidden=hidden, layers=layers).to(device)
    ddp_model = DDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(warmup):
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()
    dist.barrier()

    # Timed run
    t0 = time.perf_counter()
    for step in range(num_steps):
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        loss_fn(ddp_model(x), y).backward()
        optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()
    dist.barrier()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    ms_per_step = total_ms / num_steps
    samples_per_sec = (batch_size * world_size * num_steps) / (t1 - t0)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        param_mb = param_count * 4 / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend}")
        print(f"World size: {world_size}")
        print(f"Model params: {param_count:,} ({param_mb:.1f} MB)")
        print(f"Batch size: {batch_size} x {world_size} = {batch_size * world_size}")
        print(f"Steps: {num_steps}")
        print(f"Total time: {total_ms:.1f} ms")
        print(f"Per step: {ms_per_step:.2f} ms")
        print(f"Throughput: {samples_per_sec:.1f} samples/sec")
        print(f"{'=' * 60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mccl", choices=["mccl", "gloo"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    bench(
        backend=args.backend,
        num_steps=args.steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        hidden=args.hidden,
        layers=args.layers,
    )
