"""DDP backward smoke: gradient magnitudes must stay finite (catches ~inf corruption)."""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL DDP smoke requires macOS Apple Silicon",
)


def _ddp_backward_finite_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    dtype_name = os.environ.get("MCCL_TEST_DTYPE", "float32")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "25"))

    torch.manual_seed(77 + rank)
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    ).to(dtype=dtype, device="mps")
    ddp = DDP(model, bucket_cap_mb=bucket_mb)

    g = torch.Generator().manual_seed(500 + rank)
    x = torch.randn(32, 512, generator=g, dtype=dtype).to("mps")
    y = torch.randint(0, 10, (32,), generator=g).to("mps")
    loss = F.cross_entropy(ddp(x), y)
    loss.backward()

    max_grad = 0.0
    for p in ddp.module.parameters():
        if p.grad is not None:
            gmax = p.grad.detach().abs().max().item()
            max_grad = max(max_grad, gmax)

    if not (max_grad > 0 and max_grad < 1e6):
        raise AssertionError(
            f"rank={rank} dtype={dtype_name}: grad.abs().max()={max_grad} (expected finite, <1e6)"
        )

    # Cross-rank grad parity (same seed layout → identical reduced grads).
    buf = torch.tensor([max_grad], dtype=torch.float32, device="mps")
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
    if rank == 0 and buf.item() != max_grad:
        raise AssertionError(f"grad max mismatch across ranks: local={max_grad} global_max={buf.item()}")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("world_size", [4, 7])
def test_ddp_backward_grad_finite(dtype, world_size):
    env = {
        "MCCL_RING_ALGO": "chunked",
        "MCCL_COLLECTIVE_CONCURRENCY": "2",
        "MCCL_TEST_DTYPE": dtype,
        "MCCL_TEST_BUCKET_MB": "25",
    }
    run_workers(_ddp_backward_finite_fn, world_size=world_size, env=env, timeout=600)
