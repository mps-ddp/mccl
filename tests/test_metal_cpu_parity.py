"""Metal reduce path must match vDSP CPU reduce (separate process groups)."""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pytest
import torch

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Metal/CPU parity tests require macOS Apple Silicon",
)

SIZES = [4096, 262_144, 1_048_576]
_OUT_DIR = Path(tempfile.gettempdir()) / "mccl_metal_cpu_parity"


def _allreduce_save_fn(rank, world_size):
    import os
    from pathlib import Path

    import torch
    import torch.distributed as dist

    tag = os.environ["MCCL_TEST_TAG"]
    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])

    for n in sizes:
        torch.manual_seed(9000 + n + rank)
        base = torch.randn(n, dtype=torch.float32) * (1.0 + (torch.arange(n) % 17) * 1e-3)
        mine = (base + float(rank)).to("mps")
        dist.all_reduce(mine, op=dist.ReduceOp.SUM)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mine.cpu(), out_dir / f"{tag}_ws{world_size}_n{n}_r{rank}.pt")


def _compare_metal_cpu_fn(rank, world_size):
    import os
    from pathlib import Path

    import torch

    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "5e-5"))

    for n in sizes:
        metal = torch.load(
            out_dir / f"metal_ws{world_size}_n{n}_r{rank}.pt", weights_only=True
        )
        cpu = torch.load(
            out_dir / f"cpu_ws{world_size}_n{n}_r{rank}.pt", weights_only=True
        )
        if not torch.allclose(metal, cpu, rtol=rtol, atol=rtol):
            err = (metal - cpu).abs().max().item()
            raise AssertionError(
                f"ws={world_size} n={n} rank={rank}: Metal vs CPU max_err={err}"
            )


@pytest.mark.parametrize("world_size", [4, 6])
def test_fp32_metal_matches_cpu_reduce(world_size):
    base = {
        "MCCL_RING_ALGO": "chunked",
        "MCCL_COLLECTIVE_CONCURRENCY": "2",
        "MCCL_TEST_SIZES": ",".join(str(s) for s in SIZES),
        "MCCL_TEST_OUT_DIR": str(_OUT_DIR),
    }
    run_workers(
        _allreduce_save_fn,
        world_size=world_size,
        env={**base, "MCCL_TEST_TAG": "metal"},
        timeout=600,
    )
    run_workers(
        _allreduce_save_fn,
        world_size=world_size,
        env={**base, "MCCL_TEST_TAG": "cpu", "MCCL_FP32_CPU_REDUCE": "1"},
        timeout=600,
    )
    run_workers(_compare_metal_cpu_fn, world_size=world_size, env=base, timeout=120)
