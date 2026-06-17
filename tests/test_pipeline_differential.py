"""Pipeline vs lock-step differential: chunked ring must match at all N.

Oracle: MCCL_RING_PIPELINE=0 (lock-step).  Candidate: default pipeline ON.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pytest
import torch

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL pipeline differential requires macOS Apple Silicon",
)

SIZES = [262_144, 6_553_599, 25_165_824]
WORLD_SIZES = [3, 4, 6, 8, 12]
CONCURRENCIES = [1, 2, 3]

_OUT_DIR = Path(tempfile.gettempdir()) / "mccl_pipe_diff"


def _allreduce_save_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist

    tag = os.environ["MCCL_TEST_TAG"]
    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])

    for n in sizes:
        torch.manual_seed(31_000 + n + rank)
        base = torch.randn(n, dtype=torch.float64) * (1.0 + (torch.arange(n) % 53) * 2e-4)
        mine = (base + float(rank) * 1e-3).to(torch.float32).to("mps")
        dist.all_reduce(mine, op=dist.ReduceOp.SUM)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mine.cpu(), out_dir / f"{tag}_ws{world_size}_n{n}_r{rank}.pt")


def _compare_saved_fn(rank, world_size):
    import os

    import torch

    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    out_dir = Path(os.environ["MCCL_TEST_OUT_DIR"])
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "1e-5"))

    for n in sizes:
        oracle = torch.load(
            out_dir / f"oracle_ws{world_size}_n{n}_r{rank}.pt", weights_only=True
        )
        pipeline = torch.load(
            out_dir / f"pipeline_ws{world_size}_n{n}_r{rank}.pt", weights_only=True
        )
        if not torch.allclose(oracle, pipeline, rtol=rtol, atol=rtol):
            diff = (oracle - pipeline).abs().max().item()
            raise AssertionError(
                f"ws={world_size} n={n} rank={rank}: pipeline vs lock-step max_err={diff}"
            )


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_chunked_pipeline_matches_lockstep(world_size, concurrency):
    base = {
        "MCCL_RING_ALGO": "chunked",
        "MCCL_COLLECTIVE_CONCURRENCY": str(concurrency),
        "MCCL_TEST_SIZES": ",".join(str(s) for s in SIZES),
        "MCCL_TEST_OUT_DIR": str(_OUT_DIR),
        "MCCL_TEST_RTOL": "1e-5",
    }
    run_workers(
        _allreduce_save_fn,
        world_size=world_size,
        env={**base, "MCCL_TEST_TAG": "oracle", "MCCL_RING_PIPELINE": "0"},
        timeout=900,
    )
    run_workers(
        _allreduce_save_fn,
        world_size=world_size,
        env={**base, "MCCL_TEST_TAG": "pipeline"},
        timeout=900,
    )
    run_workers(
        _compare_saved_fn,
        world_size=world_size,
        env=base,
        timeout=120,
    )
