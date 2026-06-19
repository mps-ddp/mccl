"""Ring algorithm correctness: basic vs chunked vs float64 reference.

Gradient-sized tensors at ws=4/6/8.  Catches silent corruption (wrong
allgather schedule, pipeline fence races) that shows up as large error vs
reference — not benign fp32 associativity (~1e-7 relative).

Run on macOS arm64 with MCCL built:
  pytest -v tests/test_ring_algo_correctness.py
"""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL ring algo tests require macOS Apple Silicon",
)

# DDP-ish bucket sizes (fp32 elements): ~1 MB, ~6 MB (25 MB cap / few buckets), odd boundary.
GRAD_SIZES = [
    262_144,      # 1 MiB
    1_572_864,    # ~6 MiB
    6_553_599,    # odd, ~25 MiB
]

ALGOS = {
    "basic": {"MCCL_RING_ALGO": "basic"},
    "chunked": {"MCCL_RING_ALGO": "chunked"},
    "chunked_pipeline_off": {
        "MCCL_RING_ALGO": "chunked",
        "MCCL_RING_PIPELINE": "0",
    },
}


def _allreduce_vs_f64_reference_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist

    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    tol = float(os.environ.get("MCCL_TEST_RTOL", "5e-5"))
    algo = os.environ.get("MCCL_RING_ALGO", "(default)")

    for n in sizes:
        torch.manual_seed(17_000 + n + rank * 1_000_003)
        # Gradient-like values: mixed scale, not exactly representable.
        base = torch.randn(n, dtype=torch.float64) * (1.0 + (torch.arange(n) % 97) * 1e-4)
        mine = (base + float(rank) * 1e-3).to(torch.float32).to("mps")

        contribs = []
        for r in range(world_size):
            torch.manual_seed(17_000 + n + r * 1_000_003)
            b = torch.randn(n, dtype=torch.float64) * (
                1.0 + (torch.arange(n) % 97) * 1e-4
            )
            contribs.append((b + float(r) * 1e-3).to(torch.float32).to(torch.float64))
        expected = sum(contribs)

        dist.all_reduce(mine, op=dist.ReduceOp.SUM)
        got = mine.cpu().to(torch.float64)
        abs_err = (got - expected).abs()
        max_err = abs_err.max().item()
        ref_scale = expected.abs().max().item() + 1e-8
        rel_err = max_err / ref_scale

        # Corruption (stale buffer / wrong chunk) is orders of magnitude above
        # fp32 associativity; flag anything above a loose gradient tolerance.
        assert rel_err <= tol, (
            f"algo={algo} ws={world_size} n={n}: rel_err={rel_err:.3e} "
            f"(max_abs={max_err:.3e}) > {tol}"
        )


def _ddp_gradient_parity_fn(rank, world_size):
    """Multi-bucket backward; gradients vs single-process reference."""
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    algo = os.environ.get("MCCL_RING_ALGO", "(default)")
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "1"))

    torch.manual_seed(42)

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    model = make_model().to("mps")
    ddp = DDP(model, bucket_cap_mb=bucket_mb)

    batches = []
    for r in range(world_size):
        g = torch.Generator().manual_seed(2000 + r)
        x = torch.randn(64, 256, generator=g)
        y = torch.randint(0, 10, (64,), generator=g)
        batches.append((x, y))

    x_mine, y_mine = batches[rank]
    out = ddp(x_mine.to("mps"))
    loss = F.cross_entropy(out, y_mine.to("mps"))
    loss.backward()

    torch.manual_seed(42)
    ref = make_model()
    ref_losses = []
    for r in range(world_size):
        xr, yr = batches[r]
        ref_losses.append(F.cross_entropy(ref(xr), yr))
    (sum(ref_losses) / world_size).backward()

    rtol = float(os.environ.get("MCCL_TEST_RTOL", "5e-4"))
    for (name, p_ddp), (_, p_ref) in zip(ddp.module.named_parameters(), ref.named_parameters()):
        g_ddp = p_ddp.grad.detach().cpu()
        g_ref = p_ref.grad.detach()
        max_err = (g_ddp - g_ref).abs().max().item()
        scale = g_ref.abs().max().item() + 1e-8
        rel = max_err / scale
        assert rel < rtol, (
            f"algo={algo} ws={world_size} {name}: grad rel err {rel:.2e} > {rtol}"
        )


class TestRingAlgoVsReference:
    @pytest.mark.parametrize("algo_name,algo_env", list(ALGOS.items()))
    @pytest.mark.parametrize("world_size", [4, 6, 8])
    def test_allreduce_gradient_sizes(self, algo_name, algo_env, world_size):
        env = {
            **algo_env,
            "MCCL_COLLECTIVE_CONCURRENCY": "1",
            "MCCL_TEST_SIZES": ",".join(str(s) for s in GRAD_SIZES),
            "MCCL_TEST_RTOL": "5e-5",
        }
        run_workers(
            _allreduce_vs_f64_reference_fn,
            world_size=world_size,
            env=env,
            timeout=600,
        )

    @pytest.mark.parametrize("algo_name,algo_env", [("basic", ALGOS["basic"]), ("chunked", ALGOS["chunked"])])
    @pytest.mark.parametrize("world_size", [4, 6])
    def test_ddp_multibucket_gradient_parity(self, algo_name, algo_env, world_size):
        env = {
            **algo_env,
            "MCCL_COLLECTIVE_CONCURRENCY": "1",
            "MCCL_TEST_BUCKET_MB": "1",
            "MCCL_TEST_RTOL": "5e-4",
        }
        run_workers(
            _ddp_gradient_parity_fn,
            world_size=world_size,
            env=env,
            timeout=600,
        )


class TestChunkedConcurrencyStress:
    """chunked + concurrency=2 is the old training default; verify correctness."""

    @pytest.mark.parametrize("world_size", [4, 6, 8])
    def test_chunked_concurrency2_allreduce(self, world_size):
        env = {
            "MCCL_RING_ALGO": "chunked",
            "MCCL_COLLECTIVE_CONCURRENCY": "2",
            "MCCL_TEST_SIZES": "262144,1572864",
            "MCCL_TEST_RTOL": "5e-5",
        }
        run_workers(
            _allreduce_vs_f64_reference_fn,
            world_size=world_size,
            env=env,
            timeout=600,
        )

    @pytest.mark.parametrize("world_size", [8])
    def test_chunked_concurrency2_25mb_ws8(self, world_size):
        """25 MB fp32 bucket vs f64 reference at ws=8 (training default scale)."""
        env = {
            "MCCL_RING_ALGO": "chunked",
            "MCCL_COLLECTIVE_CONCURRENCY": "2",
            "MCCL_RING_PIPELINE": "0",
            "MCCL_TEST_SIZES": "6553600",
            "MCCL_TEST_RTOL": "5e-5",
        }
        run_workers(
            _allreduce_vs_f64_reference_fn,
            world_size=world_size,
            env=env,
            timeout=1200,
        )

    @pytest.mark.parametrize("world_size", [4, 8])
    def test_submit_job_settings_allreduce_64mb(self, world_size):
        """64 MB fp32 allreduce at current submit_job.sh (conc=2, ring pipeline ON)."""
        env = {
            "MCCL_RING_ALGO": "ring_chunked",
            "MCCL_COLLECTIVE_CONCURRENCY": "2",
            "MCCL_RING_PIPELINE": "1",
            "MCCL_OVERLAP_COMM": "1",
            "MCCL_PIPELINE_DEPTH": "1",
            "MCCL_TEST_SIZES": "16777216",  # 64 MiB fp32
            "MCCL_TEST_RTOL": "5e-5",
        }
        run_workers(
            _allreduce_vs_f64_reference_fn,
            world_size=world_size,
            env=env,
            timeout=1200,
        )

    @pytest.mark.parametrize("world_size", [4, 8])
    def test_submit_job_settings_ddp_gradient_parity(self, world_size):
        """Multi-bucket DDP backward at submit_job 64 MB bucket cap."""
        env = {
            "MCCL_RING_ALGO": "ring_chunked",
            "MCCL_COLLECTIVE_CONCURRENCY": "2",
            "MCCL_RING_PIPELINE": "1",
            "MCCL_OVERLAP_COMM": "1",
            "MCCL_TEST_BUCKET_MB": "64",
            "MCCL_TEST_RTOL": "5e-4",
        }
        run_workers(
            _ddp_gradient_parity_fn,
            world_size=world_size,
            env=env,
            timeout=1200,
        )
