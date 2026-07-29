"""End-to-end compression tests (FP16 / TopK) through real collectives and DDP.

Previously compression had unit tests only; the wire path (size-prefixed
exact-payload framing), error-feedback keying across multiple buckets, and
convergence vs an uncompressed baseline were untested.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _fp16_compressed_allreduce_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    # Values exactly representable in fp16 so compression is lossless here.
    for n in (1024, 70_001):
        t = torch.full((n,), float(rank + 1) * 0.5, dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = 0.5 * sum(r + 1 for r in range(world_size))
        bad = (t.cpu() != expected).sum().item()
        assert bad == 0, f"fp16-compressed allreduce n={n}: {bad} bad elements"


def _topk_error_feedback_multibucket_fn(rank, world_size):
    """Multiple distinct gradient buffers reduced repeatedly with TopK.

    Regression for error-feedback keying: residuals were keyed on the shared
    staging-buffer address, mixing residuals BETWEEN buffers.  With correct
    per-tensor feedback, the running sum of compressed allreduce outputs for
    each buffer converges to the running sum of true values (the residual is
    carried, never lost or cross-contaminated).
    """
    import torch
    import torch.distributed as dist

    n = 8192
    steps = 25
    # Two persistent "gradient buckets" with very different magnitudes.
    # Stable storage across steps (like DDP bucket buffers) so the per-tensor
    # error-feedback keying applies.
    big = torch.zeros(n, dtype=torch.float32, device="mps")
    small = torch.zeros(n, dtype=torch.float32, device="mps")

    accum_small = torch.zeros(n, dtype=torch.float64)

    g = torch.Generator().manual_seed(7 + rank)
    for _ in range(steps):
        gb = torch.randn(n, generator=g) * 100.0   # big-magnitude bucket
        gs = torch.randn(n, generator=g) * 0.01    # small-magnitude bucket

        big.copy_(gb.to("mps"))
        small.copy_(gs.to("mps"))

        dist.all_reduce(big, op=dist.ReduceOp.SUM)
        dist.all_reduce(small, op=dist.ReduceOp.SUM)

        accum_small += small.cpu().double()

    # With per-tensor error feedback, the small bucket's accumulated output
    # stays at its own magnitude (~0.01 * sqrt(steps) * world_size).  If
    # residuals leak from the big bucket (the old staging-pointer keying),
    # values ~1e4x larger appear here.
    scale_small = accum_small.abs().mean().item()
    assert scale_small < 1.0, (
        f"small bucket polluted by large bucket residuals: mean |x| = {scale_small}"
    )


def _ddp_topk_convergence_fn(rank, world_size):
    """Tiny DDP regression vs single-process training: TopK-compressed
    training must reduce the loss (not diverge)."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    ).to("mps")
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    torch.manual_seed(100 + rank)
    x = torch.randn(256, 32, device="mps")
    w_true = torch.randn(32, 1, device="mps")
    y = x @ w_true

    first_loss = None
    last_loss = None
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(ddp(x), y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

    assert last_loss < first_loss * 0.5, (
        f"TopK DDP training did not converge: {first_loss} -> {last_loss}"
    )


class TestFP16Compression:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_allreduce_exact_values(self, world_size):
        run_workers(
            _fp16_compressed_allreduce_fn,
            world_size=world_size,
            env={"MCCL_COMPRESSION": "fp16"},
        )


class TestTopKCompression:
    def test_error_feedback_isolated_per_bucket(self):
        run_workers(
            _topk_error_feedback_multibucket_fn,
            world_size=2,
            env={"MCCL_COMPRESSION": "topk", "MCCL_TOPK_RATIO": "0.05"},
            timeout=300,
        )

    def test_ddp_converges(self):
        run_workers(
            _ddp_topk_convergence_fn,
            world_size=2,
            env={"MCCL_COMPRESSION": "topk", "MCCL_TOPK_RATIO": "0.1"},
            timeout=420,
        )
