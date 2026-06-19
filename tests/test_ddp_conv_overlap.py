"""DDP + MPS conv backward + MCCL overlap: reproduces the cluster SIGSEGV.

The production job crashed in ``mps_convolution_backward`` →
``MPSStream::executeMPSGraph`` → ``commit`` → ``objc_release`` while MCCL
collective threads were concurrently calling ``torch::mps::synchronize()`` /
committing PyTorch's shared MPS command buffer (``stage_for_send_collective``).

This test runs real Conv2d backward under DDP with small grad buckets (so
allreduce overlaps backward compute) and ``MCCL_OVERLAP_COMM=1`` — the exact
submit-job config — for many iterations, and verifies gradient parity vs a
single-process reference.  Without the EventSync fix it SIGSEGVs; with it the
engine threads wait on an MTLSharedEvent instead of touching PyTorch's stream.
"""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL DDP conv overlap test requires macOS Apple Silicon",
)


def _ddp_conv_overlap_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    iters = int(os.environ.get("MCCL_TEST_ITERS", "25"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "1"))
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "2e-3"))
    # Heavy conv on large spatial maps → long mps_convolution_backward encode,
    # so an earlier bucket's allreduce (engine thread) overlaps a later bucket's
    # backward (autograd thread) — the window that crashed on the cluster.
    ch = int(os.environ.get("MCCL_TEST_CONV_CH", "64"))
    sp = int(os.environ.get("MCCL_TEST_CONV_SP", "256"))
    bs = int(os.environ.get("MCCL_TEST_CONV_BS", "4"))

    torch.manual_seed(42)

    def make_model():
        # Conv stack → exercises mps_convolution_backward (the crashing op),
        # plus enough params over several layers to span multiple grad buckets.
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
        )

    model = make_model().to("mps")
    ddp = DDP(model, bucket_cap_mb=bucket_mb)

    # Per-rank distinct batches so allreduce actually averages distinct grads.
    def batch_for(r):
        g = torch.Generator().manual_seed(2000 + r)
        x = torch.randn(bs, 3, sp, sp, generator=g)
        y = torch.randn(bs, 3, sp, sp, generator=g)
        return x, y

    for it in range(iters):
        x_mine, y_mine = batch_for((rank + it) % world_size)
        ddp.zero_grad(set_to_none=True)
        out = ddp(x_mine.to("mps"))
        loss = F.mse_loss(out, y_mine.to("mps"))
        loss.backward()
        # Force the grads to be materialized/read back each step.
        _ = sum(p.grad.detach().float().abs().sum().item()
                for p in ddp.module.parameters() if p.grad is not None)

    # Final-step gradient parity vs single-process reference for the same
    # global batch (all ranks averaged).
    torch.manual_seed(42)
    ref = make_model().to("mps")
    ref.zero_grad(set_to_none=True)
    ref_losses = []
    last_it = iters - 1
    for r in range(world_size):
        xr, yr = batch_for((r + last_it) % world_size)
        out_r = ref(xr.to("mps"))
        ref_losses.append(F.mse_loss(out_r, yr.to("mps")))
    (sum(ref_losses) / world_size).backward()

    max_rel = 0.0
    for (name, p_ddp), (_, p_ref) in zip(
        ddp.module.named_parameters(), ref.named_parameters()
    ):
        if p_ddp.grad is None:
            continue
        g_ddp = p_ddp.grad.detach().cpu()
        g_ref = p_ref.grad.detach().cpu()
        max_err = (g_ddp - g_ref).abs().max().item()
        scale = g_ref.abs().max().item() + 1e-8
        rel = max_err / scale
        max_rel = max(max_rel, rel)
        assert rel < rtol, (
            f"ws={world_size} {name}: grad rel err {rel:.2e} > {rtol}"
        )
    if rank == 0:
        print(f"[ddp-conv-overlap] ws={world_size} iters={iters} max_rel={max_rel:.2e}",
              flush=True)


# Submit-job MCCL env: overlap ON, event sync default ON, chunked ring, conc=1.
_SUBMIT_ENV = {
    "MCCL_OVERLAP_COMM": "1",
    "MCCL_RING_ALGO": "ring_chunked",
    "MCCL_RING_PIPELINE": "0",
    "MCCL_COLLECTIVE_CONCURRENCY": "1",
    "MCCL_PIPELINE_DEPTH": "1",
}

# Current submit_job.sh defaults (2026-06): conc=2, ring pipeline ON, 64 MB buckets.
_SUBMIT_ENV_NOW = {
    "MCCL_OVERLAP_COMM": "1",
    "MCCL_RING_ALGO": "ring_chunked",
    "MCCL_RING_PIPELINE": "1",
    "MCCL_COLLECTIVE_CONCURRENCY": "2",
    "MCCL_PIPELINE_DEPTH": "1",
}


@pytest.mark.parametrize("world_size", [4, 8])
def test_ddp_conv_overlap_gradient_parity(world_size):
    run_workers(
        _ddp_conv_overlap_fn,
        world_size=world_size,
        env={**_SUBMIT_ENV, "MCCL_TEST_ITERS": "25", "MCCL_TEST_BUCKET_MB": "1"},
        timeout=900,
    )


@pytest.mark.parametrize("world_size", [4])
def test_ddp_conv_overlap_tiny_buckets_stress(world_size):
    """Many tiny buckets → max overlap of allreduce with conv backward."""
    run_workers(
        _ddp_conv_overlap_fn,
        world_size=world_size,
        env={**_SUBMIT_ENV, "MCCL_TEST_ITERS": "40", "MCCL_TEST_BUCKET_MB": "1",
             "MCCL_COLLECTIVE_CONCURRENCY": "2"},
        timeout=900,
    )


@pytest.mark.parametrize("world_size", [4, 8])
def test_ddp_conv_overlap_submit_job_settings(world_size):
    """DDP conv backward parity at current submit_job.sh MCCL knobs (conc=2, pipeline=1)."""
    run_workers(
        _ddp_conv_overlap_fn,
        world_size=world_size,
        env={
            **_SUBMIT_ENV_NOW,
            "MCCL_TEST_ITERS": "25",
            "MCCL_TEST_BUCKET_MB": "64",
            "MCCL_TEST_RTOL": "2e-3",
        },
        timeout=1200,
    )
