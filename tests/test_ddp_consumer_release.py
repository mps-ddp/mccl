"""DDP consumer-release resilience: catches stale overlapped gradients without Lightning.

NCCL guarantees the autograd thread cannot resume MPS until collective output is
visible.  MCCL v5.7+ enforces the same via ``Work::wait`` → ``wait_for_mccl``.

These tests deliberately avoid ``.item()`` / CPU grad reads between backward and
optimizer (the old parity harness masked the bug by synchronizing).  They also
simulate train → barrier → rank-skewed val → train without waiting for all ranks
to finish val — the window that killed cluster jobs at epoch boundaries.

Would fail with the broken ``markComplete`` consumer-release path (engine thread
consumes ``release_waited_`` so ``Work::wait`` skips the GPU fence).
"""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers, _submit_job_mccl_env

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL consumer-release tests require macOS Apple Silicon",
)

_SUBMIT_ENV = _submit_job_mccl_env(64)


def _ddp_no_cpu_sync_trajectory_fn(rank, world_size):
    import os

    import torch
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    steps = int(os.environ.get("MCCL_TEST_STEPS", "30"))
    lr = float(os.environ.get("MCCL_TEST_LR", "0.05"))
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "5e-3"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "64"))
    bs = int(os.environ.get("MCCL_TEST_BS", "4"))
    sp = int(os.environ.get("MCCL_TEST_SP", "128"))
    ch = int(os.environ.get("MCCL_TEST_CH", "48"))
    burn = int(os.environ.get("MCCL_TEST_MPS_BURN", "4"))

    def conv_stack():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
        )

    def batch_for(r, step):
        g = torch.Generator().manual_seed(9000 + r * 10_007 + step)
        x = torch.randn(bs, 3, sp, sp, generator=g)
        y = torch.randn(bs, 3, sp, sp, generator=g)
        return x, y

    def mps_burn(iters=3):
        a = torch.randn(512, 512, device="mps")
        b = torch.randn(512, 512, device="mps")
        for _ in range(iters):
            a = a @ b
            b = b @ a
        del a, b

    def weight_max_rel(ddp_model, ref_model):
        max_rel = 0.0
        for (_, p_ddp), (_, p_ref) in zip(
            ddp_model.named_parameters(), ref_model.named_parameters()
        ):
            w_ddp = p_ddp.detach().cpu().float()
            w_ref = p_ref.detach().cpu().float()
            err = (w_ddp - w_ref).abs().max().item()
            scale = w_ref.abs().max().item() + 1e-8
            max_rel = max(max_rel, err / scale)
        return max_rel

    def reference_weights():
        torch.manual_seed(42)
        model = conv_stack().to("mps")
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            losses = []
            for r in range(world_size):
                x, y = batch_for(r, step)
                out = model(x.to("mps"))
                losses.append(F.mse_loss(out, y.to("mps")))
            (sum(losses) / world_size).backward()
            opt.step()
        return model

    torch.manual_seed(42)
    ddp = DDP(conv_stack().to("mps"), bucket_cap_mb=bucket_mb)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)

    for step in range(steps):
        x, y = batch_for(rank, step)
        opt.zero_grad(set_to_none=True)
        out = ddp(x.to("mps"))
        loss = F.mse_loss(out, y.to("mps"))
        loss.backward()
        mps_burn(burn)
        opt.step()
        with torch.no_grad():
            _ = ddp(x.to("mps"))

    if rank == 0:
        ref = reference_weights()
        max_rel = weight_max_rel(ddp.module, ref)
        print(
            f"[ddp-no-cpu-sync] ws={world_size} steps={steps} max_weight_rel={max_rel:.2e}",
            flush=True,
        )
        assert max_rel < rtol, f"weight rel err {max_rel:.2e} > {rtol}"


def _ddp_phase_boundary_fn(rank, world_size):
    import os
    import time

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    epochs = int(os.environ.get("MCCL_TEST_EPOCHS", "4"))
    train_steps = int(os.environ.get("MCCL_TEST_TRAIN_STEPS", "8"))
    val_steps = int(os.environ.get("MCCL_TEST_VAL_STEPS", "6"))
    stagger_ms = float(os.environ.get("MCCL_TEST_STAGGER_MS", "80"))
    lr = float(os.environ.get("MCCL_TEST_LR", "0.05"))
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "5e-3"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "64"))
    bs = int(os.environ.get("MCCL_TEST_BS", "4"))
    sp = int(os.environ.get("MCCL_TEST_SP", "128"))
    ch = int(os.environ.get("MCCL_TEST_CH", "48"))
    burn = int(os.environ.get("MCCL_TEST_MPS_BURN", "3"))

    def conv_stack():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
        )

    def batch_for(r, step):
        g = torch.Generator().manual_seed(9000 + r * 10_007 + step)
        x = torch.randn(bs, 3, sp, sp, generator=g)
        y = torch.randn(bs, 3, sp, sp, generator=g)
        return x, y

    def mps_burn(iters=3):
        a = torch.randn(512, 512, device="mps")
        b = torch.randn(512, 512, device="mps")
        for _ in range(iters):
            a = a @ b
            b = b @ a
        del a, b

    def weight_max_rel(ddp_model, ref_model):
        max_rel = 0.0
        for (_, p_ddp), (_, p_ref) in zip(
            ddp_model.named_parameters(), ref_model.named_parameters()
        ):
            w_ddp = p_ddp.detach().cpu().float()
            w_ref = p_ref.detach().cpu().float()
            err = (w_ddp - w_ref).abs().max().item()
            scale = w_ref.abs().max().item() + 1e-8
            max_rel = max(max_rel, err / scale)
        return max_rel

    def reference_weights(total_steps):
        torch.manual_seed(42)
        model = conv_stack().to("mps")
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for step in range(total_steps):
            opt.zero_grad(set_to_none=True)
            losses = []
            for r in range(world_size):
                x, y = batch_for(r, step)
                out = model(x.to("mps"))
                losses.append(F.mse_loss(out, y.to("mps")))
            (sum(losses) / world_size).backward()
            opt.step()
        return model

    torch.manual_seed(42)
    ddp = DDP(conv_stack().to("mps"), bucket_cap_mb=bucket_mb)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)
    global_step = 0

    for epoch in range(epochs):
        ddp.train()
        for _ in range(train_steps):
            x, y = batch_for(rank, global_step)
            opt.zero_grad(set_to_none=True)
            out = ddp(x.to("mps"))
            loss = F.mse_loss(out, y.to("mps"))
            loss.backward()
            mps_burn(burn)
            opt.step()
            with torch.no_grad():
                _ = ddp(x.to("mps"))
            global_step += 1

        dist.barrier()

        ddp.eval()
        for vs in range(val_steps):
            xv, yv = batch_for(rank, 50_000 + epoch * 100 + vs)
            with torch.no_grad():
                out = ddp(xv.to("mps"))
                _ = F.mse_loss(out, yv.to("mps"))
            if stagger_ms > 0:
                time.sleep(stagger_ms * (0.02 + 0.08 * (rank % 4)) / 1000.0)

        ddp.train()

    if rank == 0:
        ref = reference_weights(global_step)
        max_rel = weight_max_rel(ddp.module, ref)
        print(
            f"[ddp-phase-boundary] ws={world_size} epochs={epochs} "
            f"steps={global_step} max_weight_rel={max_rel:.2e}",
            flush=True,
        )
        assert max_rel < rtol, f"weight rel err {max_rel:.2e} > {rtol}"


def _ddp_barrier_allreduce_burst_fn(rank, world_size):
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    cycles = int(os.environ.get("MCCL_TEST_CYCLES", "12"))
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "64"))
    bs, sp, ch = 4, 128, 48
    rtol = float(os.environ.get("MCCL_TEST_RTOL", "5e-3"))
    lr = 0.05

    def conv_stack():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch * 2, ch, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
        )

    def batch_for(r, step):
        g = torch.Generator().manual_seed(9000 + r * 10_007 + step)
        x = torch.randn(bs, 3, sp, sp, generator=g)
        y = torch.randn(bs, 3, sp, sp, generator=g)
        return x, y

    def mps_burn(iters=3):
        a = torch.randn(512, 512, device="mps")
        b = torch.randn(512, 512, device="mps")
        for _ in range(iters):
            a = a @ b
            b = b @ a
        del a, b

    def weight_max_rel(ddp_model, ref_model):
        max_rel = 0.0
        for (_, p_ddp), (_, p_ref) in zip(
            ddp_model.named_parameters(), ref_model.named_parameters()
        ):
            w_ddp = p_ddp.detach().cpu().float()
            w_ref = p_ref.detach().cpu().float()
            err = (w_ddp - w_ref).abs().max().item()
            scale = w_ref.abs().max().item() + 1e-8
            max_rel = max(max_rel, err / scale)
        return max_rel

    def reference_weights():
        torch.manual_seed(7)
        model = conv_stack().to("mps")
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for step in range(cycles):
            opt.zero_grad(set_to_none=True)
            losses = []
            for r in range(world_size):
                x, y = batch_for(r, step)
                out = model(x.to("mps"))
                losses.append(F.mse_loss(out, y.to("mps")))
            (sum(losses) / world_size).backward()
            opt.step()
        return model

    torch.manual_seed(7)
    ddp = DDP(conv_stack().to("mps"), bucket_cap_mb=bucket_mb)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)

    for cycle in range(cycles):
        x, y = batch_for(rank, cycle)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(ddp(x.to("mps")), y.to("mps"))
        loss.backward()
        mps_burn(5)
        opt.step()
        dist.barrier()
        buf = torch.randn(1_048_576, device="mps") + float(rank)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        mps_burn(2)
        with torch.no_grad():
            _ = ddp(x.to("mps"))

    if rank == 0:
        ref = reference_weights()
        max_rel = weight_max_rel(ddp.module, ref)
        print(
            f"[ddp-barrier-burst] ws={world_size} cycles={cycles} "
            f"max_weight_rel={max_rel:.2e}",
            flush=True,
        )
        assert max_rel < rtol, f"weight rel err {max_rel:.2e} > {rtol}"


@pytest.mark.parametrize("world_size", [4, 8])
def test_ddp_no_cpu_sync_weight_trajectory(world_size):
    run_workers(
        _ddp_no_cpu_sync_trajectory_fn,
        world_size=world_size,
        env={**_SUBMIT_ENV, "MCCL_TEST_STEPS": "35", "MCCL_TEST_BUCKET_MB": "64"},
        timeout=1200,
    )


@pytest.mark.parametrize("world_size", [4, 8])
def test_ddp_phase_boundary_rank_skew(world_size):
    run_workers(
        _ddp_phase_boundary_fn,
        world_size=world_size,
        env={
            **_SUBMIT_ENV,
            "MCCL_TEST_EPOCHS": "5",
            "MCCL_TEST_TRAIN_STEPS": "10",
            "MCCL_TEST_VAL_STEPS": "8",
            "MCCL_TEST_STAGGER_MS": "120",
        },
        timeout=1500,
    )


@pytest.mark.parametrize("world_size", [4])
def test_ddp_barrier_allreduce_mps_burst(world_size):
    run_workers(
        _ddp_barrier_allreduce_burst_fn,
        world_size=world_size,
        env={**_SUBMIT_ENV, "MCCL_TEST_CYCLES": "15"},
        timeout=1200,
    )
