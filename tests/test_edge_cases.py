"""Edge-case contract tests: world_size=1, zero-size tensors, non-contiguous
rejection, and unsupported-dtype rejection at the call site."""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _ws1_all_ops_fn(rank, world_size):
    """world_size == 1: every collective must complete immediately.
    Regression: broadcast used to hang forever (its completion counter
    started at zero peers), and the transport refused to construct."""
    import torch
    import torch.distributed as dist

    t = torch.full((1000,), 3.0, device="mps")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert (t.cpu() == 3.0).all()

    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    assert (t.cpu() == 3.0).all()

    b = torch.full((64,), 7.0, device="mps")
    dist.broadcast(b, src=0)  # the historical hang
    assert (b.cpu() == 7.0).all()

    outs = [torch.zeros(1000, device="mps")]
    dist.all_gather(outs, t)
    assert (outs[0].cpu() == 3.0).all()

    out = torch.zeros(1000, device="mps")
    dist.reduce_scatter(out, [t], op=dist.ReduceOp.SUM)
    assert (out.cpu() == 3.0).all()

    dist.barrier()


def _zero_size_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    t = torch.zeros(0, dtype=torch.float32, device="mps")
    w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    w.wait()

    b = torch.zeros(0, dtype=torch.float32, device="mps")
    dist.broadcast(b, src=0)

    # Follow up with a real collective to prove no wire desync occurred.
    real = torch.full((128,), float(rank + 1), device="mps")
    dist.all_reduce(real, op=dist.ReduceOp.SUM)
    assert (real.cpu() == sum(r + 1 for r in range(world_size))).all()


def _noncontiguous_rejected_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    m = torch.ones(64, 64, device="mps")
    col = m[:, 0]  # non-contiguous view
    assert not col.is_contiguous()
    try:
        dist.all_reduce(col, op=dist.ReduceOp.SUM)
    except Exception:
        pass  # expected: loud rejection, not a silent clone
    else:
        raise AssertionError(
            "non-contiguous allreduce should be rejected (silent clone drops results)"
        )

    # Group must still be usable afterwards.
    t = torch.full((128,), float(rank + 1), device="mps")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert (t.cpu() == sum(r + 1 for r in range(world_size))).all()


def _bad_dtype_rejected_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    t = torch.ones(128, dtype=torch.int64, device="mps")
    try:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    except Exception:
        pass  # expected: rejected at validation, not deep in an engine thread
    else:
        raise AssertionError("int64 allreduce should be rejected at the call site")

    # Group must still be usable afterwards.
    f = torch.full((128,), float(rank + 1), device="mps")
    dist.all_reduce(f, op=dist.ReduceOp.SUM)
    assert (f.cpu() == sum(r + 1 for r in range(world_size))).all()


def _broadcast_nonroot_values_fn(rank, world_size):
    """Broadcast must deliver root's data to all ranks for various sizes,
    including ones spanning multiple transport chunks."""
    import torch
    import torch.distributed as dist

    for n in (1, 4096, 1_000_003):
        if rank == 0:
            t = torch.arange(n, dtype=torch.float32, device="mps") % 97
        else:
            t = torch.full((n,), -1.0, dtype=torch.float32, device="mps")
        dist.broadcast(t, src=0)
        expected = torch.arange(n, dtype=torch.float32) % 97
        assert torch.equal(t.cpu(), expected), f"broadcast n={n} rank={rank}"


class TestWorldSizeOne:
    def test_all_ops_complete(self):
        run_workers(_ws1_all_ops_fn, world_size=1, timeout=120)


class TestZeroSize:
    @pytest.mark.parametrize("world_size", [2])
    def test_zero_size_noop(self, world_size):
        run_workers(_zero_size_fn, world_size=world_size)


class TestValidation:
    @pytest.mark.parametrize("world_size", [2])
    def test_noncontiguous_rejected(self, world_size):
        run_workers(_noncontiguous_rejected_fn, world_size=world_size)

    @pytest.mark.parametrize("world_size", [2])
    def test_unsupported_dtype_rejected(self, world_size):
        run_workers(_bad_dtype_rejected_fn, world_size=world_size)


class TestBroadcast:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_values_all_sizes(self, world_size):
        run_workers(_broadcast_nonroot_values_fn, world_size=world_size, timeout=300)
