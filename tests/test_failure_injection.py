"""Failure-injection tests: peer death, watchdog behavior under queue
backpressure, and clean error propagation (no std::terminate, no silent
hangs)."""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _peer_death_fn(rank, world_size):
    """Rank 1 dies mid-run; rank 0's next collective must fail with a Python
    exception (watchdog/health abort), NOT hang and NOT kill the process via
    std::terminate."""
    import os
    import sys
    import time
    import torch
    import torch.distributed as dist

    # A healthy collective first.
    t = torch.ones(1024, device="mps")
    dist.all_reduce(t)

    if rank == 1:
        os._exit(0)  # simulate sudden peer death (clean exit code for harness)

    time.sleep(1.0)
    t2 = torch.ones(500_000, device="mps")
    try:
        work = dist.all_reduce(t2, async_op=True)
        work.wait()
    except Exception as e:
        print(f"rank {rank}: got expected error after peer death: {e}", file=sys.stderr)
        os._exit(0)
    # Either the collective errored (good) or, if the transport buffered the
    # send, it may appear complete; a barrier must then surely fail.
    try:
        dist.barrier()
    except Exception as e:
        print(f"rank {rank}: barrier errored after peer death: {e}", file=sys.stderr)
        os._exit(0)
    raise AssertionError("collectives kept succeeding after peer death")


def _watchdog_backpressure_fn(rank, world_size):
    """Deep queue of healthy ops with a watchdog SHORTER than the total queue
    drain time.  The watchdog clock starts when an op begins executing (not
    when it is submitted), so no healthy op should be aborted even though the
    last ops sit in the queue far longer than the watchdog timeout.  With the
    old submission-time clock this reliably aborts."""
    import torch
    import torch.distributed as dist

    # 200 x 4MB allreduces: total drain time is many seconds (well past the
    # 3s watchdog), while each individual op executes in tens of ms.
    works = []
    tensors = []
    for i in range(200):
        t = torch.full((1_000_000,), float(i % 50 + 1) * (rank + 1), device="mps")
        tensors.append(t)
        works.append(dist.all_reduce(t, async_op=True))
    for w in works:
        w.wait()

    total = sum(r + 1 for r in range(world_size))
    for i, t in enumerate(tensors):
        assert (t.cpu() == float(i % 50 + 1) * total).all(), f"op {i} corrupted"


class TestPeerDeath:
    def test_clean_abort_on_peer_death(self):
        # Workers self-exit(0) on the expected error path; a hang trips the
        # harness timeout, and std::terminate yields a non-zero exit code.
        run_workers(_peer_death_fn, world_size=2, timeout=120)


class TestWatchdog:
    def test_no_spurious_abort_under_backpressure(self):
        run_workers(
            _watchdog_backpressure_fn,
            world_size=2,
            env={"MCCL_WATCHDOG_TIMEOUT_MS": "3000"},
            timeout=420,
        )
