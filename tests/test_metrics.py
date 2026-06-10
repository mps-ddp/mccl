"""Metrics regression: after real collectives, mccl.get_metrics() must report
non-zero ops, bytes, latency, and phase breakdowns on EVERY rank.

Guards two past failure modes:
  - ProgressEngine stamped op_execute_start with its OWN counter (a different
    namespace from the collective seq), zeroing queue-wait/execution splits;
  - the pipelined ring / tree / broadcast paths recorded bytes but never
    record_phase, so avg_network_ms / avg_reduce_ms read 0 at ws >= 3.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _metrics_populated_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    import mccl

    # Ring path (large): several iterations so percentiles are meaningful.
    n_ring = 400_000  # 1.6 MB fp32 > threshold at any ws scaling
    for it in range(5):
        t = torch.full((n_ring,), float(it + 1) * (rank + 1),
                       dtype=torch.float32, device="mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = float(it + 1) * sum(r + 1 for r in range(world_size))
        assert (t.cpu() == expected).all()

    # Tree path (small) + broadcast for op variety.
    s = torch.full((1024,), float(rank + 1), dtype=torch.float32, device="mps")
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    b = (torch.arange(2048, dtype=torch.float32).to("mps")
         if rank == 0 else torch.zeros(2048, dtype=torch.float32, device="mps"))
    dist.broadcast(b, src=0)

    m = mccl.get_metrics()
    assert m is not None, "get_metrics() returned None with an active group"

    # Counters
    assert m.total_ops >= 7, f"total_ops={m.total_ops}"
    assert m.total_errors == 0, f"total_errors={m.total_errors}"
    assert m.total_bytes_sent > 0, "no bytes recorded as sent"
    assert m.total_bytes_recv > 0, "no bytes recorded as received"

    # Latency must be real: 1.6 MB over loopback takes measurable time.
    assert m.avg_latency_ms > 0.0, f"avg_latency_ms={m.avg_latency_ms}"
    assert m.p50_latency_ms > 0.0, f"p50_latency_ms={m.p50_latency_ms}"
    assert m.p99_latency_ms >= m.p50_latency_ms

    # Execution split: op_execute_start must be stamped with the collective
    # seq (regression for the engine-counter mixup).
    assert m.avg_execution_ms > 0.0, f"avg_execution_ms={m.avg_execution_ms}"
    assert m.avg_queue_wait_ms >= 0.0

    # Phase breakdown: the ring/tree/broadcast paths must record their
    # network phase (wall time of the transfer pipeline) and the allreduces
    # their reduce time (vDSP wall / Metal encode time — small but nonzero).
    assert m.avg_network_ms > 0.0, f"avg_network_ms={m.avg_network_ms}"
    assert m.avg_reduce_ms > 0.0, f"avg_reduce_ms={m.avg_reduce_ms}"

    assert m.peak_throughput_gbps > 0.0, \
        f"peak_throughput_gbps={m.peak_throughput_gbps}"

    if rank == 0:
        print(
            f"\n[metrics ws={world_size}] ops={m.total_ops} "
            f"sent={m.total_bytes_sent / 1e6:.1f}MB "
            f"recv={m.total_bytes_recv / 1e6:.1f}MB "
            f"avg={m.avg_latency_ms:.3f}ms p50={m.p50_latency_ms:.3f}ms "
            f"p99={m.p99_latency_ms:.3f}ms "
            f"queue={m.avg_queue_wait_ms:.3f}ms exec={m.avg_execution_ms:.3f}ms "
            f"net={m.avg_network_ms:.3f}ms reduce={m.avg_reduce_ms:.3f}ms "
            f"peak={m.peak_throughput_gbps:.2f}Gbps",
            flush=True,
        )

    # reset_metrics() really clears.
    mccl.reset_metrics()
    m2 = mccl.get_metrics()
    assert m2.total_ops == 0 and m2.total_bytes_sent == 0

    # Keep ranks aligned so no process exits while peers still transfer.
    dist.barrier()


class TestMetricsPopulated:
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_stats_nonzero_after_traffic(self, world_size):
        run_workers(_metrics_populated_fn, world_size=world_size, timeout=300)
