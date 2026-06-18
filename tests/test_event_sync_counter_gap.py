"""EventSync per-event counters: fence churn must not poison mps_event waits."""

import platform

import pytest


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="EventSync requires macOS Apple Silicon + MPS",
)
@pytest.mark.parametrize("ws", [4, 6, 8])
def test_event_sync_fence_churn_does_not_hang(ws):
  import mccl._C as mccl_c

  ok = mccl_c._test_event_sync_fence_churn(ws)
  assert ok, "event_sync_init failed or unavailable"
