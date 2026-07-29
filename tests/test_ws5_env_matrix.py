"""ws=5 env matrix: compare test-prod vs submit-conservative MCCL settings."""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers
from test_ring_algo_correctness import _allreduce_vs_f64_reference_fn

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="requires macOS Apple Silicon + MCCL",
)

# test_broadcast_object_list.py base harness
_TEST_HARNESS = {
    "MCCL_OVERLAP_COMM": "1",
    "MCCL_COLLECTIVE_CONCURRENCY": "1",
}

# test_mccl_mask_unet_bb_ddp _PROD_MCCL_ENV (ws=8 prod stack)
_PROD_STACK = {
    "MCCL_RING_PIPELINE": "0",
    "MCCL_RING_ALGO": "ring_chunked",
    "MCCL_OVERLAP_COMM": "1",
    "MCCL_PIPELINE_DEPTH": "1",
    "MCCL_COLLECTIVE_CONCURRENCY": "1",
    "DDP_BUCKET_MB": "25",
}

# current submit_job.sh defaults
_SUBMIT_NOW = {
    "MCCL_RING_PIPELINE": "0",
    "MCCL_RING_ALGO": "basic",
    "MCCL_OVERLAP_COMM": "0",
    "MCCL_EVENT_SYNC": "0",
    "MCCL_PIPELINE_DEPTH": "1",
    "MCCL_COLLECTIVE_CONCURRENCY": "1",
    "DDP_BUCKET_MB": "25",
}

# overlap on + event sync on (MCCL default when EVENT_SYNC unset)
_OVERLAP_EVENT_ON = {
    **_TEST_HARNESS,
    "MCCL_RING_ALGO": "chunked",
    "MCCL_RING_PIPELINE": "0",
    "MCCL_TEST_SIZES": "6553600",
    "MCCL_TEST_RTOL": "5e-5",
}


@pytest.mark.parametrize(
    "label,base",
    [
        ("test_harness_overlap1", {**_TEST_HARNESS, "MCCL_RING_ALGO": "chunked", "MCCL_TEST_SIZES": "6553600", "MCCL_TEST_RTOL": "5e-5"}),
        ("prod_stack_overlap1_chunked", {**_PROD_STACK, "MCCL_RING_ALGO": "chunked", "MCCL_TEST_SIZES": "6553600", "MCCL_TEST_RTOL": "5e-5"}),
        ("submit_overlap0_eventsync0_basic", {**_SUBMIT_NOW, "MCCL_TEST_SIZES": "6553600", "MCCL_TEST_RTOL": "5e-5"}),
        ("overlap1_eventsync0_chunked", {**_TEST_HARNESS, "MCCL_RING_ALGO": "chunked", "MCCL_EVENT_SYNC": "0", "MCCL_TEST_SIZES": "6553600", "MCCL_TEST_RTOL": "5e-5"}),
        ("overlap1_eventsync1_chunked", {**_OVERLAP_EVENT_ON, "MCCL_EVENT_SYNC": "1"}),
        ("env_c2_capped_ws5", {**_TEST_HARNESS, "MCCL_RING_ALGO": "chunked", "MCCL_COLLECTIVE_CONCURRENCY": "2", "MCCL_TEST_SIZES": "6553600", "MCCL_TEST_RTOL": "5e-5"}),
    ],
)
def test_ws5_25mb_allreduce_env_matrix(label, base):
    run_workers(_allreduce_vs_f64_reference_fn, world_size=5, env=base, timeout=600)
