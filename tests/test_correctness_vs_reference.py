"""Collective correctness vs an analytically computed reference.

Every rank fills its tensor with a deterministic pattern derived from its
rank; the expected reduced/gathered value is computed locally on CPU in
float64 and compared elementwise.  Covers:

- dtypes: float32 / float16 / bfloat16
- sizes spanning the small-message, plain-ring, and chunked-ring paths,
  including odd / non-power-of-two / chunk-boundary element counts
- ops: SUM / AVG / MAX
- algorithm env matrix: chunked ring (default), basic ring, fp32 CPU reduce
- bit-level determinism of repeated fp32 allreduces
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)

# Sizes chosen to hit every code path:
#   1, 3            -> tiny (star/small path)
#   1023, 4097      -> small path, non-power-of-2, around MCCL_GPU_THRESHOLD
#   70_001          -> > small_msg_threshold (ring path), odd numel
#   262_147         -> multiple ring chunks, prime-ish
#   2_100_001       -> 8 MB+ fp32 (2-rank RS+AG fast path), odd numel
SIZES = [1, 3, 1023, 4097, 70_001, 262_147]
LARGE_SIZES = [2_100_001]


def _allreduce_check_fn(rank, world_size):
    import os
    import torch
    import torch.distributed as dist

    sizes = [int(s) for s in os.environ["MCCL_TEST_SIZES"].split(",")]
    dtype = getattr(torch, os.environ["MCCL_TEST_DTYPE"])
    op_name = os.environ["MCCL_TEST_OP"]
    op = getattr(dist.ReduceOp, op_name)

    for n in sizes:
        # Deterministic per-rank pattern, exactly representable in fp16
        # so reductions are exact and tolerance can be tight.
        base = torch.arange(n, dtype=torch.float64) % 13 - 6.0
        mine = (base + float(rank + 1)).to(dtype)
        t = mine.to("mps")

        contribs = [
            ((base + float(r + 1)).to(dtype).to(torch.float64))
            for r in range(world_size)
        ]
        if op_name == "SUM":
            expected = sum(contribs)
        elif op_name == "AVG":
            expected = sum(contribs) / world_size
        elif op_name == "MAX":
            expected = contribs[0]
            for c in contribs[1:]:
                expected = torch.maximum(expected, c)
        else:
            raise AssertionError(f"unhandled op {op_name}")

        dist.all_reduce(t, op=op)
        got = t.cpu().to(torch.float64)

        # Values are small integers (exact in all dtypes); ring reduction
        # order cannot introduce error for exact values, so allow only a
        # tiny dtype-rounding slack.
        tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dtype]
        max_err = (got - expected).abs().max().item()
        assert max_err <= tol, (
            f"n={n} dtype={dtype} op={op_name}: max_err={max_err} > {tol}"
        )


ALGO_ENVS = {
    "chunked_default": {},
    "basic_ring": {"MCCL_RING_ALGO": "basic"},
    "fp32_cpu_reduce": {"MCCL_FP32_CPU_REDUCE": "1"},
}


class TestAllreduceVsReference:
    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    @pytest.mark.parametrize("op", ["SUM", "AVG", "MAX"])
    @pytest.mark.parametrize("algo", list(ALGO_ENVS.keys()))
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_allreduce(self, dtype, op, algo, world_size):
        if op == "MAX" and algo == "fp32_cpu_reduce" and dtype != "float32":
            pytest.skip("cpu reduce env only affects fp32")
        env = {
            **ALGO_ENVS[algo],
            "MCCL_TEST_SIZES": ",".join(str(s) for s in SIZES),
            "MCCL_TEST_DTYPE": dtype,
            "MCCL_TEST_OP": op,
        }
        run_workers(_allreduce_check_fn, world_size=world_size, env=env)

    @pytest.mark.parametrize("algo", ["chunked_default", "fp32_cpu_reduce"])
    def test_allreduce_large_odd_numel(self, algo):
        """8 MB+ odd-element-count fp32: regression for the two-rank RS+AG
        byte-split corruption (boundary element spliced from both halves)."""
        env = {
            **ALGO_ENVS[algo],
            "MCCL_TEST_SIZES": ",".join(str(s) for s in LARGE_SIZES),
            "MCCL_TEST_DTYPE": "float32",
            "MCCL_TEST_OP": "SUM",
        }
        run_workers(_allreduce_check_fn, world_size=2, env=env, timeout=300)

    @pytest.mark.parametrize("world_size", [3, 4])
    def test_chunked_ring_schedule(self, world_size):
        """ws >= 3 chunked ring (the corrected allgather schedule): every
        chunk must carry every rank's contribution."""
        env = {
            "MCCL_RING_ALGO": "chunked",
            "MCCL_TEST_SIZES": "70001,262147",
            "MCCL_TEST_DTYPE": "float32",
            "MCCL_TEST_OP": "SUM",
        }
        run_workers(_allreduce_check_fn, world_size=world_size, env=env, timeout=300)


def _determinism_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    n = 300_007
    torch.manual_seed(1234 + rank)
    src = torch.randn(n, dtype=torch.float32)

    results = []
    for _ in range(2):
        t = src.clone().to("mps")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        results.append(t.cpu())

    # Same inputs, same algorithm, same step count -> bitwise identical.
    assert torch.equal(results[0], results[1]), "fp32 allreduce is nondeterministic"


class TestDeterminism:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_repeated_allreduce_bitwise_equal(self, world_size):
        run_workers(_determinism_fn, world_size=world_size)


def _allgather_check_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    for n in (5, 4097, 70_001):
        mine = (torch.arange(n, dtype=torch.float32) + 1000.0 * (rank + 1)).to("mps")
        outs = [torch.zeros(n, dtype=torch.float32, device="mps") for _ in range(world_size)]
        dist.all_gather(outs, mine)
        for r in range(world_size):
            expected = torch.arange(n, dtype=torch.float32) + 1000.0 * (r + 1)
            assert torch.equal(outs[r].cpu(), expected), f"allgather n={n} rank slot {r}"


def _reduce_scatter_check_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    n = 4096
    ins = [
        (torch.full((n,), float(rank + 1) * (i + 1), dtype=torch.float32)).to("mps")
        for i in range(world_size)
    ]
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    dist.reduce_scatter(out, ins, op=dist.ReduceOp.SUM)
    expected = sum(float(r + 1) * (rank + 1) for r in range(world_size))
    assert torch.allclose(
        out.cpu(), torch.full((n,), expected)
    ), f"reduce_scatter rank {rank}"


def _broadcast_check_fn(rank, world_size):
    import torch
    import torch.distributed as dist

    for n in (7, 70_001):
        t = (
            torch.arange(n, dtype=torch.float32).to("mps")
            if rank == 0
            else torch.zeros(n, dtype=torch.float32, device="mps")
        )
        dist.broadcast(t, src=0)
        assert torch.equal(t.cpu(), torch.arange(n, dtype=torch.float32))


class TestOtherCollectivesVsReference:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_allgather(self, world_size):
        run_workers(_allgather_check_fn, world_size=world_size)

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_reduce_scatter(self, world_size):
        run_workers(_reduce_scatter_check_fn, world_size=world_size)

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_broadcast(self, world_size):
        run_workers(_broadcast_check_fn, world_size=world_size)
