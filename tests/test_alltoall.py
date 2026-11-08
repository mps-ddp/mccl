"""all_to_all_single (ProcessGroupMCCL::alltoall_base): equal and uneven splits, large messages,
the blit path for non-unified buffers, and sequencing next to other collectives.

Spawns processes on the same host. Requires macOS Apple Silicon with MCCL built.
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Process group tests require macOS on Apple Silicon",
)


def _equal_splits(rank, world_size):
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        rows, cols = 3, 5
        inp = torch.empty(world_size * rows, cols, device="mps", dtype=dtype)
        for j in range(world_size):
            inp[j * rows:(j + 1) * rows] = rank * world_size + j
        out = torch.full_like(inp, -1)
        dist.all_to_all_single(out, inp)
        for j in range(world_size):
            got = out[j * rows:(j + 1) * rows]
            assert torch.all(got == j * world_size + rank), (dtype, j, got)


def _random_vs_allgather(rank, world_size):
    torch.manual_seed(rank)
    inp = torch.randn(world_size * 3510, 20, 128, device="mps", dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    dist.all_to_all_single(out, inp)
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    n = inp.shape[0] // world_size
    for j in range(world_size):
        assert torch.equal(out[j * n:(j + 1) * n], gathered[j][rank * n:(rank + 1) * n]), j


def _large_known_values(rank, world_size):
    n = 3510
    base = torch.arange(world_size * n * 64, dtype=torch.float32).reshape(world_size * n, 64)
    inp = (base + rank * 1e6).to("mps")
    out = torch.full_like(inp, -1)
    dist.all_to_all_single(out, inp)
    for j in range(world_size):
        exp = (base[rank * n:(rank + 1) * n] + j * 1e6).to("mps")
        assert torch.equal(out[j * n:(j + 1) * n], exp), j


def _uneven_splits(rank, world_size):
    def rows(src, dst):
        if (src == 0) != (dst == 0):
            return 0  # rank 0 keeps only its own chunk: it sends and receives nothing
        return src + dst + 1

    in_splits = [rows(rank, j) for j in range(world_size)]
    out_splits = [rows(j, rank) for j in range(world_size)]
    inp = torch.cat([torch.full((n, 4), float(rank * 10 + j), device="mps")
                     for j, n in enumerate(in_splits)])
    out = torch.full((sum(out_splits), 4), -1.0, device="mps")
    dist.all_to_all_single(out, inp, out_splits, in_splits)
    off = 0
    for j, n in enumerate(out_splits):
        assert torch.all(out[off:off + n] == j * 10 + rank), (j, out[off:off + n])
        off += n
    t = torch.full((8,), float(rank + 1), device="mps")
    dist.all_reduce(t)
    assert torch.all(t == world_size * (world_size + 1) / 2), t


def _interleaved(rank, world_size):
    for it in range(20):
        inp = torch.arange(world_size * 64, device="mps", dtype=torch.float32) + 1000 * rank + it
        out = torch.empty_like(inp)
        work = dist.all_to_all_single(out, inp, async_op=True)
        t = torch.full((1000,), float(rank), device="mps")
        dist.all_reduce(t)
        work.wait()
        assert torch.all(t == sum(range(world_size))), t
        for j in range(world_size):
            exp = torch.arange(rank * 64, (rank + 1) * 64, device="mps", dtype=torch.float32) + 1000 * j + it
            assert torch.equal(out[j * 64:(j + 1) * 64], exp), (it, j)


@pytest.mark.parametrize("ws", [2, 3, 4])
def test_equal_splits(ws):
    run_workers(_equal_splits, ws, timeout=120)


@pytest.mark.parametrize("ws", [2, 4])
def test_large_random_vs_allgather(ws):
    run_workers(_random_vs_allgather, ws, timeout=180)


@pytest.mark.parametrize("ws", [2, 3])
def test_large_blit_path(ws):
    # Checked against known values: allgather is wrong with MCCL_UNIFIED_COLLECTIVE=0 (see CHANGELOG).
    run_workers(_large_known_values, ws, env={"MCCL_UNIFIED_COLLECTIVE": "0"}, timeout=180)


@pytest.mark.parametrize("ws", [3, 4])
def test_uneven_splits_and_sequencing(ws):
    run_workers(_uneven_splits, ws, timeout=120)


@pytest.mark.parametrize("ws", [2, 3])
def test_interleaved_with_allreduce(ws):
    run_workers(_interleaved, ws, timeout=120)


_STREAM_ORDERED = {"MCCL_STREAM_ORDERED_A2A": "1"}


@pytest.mark.parametrize("ws", [2, 3])
def test_stream_ordered_equal_splits(ws):
    run_workers(_equal_splits, ws, env=_STREAM_ORDERED, timeout=120)


@pytest.mark.parametrize("ws", [2, 4])
def test_stream_ordered_large_random_vs_allgather(ws):
    run_workers(_random_vs_allgather, ws, env=_STREAM_ORDERED, timeout=180)


@pytest.mark.parametrize("ws", [2, 3])
def test_stream_ordered_interleaved_with_allreduce(ws):
    run_workers(_interleaved, ws, env=_STREAM_ORDERED, timeout=120)


def test_stream_ordered_uneven_splits_and_sequencing():
    run_workers(_uneven_splits, 3, env=_STREAM_ORDERED, timeout=120)
