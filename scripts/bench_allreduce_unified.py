#!/usr/bin/env python3
"""End-to-end allreduce bench: blit+Metal vs unified Metal (MCCL_FP32_CPU_REDUCE=0)."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from mccl_test_utils import run_workers, _submit_job_mccl_env


def _bench_fn(rank, world_size):
    import os
    import time

    import torch
    import torch.distributed as dist

    nbytes = int(os.environ["BENCH_NBYTES"])
    iters = int(os.environ["BENCH_ITERS"])
    n = nbytes // 4

    if rank == 0:
        from mccl._C import _mps_storage_mode

        probe = torch.randn(8, device="mps")
        if _mps_storage_mode(probe) != "shared":
            print("SKIP", flush=True)
            return

    for _ in range(2):
        x = torch.randn(n, device="mps")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        x = torch.randn(n, device="mps")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.mps.synchronize()
    if rank == 0:
        ms = (time.perf_counter() - t0) / iters * 1000.0
        print(f"MS_PER_OP {ms:.4f}", flush=True)


def _run(unified: bool, world_size: int, nbytes: int, iters: int) -> float:
    import io
    from contextlib import redirect_stdout

    env = {
        **_submit_job_mccl_env(nbytes // (1024 * 1024)),
        "MCCL_FP32_CPU_REDUCE": "0",
        "MCCL_RING_PIPELINE": "0",
        "BENCH_NBYTES": str(nbytes),
        "BENCH_ITERS": str(iters),
    }
    if unified:
        env["MCCL_UNIFIED_COLLECTIVE"] = "1"
    else:
        env["MCCL_UNIFIED_COLLECTIVE"] = "0"

    buf = io.StringIO()

    class _Capture:
        def write(self, s):
            buf.write(s)

        def flush(self):
            pass

    # run_workers doesn't capture stdout; use timing file instead
    import tempfile
    from pathlib import Path

    out = Path(tempfile.gettempdir()) / f"mccl_bench_{'u' if unified else 'b'}.txt"

    def timed(rank, world_size):
        import os
        import time
        from pathlib import Path

        import torch
        import torch.distributed as dist

        nbytes = int(os.environ["BENCH_NBYTES"])
        iters = int(os.environ["BENCH_ITERS"])
        n = nbytes // 4
        tag = Path(os.environ["BENCH_OUT"])

        if rank == 0:
            from mccl._C import _mps_storage_mode

            if _mps_storage_mode(torch.randn(8, device="mps")) != "shared":
                tag.write_text("SKIP")
                return

        for _ in range(2):
            x = torch.randn(n, device="mps")
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            x = torch.randn(n, device="mps")
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.mps.synchronize()
        if rank == 0:
            ms = (time.perf_counter() - t0) / iters * 1000.0
            tag.write_text(f"{ms:.6f}")

    env["BENCH_OUT"] = str(out)
    run_workers(timed, world_size=world_size, env=env, timeout=600)
    text = out.read_text().strip()
    if text == "SKIP":
        return -1.0
    return float(text)


def main() -> int:
    world_size = int(os.environ.get("BENCH_WS", "4"))
    nbytes = int(os.environ.get("BENCH_NBYTES", str(25 * 1024 * 1024)))
    iters = int(os.environ.get("BENCH_ITERS", "12"))
    mb = nbytes / 1e6

    blit = _run(False, world_size, nbytes, iters)
    if blit < 0:
        print("SKIP: need torch 2.12+ shared MPS storage")
        return 0
    uni = _run(True, world_size, nbytes, iters)

    print(f"allreduce ws={world_size} {mb:.1f}MB Metal GPU (no CPU reduce):")
    print(f"  blit:     {blit:.2f} ms/op")
    print(f"  unified:  {uni:.2f} ms/op")
    print(f"  speedup:  {blit / max(uni, 1e-6):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
