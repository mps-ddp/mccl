#!/usr/bin/env python3
"""Benchmark MCCL staging: blit vs unified cpu_ptr send (torch 2.12+ shared storage)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run_once(unified: bool, nbytes: int, iters: int) -> float:
    env = os.environ.copy()
    if unified:
        env["MCCL_UNIFIED_COLLECTIVE"] = "1"
    else:
        env.pop("MCCL_UNIFIED_COLLECTIVE", None)
    script = textwrap.dedent(
        f"""
        import os, time, torch
        import mccl
        from mccl._C import (
            _mps_storage_mode,
            _stage_for_send_collective_bench,
            _tensor_cpu_accessible,
        )
        n = {nbytes} // 4
        iters = {iters}
        t = torch.randn(n, device="mps", dtype=torch.float32)
        torch.mps.synchronize()
        if _mps_storage_mode(t) != "shared":
            print("SKIP")
            raise SystemExit(0)
        ms = _stage_for_send_collective_bench(t, iters)
        print(ms)
        """
    )
    out = subprocess.check_output([sys.executable, "-c", script], env=env, text=True).strip()
    if out == "SKIP":
        return -1.0
    return float(out)


def main() -> int:
    nbytes = int(os.environ.get("BENCH_NBYTES", str(25 * 1024 * 1024)))
    iters = int(os.environ.get("BENCH_ITERS", "30"))

    blit_ms = _run_once(False, nbytes, iters)
    if blit_ms < 0:
        print("SKIP: MPS storage not shared (need torch 2.12+)")
        return 0
    uni_ms = _run_once(True, nbytes, iters)
    mb = nbytes / 1e6
    print(f"stage_for_send_collective @ {mb:.1f} MB, {iters} iters:")
    print(f"  blit (default):     {blit_ms:.3f} ms")
    print(f"  unified (cpu_ptr):  {uni_ms:.3f} ms")
    print(f"  speedup:            {blit_ms / max(uni_ms, 1e-6):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
