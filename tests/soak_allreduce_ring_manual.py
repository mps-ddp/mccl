#!/usr/bin/env python3
"""Manual soak: 3-rank MCCL plain-ring all_reduce (no pytest).

For long runs or Metal validation, execute directly on Apple Silicon, for example:

    MCCL_RING_ALGO=basic MCCL_STRESS_ITERS=100000 \\
        python tests/soak_allreduce_ring_manual.py

Environment:
    MCCL_STRESS_ITERS — iterations per rank (default: 50000)
    MCCL_RING_ALGO — ring variant (default: basic, via setdefault in child)
    MCCL_SOAK_PORT — base MASTER_PORT (default: 34900; MCCL_PORT_BASE is port+100)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

# Plain ring path: numel * 4 bytes > default MCCL_SMALL_MSG_THRESHOLD
_RING_NUMEL = 70_000


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        print("This soak requires macOS on Apple Silicon.", file=sys.stderr)
        return 2

    port = int(os.environ.get("MCCL_SOAK_PORT", "34900"))
    n_iters = int(os.environ.get("MCCL_STRESS_ITERS", "50000"))
    world_size = 3

    script = f"""
import os, sys
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '{port}'
os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'
os.environ['MCCL_PORT_BASE'] = '{port + 100}'
os.environ.setdefault('MCCL_RING_ALGO', 'basic')
os.environ.setdefault('MCCL_LOG_LEVEL', 'WARN')
rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_iters = int(sys.argv[3])
import torch, torch.distributed as dist
import mccl
dist.init_process_group(backend='mccl', rank=rank, world_size=world_size,
                        device_id=torch.device('mps:0'))
try:
    sizes = (1024, {_RING_NUMEL}, 4096, {_RING_NUMEL // 2})
    for i in range(n_iters):
        s = sizes[i % len(sizes)]
        t = torch.randn(s, device='mps', dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if i % 500 == 0 and i > 0:
            torch.mps.synchronize()
finally:
    dist.destroy_process_group()
    os._exit(0)
"""
    procs = []
    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, "-c", script, str(r), str(world_size), str(n_iters)]
        )
        procs.append(p)
        time.sleep(0.5)
    rc = 0
    for p in procs:
        rc = max(rc, p.wait())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
