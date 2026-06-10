"""Shared subprocess harness for MCCL distributed tests.

Spawns ``world_size`` python subprocesses, each initializing the MCCL
process group and executing the source of a given function ``fn(rank,
world_size)``.  Workers run with hard timeouts so a hang fails the test
instead of wedging CI.
"""

import inspect
import itertools
import os
import subprocess
import sys
import textwrap
import time

# Each call grabs a fresh port window to avoid TIME_WAIT collisions between
# tests.  MCCL_PORT_BASE = port + 100 must not collide with MASTER_PORT.
_port_counter = itertools.count(0)
_PORT_LO = 36000
_PORT_STRIDE = 211  # > 100 (port_base offset) + world_size head room


def next_port() -> int:
    return _PORT_LO + (next(_port_counter) * _PORT_STRIDE) % 24000


def run_workers(
    fn,
    world_size: int = 2,
    *,
    port: int | None = None,
    env: dict | None = None,
    timeout: float = 180.0,
    expect_failure: bool = False,
):
    """Run ``fn(rank, world_size)`` in ``world_size`` subprocesses.

    env: extra environment variables (e.g. {"MCCL_RING_ALGO": "basic"}).
    expect_failure: assert that at least one worker exits non-zero.
    """
    if port is None:
        port = next_port()

    src = textwrap.dedent(inspect.getsource(fn))
    env_lines = ""
    for k, v in (env or {}).items():
        env_lines += f"os.environ[{k!r}] = {v!r}\n"

    script = (
        "import os, sys, torch, torch.distributed as dist\n"
        "os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        "os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        "os.environ.setdefault('MCCL_LOG_LEVEL', 'INFO')\n"
        f"{env_lines}"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        "import mccl\n"
        "dist.init_process_group(backend='mccl', rank=rank, world_size=world_size, "
        "device_id=torch.device('mps:0'))\n"
        "try:\n"
        f"{textwrap.indent(src, '    ')}"
        "    fn(rank, world_size)\n"
        "finally:\n"
        "    dist.destroy_process_group()\n"
        "    os._exit(0)\n"
    )

    procs = []
    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, "-c", script, str(r), str(world_size)],
            env={**os.environ},
        )
        procs.append(p)
        time.sleep(0.4)

    codes = []
    deadline = time.monotonic() + timeout
    try:
        for p in procs:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                codes.append(p.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                for q in procs:
                    if q.poll() is None:
                        q.kill()
                raise AssertionError(
                    f"Worker hung past {timeout}s timeout (likely deadlock)"
                )
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()

    if expect_failure:
        assert any(c != 0 for c in codes), (
            f"Expected at least one worker to fail, all exited 0: {codes}"
        )
    else:
        assert all(c == 0 for c in codes), f"Worker exit codes: {codes}"
    return codes
