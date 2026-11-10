"""Workers exit normally after ``destroy_process_group()``.

The shared harness ends every worker with ``os._exit(0)``, which skips C++
static destructors and so hides exit-time bugs.  These workers return
normally while still holding the group, as DDP does.
``destroy_process_group()`` must shut MCCL down promptly (threads joined,
peer sockets closed) even though the backend object stays alive, and a peer
closing its sockets while this process runs its static destructors must not
abort it (a reader logging "peer closed connection" through a destroyed
logger mutex used to raise ``std::system_error`` and SIGABRT the process).
"""

import os
import platform
import subprocess
import sys
import textwrap
import time

import pytest

from mccl_test_utils import next_port

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Process group tests require macOS on Apple Silicon",
)

# Runs to repeat: the abort depended on exit timing between the two ranks.
ROUNDS = 8

_WORKER = textwrap.dedent("""
    import os, subprocess, sys, time
    import torch, torch.distributed as dist

    def mccl_sockets(port_base, world_size):
        # TCP sockets of this process on MCCL's listen ports (the store's
        # MASTER_PORT is excluded).
        out = subprocess.run(["lsof", "-a", "-p", str(os.getpid()), "-iTCP",
                              "-n", "-P", "-F", "n"],
                             capture_output=True, text=True).stdout
        ports = {f":{port_base + r}" for r in range(world_size)}
        return sum(1 for line in out.splitlines()
                   if line.startswith("n") and any(p in line for p in ports))

    rank, world_size, port = (int(a) for a in sys.argv[1:4])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    os.environ["MCCL_PORT_BASE"] = str(port + 100)
    os.environ["MCCL_LOG_LEVEL"] = "WARN"
    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size,
                            device_id=torch.device("mps:0"))
    t = torch.full((1024,), float(rank + 1), device="mps")
    dist.all_reduce(t)
    torch.mps.synchronize()
    assert t[0].item() == world_size * (world_size + 1) / 2
    during = mccl_sockets(port + 100, world_size)
    held = dist.group.WORLD   # DDP's reducer keeps the group alive like this
    start = time.monotonic()
    dist.destroy_process_group()
    seconds = time.monotonic() - start
    after = mccl_sockets(port + 100, world_size)
    print(f"EXIT rank={rank} sockets={during}/{after} destroy={seconds:.3f}",
          flush=True)
    # Normal interpreter exit with ``held`` alive: static destructors run
    # while the peer exits.
""")


def _round(port: int) -> list[tuple[int, str, str]]:
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(r), "2", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ},
        )
        for r in range(2)
    ]
    results = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise
        results.append((p.returncode, out, err))
    return results


@pytest.mark.timeout(600)
def test_workers_exit_cleanly_after_destroy():
    for i in range(ROUNDS):
        for rank, (code, out, err) in enumerate(_round(next_port())):
            where = f"round {i} rank {rank}"
            assert "terminating due to" not in err, f"{where} aborted:\n{err}"
            assert code == 0, f"{where} exit code {code}:\n{out}\n{err}"
            line = next(ln for ln in out.splitlines() if ln.startswith("EXIT"))
            fields = dict(f.split("=") for f in line.split()[1:])
            during, after = (int(n) for n in fields["sockets"].split("/"))
            assert during > 0, f"{where}: no MCCL sockets found ({line})"
            assert after == 0, f"{where}: MCCL sockets outlive destroy ({line})"
            assert float(fields["destroy"]) < 2.0, f"{where}: slow destroy ({line})"
        time.sleep(0.2)
