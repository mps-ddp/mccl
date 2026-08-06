#!/usr/bin/env python3
"""NCCL-tests-style bus bandwidth benchmark for MCCL.

Reports, per message size:
  - algbw  = bytes / time                      (algorithmic bandwidth)
  - busbw  = algbw * 2*(n-1)/n  for allreduce  (bus bandwidth, NCCL convention)

Also measures:
  - compute/comm overlap ratio: wall time of (backward-like compute + async
    allreduce) vs the sum of each alone.  1.0 = perfect overlap.
  - optional GPU utilization sampling via `sudo powermetrics` (--powermetrics).

Usage:
    python tests/bench_busbw.py                  # full sweep, ws=2
    python tests/bench_busbw.py --quick          # short sweep for CI
    python tests/bench_busbw.py --world-size 3
    MCCL_RING_ALGO=basic python tests/bench_busbw.py   # compare algorithms
"""

import argparse
import os
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--port", type=int, default=47000)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="short sweep for CI")
    p.add_argument("--min-bytes", type=int, default=4 * 1024)
    p.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    p.add_argument("--concurrency", type=int, default=None,
                   help="MCCL_COLLECTIVE_CONCURRENCY for the run (ws>=3)")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--bucket-mb", type=int, default=25,
                   help="representative overlap bucket size in MiB")
    p.add_argument("--link-gbps", type=float, default=None,
                   help="physical link rate; prints busbw as %% of the "
                        "NCCL ring bound 2(N-1)/N x link")
    p.add_argument("--powermetrics", action="store_true",
                   help="sample GPU utilization via sudo powermetrics")
    p.add_argument("--json", metavar="PATH",
                   help="write rank-0 benchmark rows, metrics, and config as JSON")
    p.add_argument("--rank", type=int, default=None,
                   help="external rank for multi-Mac runs (one worker per host)")
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--listen-addr", default=None,
                   help="MCCL address advertised by this host (defaults to master address)")
    return p.parse_args()


WORKER = r"""
import json, os, platform, socket, sys, time, torch, torch.distributed as dist

rank = int(sys.argv[1]); world_size = int(sys.argv[2])
iters = int(os.environ["BENCH_ITERS"]); warmup = int(os.environ["BENCH_WARMUP"])
sizes = [int(s) for s in os.environ["BENCH_SIZES"].split(",")]
dtype = getattr(torch, os.environ["BENCH_DTYPE"])
element_size = torch.empty((), dtype=dtype).element_size()
bucket_bytes = int(os.environ["BENCH_BUCKET_MB"]) * 1024 * 1024

import mccl
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size,
                        device_id=torch.device("mps:0"))

def bench_allreduce(numel):
    t = torch.randn(numel, dtype=dtype, device="mps")
    torch.mps.synchronize()
    for _ in range(warmup):
        dist.all_reduce(t)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    dt = (time.perf_counter() - t0) / iters
    return dt

def bench_overlap(numel):
    # Comm alone
    t = torch.randn(numel, dtype=dtype, device="mps")
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    comm = (time.perf_counter() - t0) / iters

    # Compute alone (backward-ish matmul chain)
    a = torch.randn(1024, 1024, device="mps")
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b = a
        for _ in range(8):
            b = b @ a
        torch.mps.synchronize()
    comp = (time.perf_counter() - t0) / iters

    # Overlapped: launch async allreduce, run compute, wait
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        w = dist.all_reduce(t, async_op=True)
        b = a
        for _ in range(8):
            b = b @ a
        torch.mps.synchronize()
        w.wait()
    both = (time.perf_counter() - t0) / iters
    return comm, comp, both

link_gbps = float(os.environ.get("BENCH_LINK_GBPS", "0") or 0)
rows = []

if rank == 0:
    print(f"# MCCL busbw  ws={world_size} iters={iters} "
          f"ring_algo={os.environ.get('MCCL_RING_ALGO','chunked(default)')} "
          f"pipeline={os.environ.get('MCCL_RING_PIPELINE','1(default)')} "
          f"concurrency={os.environ.get('MCCL_COLLECTIVE_CONCURRENCY','2(default)')} "
          f"dtype={dtype} bucket_mb={bucket_bytes // (1024 * 1024)} "
          f"fp32_cpu_reduce={os.environ.get('MCCL_FP32_CPU_REDUCE','0')}")
    hdr = f"{'bytes':>14} {'numel':>12} {'time(ms)':>10} {'algbw(GB/s)':>12} {'busbw(GB/s)':>12}"
    if link_gbps > 0:
        hdr += f" {'%ring-bound':>12}"
    print(hdr)

for numel in sizes:
    dt = bench_allreduce(numel)
    nbytes = numel * element_size
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world_size - 1) / world_size
    if rank == 0:
        rows.append({
            "bytes": nbytes,
            "numel": numel,
            "time_ms": dt * 1e3,
            "algbw_gbps": algbw,
            "busbw_gbps": busbw,
            "ring_bound_percent": (
                100.0 * busbw / (link_gbps / 8.0) if link_gbps > 0 else None
            ),
        })
        line = f"{nbytes:>14} {numel:>12} {dt*1e3:>10.3f} {algbw:>12.2f} {busbw:>12.2f}"
        if link_gbps > 0:
            # NCCL ring bound: busbw cannot exceed the per-link rate; report
            # how much of the physical link the ring is actually using.
            link_gBps = link_gbps / 8.0
            line += f" {100.0 * busbw / link_gBps:>11.1f}%"
        print(line, flush=True)

# Overlap at the requested representative DDP bucket size.
comm, comp, both = bench_overlap(bucket_bytes // element_size)
if rank == 0:
    ideal = max(comm, comp)
    serial = comm + comp
    overlap_ratio_raw = (serial - both) / max(serial - ideal, 1e-9)
    overlap_ratio = max(0.0, min(1.0, overlap_ratio_raw))
    print(f"# overlap@{bucket_bytes // (1024 * 1024)}MB: "
          f"comm={comm*1e3:.1f}ms comp={comp*1e3:.1f}ms "
          f"together={both*1e3:.1f}ms  overlap_efficiency={overlap_ratio:.2f} "
          f"(1.0 = perfect, 0.0 = fully serial)")
    json_path = os.environ.get("BENCH_JSON")
    if json_path:
        metrics = mccl.get_metrics()
        metric_names = (
            "total_ops", "total_bytes_sent", "total_bytes_recv", "total_errors",
            "avg_latency_ms", "p50_latency_ms", "p99_latency_ms",
            "peak_throughput_gbps", "avg_sync_ms", "avg_network_ms",
            "avg_reduce_ms", "avg_queue_wait_ms", "avg_execution_ms",
        )
        payload = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "world_size": world_size,
            "iterations": iters,
            "warmup": warmup,
            "link_gbps": link_gbps or None,
            "config": {
                key: value for key, value in sorted(os.environ.items())
                if key.startswith("MCCL_") or key in ("DDP_BUCKET_MB",)
            },
            "rows": rows,
            "overlap": {
                "bucket_bytes": bucket_bytes,
                "comm_ms": comm * 1e3,
                "compute_ms": comp * 1e3,
                "together_ms": both * 1e3,
                "efficiency": overlap_ratio,
                "efficiency_raw": overlap_ratio_raw,
            },
            "metrics": {
                name: getattr(metrics, name) for name in metric_names
            } if metrics else None,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"# wrote JSON results to {json_path}", flush=True)

dist.destroy_process_group()
os._exit(0)
"""


def main():
    args = parse_args()
    if args.rank is not None and not (0 <= args.rank < args.world_size):
        raise SystemExit("--rank must be in [0, world-size)")

    sizes = []
    element_size = 2 if args.dtype == "bfloat16" else 4
    n = max(args.min_bytes // element_size, 1024)
    max_n = args.max_bytes // element_size
    if args.quick:
        max_n = min(max_n, 16 * 1024 * 1024)
        args.iters = 5
        args.warmup = 2
    while n <= max_n:
        sizes.append(n)
        n *= 4

    env = {
        **os.environ,
        "MASTER_ADDR": args.master_addr,
        "MASTER_PORT": str(args.port),
        "MCCL_PORT_BASE": str(args.port + 100),
        "BENCH_ITERS": str(args.iters),
        "BENCH_WARMUP": str(args.warmup),
        "BENCH_SIZES": ",".join(str(s) for s in sizes),
        "BENCH_DTYPE": args.dtype,
        "BENCH_BUCKET_MB": str(args.bucket_mb),
    }
    if args.concurrency is not None:
        env["MCCL_COLLECTIVE_CONCURRENCY"] = str(args.concurrency)
    if args.listen_addr is not None:
        env["MCCL_LISTEN_ADDR"] = args.listen_addr
    elif args.rank is None:
        env["MCCL_LISTEN_ADDR"] = "127.0.0.1"
    if args.link_gbps is not None:
        env["BENCH_LINK_GBPS"] = str(args.link_gbps)
    if args.json is not None:
        env["BENCH_JSON"] = args.json
    env.setdefault("MCCL_LOG_LEVEL", "ERROR")

    pm_proc = None
    if args.powermetrics:
        pm_proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000"],
            stdout=open("powermetrics_gpu.log", "w"),
            stderr=subprocess.DEVNULL,
        )
        print("# powermetrics sampling -> powermetrics_gpu.log")

    procs = []
    ranks = [args.rank] if args.rank is not None else range(args.world_size)
    for r in ranks:
        p = subprocess.Popen(
            [sys.executable, "-c", WORKER, str(r), str(args.world_size)], env=env
        )
        procs.append(p)
        time.sleep(0.4)

    rc = 0
    for p in procs:
        rc |= p.wait()

    if pm_proc:
        pm_proc.terminate()

    sys.exit(rc)


if __name__ == "__main__":
    main()
