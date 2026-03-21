#!/usr/bin/env python3
"""
Compare training throughput from two ``--save-stats`` JSON files produced by
``examples/ddp_dummy_train.py``.

1. Single-device MPS::

    python examples/ddp_dummy_train.py --baseline --save-stats baseline_stats.json

2. MCCL DDP (example: 2 ranks on one Mac)::

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/ddp_dummy_train.py --save-stats ddp_stats.json

3. Build arrays + optional plot::

    python examples/benchmark_throughput.py \\
        --baseline baseline_stats.json --ddp ddp_stats.json -o throughput_bench

Writes ``<output>.npz`` (NumPy arrays) and ``<output>.png`` if matplotlib is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs DDP stats JSON from ddp_dummy_train.py --save-stats.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="JSON from: python examples/ddp_dummy_train.py --baseline --save-stats ...",
    )
    parser.add_argument(
        "--ddp",
        type=Path,
        required=True,
        help="JSON from: torchrun ... examples/ddp_dummy_train.py --save-stats ...",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="benchmark_throughput",
        metavar="PREFIX",
        help="Output prefix for .npz and .png (default: benchmark_throughput)",
    )
    args = parser.parse_args()

    for label, p in ("baseline", args.baseline), ("ddp", args.ddp):
        if not p.is_file():
            print(f"error: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    bl = _load(args.baseline)
    ddp = _load(args.ddp)

    required = (
        "step_times",
        "losses",
        "avg_step_time_s",
        "throughput_samples_per_sec",
        "batch_size_per_rank",
        "world_size",
        "global_batch_size",
        "total_params",
        "mode",
    )
    for name, data in ("baseline", bl), ("ddp", ddp):
        missing = [k for k in required if k not in data]
        if missing:
            print(f"error: {name} JSON missing keys: {missing}", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
    except ImportError:
        print("error: numpy is required (pip install numpy)", file=sys.stderr)
        sys.exit(1)

    bl_times = np.asarray(bl["step_times"], dtype=np.float64)
    ddp_times = np.asarray(ddp["step_times"], dtype=np.float64)
    bl_loss = np.asarray(bl["losses"], dtype=np.float64)
    ddp_loss = np.asarray(ddp["losses"], dtype=np.float64)

    out_npz = Path(f"{args.output}.npz")
    np.savez(
        out_npz,
        baseline_step_times=bl_times,
        ddp_step_times=ddp_times,
        baseline_losses=bl_loss,
        ddp_losses=ddp_loss,
        baseline_throughput=bl["throughput_samples_per_sec"],
        ddp_throughput=ddp["throughput_samples_per_sec"],
        baseline_avg_step_s=bl["avg_step_time_s"],
        ddp_avg_step_s=ddp["avg_step_time_s"],
        baseline_global_batch=bl["global_batch_size"],
        ddp_global_batch=ddp["global_batch_size"],
        baseline_world_size=bl["world_size"],
        ddp_world_size=ddp["world_size"],
    )
    print(f"wrote {out_npz.resolve()}")

    bl_tput = float(bl["throughput_samples_per_sec"])
    ddp_tput = float(ddp["throughput_samples_per_sec"])
    ratio = bl_tput / ddp_tput if ddp_tput > 0 else float("inf")

    print("\n=== Throughput comparison ===")
    print(f"  baseline ({bl['mode']}):  {bl_tput:,.1f} samples/s  "
          f"(global_batch={bl['global_batch_size']}, world={bl['world_size']})")
    print(f"  ddp      ({ddp['mode']}):  {ddp_tput:,.1f} samples/s  "
          f"(global_batch={ddp['global_batch_size']}, world={ddp['world_size']})")
    print(f"  baseline / ddp throughput ratio: {ratio:.2f}x")
    print(f"  params (baseline): {bl['total_params']:,}  (ddp): {ddp['total_params']:,}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        steps_bl = np.arange(len(bl_times))
        steps_ddp = np.arange(len(ddp_times))
        axes[0].plot(steps_bl, bl_times * 1000.0, label="baseline (MPS)", alpha=0.85)
        axes[0].plot(steps_ddp, ddp_times * 1000.0, label="DDP (MCCL)", alpha=0.85)
        axes[0].set_ylabel("Step time (ms)")
        axes[0].set_title("Per-step wall time")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps_bl, bl_loss, label="baseline loss", alpha=0.85)
        axes[1].plot(steps_ddp, ddp_loss, label="ddp loss", alpha=0.85)
        axes[1].set_xlabel("Training step (after warmup, zero-based in file)")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        out_png = Path(f"{args.output}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png.resolve()}")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        names = ["baseline\n(MPS)", "DDP\n(MCCL)"]
        vals = [bl_tput, ddp_tput]
        colors = ["#2ecc71", "#3498db"]
        ax2.bar(names, vals, color=colors)
        ax2.set_ylabel("Throughput (samples / sec)")
        ax2.set_title("Average throughput (from JSON)")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        out_bar = Path(f"{args.output}_bars.png")
        fig2.savefig(out_bar, dpi=150)
        plt.close(fig2)
        print(f"wrote {out_bar.resolve()}")
    except ImportError:
        print(
            "(matplotlib not installed; skipped .png — pip install matplotlib)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
