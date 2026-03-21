#!/usr/bin/env python3
"""
Compare training throughput from two ``--save-stats`` JSON files produced by
``examples/ddp_dummy_train.py``.

1. **Baseline** = single machine **M1 Max**, one MPS device::

    python examples/ddp_dummy_train.py --baseline --save-stats baseline_stats.json

2. MCCL DDP (example: 2 ranks on one Mac)::

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/ddp_dummy_train.py --save-stats ddp_stats.json

3. Build arrays + optional plot::

    python examples/benchmark_throughput.py \\
        --baseline baseline_stats.json --ddp ddp_stats.json -o throughput_bench

Writes ``<output>.npz`` (NumPy arrays) and ``<output>.png`` if matplotlib is installed.
The loss subplot uses **cumulative wall time** (sum of per-step times in the JSON), not step index.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# What ``--save-stats`` from ``ddp_dummy_train.py --baseline`` represents in plots / stdout.
# Override with ``--baseline-label`` if your JSON came from another machine.
_DEFAULT_BASELINE_LABEL = "single M1 Max (MPS)"


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
    parser.add_argument(
        "--baseline-label",
        default=_DEFAULT_BASELINE_LABEL,
        metavar="TEXT",
        help=f"Label for baseline JSON in charts (default: {_DEFAULT_BASELINE_LABEL!r})",
    )
    args = parser.parse_args()
    bl_label = args.baseline_label

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
    bl_time_s = np.cumsum(bl_times)
    ddp_time_s = np.cumsum(ddp_times)

    out_npz = Path(f"{args.output}.npz")
    np.savez(
        out_npz,
        baseline_step_times=bl_times,
        ddp_step_times=ddp_times,
        baseline_losses=bl_loss,
        ddp_losses=ddp_loss,
        baseline_cumulative_time_s=bl_time_s,
        ddp_cumulative_time_s=ddp_time_s,
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
    ddp_frac = (ddp_tput / bl_tput * 100.0) if bl_tput > 0 else 0.0

    if bl["global_batch_size"] != ddp["global_batch_size"]:
        print(
            "\nwarning: global_batch differs between JSON files — "
            "throughput comparison is not apples-to-apples.",
            file=sys.stderr,
        )

    print("\n=== Throughput comparison ===")
    print(f"  {bl_label}:  {bl_tput:,.1f} samples/s  "
          f"(global_batch={bl['global_batch_size']}, world={bl['world_size']})")
    print(f"  DDP (MCCL):  {ddp_tput:,.1f} samples/s  "
          f"(global_batch={ddp['global_batch_size']}, world={ddp['world_size']})")
    print(f"  {bl_label} / DDP throughput ratio: {ratio:.2f}x  "
          f"(DDP achieves ~{ddp_frac:.0f}% of baseline samples/s)")
    print(f"  params ({bl_label}): {bl['total_params']:,}  (ddp): {ddp['total_params']:,}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        steps_bl = np.arange(len(bl_times))
        steps_ddp = np.arange(len(ddp_times))
        bl_mean_ms = float(np.mean(bl_times) * 1000.0)
        ddp_mean_ms = float(np.mean(ddp_times) * 1000.0)
        axes[0].plot(
            steps_bl,
            bl_times * 1000.0,
            label=f"{bl_label}, mean {bl_mean_ms:.1f} ms/step",
            alpha=0.85,
            linewidth=1.2,
        )
        axes[0].plot(
            steps_ddp,
            ddp_times * 1000.0,
            label=f"DDP (MCCL), mean {ddp_mean_ms:.1f} ms/step",
            alpha=0.85,
            linewidth=1.2,
        )
        axes[0].axhline(bl_mean_ms, color="C0", linestyle=":", alpha=0.5)
        axes[0].axhline(ddp_mean_ms, color="C1", linestyle=":", alpha=0.5)
        axes[0].set_ylabel("Step time (ms)")
        axes[0].set_title("Per-step wall time (matched workload JSONs)")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            bl_time_s,
            bl_loss,
            label=f"{bl_label} loss",
            alpha=0.85,
            linewidth=1.2,
        )
        axes[1].plot(
            ddp_time_s,
            ddp_loss,
            label="ddp loss",
            alpha=0.85,
            linewidth=1.2,
        )
        axes[1].set_xlabel("Wall time (s, cumulative from timed train steps)")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        nparams = int(bl["total_params"])
        gbatch = int(bl["global_batch_size"])
        fig.suptitle(
            f"MCCL benchmark  |  global_batch={gbatch}  |  ~{nparams / 1e6:.1f}M params  |  "
            f"{bl_label} / DDP throughput = {ratio:.2f}×",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        out_png = Path(f"{args.output}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png.resolve()}")

        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        names = [bl_label.replace(" ", "\n"), "DDP\n(MCCL)"]
        vals = [bl_tput, ddp_tput]
        colors = ["#2ecc71", "#3498db"]
        bars = ax2.bar(names, vals, color=colors, width=0.55)
        ax2.set_ylabel("Throughput (samples / sec)")
        ax2.set_title(
            f"Average throughput  |  global_batch={gbatch}  |  {bl_label} / DDP = {ratio:.2f}×",
        )
        ax2.grid(True, axis="y", alpha=0.3)
        ymax = max(vals) * 1.18 if vals else 1.0
        ax2.set_ylim(0, ymax)
        for bar, v in zip(bars, vals, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="medium",
            )
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
