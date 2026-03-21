#!/usr/bin/env bash
# Run single-GPU baseline then local 2-rank DDP with the same TRAIN_STEPS, BATCH_SIZE,
# and model env (INPUT_DIM, NUM_CLASSES, MODEL_HIDDEN, MODEL_DEPTH) for side-by-side timing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1

# Shared knobs — override before calling this script if you want longer runs
export TRAIN_STEPS="${TRAIN_STEPS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
# Fair baseline vs local 2-rank DDP: same **global** batch (per_rank × 2).
# Override explicitly for multi-node (e.g. BASELINE_BATCH_SIZE=$((BATCH_SIZE*2))).
export BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-$((BATCH_SIZE * 2))}"

# DDP: many allreduce buckets per backward — each needs a GPU barrier. FULL is required.
# Explicit default so a stale shell MCCL_SYNC_MODE=coalesced cannot break multi-bucket runs.
export MCCL_SYNC_MODE="${MCCL_SYNC_MODE:-full}"

# Model dims: ddp_dummy_train.py defaults to a small MLP (no env needed).
# For ~1B-param stress: MCCL_STRESS_MODEL=1 bash examples/compare_single_vs_ddp.sh
# Or set INPUT_DIM, MODEL_HIDDEN, MODEL_DEPTH, NUM_CLASSES explicitly.

echo "=========================================="
echo "  Single-GPU baseline (no allreduce)"
echo "=========================================="
python "$ROOT/examples/ddp_dummy_train.py" --baseline

echo ""
echo "=========================================="
echo "  DDP local (2 ranks, MCCL, same env)"
echo "=========================================="
torchrun --nproc_per_node=2 --nnodes=1 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
  "$ROOT/examples/ddp_dummy_train.py"

echo ""
echo "Done. Compare 'Avg step time' / throughput and MCCL phase metrics above."
echo "Multi-node checklist / bucket sweeps: docs/MULTINODE.md"
