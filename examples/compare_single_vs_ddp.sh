#!/usr/bin/env bash
# Run single-GPU baseline then local 2-rank DDP with the same TRAIN_STEPS, BATCH_SIZE,
# and model env (INPUT_DIM, NUM_CLASSES, MODEL_HIDDEN, MODEL_DEPTH) for side-by-side timing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1

# Shared knobs — override before calling this script if you want longer runs
export TRAIN_STEPS="${TRAIN_STEPS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-16}"

# ~126M-param default (see ddp_dummy_train.py); quick smoke test, e.g.:
#   MODEL_DEPTH=2 MODEL_HIDDEN=512 BATCH_SIZE=32 bash examples/compare_single_vs_ddp.sh
export INPUT_DIM="${INPUT_DIM:-2048}"
export NUM_CLASSES="${NUM_CLASSES:-128}"
export MODEL_HIDDEN="${MODEL_HIDDEN:-4096}"
export MODEL_DEPTH="${MODEL_DEPTH:-8}"

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
echo "Done. Compare 'Avg step time' / throughput above."
