#!/usr/bin/env bash
# Quick perf comparison: baseline vs DDP (local 2-rank).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Single GPU baseline ==="
BASELINE_BATCH_SIZE=8 python examples/ddp_dummy_train.py --baseline

echo ""
echo "=== 2-rank DDP (one machine) ==="
DDP_BUCKET_MB=100 torchrun --nproc_per_node=2 --nnodes=1 \
  --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py

echo ""
echo "Done. Compare step times above."
