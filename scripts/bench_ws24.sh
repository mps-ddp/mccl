#!/usr/bin/env bash
# Manual 24-rank bus bandwidth sweep on a real Ethernet/TB cluster.
# Usage (on rank-0 node):
#   torchrun --nnodes=24 --nproc_per_node=1 --node_rank=$RANK \
#     --master_addr=$MASTER_ADDR --master_port=29500 \
#     tests/bench_busbw.py --world-size 24 --link-gbps 10 --concurrency 2
#
# Or run this wrapper after exporting MASTER_ADDR and launching from each node:
#   MCCL_LINK_PROFILE=ethernet bash scripts/bench_ws24.sh

set -euo pipefail

LINK_GBPS="${LINK_GBPS:-10}"
CONCURRENCY="${MCCL_COLLECTIVE_CONCURRENCY:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

export MCCL_LINK_PROFILE="${MCCL_LINK_PROFILE:-ethernet}"
export MCCL_COLLECTIVE_CONCURRENCY="${CONCURRENCY}"
export MCCL_PIPELINE_DEPTH="${MCCL_PIPELINE_DEPTH:-3}"

echo "MCCL 24-rank busbw | LINK_GBPS=${LINK_GBPS} concurrency=${CONCURRENCY}"
python tests/bench_busbw.py \
  --world-size 24 \
  --link-gbps "${LINK_GBPS}" \
  --concurrency "${CONCURRENCY}" \
  --port "${MASTER_PORT}"
