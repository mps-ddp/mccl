#!/usr/bin/env bash
set -euo pipefail

#
# Launch a multi-host 3+ DDP correctness test with MCCL.
#
# Usage on every host:
#   RANK=<rank> WORLD_SIZE=<n> MASTER_ADDR=<rank0-ip> ./scripts/launch_multi_host_ddp.sh
#
# Example for 3 hosts:
#   Host 0: RANK=0 WORLD_SIZE=3 MASTER_ADDR=192.168.1.10 ./scripts/launch_multi_host_ddp.sh
#   Host 1: RANK=1 WORLD_SIZE=3 MASTER_ADDR=192.168.1.10 ./scripts/launch_multi_host_ddp.sh
#   Host 2: RANK=2 WORLD_SIZE=3 MASTER_ADDR=192.168.1.10 ./scripts/launch_multi_host_ddp.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

export RANK="${RANK:?Set RANK=0..WORLD_SIZE-1}"
export WORLD_SIZE="${WORLD_SIZE:?Set WORLD_SIZE>=3}"
export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the IP of rank 0}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "$WORLD_SIZE" -lt 3 ]]; then
  echo "WORLD_SIZE must be >= 3 for launch_multi_host_ddp.sh" >&2
  exit 1
fi

source "$SCRIPT_DIR/env.sh"

echo ""
echo "========================================"
echo " MCCL Multi-Host 3+ DDP Launch"
echo " Rank: $RANK / $WORLD_SIZE"
echo " Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================================"
echo ""

cd "$REPO_DIR"

python -u tests/test_multi_host_ddp.py
