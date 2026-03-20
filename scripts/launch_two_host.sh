#!/usr/bin/env bash
set -euo pipefail

#
# Launch a two-host DDP test with MCCL.
#
# Usage (on each host):
#   Host 0 (master):
#     RANK=0 WORLD_SIZE=2 MASTER_ADDR=<host0-ip> ./scripts/launch_two_host.sh
#
#   Host 1:
#     RANK=1 WORLD_SIZE=2 MASTER_ADDR=<host0-ip> ./scripts/launch_two_host.sh
#
# The MASTER_ADDR must be reachable from both hosts.
# For Thunderbolt bridge, set MCCL_LISTEN_ADDR to the bridge IP.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
export RANK="${RANK:?Set RANK=0 or RANK=1}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the IP of rank 0}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Source environment
source "$SCRIPT_DIR/env.sh"

echo ""
echo "========================================"
echo " MCCL Two-Host DDP Launch"
echo " Rank: $RANK / $WORLD_SIZE"
echo " Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================================"
echo ""

cd "$REPO_DIR"

python -u tests/test_two_host_ddp.py
