#!/usr/bin/env bash
# Production-oriented env for direct Thunderbolt IP (169.254.x.x) between Macs.
# Source on every node before torchrun, after MASTER_ADDR / MASTER_PORT are set:
#   source scripts/thunderbolt_prod.sh
# See docs/MULTINODE.md — "Production Thunderbolt profile".

export MCCL_LINK_PROFILE="${MCCL_LINK_PROFILE:-thunderbolt}"

# Optional: halve cross-host bytes (validate loss / stability for your model)
# export MCCL_COMPRESSION="${MCCL_COMPRESSION:-fp16}"

echo "[mccl] MCCL_LINK_PROFILE=$MCCL_LINK_PROFILE (Thunderbolt TCP tuning active)"
