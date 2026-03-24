#!/usr/bin/env bash
#
# Source this on each Apple Silicon host before running MCCL.
# Adjust MCCL_LISTEN_ADDR and MCCL_IFNAME for your network topology.
#

# PyTorch distributed store
export MASTER_ADDR="${MASTER_ADDR:-192.168.1.100}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# MCCL transport config
export MCCL_LISTEN_ADDR="${MCCL_LISTEN_ADDR:-0.0.0.0}"
export MCCL_PORT_BASE="${MCCL_PORT_BASE:-29600}"
export MCCL_IFNAME="${MCCL_IFNAME:-}"          # e.g. "bridge0" for Thunderbolt
export MCCL_CHUNK_BYTES="${MCCL_CHUNK_BYTES:-4194304}"   # 4 MB
export MCCL_SMALL_MSG_THRESHOLD="${MCCL_SMALL_MSG_THRESHOLD:-262144}"  # 256 KiB default
export MCCL_LOG_LEVEL="${MCCL_LOG_LEVEL:-INFO}"

# Metal shader path (if not installed in the standard location)
export MCCL_SHADER_PATH="${MCCL_SHADER_PATH:-$(dirname "$0")/../csrc/metal/shaders.metal}"

echo "[mccl env] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[mccl env] MCCL_LISTEN_ADDR=$MCCL_LISTEN_ADDR MCCL_PORT_BASE=$MCCL_PORT_BASE"
echo "[mccl env] MCCL_IFNAME=$MCCL_IFNAME MCCL_LOG_LEVEL=$MCCL_LOG_LEVEL"
