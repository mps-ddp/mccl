#!/usr/bin/env bash
# Max-throughput MCCL env for wired 10 Gbps (or faster) links between Macs with
# plenty of RAM. Source on every node before torchrun / Ray, after MASTER_ADDR is set:
#   source scripts/10gbe_maxperf.sh
#
# One-time on EACH Mac (persist in /etc/sysctl.conf for reboots):
#   sudo sysctl -w kern.ipc.maxsockbuf=134217728
#
# Set your 10GbE interface (ifname from `networksetup -listallhardwareports`):
#   export MCCL_IFNAME=en0
#
# Verify: scripts/check_sockbuf.sh

export MCCL_LINK_PROFILE="${MCCL_LINK_PROFILE:-thunderbolt}"
export MCCL_SOCK_BUFSIZE="${MCCL_SOCK_BUFSIZE:-33554432}"   # 32 MiB snd+rcv
export MCCL_CHUNK_BYTES="${MCCL_CHUNK_BYTES:-33554432}"     # 32 MiB transport frames
export MCCL_OVERLAP_COMM="${MCCL_OVERLAP_COMM:-1}"
export MCCL_RING_PIPELINE="${MCCL_RING_PIPELINE:-1}"
export MCCL_COLLECTIVE_CONCURRENCY="${MCCL_COLLECTIVE_CONCURRENCY:-2}"
export MCCL_TCP_LOWAT="${MCCL_TCP_LOWAT:-131072}"
export DDP_BUCKET_MB="${DDP_BUCKET_MB:-1024}"
export MCCL_LOG_LEVEL="${MCCL_LOG_LEVEL:-WARN}"

echo "[mccl] 10GbE max-perf profile active"
echo "[mccl]   MCCL_SOCK_BUFSIZE=$MCCL_SOCK_BUFSIZE MCCL_CHUNK_BYTES=$MCCL_CHUNK_BYTES"
echo "[mccl]   DDP_BUCKET_MB=$DDP_BUCKET_MB MCCL_IFNAME=${MCCL_IFNAME:-<unset>}"
