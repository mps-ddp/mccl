#!/usr/bin/env bash
# Report macOS socket buffer limits vs MCCL's requested MCCL_SOCK_BUFSIZE.
# Exit 0 if the kernel ceiling is high enough; 1 if buffers will be clamped.

set -euo pipefail

REQUESTED="${MCCL_SOCK_BUFSIZE:-33554432}"
MAXSOCK="$(sysctl -n kern.ipc.maxsockbuf 2>/dev/null || echo 0)"
# MCCL warns when getsockopt snd/rcv < requested; need maxsockbuf >= requested.
NEEDED="$REQUESTED"

echo "kern.ipc.maxsockbuf     = $MAXSOCK"
echo "MCCL_SOCK_BUFSIZE       = $REQUESTED (requested snd+rcv each)"
echo "recommended sysctl min  = $NEEDED (for full size; MCCL suggests 2x for headroom: $((REQUESTED * 2)))"

if [[ "$MAXSOCK" -lt "$NEEDED" ]]; then
    echo ""
    echo "CLAMPED: raise on every node:"
    echo "  sudo sysctl -w kern.ipc.maxsockbuf=$((REQUESTED * 2))"
    exit 1
fi

echo ""
echo "OK: kernel ceiling supports requested MCCL socket buffers."
exit 0
