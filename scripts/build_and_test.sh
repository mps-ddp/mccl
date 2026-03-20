#!/usr/bin/env bash
set -euo pipefail

#
# One-command build + full test suite on Apple Silicon.
#
# Usage:
#   ./scripts/build_and_test.sh          # full build + all local tests
#   ./scripts/build_and_test.sh --quick  # build + fast tests only
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

echo "========================================"
echo " MCCL Build & Test"
echo " Machine: $(uname -m) / $(sw_vers -productVersion)"
echo " Python:  $(python3 --version 2>&1)"
echo "========================================"

# ── Step 1: Environment ──────────────────────────────────────────────
if [[ ! -d .venv ]]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/6] Using existing virtual environment"
fi
source .venv/bin/activate

# ── Step 2: Dependencies ─────────────────────────────────────────────
echo "[2/6] Installing dependencies..."
pip install --quiet torch pytest pytest-timeout numpy

# ── Step 3: Build ────────────────────────────────────────────────────
echo "[3/6] Building MCCL extension..."
pip install -e ".[dev]" 2>&1 | tail -5
echo "Build complete."

# ── Step 4: Verify import ────────────────────────────────────────────
echo "[4/6] Verifying import..."
python3 -c "
import mccl
print(f'  MCCL {mccl.__version__} imported')
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
from mccl._C import __version__, __protocol_version__
print(f'  Native extension: v{__version__} protocol={__protocol_version__}')
"

# ── Step 5: Set shader path ──────────────────────────────────────────
export MCCL_SHADER_PATH="$REPO_DIR/csrc/metal/shaders.metal"

# ── Step 6: Run tests ────────────────────────────────────────────────
echo "[5/6] Running tests..."

echo ""
echo "--- Protocol tests (pure logic) ---"
python3 -m pytest tests/test_protocol.py -v --timeout=30

echo ""
echo "--- Compression tests ---"
python3 -m pytest tests/test_compression.py -v --timeout=30

echo ""
echo "--- Build/import tests ---"
python3 -m pytest tests/test_build.py -v --timeout=30

echo ""
echo "--- Metal kernel tests ---"
python3 -m pytest tests/test_local_kernels.py -v --timeout=60

if [[ "$QUICK" == "false" ]]; then
    echo ""
    echo "--- Process group local tests (spawns 2 processes) ---"
    export MCCL_LOG_LEVEL=WARN
    python3 -m pytest tests/test_process_group_local.py -v --timeout=180

    echo ""
    echo "--- v2 collective tests ---"
    python3 -m pytest tests/test_v2_collectives.py -v --timeout=180

    echo ""
    echo "--- Soak tests (long-running) ---"
    python3 -m pytest tests/test_soak.py -v --timeout=900
fi

echo ""
echo "[6/6] All tests passed."
echo "========================================"
echo " MCCL is ready for multi-host testing."
echo " See TESTING.md for two-host DDP setup."
echo "========================================"
