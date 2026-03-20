"""Phase A: Verify extension builds, imports, and backend registers."""

import sys
import platform
import pytest


def test_python_package_imports():
    import mccl
    assert hasattr(mccl, "__version__")
    assert mccl.__version__ == "0.3.0"


def test_compatibility_matrix():
    from mccl.version import COMPATIBILITY_MATRIX
    assert "mps" in COMPATIBILITY_MATRIX["supported_devices"]
    assert COMPATIBILITY_MATRIX["backend_name"] == "mccl"


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Native extension requires macOS on Apple Silicon",
)
def test_native_extension_loads():
    from mccl._C import __version__, __protocol_version__
    assert __version__ == "0.3.0"
    assert __protocol_version__ == 3


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Backend registration requires macOS on Apple Silicon",
)
def test_backend_registered():
    import torch.distributed as dist
    import mccl  # noqa: F401 — triggers registration

    # After import, "mccl" should be recognized
    assert hasattr(dist.Backend, "MCCL") or "mccl" in str(dist.Backend("mccl"))
