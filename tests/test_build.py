"""Phase A: Verify extension builds, imports, and backend registers."""

from pathlib import Path

import platform
import subprocess
import pytest


def _xcrun_metal_available() -> bool:
    try:
        subprocess.run(
            ["xcrun", "--find", "metal"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def test_python_package_imports():
    import mccl
    assert hasattr(mccl, "__version__")
    assert mccl.__version__ == "0.3.3"


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
    assert __version__ == "0.3.4"
    assert __protocol_version__ == 3


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Metallib is built next to the extension on macOS Apple Silicon",
)
def test_metallib_shipped_next_to_extension():
    """If ``xcrun metal`` was available at build time, metallib must exist beside _C."""
    import mccl._C as mccl_c

    ext_dir = Path(mccl_c.__file__).resolve().parent
    mlib = ext_dir / "mccl_shaders.metallib"
    if mlib.is_file():
        return
    if _xcrun_metal_available():
        pytest.fail(
            f"Expected {mlib} when xcrun metal is available (build should not skip metallib)"
        )
    pytest.skip(
        "No mccl_shaders.metallib (metallib skipped without Xcode metal CLI; JIT uses shaders.metal)"
    )


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="shaders.metal is installed next to the extension on macOS Apple Silicon",
)
def test_shaders_metal_shipped_next_to_extension():
    """JIT fallback: setup.py copies shaders.metal beside _C.so for pip installs."""
    import mccl._C as mccl_c

    ext_dir = Path(mccl_c.__file__).resolve().parent
    metal = ext_dir / "shaders.metal"
    assert metal.is_file(), f"Expected {metal} (setup.py _install_shaders_metal_next_to_extension)"


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Backend registration requires macOS on Apple Silicon",
)
def test_backend_registered():
    import torch.distributed as dist
    import mccl  # noqa: F401 — triggers registration

    # After import, "mccl" should be recognized
    assert hasattr(dist.Backend, "MCCL") or "mccl" in str(dist.Backend("mccl"))
