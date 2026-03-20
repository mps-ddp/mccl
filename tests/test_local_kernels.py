"""Phase B: Local Metal kernel correctness tests.

Exercises MCCL's own Metal compute shaders via the thin _C bindings added in
Registration.cpp (_metal_accumulate_chunk, _metal_elementwise_min/max/product,
_metal_scale_inplace, _metal_accumulate_and_scale).

These tests run on a single Apple Silicon host — no distributed setup needed.
Each test calls the actual MCCL kernel, not PyTorch's own MPS ops.
"""

import platform
import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="Metal kernel tests require macOS on Apple Silicon",
)


@pytest.fixture(autouse=True, scope="module")
def load_mccl():
    """Import MCCL and verify the kernel helpers are exposed."""
    import mccl  # noqa: F401 — triggers backend registration
    from mccl._C import (
        _metal_accumulate_chunk,
        _metal_elementwise_min,
        _metal_elementwise_max,
        _metal_elementwise_product,
        _metal_scale_inplace,
        _metal_accumulate_and_scale,
    )
    return {
        "accumulate": _metal_accumulate_chunk,
        "min": _metal_elementwise_min,
        "max": _metal_elementwise_max,
        "product": _metal_elementwise_product,
        "scale": _metal_scale_inplace,
        "acc_scale": _metal_accumulate_and_scale,
    }


# ── Helper ───────────────────────────────────────────────────────────

def _kernels():
    from mccl._C import (
        _metal_accumulate_chunk,
        _metal_elementwise_min,
        _metal_elementwise_max,
        _metal_elementwise_product,
        _metal_scale_inplace,
        _metal_accumulate_and_scale,
    )
    return {
        "accumulate": _metal_accumulate_chunk,
        "min": _metal_elementwise_min,
        "max": _metal_elementwise_max,
        "product": _metal_elementwise_product,
        "scale": _metal_scale_inplace,
        "acc_scale": _metal_accumulate_and_scale,
    }


# ── metal_accumulate_chunk (dst += src) ──────────────────────────────

class TestAccumulateChunk:
    def test_f32_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        b = torch.tensor([4.0, 5.0, 6.0], device="mps")
        result = k["accumulate"](a.clone(), b)
        expected = torch.tensor([5.0, 7.0, 9.0], device="mps")
        assert torch.allclose(result, expected), f"got {result}"

    def test_f32_zeros(self):
        k = _kernels()
        a = torch.zeros(256, device="mps")
        b = torch.ones(256, device="mps") * 3.0
        result = k["accumulate"](a, b)
        assert torch.allclose(result, b)

    def test_f32_non_aligned_size(self):
        k = _kernels()
        # Sizes that are not multiples of the Metal vectorisation width (4)
        for size in (1, 3, 7, 13, 97, 997):
            a = torch.ones(size, device="mps") * 2.0
            b = torch.ones(size, device="mps") * 3.0
            result = k["accumulate"](a, b)
            expected = torch.ones(size, device="mps") * 5.0
            assert torch.allclose(result, expected), f"size={size}: got {result}"

    def test_f32_large(self):
        k = _kernels()
        n = 1_000_000
        a = torch.randn(n, device="mps")
        b = torch.randn(n, device="mps")
        a_cpu = a.cpu().clone()
        b_cpu = b.cpu().clone()
        result = k["accumulate"](a, b)
        expected = (a_cpu + b_cpu).to("mps")
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_f16_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
        b = torch.tensor([4.0, 5.0, 6.0], device="mps", dtype=torch.float16)
        result = k["accumulate"](a.clone(), b)
        expected = torch.tensor([5.0, 7.0, 9.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)

    def test_f16_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13):
            a = torch.ones(size, device="mps", dtype=torch.float16) * 2.0
            b = torch.ones(size, device="mps", dtype=torch.float16) * 3.0
            result = k["accumulate"](a, b)
            expected = torch.ones(size, device="mps", dtype=torch.float16) * 5.0
            assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), \
                f"f16 size={size}: got {result}"


# ── metal_elementwise_min ────────────────────────────────────────────

class TestElementwiseMin:
    def test_f32_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 5.0, 3.0], device="mps")
        b = torch.tensor([4.0, 2.0, 6.0], device="mps")
        result = k["min"](a.clone(), b)
        expected = torch.tensor([1.0, 2.0, 3.0], device="mps")
        assert torch.allclose(result, expected)

    def test_f32_all_less(self):
        k = _kernels()
        a = torch.ones(100, device="mps") * -1.0
        b = torch.ones(100, device="mps") * 1.0
        result = k["min"](a.clone(), b)
        assert torch.allclose(result, a)

    def test_f32_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13, 97):
            a = torch.randn(size, device="mps")
            b = torch.randn(size, device="mps")
            result = k["min"](a.clone(), b.clone())
            expected = torch.minimum(a, b)
            assert torch.allclose(result, expected), f"min size={size}"

    def test_f16_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 5.0, 3.0], device="mps", dtype=torch.float16)
        b = torch.tensor([4.0, 2.0, 6.0], device="mps", dtype=torch.float16)
        result = k["min"](a.clone(), b)
        expected = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


# ── metal_elementwise_max ────────────────────────────────────────────

class TestElementwiseMax:
    def test_f32_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 5.0, 3.0], device="mps")
        b = torch.tensor([4.0, 2.0, 6.0], device="mps")
        result = k["max"](a.clone(), b)
        expected = torch.tensor([4.0, 5.0, 6.0], device="mps")
        assert torch.allclose(result, expected)

    def test_f32_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13, 97):
            a = torch.randn(size, device="mps")
            b = torch.randn(size, device="mps")
            result = k["max"](a.clone(), b.clone())
            expected = torch.maximum(a, b)
            assert torch.allclose(result, expected), f"max size={size}"

    def test_f16_basic(self):
        k = _kernels()
        a = torch.tensor([1.0, 5.0, 3.0], device="mps", dtype=torch.float16)
        b = torch.tensor([4.0, 2.0, 6.0], device="mps", dtype=torch.float16)
        result = k["max"](a.clone(), b)
        expected = torch.tensor([4.0, 5.0, 6.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


# ── metal_elementwise_product ────────────────────────────────────────

class TestElementwiseProduct:
    def test_f32_basic(self):
        k = _kernels()
        a = torch.tensor([2.0, 3.0, 4.0], device="mps")
        b = torch.tensor([5.0, 6.0, 7.0], device="mps")
        result = k["product"](a.clone(), b)
        expected = torch.tensor([10.0, 18.0, 28.0], device="mps")
        assert torch.allclose(result, expected)

    def test_f32_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13):
            a = torch.randn(size, device="mps").abs() + 0.1
            b = torch.randn(size, device="mps").abs() + 0.1
            result = k["product"](a.clone(), b.clone())
            expected = a * b
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), \
                f"product size={size}"

    def test_f16_basic(self):
        k = _kernels()
        a = torch.tensor([2.0, 3.0, 4.0], device="mps", dtype=torch.float16)
        b = torch.tensor([5.0, 6.0, 7.0], device="mps", dtype=torch.float16)
        result = k["product"](a.clone(), b)
        expected = torch.tensor([10.0, 18.0, 28.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


# ── metal_scale_inplace ──────────────────────────────────────────────

class TestScaleInplace:
    def test_f32_half(self):
        k = _kernels()
        a = torch.tensor([2.0, 4.0, 6.0], device="mps")
        result = k["scale"](a.clone(), 0.5)
        expected = torch.tensor([1.0, 2.0, 3.0], device="mps")
        assert torch.allclose(result, expected)

    def test_f32_identity(self):
        k = _kernels()
        a = torch.randn(1000, device="mps")
        result = k["scale"](a.clone(), 1.0)
        assert torch.allclose(result, a, rtol=1e-6, atol=1e-6)

    def test_f32_world_size_avg(self):
        """Verify the AVG allreduce scaling factor (1/N) is numerically correct."""
        k = _kernels()
        for world_size in (2, 3, 4, 8):
            a = torch.ones(256, device="mps") * float(world_size)
            result = k["scale"](a, 1.0 / world_size)
            expected = torch.ones(256, device="mps")
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), \
                f"scale 1/{world_size} failed"

    def test_f32_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13, 97):
            a = torch.randn(size, device="mps")
            scale = 0.25
            result = k["scale"](a.clone(), scale)
            expected = a * scale
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), \
                f"scale size={size}"

    def test_f16_half(self):
        k = _kernels()
        a = torch.tensor([2.0, 4.0, 6.0], device="mps", dtype=torch.float16)
        result = k["scale"](a.clone(), 0.5)
        expected = torch.tensor([1.0, 2.0, 3.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


# ── metal_accumulate_and_scale ───────────────────────────────────────

class TestAccumulateAndScale:
    def test_f32_avg_two_ranks(self):
        """(dst + src) * 0.5 is the AVG allreduce for world_size=2."""
        k = _kernels()
        a = torch.tensor([1.0, 3.0, 5.0], device="mps")
        b = torch.tensor([3.0, 5.0, 7.0], device="mps")
        result = k["acc_scale"](a.clone(), b, 0.5)
        expected = torch.tensor([2.0, 4.0, 6.0], device="mps")
        assert torch.allclose(result, expected)

    def test_f32_non_aligned(self):
        k = _kernels()
        for size in (1, 3, 7, 13, 97):
            a = torch.randn(size, device="mps")
            b = torch.randn(size, device="mps")
            scale = 0.5
            result = k["acc_scale"](a.clone(), b, scale)
            expected = (a + b) * scale
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), \
                f"acc_scale size={size}"

    def test_f16_avg(self):
        k = _kernels()
        a = torch.tensor([1.0, 3.0, 5.0], device="mps", dtype=torch.float16)
        b = torch.tensor([3.0, 5.0, 7.0], device="mps", dtype=torch.float16)
        result = k["acc_scale"](a.clone(), b, 0.5)
        expected = torch.tensor([2.0, 4.0, 6.0], device="mps", dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
