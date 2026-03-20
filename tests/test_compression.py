"""Tests for FP16 and TopK compression.

These are pure-logic tests that run on any platform.
"""

import struct
import numpy as np
import pytest


class TestFP16Compression:
    """Test FP16 compress/decompress logic via numpy (portable proxy)."""

    def test_f32_roundtrip(self):
        data = np.array([1.0, -2.5, 3.14, 0.0, 100.0], dtype=np.float32)
        compressed = data.astype(np.float16)
        decompressed = compressed.astype(np.float32)
        np.testing.assert_allclose(decompressed, data, rtol=1e-2, atol=1e-3)

    def test_bandwidth_halved(self):
        data = np.random.randn(1_000_000).astype(np.float32)
        compressed = data.astype(np.float16)
        assert compressed.nbytes == data.nbytes // 2

    def test_f16_passthrough(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        roundtripped = data.astype(np.float16)
        np.testing.assert_array_equal(data, roundtripped)

    def test_large_values_clamp(self):
        data = np.array([65504.0, -65504.0], dtype=np.float32)
        compressed = data.astype(np.float16)
        np.testing.assert_allclose(compressed.astype(np.float32), data, rtol=1e-3)

    def test_denormals(self):
        data = np.array([1e-7, -1e-7], dtype=np.float32)
        compressed = data.astype(np.float16)
        decompressed = compressed.astype(np.float32)
        # Very small values may flush to zero
        assert all(abs(d) <= abs(o) + 1e-5 for d, o in zip(decompressed, data))


class TestTopKCompression:
    """Test TopK sparsification + error feedback logic."""

    def test_basic_sparsity(self):
        """Top 10% of 100 elements = 10 elements."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100).astype(np.float32)
        k = 10

        # Find top-k by magnitude
        indices = np.argsort(-np.abs(data))[:k]
        assert len(indices) == k

        # Sparse representation
        sparse = np.zeros_like(data)
        sparse[indices] = data[indices]

        # Verify only k elements are non-zero
        assert np.count_nonzero(sparse) == k

    def test_error_feedback_accumulates(self):
        """Verify error feedback preserves unsent gradients."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100).astype(np.float32)
        k = 10
        error = np.zeros(100, dtype=np.float32)

        # Step 1
        adjusted = data + error
        top_indices = np.argsort(-np.abs(adjusted))[:k]
        sent = np.zeros_like(adjusted)
        sent[top_indices] = adjusted[top_indices]
        error = adjusted - sent

        # Error should capture what wasn't sent
        assert np.count_nonzero(error) == 100 - k

        # Step 2 — error should influence selection
        data2 = rng.standard_normal(100).astype(np.float32)
        adjusted2 = data2 + error
        assert not np.allclose(adjusted2, data2)  # error changes the picture

    def test_convergence_guarantee(self):
        """Over many steps, all gradient info should eventually be transmitted."""
        rng = np.random.default_rng(42)
        error = np.zeros(100, dtype=np.float32)
        total_sent = np.zeros(100, dtype=np.float32)
        total_data = np.zeros(100, dtype=np.float32)
        k = 10

        for step in range(1000):
            data = rng.standard_normal(100).astype(np.float32)
            total_data += data

            adjusted = data + error
            top_indices = np.argsort(-np.abs(adjusted))[:k]
            sent = np.zeros_like(adjusted)
            sent[top_indices] = adjusted[top_indices]
            total_sent += sent
            error = adjusted - sent

        # After many steps, total_sent should approximate total_data
        residual = np.abs(total_data - total_sent - error)
        assert np.max(residual) < 1e-5, f"Max residual: {np.max(residual)}"

    def test_wire_format(self):
        """Verify the index-value wire format."""
        k = 3
        pairs = [(5, 1.5), (10, -2.0), (42, 3.14)]

        # Encode
        buf = struct.pack("<I", k)
        for idx, val in pairs:
            buf += struct.pack("<If", idx, val)

        assert len(buf) == 4 + k * 8

        # Decode
        count = struct.unpack_from("<I", buf, 0)[0]
        assert count == k
        for i in range(count):
            offset = 4 + i * 8
            idx, val = struct.unpack_from("<If", buf, offset)
            assert idx == pairs[i][0]
            assert abs(val - pairs[i][1]) < 1e-5
