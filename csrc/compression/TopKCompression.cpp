#include "compression/TopKCompression.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <numeric>

namespace mccl {

TopKCompressor::TopKCompressor(double k_ratio) : k_ratio_(k_ratio) {
    MCCL_CHECK(k_ratio > 0 && k_ratio <= 1.0,
               "k_ratio must be in (0, 1], got " + std::to_string(k_ratio));
}

struct IndexValue {
    uint32_t index;
    float value;
};

size_t TopKCompressor::compress(const void* src, size_t nbytes,
                                void* dst, size_t dst_capacity,
                                at::ScalarType dtype) {
    MCCL_CHECK(dtype == at::kFloat,
               "TopK compression currently supports float32 only");

    size_t count = nbytes / sizeof(float);
    uint32_t k = std::max(uint32_t(1),
                          static_cast<uint32_t>(count * k_ratio_));

    const float* data = static_cast<const float*>(src);

    // Ensure error feedback buffer is sized.
    // NOTE: error_buf_ is keyed only on element count, not on tensor identity.
    // If two different parameter tensors share the same element count (common in
    // practice), their error feedback states are merged into the same buffer,
    // corrupting the residual signal for both.  A future fix would use a
    // per-tensor-id error buffer map.  For now, TopK compression is documented
    // as experimental.
    if (error_buf_.size() != count) {
        error_buf_.assign(count, 0.0f);
        last_count_ = count;
    }

    // Add error feedback to current gradients
    std::vector<float> adjusted(count);
    for (size_t i = 0; i < count; i++) {
        adjusted[i] = data[i] + error_buf_[i];
    }

    // Find top-k by magnitude using partial sort
    std::vector<uint32_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);

    std::nth_element(
        indices.begin(), indices.begin() + k, indices.end(),
        [&adjusted](uint32_t a, uint32_t b) {
            return std::fabs(adjusted[a]) > std::fabs(adjusted[b]);
        }
    );

    // Sort the top-k by index for cache-friendly access at receiver
    std::sort(indices.begin(), indices.begin() + k);

    // Write compressed output: [k][index, value] pairs
    size_t output_size = sizeof(uint32_t) + k * sizeof(IndexValue);
    MCCL_CHECK(dst_capacity >= output_size, "TopK compress: buffer too small");

    uint8_t* out = static_cast<uint8_t*>(dst);
    memcpy(out, &k, sizeof(uint32_t));
    out += sizeof(uint32_t);

    // Update error feedback: residual = adjusted - what_we_sent
    // Reset sent elements, keep unsent as error
    std::vector<bool> sent(count, false);

    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = indices[i];
        IndexValue iv;
        iv.index = idx;
        iv.value = adjusted[idx];
        memcpy(out, &iv, sizeof(IndexValue));
        out += sizeof(IndexValue);
        sent[idx] = true;
    }

    // Error feedback: unsent values become the residual for next iteration
    for (size_t i = 0; i < count; i++) {
        error_buf_[i] = sent[i] ? 0.0f : adjusted[i];
    }

    double sparsity = 100.0 * (1.0 - (double)k / count);
    MCCL_TRACE("TopK compress: %zu elements → %u values (%.1f%% sparse), "
               "output %zu bytes", count, k, sparsity, output_size);

    return output_size;
}

void TopKCompressor::decompress(const void* src, size_t compressed_size,
                                void* dst, size_t nbytes,
                                at::ScalarType dtype) {
    MCCL_CHECK(dtype == at::kFloat,
               "TopK decompression currently supports float32 only");

    size_t count = nbytes / sizeof(float);

    // Zero the output first
    memset(dst, 0, nbytes);

    const uint8_t* in = static_cast<const uint8_t*>(src);
    uint32_t k;
    memcpy(&k, in, sizeof(uint32_t));
    in += sizeof(uint32_t);

    MCCL_CHECK(compressed_size == sizeof(uint32_t) + k * sizeof(IndexValue),
               "TopK decompress: size mismatch");

    float* out_data = static_cast<float*>(dst);

    for (uint32_t i = 0; i < k; i++) {
        IndexValue iv;
        memcpy(&iv, in, sizeof(IndexValue));
        in += sizeof(IndexValue);

        MCCL_CHECK(iv.index < count,
                   "TopK decompress: index out of bounds");
        out_data[iv.index] = iv.value;
    }
}

size_t TopKCompressor::max_compressed_size(size_t nbytes) const {
    size_t count = nbytes / sizeof(float);
    uint32_t k = static_cast<uint32_t>(count * k_ratio_) + 1;
    return sizeof(uint32_t) + k * sizeof(IndexValue);
}

void TopKCompressor::reset_error_feedback() {
    std::fill(error_buf_.begin(), error_buf_.end(), 0.0f);
    MCCL_DEBUG("TopK: error feedback buffer reset");
}

} // namespace mccl
