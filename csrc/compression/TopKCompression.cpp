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

namespace {

// Persistent per-thread scratch: avoids two large allocations (plus a bit
// vector) per compress call.
struct TopKScratch {
    std::vector<float> adjusted;
    std::vector<uint32_t> indices;
};

TopKScratch& scratch() {
    thread_local TopKScratch s;
    return s;
}

} // anonymous namespace

size_t TopKCompressor::compress(const void* src, size_t nbytes,
                                void* dst, size_t dst_capacity,
                                at::ScalarType dtype,
                                uint64_t stable_id) {
    MCCL_CHECK(dtype == at::kFloat,
               "TopK compression currently supports float32 only");

    size_t count = nbytes / sizeof(float);
    uint32_t k = std::max(uint32_t(1),
                          static_cast<uint32_t>(count * k_ratio_));

    const float* data = static_cast<const float*>(src);

    std::lock_guard<std::mutex> lock(mu_);

    // Per-tensor error feedback keyed on the caller-supplied stable identity.
    // (The old code keyed on `src`, which is the shared staging buffer for
    // every tensor on the blit path — residuals from one gradient bucket
    // were silently added to other buckets.)
    if (error_buffers_.size() >= kMaxErrorBuffers &&
        error_buffers_.find(stable_id) == error_buffers_.end()) {
        MCCL_WARN("TopK: evicting all %zu error feedback buffers "
                  "(unstable tensor identities?)", error_buffers_.size());
        error_buffers_.clear();
    }
    auto it = error_buffers_.find(stable_id);
    if (it == error_buffers_.end() || it->second.size() != count) {
        error_buffers_[stable_id].assign(count, 0.0f);
    }
    std::vector<float>& error_buf = error_buffers_[stable_id];

    // Add error feedback to current gradients
    TopKScratch& sc = scratch();
    sc.adjusted.resize(count);
    float* adjusted = sc.adjusted.data();
    for (size_t i = 0; i < count; i++) {
        adjusted[i] = data[i] + error_buf[i];
    }

    // Find top-k by magnitude using partial selection
    sc.indices.resize(count);
    uint32_t* indices = sc.indices.data();
    std::iota(indices, indices + count, 0);

    std::nth_element(
        indices, indices + k, indices + count,
        [adjusted](uint32_t a, uint32_t b) {
            return std::fabs(adjusted[a]) > std::fabs(adjusted[b]);
        }
    );

    // Sort the top-k by index for cache-friendly access at receiver
    std::sort(indices, indices + k);

    // Write compressed output: [k][index, value] pairs
    size_t output_size = sizeof(uint32_t) + k * sizeof(IndexValue);
    MCCL_CHECK(dst_capacity >= output_size, "TopK compress: buffer too small");

    uint8_t* out = static_cast<uint8_t*>(dst);
    memcpy(out, &k, sizeof(uint32_t));
    out += sizeof(uint32_t);

    // Error feedback: residual = adjusted - sent.  Copy the adjusted values
    // wholesale, then zero exactly the k entries that went on the wire —
    // one memcpy + k stores instead of a bit vector and a full second pass.
    memcpy(error_buf.data(), adjusted, count * sizeof(float));

    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = indices[i];
        IndexValue iv;
        iv.index = idx;
        iv.value = adjusted[idx];
        memcpy(out, &iv, sizeof(IndexValue));
        out += sizeof(IndexValue);
        error_buf[idx] = 0.0f;
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
    std::lock_guard<std::mutex> lock(mu_);
    error_buffers_.clear();
    MCCL_DEBUG("TopK: all error feedback buffers reset");
}

void TopKCompressor::reset_error_feedback_for_tensor(uint64_t stable_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = error_buffers_.find(stable_id);
    if (it != error_buffers_.end()) {
        std::fill(it->second.begin(), it->second.end(), 0.0f);
        MCCL_DEBUG("TopK: error feedback reset for tensor id %llu",
                   (unsigned long long)stable_id);
    }
}

size_t TopKCompressor::error_feedback_buffer_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return error_buffers_.size();
}

} // namespace mccl
