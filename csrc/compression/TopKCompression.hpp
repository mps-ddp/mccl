#pragma once

#include "compression/Compression.hpp"
#include <vector>

namespace mccl {

/// Top-K sparsification with error feedback for gradient compression.
///
/// Only the largest-magnitude K elements are transmitted each step.
/// Residual (un-sent) values are accumulated into an error feedback buffer
/// and added to the next step's gradients — this is critical for
/// convergence guarantees.
///
/// Without error feedback, top-k sparsification can cause training divergence.
/// See: "Deep Gradient Compression" (Lin et al., ICLR 2018).
///
/// Wire format: [uint32_t count][count x (uint32_t index, float value)]
class TopKCompressor : public Compressor {
public:
    /// k_ratio: fraction of elements to keep (e.g. 0.01 = top 1%)
    explicit TopKCompressor(double k_ratio);

    size_t compress(const void* src, size_t nbytes,
                    void* dst, size_t dst_capacity,
                    at::ScalarType dtype) override;

    void decompress(const void* src, size_t compressed_size,
                    void* dst, size_t nbytes,
                    at::ScalarType dtype) override;

    size_t max_compressed_size(size_t nbytes) const override;

    std::string name() const override { return "topk"; }

    /// Reset the error feedback buffer (e.g. between training runs).
    void reset_error_feedback();

    /// Get the current error feedback buffer (for diagnostics).
    const std::vector<float>& error_feedback() const { return error_buf_; }

private:
    double k_ratio_;
    std::vector<float> error_buf_;
    size_t last_count_ = 0;
};

} // namespace mccl
