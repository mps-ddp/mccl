#pragma once

#include "compression/Compression.hpp"

namespace mccl {

/// FP16 transport compression.
///
/// Casts float32 gradient data to float16 for transport, halving bandwidth.
/// Decompression widens back to float32 at the receiver.
///
/// Only operates on float32 input — float16 input passes through unchanged.
/// This is a lossy compression; acceptable for gradient communication
/// where training can absorb the quantization noise.
class FP16Compressor : public Compressor {
public:
    FP16Compressor() = default;

    size_t compress(const void* src, size_t nbytes,
                    void* dst, size_t dst_capacity,
                    at::ScalarType dtype) override;

    void decompress(const void* src, size_t compressed_size,
                    void* dst, size_t nbytes,
                    at::ScalarType dtype) override;

    size_t max_compressed_size(size_t nbytes) const override;

    std::string name() const override { return "fp16"; }
};

} // namespace mccl
