#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mccl {

/// Compression policy — controls which strategy is used at transport time.
enum class CompressionMode : int {
    NONE    = 0,   // v1 default — no compression
    FP16    = 1,   // v1.1 — cast f32 gradients to f16 for transport
    TOPK    = 2,   // v1.2 — top-k sparsification with error feedback
};

/// Abstract compression interface.
///
/// Compress/decompress operate on flat byte buffers.
/// The compressed buffer is what goes over the wire.
class Compressor {
public:
    virtual ~Compressor() = default;

    /// Compress `src` (nbytes of original data) into `dst`.
    /// Returns the compressed size in bytes.
    /// `dst` must be pre-allocated to at least `max_compressed_size(nbytes)`.
    virtual size_t compress(const void* src, size_t nbytes,
                            void* dst, size_t dst_capacity,
                            at::ScalarType dtype) = 0;

    /// Decompress `src` (compressed_size bytes) into `dst` (nbytes original).
    virtual void decompress(const void* src, size_t compressed_size,
                            void* dst, size_t nbytes,
                            at::ScalarType dtype) = 0;

    /// Upper bound on compressed output for a given input size.
    virtual size_t max_compressed_size(size_t nbytes) const = 0;

    virtual std::string name() const = 0;
};

/// Factory for creating compressors based on mode.
std::unique_ptr<Compressor> make_compressor(CompressionMode mode, double param = 0.0);

} // namespace mccl
