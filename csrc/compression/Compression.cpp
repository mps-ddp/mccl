#include "compression/Compression.hpp"
#include "compression/FP16Compression.hpp"
#include "compression/TopKCompression.hpp"
#include "common/Errors.hpp"

namespace mccl {

std::unique_ptr<Compressor> make_compressor(CompressionMode mode, double param) {
    switch (mode) {
        case CompressionMode::NONE:
            return nullptr;
        case CompressionMode::FP16:
            return std::make_unique<FP16Compressor>();
        case CompressionMode::TOPK:
            return std::make_unique<TopKCompressor>(param > 0 ? param : 0.01);
        default:
            throw MCCLError("Unknown compression mode: " +
                            std::to_string(static_cast<int>(mode)));
    }
}

} // namespace mccl
