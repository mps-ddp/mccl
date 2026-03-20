#include "compression/FP16Compression.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <cstring>

// Minimal f32↔f16 conversion using hardware intrinsics on ARM
#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace mccl {

namespace {

void f32_to_f16(const float* src, uint16_t* dst, size_t count) {
#ifdef __aarch64__
    size_t i = 0;
    // 8-wide NEON: two 4-lane converts → one 128-bit store
    for (; i + 8 <= count; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        float16x4_t h_lo = vcvt_f16_f32(lo);
        float16x4_t h_hi = vcvt_f16_f32(hi);
        float16x8_t combined = vcombine_f16(h_lo, h_hi);
        vst1q_f16(reinterpret_cast<__fp16*>(dst + i), combined);
    }
    // 4-wide tail
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h);
    }
    for (; i < count; i++) {
        __fp16 h = static_cast<__fp16>(src[i]);
        memcpy(&dst[i], &h, sizeof(uint16_t));
    }
#else
    // Portable fallback — not performance-critical, only for non-ARM builds
    for (size_t i = 0; i < count; i++) {
        // IEEE 754 f32→f16 with round-to-nearest
        uint32_t x;
        memcpy(&x, &src[i], 4);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;

        if (exp <= 0) {
            dst[i] = static_cast<uint16_t>(sign); // flush to zero
        } else if (exp >= 31) {
            dst[i] = static_cast<uint16_t>(sign | 0x7C00); // inf
        } else {
            dst[i] = static_cast<uint16_t>(sign | (exp << 10) | mant);
        }
    }
#endif
}

void f16_to_f32(const uint16_t* src, float* dst, size_t count) {
#ifdef __aarch64__
    size_t i = 0;
    // 8-wide NEON: one 128-bit load → two 4-lane conversions → two 128-bit stores
    for (; i + 8 <= count; i += 8) {
        float16x8_t combined = vld1q_f16(reinterpret_cast<const __fp16*>(src + i));
        float16x4_t lo = vget_low_f16(combined);
        float16x4_t hi = vget_high_f16(combined);
        vst1q_f32(dst + i,     vcvt_f32_f16(lo));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(hi));
    }
    for (; i + 4 <= count; i += 4) {
        float16x4_t h = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(h));
    }
    for (; i < count; i++) {
        __fp16 h;
        memcpy(&h, &src[i], sizeof(uint16_t));
        dst[i] = static_cast<float>(h);
    }
#else
    for (size_t i = 0; i < count; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        uint32_t f;
        if (exp == 0) {
            f = sign; // zero or denormal → zero
        } else if (exp == 31) {
            f = sign | 0x7F800000 | (mant << 13); // inf/nan
        } else {
            f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        memcpy(&dst[i], &f, 4);
    }
#endif
}

} // anonymous namespace


size_t FP16Compressor::compress(const void* src, size_t nbytes,
                                void* dst, size_t dst_capacity,
                                at::ScalarType dtype) {
    if (dtype == at::kHalf) {
        MCCL_CHECK(dst_capacity >= nbytes, "FP16 compress: buffer too small");
        memcpy(dst, src, nbytes);
        return nbytes;
    }

    MCCL_CHECK(dtype == at::kFloat, "FP16 compression only supports float32 input");

    size_t count = nbytes / sizeof(float);
    size_t compressed_bytes = count * sizeof(uint16_t);
    MCCL_CHECK(dst_capacity >= compressed_bytes, "FP16 compress: buffer too small");

    f32_to_f16(static_cast<const float*>(src),
               static_cast<uint16_t*>(dst), count);

    MCCL_TRACE("FP16 compress: %zu → %zu bytes (%.0f%% reduction)",
               nbytes, compressed_bytes,
               100.0 * (1.0 - (double)compressed_bytes / nbytes));

    return compressed_bytes;
}

void FP16Compressor::decompress(const void* src, size_t compressed_size,
                                void* dst, size_t nbytes,
                                at::ScalarType dtype) {
    if (dtype == at::kHalf) {
        MCCL_CHECK(compressed_size == nbytes, "FP16 decompress: size mismatch");
        memcpy(dst, src, nbytes);
        return;
    }

    MCCL_CHECK(dtype == at::kFloat, "FP16 decompression only supports float32 output");

    size_t count = nbytes / sizeof(float);
    MCCL_CHECK(compressed_size == count * sizeof(uint16_t),
               "FP16 decompress: compressed size mismatch");

    f16_to_f32(static_cast<const uint16_t*>(src),
               static_cast<float*>(dst), count);
}

size_t FP16Compressor::max_compressed_size(size_t nbytes) const {
    // At worst (already f16), same size; for f32, half.
    return nbytes;
}

} // namespace mccl
