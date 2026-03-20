#pragma once

#include <c10d/Types.hpp>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cstddef>
#include <cstdint>

namespace mccl {

/// CPU-side element-wise reduction using Apple's Accelerate framework (vDSP).
/// Operates directly on raw float* pointers — designed for f32 MPS shared-memory
/// tensors where the CPU pointer is accessible without staging.
/// AMX-accelerated on Apple Silicon.

/// dst[i] += src[i]  (in-place on dst)
void cpu_accumulate(float* dst, const float* src, int64_t count);

/// buf[i] *= scale  (in-place)
void cpu_scale_inplace(float* buf, int64_t count, float scale);

/// dst[i] = (dst[i] + src[i]) * scale  (fused add+scale, in-place on dst)
void cpu_accumulate_and_scale(float* dst, const float* src,
                              int64_t count, float scale);

/// dst[i] = min(dst[i], src[i])  (in-place on dst)
void cpu_elementwise_min(float* dst, const float* src, int64_t count);

/// dst[i] = max(dst[i], src[i])  (in-place on dst)
void cpu_elementwise_max(float* dst, const float* src, int64_t count);

/// dst[i] *= src[i]  (in-place on dst)
void cpu_elementwise_product(float* dst, const float* src, int64_t count);

/// Small-tensor f16 CPU fallback: widen to f32, reduce with vDSP, narrow back.
void cpu_reduce_op_half(c10::Half* dst, const c10::Half* src, int64_t count,
                        c10d::ReduceOp::RedOpType op);
void cpu_scale_inplace_half(c10::Half* buf, int64_t count, float scale);
void cpu_accumulate_and_scale_half(c10::Half* dst, const c10::Half* src,
                                   int64_t count, float scale);

/// Small-tensor bf16 CPU fallback: widen to f32, reduce with vDSP, narrow back.
void cpu_reduce_op_bf16(c10::BFloat16* dst, const c10::BFloat16* src, int64_t count,
                        c10d::ReduceOp::RedOpType op);
void cpu_scale_inplace_bf16(c10::BFloat16* buf, int64_t count, float scale);
void cpu_accumulate_and_scale_bf16(c10::BFloat16* dst, const c10::BFloat16* src,
                                   int64_t count, float scale);

/// Dispatch the correct CPU reduce operation based on ReduceOp.
/// dst and src are raw float* pointers, count is number of float elements.
void cpu_reduce_op(float* dst, const float* src, int64_t count,
                   c10d::ReduceOp::RedOpType op);

} // namespace mccl
