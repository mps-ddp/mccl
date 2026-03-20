#include "metal/AccelerateOps.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <Accelerate/Accelerate.h>
#include <vector>

namespace mccl {

namespace {

template <typename SrcT>
void widen_to_float(const SrcT* src, float* dst, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

template <typename DstT>
void narrow_from_float(const float* src, DstT* dst, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        dst[i] = static_cast<DstT>(src[i]);
    }
}

template <typename T>
void cpu_reduce_via_float(T* dst, const T* src, int64_t count,
                          c10d::ReduceOp::RedOpType op) {
    thread_local std::vector<float> lhs;
    thread_local std::vector<float> rhs;
    lhs.resize(static_cast<size_t>(count));
    rhs.resize(static_cast<size_t>(count));

    widen_to_float(dst, lhs.data(), count);
    widen_to_float(src, rhs.data(), count);
    cpu_reduce_op(lhs.data(), rhs.data(), count, op);
    narrow_from_float(lhs.data(), dst, count);
}

template <typename T>
void cpu_scale_via_float(T* buf, int64_t count, float scale) {
    thread_local std::vector<float> tmp;
    tmp.resize(static_cast<size_t>(count));

    widen_to_float(buf, tmp.data(), count);
    cpu_scale_inplace(tmp.data(), count, scale);
    narrow_from_float(tmp.data(), buf, count);
}

template <typename T>
void cpu_accumulate_and_scale_via_float(T* dst, const T* src,
                                        int64_t count, float scale) {
    thread_local std::vector<float> lhs;
    thread_local std::vector<float> rhs;
    lhs.resize(static_cast<size_t>(count));
    rhs.resize(static_cast<size_t>(count));

    widen_to_float(dst, lhs.data(), count);
    widen_to_float(src, rhs.data(), count);
    cpu_accumulate_and_scale(lhs.data(), rhs.data(), count, scale);
    narrow_from_float(lhs.data(), dst, count);
}

} // anonymous namespace

void cpu_accumulate(float* dst, const float* src, int64_t count) {
    vDSP_vadd(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
}

void cpu_scale_inplace(float* buf, int64_t count, float scale) {
    vDSP_vsmul(buf, 1, &scale, buf, 1, static_cast<vDSP_Length>(count));
}

void cpu_accumulate_and_scale(float* dst, const float* src,
                              int64_t count, float scale) {
    // vDSP_vasm requires non-aliasing pointers for __A, __B, and __D.
    // dst is both __A (first addend) and __D (output), which violates that
    // contract and produces undefined results.  Use two separate vDSP calls
    // instead: vDSP_vadd and vDSP_vsmul both permit in-place (dst == output).
    vDSP_vadd(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
    vDSP_vsmul(dst, 1, &scale, dst, 1, static_cast<vDSP_Length>(count));
}

void cpu_elementwise_min(float* dst, const float* src, int64_t count) {
    vDSP_vmin(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
}

void cpu_elementwise_max(float* dst, const float* src, int64_t count) {
    vDSP_vmax(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
}

void cpu_elementwise_product(float* dst, const float* src, int64_t count) {
    vDSP_vmul(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
}

void cpu_reduce_op_half(c10::Half* dst, const c10::Half* src, int64_t count,
                        c10d::ReduceOp::RedOpType op) {
    cpu_reduce_via_float(dst, src, count, op);
}

void cpu_scale_inplace_half(c10::Half* buf, int64_t count, float scale) {
    cpu_scale_via_float(buf, count, scale);
}

void cpu_accumulate_and_scale_half(c10::Half* dst, const c10::Half* src,
                                   int64_t count, float scale) {
    cpu_accumulate_and_scale_via_float(dst, src, count, scale);
}

void cpu_reduce_op_bf16(c10::BFloat16* dst, const c10::BFloat16* src, int64_t count,
                        c10d::ReduceOp::RedOpType op) {
    cpu_reduce_via_float(dst, src, count, op);
}

void cpu_scale_inplace_bf16(c10::BFloat16* buf, int64_t count, float scale) {
    cpu_scale_via_float(buf, count, scale);
}

void cpu_accumulate_and_scale_bf16(c10::BFloat16* dst, const c10::BFloat16* src,
                                   int64_t count, float scale) {
    cpu_accumulate_and_scale_via_float(dst, src, count, scale);
}

void cpu_reduce_op(float* dst, const float* src, int64_t count,
                   c10d::ReduceOp::RedOpType op) {
    switch (op) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
            cpu_accumulate(dst, src, count);
            break;
        case c10d::ReduceOp::MIN:
            cpu_elementwise_min(dst, src, count);
            break;
        case c10d::ReduceOp::MAX:
            cpu_elementwise_max(dst, src, count);
            break;
        case c10d::ReduceOp::PRODUCT:
            cpu_elementwise_product(dst, src, count);
            break;
        default:
            throw MCCLError("Unsupported ReduceOp for CPU path: " +
                            std::to_string(static_cast<int>(op)));
    }
}

} // namespace mccl
