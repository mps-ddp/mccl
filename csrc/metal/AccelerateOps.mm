#include "metal/AccelerateOps.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <vector>
#include <algorithm>

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

constexpr int64_t MT_THRESHOLD = 1024 * 1024;  // 1M floats = 4MB
constexpr int     MT_MAX_JOBS  = 8;

int mt_jobs(int64_t count) {
    if (count < MT_THRESHOLD) return 1;
    return std::min(static_cast<int>(count / MT_THRESHOLD), MT_MAX_JOBS);
}

using BinaryVDSPFn = void(*)(const float*, vDSP_Stride, const float*, vDSP_Stride,
                              float*, vDSP_Stride, vDSP_Length);

void parallel_binary(float* dst, const float* src, int64_t count, BinaryVDSPFn fn) {
    int jobs = mt_jobs(count);
    if (jobs <= 1) {
        fn(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
        return;
    }
    int64_t chunk = (count + jobs - 1) / jobs;
    dispatch_apply(static_cast<size_t>(jobs),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t i) {
        int64_t start = static_cast<int64_t>(i) * chunk;
        int64_t end = std::min(start + chunk, count);
        if (start < end) {
            fn(dst + start, 1, src + start, 1, dst + start, 1,
               static_cast<vDSP_Length>(end - start));
        }
    });
}

} // anonymous namespace

void cpu_accumulate(float* dst, const float* src, int64_t count) {
    parallel_binary(dst, src, count, vDSP_vadd);
}

void cpu_scale_inplace(float* buf, int64_t count, float scale) {
    int jobs = mt_jobs(count);
    if (jobs <= 1) {
        vDSP_vsmul(buf, 1, &scale, buf, 1, static_cast<vDSP_Length>(count));
        return;
    }
    int64_t chunk = (count + jobs - 1) / jobs;
    dispatch_apply(static_cast<size_t>(jobs),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t i) {
        int64_t start = static_cast<int64_t>(i) * chunk;
        int64_t end = std::min(start + chunk, count);
        if (start < end) {
            vDSP_vsmul(buf + start, 1, &scale, buf + start, 1,
                       static_cast<vDSP_Length>(end - start));
        }
    });
}

void cpu_accumulate_and_scale(float* dst, const float* src,
                              int64_t count, float scale) {
    int jobs = mt_jobs(count);
    if (jobs <= 1) {
        vDSP_vadd(dst, 1, src, 1, dst, 1, static_cast<vDSP_Length>(count));
        vDSP_vsmul(dst, 1, &scale, dst, 1, static_cast<vDSP_Length>(count));
        return;
    }
    int64_t chunk = (count + jobs - 1) / jobs;
    dispatch_apply(static_cast<size_t>(jobs),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t i) {
        int64_t start = static_cast<int64_t>(i) * chunk;
        int64_t end = std::min(start + chunk, count);
        if (start < end) {
            vDSP_Length len = static_cast<vDSP_Length>(end - start);
            vDSP_vadd(dst + start, 1, src + start, 1, dst + start, 1, len);
            vDSP_vsmul(dst + start, 1, &scale, dst + start, 1, len);
        }
    });
}

void cpu_elementwise_min(float* dst, const float* src, int64_t count) {
    parallel_binary(dst, src, count, vDSP_vmin);
}

void cpu_elementwise_max(float* dst, const float* src, int64_t count) {
    parallel_binary(dst, src, count, vDSP_vmax);
}

void cpu_elementwise_product(float* dst, const float* src, int64_t count) {
    parallel_binary(dst, src, count, vDSP_vmul);
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
