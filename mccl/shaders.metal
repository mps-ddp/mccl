#include <metal_stdlib>
using namespace metal;

namespace {

#if defined(__HAVE_BFLOAT__)
// Metal's min/max overloads are ambiguous for scalar bfloat and unavailable for bfloat
// vectors; use float intermediates (same numeric range as bf16).
inline bfloat bf_min(bfloat a, bfloat b) { return bfloat(metal::min(float(a), float(b))); }
inline bfloat bf_max(bfloat a, bfloat b) { return bfloat(metal::max(float(a), float(b))); }
inline bfloat4 bf_min4(bfloat4 a, bfloat4 b) {
    return bfloat4(metal::min(float4(a), float4(b)));
}
inline bfloat4 bf_max4(bfloat4 a, bfloat4 b) {
    return bfloat4(metal::max(float4(a), float4(b)));
}
#endif

constant uint kElementsPerThread = 8;
constant uint kOpAdd = 0;
constant uint kOpMin = 1;
constant uint kOpMax = 2;
constant uint kOpMul = 3;

template <typename T, typename VecT>
inline void load2(device const T* src, uint vec_index, thread VecT& v0, thread VecT& v1) {
    device const VecT* src_vec = reinterpret_cast<device const VecT*>(src);
    v0 = src_vec[vec_index];
    v1 = src_vec[vec_index + 1];
}

template <typename T, typename VecT>
inline void store2(device T* dst, uint vec_index, VecT v0, VecT v1) {
    device VecT* dst_vec = reinterpret_cast<device VecT*>(dst);
    dst_vec[vec_index] = v0;
    dst_vec[vec_index + 1] = v1;
}

template <typename T, typename VecT>
inline void load2_scalar(device const T* src, uint base, thread VecT& v0, thread VecT& v1) {
    v0 = VecT(src[base], src[base + 1], src[base + 2], src[base + 3]);
    v1 = VecT(src[base + 4], src[base + 5], src[base + 6], src[base + 7]);
}

template <typename T>
inline void scale_tail(device T* dst, T scale, uint base, uint count) {
    for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
        dst[i] *= scale;
    }
}

template <typename T>
inline void accumulate_scale_tail(device T* dst, device const T* src, T scale, uint base, uint count) {
    for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
        dst[i] = (dst[i] + src[i]) * scale;
    }
}

template <uint Op, typename VecT>
struct BinaryApply;

template <typename VecT>
struct BinaryApply<kOpAdd, VecT> {
    static inline VecT vec(VecT a, VecT b) { return a + b; }
};

template <typename VecT>
struct BinaryApply<kOpMin, VecT> {
    static inline VecT vec(VecT a, VecT b) { return min(a, b); }
};

template <typename VecT>
struct BinaryApply<kOpMax, VecT> {
    static inline VecT vec(VecT a, VecT b) { return max(a, b); }
};

template <typename VecT>
struct BinaryApply<kOpMul, VecT> {
    static inline VecT vec(VecT a, VecT b) { return a * b; }
};

#if defined(__HAVE_BFLOAT__)
template <>
struct BinaryApply<kOpMin, bfloat4> {
    static inline bfloat4 vec(bfloat4 a, bfloat4 b) { return bf_min4(a, b); }
};

template <>
struct BinaryApply<kOpMax, bfloat4> {
    static inline bfloat4 vec(bfloat4 a, bfloat4 b) { return bf_max4(a, b); }
};
#endif

// Split per (Op, T): a single template with if(Op==...) still type-checks every branch for bfloat.
template <uint Op, typename T>
struct ApplyBinaryTail;

template <typename T>
struct ApplyBinaryTail<kOpAdd, T> {
    static inline void apply(device T* dst, device const T* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] += src[i];
        }
    }
};

template <typename T>
struct ApplyBinaryTail<kOpMin, T> {
    static inline void apply(device T* dst, device const T* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] = min(dst[i], src[i]);
        }
    }
};

template <typename T>
struct ApplyBinaryTail<kOpMax, T> {
    static inline void apply(device T* dst, device const T* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] = max(dst[i], src[i]);
        }
    }
};

template <typename T>
struct ApplyBinaryTail<kOpMul, T> {
    static inline void apply(device T* dst, device const T* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] *= src[i];
        }
    }
};

#if defined(__HAVE_BFLOAT__)
template <>
struct ApplyBinaryTail<kOpMin, bfloat> {
    static inline void apply(device bfloat* dst, device const bfloat* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] = bf_min(dst[i], src[i]);
        }
    }
};

template <>
struct ApplyBinaryTail<kOpMax, bfloat> {
    static inline void apply(device bfloat* dst, device const bfloat* src, uint base, uint count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            dst[i] = bf_max(dst[i], src[i]);
        }
    }
};
#endif

template <uint Op, typename T>
inline void apply_binary_tail(device T* dst, device const T* src, uint base, uint count) {
    ApplyBinaryTail<Op, T>::apply(dst, src, base, count);
}

template <uint Op, typename T, typename VecT>
inline void binary_vec_op(device T* dst, device const T* src, uint base, bool aligned,
                          uint count, uint gid) {
    if (base + kElementsPerThread <= count) {
        VecT d0, d1, s0, s1;
        if (aligned) {
            uint vec_index = gid * 2;
            load2<T, VecT>(dst, vec_index, d0, d1);
            load2<T, VecT>(src, vec_index, s0, s1);
            store2<T, VecT>(dst, vec_index, BinaryApply<Op, VecT>::vec(d0, s0),
                            BinaryApply<Op, VecT>::vec(d1, s1));
        } else {
            load2_scalar<T, VecT>(dst, base, d0, d1);
            load2_scalar<T, VecT>(src, base, s0, s1);
            VecT o0 = BinaryApply<Op, VecT>::vec(d0, s0);
            VecT o1 = BinaryApply<Op, VecT>::vec(d1, s1);
            dst[base] = o0[0];
            dst[base + 1] = o0[1];
            dst[base + 2] = o0[2];
            dst[base + 3] = o0[3];
            dst[base + 4] = o1[0];
            dst[base + 5] = o1[1];
            dst[base + 6] = o1[2];
            dst[base + 7] = o1[3];
        }
    } else {
        apply_binary_tail<Op, T>(dst, src, base, count);
    }
}

} // namespace

// ── Vectorized accumulate: dst[i] += src[i] ────────────────────────
// Each thread processes 8 elements (two vector loads). Grid size = ceil(count / 8).
// Host code passes an alignment flag so narrowed tensor views safely fall back
// to scalar loads while aligned buffers use coalesced vector memory ops.

kernel void accumulate_chunk_f32(
    device float* dst        [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, float, float4>(dst, src, base, aligned, count, gid);
}

kernel void accumulate_chunk_f16(
    device half* dst         [[buffer(0)]],
    device const half* src   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, half, half4>(dst, src, base, aligned, count, gid);
}

#if defined(__HAVE_BFLOAT__)
kernel void accumulate_chunk_bf16(
    device bfloat* dst        [[buffer(0)]],
    device const bfloat* src  [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, bfloat, bfloat4>(dst, src, base, aligned, count, gid);
}
#endif

// ── Vectorized scale: buf[i] *= scale ──────────────────────────────

kernel void scale_inplace_f32(
    device float* buf        [[buffer(0)]],
    constant float& scale    [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        if (aligned) {
            uint vec_index = gid * 2;
            float4 b0, b1;
            load2<float, float4>(buf, vec_index, b0, b1);
            store2<float, float4>(buf, vec_index, b0 * scale, b1 * scale);
        } else {
            float4 b0, b1;
            load2_scalar<float, float4>(buf, base, b0, b1);
            b0 *= scale;
            b1 *= scale;
            buf[base] = b0[0];
            buf[base + 1] = b0[1];
            buf[base + 2] = b0[2];
            buf[base + 3] = b0[3];
            buf[base + 4] = b1[0];
            buf[base + 5] = b1[1];
            buf[base + 6] = b1[2];
            buf[base + 7] = b1[3];
        }
    } else {
        scale_tail<float>(buf, scale, base, count);
    }
}

kernel void scale_inplace_f16(
    device half* buf         [[buffer(0)]],
    constant half& scale     [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        if (aligned) {
            uint vec_index = gid * 2;
            half4 b0, b1;
            load2<half, half4>(buf, vec_index, b0, b1);
            store2<half, half4>(buf, vec_index, b0 * scale, b1 * scale);
        } else {
            half4 b0, b1;
            load2_scalar<half, half4>(buf, base, b0, b1);
            b0 *= scale;
            b1 *= scale;
            buf[base] = b0[0];
            buf[base + 1] = b0[1];
            buf[base + 2] = b0[2];
            buf[base + 3] = b0[3];
            buf[base + 4] = b1[0];
            buf[base + 5] = b1[1];
            buf[base + 6] = b1[2];
            buf[base + 7] = b1[3];
        }
    } else {
        scale_tail<half>(buf, scale, base, count);
    }
}

#if defined(__HAVE_BFLOAT__)
kernel void scale_inplace_bf16(
    device bfloat* buf       [[buffer(0)]],
    constant bfloat& scale   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        if (aligned) {
            uint vec_index = gid * 2;
            bfloat4 b0, b1;
            load2<bfloat, bfloat4>(buf, vec_index, b0, b1);
            store2<bfloat, bfloat4>(buf, vec_index, b0 * scale, b1 * scale);
        } else {
            bfloat4 b0, b1;
            load2_scalar<bfloat, bfloat4>(buf, base, b0, b1);
            b0 *= scale;
            b1 *= scale;
            buf[base] = b0[0];
            buf[base + 1] = b0[1];
            buf[base + 2] = b0[2];
            buf[base + 3] = b0[3];
            buf[base + 4] = b1[0];
            buf[base + 5] = b1[1];
            buf[base + 6] = b1[2];
            buf[base + 7] = b1[3];
        }
    } else {
        scale_tail<bfloat>(buf, scale, base, count);
    }
}
#endif

// ── Element-wise min: dst[i] = min(dst[i], src[i]) ─────────────────

kernel void elementwise_min_f32(
    device float* dst        [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, float, float4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_min_f16(
    device half* dst         [[buffer(0)]],
    device const half* src   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, half, half4>(dst, src, base, aligned, count, gid);
}

#if defined(__HAVE_BFLOAT__)
kernel void elementwise_min_bf16(
    device bfloat* dst        [[buffer(0)]],
    device const bfloat* src  [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, bfloat, bfloat4>(dst, src, base, aligned, count, gid);
}
#endif

// ── Element-wise max: dst[i] = max(dst[i], src[i]) ─────────────────

kernel void elementwise_max_f32(
    device float* dst        [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, float, float4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_max_f16(
    device half* dst         [[buffer(0)]],
    device const half* src   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, half, half4>(dst, src, base, aligned, count, gid);
}

#if defined(__HAVE_BFLOAT__)
kernel void elementwise_max_bf16(
    device bfloat* dst        [[buffer(0)]],
    device const bfloat* src  [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, bfloat, bfloat4>(dst, src, base, aligned, count, gid);
}
#endif

// ── Element-wise product: dst[i] *= src[i] ──────────────────────────

kernel void elementwise_product_f32(
    device float* dst        [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, float, float4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_product_f16(
    device half* dst         [[buffer(0)]],
    device const half* src   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    constant bool& aligned   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, half, half4>(dst, src, base, aligned, count, gid);
}

#if defined(__HAVE_BFLOAT__)
kernel void elementwise_product_bf16(
    device bfloat* dst        [[buffer(0)]],
    device const bfloat* src  [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, bfloat, bfloat4>(dst, src, base, aligned, count, gid);
}
#endif

// ── Fused accumulate + scale: dst = (dst + src) * scale ────────────

kernel void accumulate_scale_f32(
    device float* dst        [[buffer(0)]],
    device const float* src  [[buffer(1)]],
    constant float& scale    [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    constant bool& aligned   [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        float4 d0, d1, s0, s1;
        if (aligned) {
            uint vec_index = gid * 2;
            load2<float, float4>(dst, vec_index, d0, d1);
            load2<float, float4>(src, vec_index, s0, s1);
            store2<float, float4>(dst, vec_index, (d0 + s0) * scale, (d1 + s1) * scale);
        } else {
            load2_scalar<float, float4>(dst, base, d0, d1);
            load2_scalar<float, float4>(src, base, s0, s1);
            float4 o0 = (d0 + s0) * scale;
            float4 o1 = (d1 + s1) * scale;
            dst[base] = o0[0];
            dst[base + 1] = o0[1];
            dst[base + 2] = o0[2];
            dst[base + 3] = o0[3];
            dst[base + 4] = o1[0];
            dst[base + 5] = o1[1];
            dst[base + 6] = o1[2];
            dst[base + 7] = o1[3];
        }
    } else {
        accumulate_scale_tail<float>(dst, src, scale, base, count);
    }
}

kernel void accumulate_scale_f16(
    device half* dst         [[buffer(0)]],
    device const half* src   [[buffer(1)]],
    constant half& scale     [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    constant bool& aligned   [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        half4 d0, d1, s0, s1;
        if (aligned) {
            uint vec_index = gid * 2;
            load2<half, half4>(dst, vec_index, d0, d1);
            load2<half, half4>(src, vec_index, s0, s1);
            store2<half, half4>(dst, vec_index, (d0 + s0) * scale, (d1 + s1) * scale);
        } else {
            load2_scalar<half, half4>(dst, base, d0, d1);
            load2_scalar<half, half4>(src, base, s0, s1);
            half4 o0 = (d0 + s0) * scale;
            half4 o1 = (d1 + s1) * scale;
            dst[base] = o0[0];
            dst[base + 1] = o0[1];
            dst[base + 2] = o0[2];
            dst[base + 3] = o0[3];
            dst[base + 4] = o1[0];
            dst[base + 5] = o1[1];
            dst[base + 6] = o1[2];
            dst[base + 7] = o1[3];
        }
    } else {
        accumulate_scale_tail<half>(dst, src, scale, base, count);
    }
}

#if defined(__HAVE_BFLOAT__)
kernel void accumulate_scale_bf16(
    device bfloat* dst        [[buffer(0)]],
    device const bfloat* src  [[buffer(1)]],
    constant bfloat& scale    [[buffer(2)]],
    constant uint& count      [[buffer(3)]],
    constant bool& aligned    [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        bfloat4 d0, d1, s0, s1;
        if (aligned) {
            uint vec_index = gid * 2;
            load2<bfloat, bfloat4>(dst, vec_index, d0, d1);
            load2<bfloat, bfloat4>(src, vec_index, s0, s1);
            store2<bfloat, bfloat4>(dst, vec_index, (d0 + s0) * scale, (d1 + s1) * scale);
        } else {
            load2_scalar<bfloat, bfloat4>(dst, base, d0, d1);
            load2_scalar<bfloat, bfloat4>(src, base, s0, s1);
            bfloat4 o0 = (d0 + s0) * scale;
            bfloat4 o1 = (d1 + s1) * scale;
            dst[base] = o0[0];
            dst[base + 1] = o0[1];
            dst[base + 2] = o0[2];
            dst[base + 3] = o0[3];
            dst[base + 4] = o1[0];
            dst[base + 5] = o1[1];
            dst[base + 6] = o1[2];
            dst[base + 7] = o1[3];
        }
    } else {
        accumulate_scale_tail<bfloat>(dst, src, scale, base, count);
    }
}
#endif
