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

// ── Integral / bool reductions (DDP usage maps, Lightning sync) ─────

kernel void accumulate_chunk_i32(
    device int* dst           [[buffer(0)]],
    device const int* src     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, int, int4>(dst, src, base, aligned, count, gid);
}

kernel void accumulate_chunk_i64(
    device long* dst          [[buffer(0)]],
    device const long* src    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, long, long4>(dst, src, base, aligned, count, gid);
}

kernel void accumulate_chunk_u8(
    device uchar* dst         [[buffer(0)]],
    device const uchar* src   [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpAdd, uchar, uchar4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_min_i32(
    device int* dst           [[buffer(0)]],
    device const int* src     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, int, int4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_max_i32(
    device int* dst           [[buffer(0)]],
    device const int* src     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, int, int4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_product_i32(
    device int* dst           [[buffer(0)]],
    device const int* src     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, int, int4>(dst, src, base, aligned, count, gid);
}

kernel void scale_inplace_i32(
    device int* buf           [[buffer(0)]],
    constant float& scale     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    if (base + kElementsPerThread <= count) {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            buf[i] = int(round(float(buf[i]) * scale));
        }
    } else {
        for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
            buf[i] = int(round(float(buf[i]) * scale));
        }
    }
}

kernel void scale_inplace_i64(
    device long* buf          [[buffer(0)]],
    constant float& scale     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
        buf[i] = long(float(buf[i]) * scale);
    }
}

kernel void scale_inplace_u8(
    device uchar* buf         [[buffer(0)]],
    constant float& scale     [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    for (uint i = base; i < min(base + kElementsPerThread, count); ++i) {
        buf[i] = uchar(float(buf[i]) * scale);
    }
}

kernel void elementwise_min_i64(
    device long* dst          [[buffer(0)]],
    device const long* src    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, long, long4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_max_i64(
    device long* dst          [[buffer(0)]],
    device const long* src    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, long, long4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_product_i64(
    device long* dst          [[buffer(0)]],
    device const long* src    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, long, long4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_min_u8(
    device uchar* dst         [[buffer(0)]],
    device const uchar* src   [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMin, uchar, uchar4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_max_u8(
    device uchar* dst         [[buffer(0)]],
    device const uchar* src   [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMax, uchar, uchar4>(dst, src, base, aligned, count, gid);
}

kernel void elementwise_product_u8(
    device uchar* dst         [[buffer(0)]],
    device const uchar* src   [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant bool& aligned    [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    uint base = gid * kElementsPerThread;
    binary_vec_op<kOpMul, uchar, uchar4>(dst, src, base, aligned, count, gid);
}

// ── STFT transforms (batched radix-2 complex FFT per frame) ───────────────

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline uint bit_reverse(uint x, uint bits) {
    uint y = 0;
    for (uint i = 0; i < bits; ++i) {
        y = (y << 1) | (x & 1u);
        x >>= 1;
    }
    return y;
}

/// In-place radix-2 FFT on ``data[0..n-1]`` (n power of two).
inline void fft_radix2_device(thread float2* data, uint n, uint log2n) {
    for (uint i = 0; i < n; ++i) {
        uint j = bit_reverse(i, log2n);
        if (j > i) {
            float2 tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
    for (uint len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI_F / float(len);
        float2 wlen = float2(cos(ang), sin(ang));
        for (uint i = 0; i < n; i += len) {
            float2 w = float2(1.0f, 0.0f);
            for (uint j = 0; j < len / 2; ++j) {
                float2 u = data[i + j];
                float2 v = cmul(w, data[i + j + len / 2]);
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w = cmul(w, wlen);
            }
        }
    }
}

/// One thread per (batch, frame): windowed real frame -> complex spectrum bins.
kernel void stft_rfft_frame(
    device const float* padded_waveform [[buffer(0)]],
    device const float* window          [[buffer(1)]],
    device float* spec_real             [[buffer(2)]],
    device float* spec_imag             [[buffer(3)]],
    constant uint& batch                [[buffer(4)]],
    constant uint& signal_length        [[buffer(5)]],
    constant uint& padded_length        [[buffer(6)]],
    constant uint& n_fft                [[buffer(7)]],
    constant uint& hop                  [[buffer(8)]],
    constant uint& n_frames             [[buffer(9)]],
    constant uint& n_freq               [[buffer(10)]],
    constant uint& log2n                [[buffer(11)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    const uint b = gid.x;
    const uint f = gid.y;
    if (b >= batch || f >= n_frames) {
        return;
    }

    const uint wave_base = b * padded_length;
    const uint frame_start = f * hop;

    thread float2 local[4096];
    for (uint i = 0; i < n_fft; ++i) {
        const uint idx = frame_start + i;
        float v = 0.0f;
        if (idx < padded_length) {
            v = padded_waveform[wave_base + idx];
        }
        local[i] = float2(v * window[i], 0.0f);
    }

    fft_radix2_device(local, n_fft, log2n);

    for (uint k = 0; k < n_freq; ++k) {
        const uint out_idx = (b * n_freq + k) * n_frames + f;
        spec_real[out_idx] = local[k].x;
        spec_imag[out_idx] = local[k].y;
    }
}

/// Inverse FFT per frame + window (iSTFT frame synthesis).
kernel void stft_irfft_frame(
    device const float* spec_real       [[buffer(0)]],
    device const float* spec_imag       [[buffer(1)]],
    device const float* window          [[buffer(2)]],
    device float* frames_out            [[buffer(3)]],
    constant uint& batch                [[buffer(4)]],
    constant uint& n_fft                [[buffer(5)]],
    constant uint& n_frames             [[buffer(6)]],
    constant uint& n_freq               [[buffer(7)]],
    constant uint& log2n                [[buffer(8)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    const uint b = gid.x;
    const uint f = gid.y;
    if (b >= batch || f >= n_frames) {
        return;
    }

    thread float2 local[4096];
    for (uint k = 0; k < n_freq; ++k) {
        const uint in_idx = (b * n_freq + k) * n_frames + f;
        local[k] = float2(spec_real[in_idx], spec_imag[in_idx]);
    }
    for (uint k = n_freq; k < n_fft; ++k) {
        local[k] = float2(0.0f, 0.0f);
    }

    // Inverse via conjugate-forward-conjugate / n.
    for (uint i = 0; i < n_fft; ++i) {
        local[i].y = -local[i].y;
    }
    fft_radix2_device(local, n_fft, log2n);
    const float inv_n = 1.0f / float(n_fft);
    for (uint i = 0; i < n_fft; ++i) {
        local[i] *= inv_n;
        local[i].y = -local[i].y;
    }

    const uint out_base = (b * n_frames + f) * n_fft;
    for (uint i = 0; i < n_fft; ++i) {
        frames_out[out_base + i] = local[i].x * window[i];
    }
}

/// Overlap-add windowed frames into padded output (one thread per sample).
kernel void stft_overlap_add(
    device const float* frames          [[buffer(0)]],
    device float* output                [[buffer(1)]],
    device float* wsum                  [[buffer(2)]],
    constant uint& batch                [[buffer(3)]],
    constant uint& padded_length        [[buffer(4)]],
    constant uint& n_fft                [[buffer(5)]],
    constant uint& hop                  [[buffer(6)]],
    constant uint& n_frames             [[buffer(7)]],
    constant float& eps                 [[buffer(8)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    const uint b = gid.x;
    const uint t = gid.y;
    if (b >= batch || t >= padded_length) {
        return;
    }

    float acc = 0.0f;
    float wacc = 0.0f;
    const uint frame_base = b * n_frames * n_fft;
    for (uint f = 0; f < n_frames; ++f) {
        const int rel = int(t) - int(f * hop);
        if (rel >= 0 && rel < int(n_fft)) {
            const float v = frames[frame_base + f * n_fft + uint(rel)];
            acc += v;
            wacc += 1.0f;
        }
    }
    const uint out_idx = b * padded_length + t;
    output[out_idx] = acc;
    wsum[out_idx] = max(wacc, eps);
}

kernel void stft_normalize_ola(
    device float* output [[buffer(0)]],
    device const float* wsum [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    output[gid] /= wsum[gid];
}
