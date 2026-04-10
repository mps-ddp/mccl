#pragma once

#include <torch/torch.h>
#include <c10d/Types.hpp>
#include <cstddef>
#include <cstdint>

namespace mccl {

void metal_kernels_init();

/// dst += src element-wise on GPU via Metal compute.
void metal_accumulate_chunk(const at::Tensor& dst, const at::Tensor& src);

/// dst = min(dst, src) element-wise.
void metal_elementwise_min(const at::Tensor& dst, const at::Tensor& src);

/// dst = max(dst, src) element-wise.
void metal_elementwise_max(const at::Tensor& dst, const at::Tensor& src);

/// dst *= src element-wise.
void metal_elementwise_product(const at::Tensor& dst, const at::Tensor& src);

/// Dispatch the correct reduce kernel based on ReduceOp.
void metal_reduce_op(const at::Tensor& dst, const at::Tensor& src,
                     c10d::ReduceOp::RedOpType op);

/// buf *= scale element-wise on GPU via Metal compute.
void metal_scale_inplace(const at::Tensor& buf, double scale);

/// Fused: dst = (dst + src) * scale in a single kernel launch.
void metal_accumulate_and_scale(const at::Tensor& dst, const at::Tensor& src,
                                double scale);

/// Block until all MCCL Metal commands have completed.
void metal_sync();

/// Drain MCCL's Metal queue only (no ``torch::mps::synchronize``). Use from
/// ProgressEngine / net threads; full ``metal_sync()`` is unsafe there.
void metal_sync_queue_only();

} // namespace mccl
