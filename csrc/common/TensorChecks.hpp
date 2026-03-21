#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include "common/Errors.hpp"

namespace mccl {

inline void check_single_tensor(const at::Tensor& tensor) {
    MCCL_CHECK_TENSOR(
        tensor.is_mps() || tensor.is_cpu(),
        "MCCL requires MPS or CPU tensors (unified memory), got device: " + tensor.device().str()
    );

    MCCL_CHECK_TENSOR(
        tensor.is_contiguous(),
        "MCCL v1 requires contiguous tensors. "
        "Call .contiguous() before passing to collective."
    );

    auto dtype = tensor.scalar_type();
    MCCL_CHECK_TENSOR(
        dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 || 
        dtype == at::kLong || dtype == at::kInt || dtype == at::kBool || 
        dtype == at::kByte || dtype == at::kDouble,
        "MCCL supports float32/float16/bfloat16 (training), int64/int32/bool/uint8 (metadata), and float64, got: " +
        std::string(at::toString(dtype))
    );

    MCCL_CHECK_TENSOR(
        tensor.numel() > 0,
        "MCCL does not accept empty tensors"
    );
}

inline void check_tensor_list(const std::vector<at::Tensor>& tensors) {
    MCCL_CHECK_TENSOR(
        tensors.size() == 1,
        "MCCL collectives expect exactly one tensor per rank per call"
    );
    check_single_tensor(tensors[0]);
}

inline void check_same_shape_dtype(const at::Tensor& a, const at::Tensor& b) {
    MCCL_CHECK_TENSOR(
        a.sizes() == b.sizes(),
        "Shape mismatch in collective"
    );
    MCCL_CHECK_TENSOR(
        a.scalar_type() == b.scalar_type(),
        "Dtype mismatch in collective"
    );
}

inline size_t tensor_nbytes(const at::Tensor& tensor) {
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

} // namespace mccl
