#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <mutex>
#include <unordered_map>

namespace mccl {

/// Gradient validation instrumentation for debugging convergence issues.
/// Computes checksums of tensors before/after allreduce to detect sync problems.
class GradientValidator {
public:
    GradientValidator();
    ~GradientValidator();

    /// Enable/disable validation (controlled by MCCL_VALIDATE_GRAD_SYNC env var)
    bool is_enabled() const { return enabled_; }

    /// Compute checksum of a tensor (sum of absolute values)
    double compute_checksum(const at::Tensor& tensor);

    /// Record pre-allreduce checksum
    void record_pre_allreduce(uint32_t seq, const at::Tensor& tensor);

    /// Record post-allreduce checksum and validate
    void record_post_allreduce(uint32_t seq, const at::Tensor& tensor, int rank, int world_size);

    /// Log summary of validation results
    void log_summary() const;

private:
    bool enabled_;
    mutable std::mutex mu_;
    
    struct ValidationRecord {
        double pre_checksum = 0.0;
        double post_checksum = 0.0;
        bool validated = false;
        bool mismatch = false;
    };
    
    std::unordered_map<uint32_t, ValidationRecord> records_;
    uint64_t total_validations_ = 0;
    uint64_t total_mismatches_ = 0;
};

} // namespace mccl
