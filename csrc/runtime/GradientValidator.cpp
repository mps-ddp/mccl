#include "runtime/GradientValidator.hpp"
#include "common/Logging.hpp"
#include "metal/MPSInterop.hpp"
#include <cmath>
#include <cstdlib>

namespace mccl {

GradientValidator::GradientValidator() {
    enabled_ = false;
    if (auto* v = std::getenv("MCCL_VALIDATE_GRAD_SYNC")) {
        std::string s(v);
        enabled_ = (s == "1" || s == "true" || s == "yes");
    }
    
    if (enabled_) {
        MCCL_INFO("Gradient validation enabled (MCCL_VALIDATE_GRAD_SYNC=1)");
    }
}

GradientValidator::~GradientValidator() {
    if (enabled_) {
        log_summary();
    }
}

double GradientValidator::compute_checksum(const at::Tensor& tensor) {
    if (!enabled_) return 0.0;
    
    try {
        // Ensure tensor is on CPU for checksum computation
        at::Tensor cpu_tensor;
        if (tensor.is_mps()) {
            // For MPS tensors, check if CPU-accessible
            if (tensor_cpu_accessible(tensor)) {
                MPSBufferView view = extract_mps_buffer(tensor);
                // Compute checksum directly from CPU-accessible memory
                if (tensor.scalar_type() == at::kFloat) {
                    const float* data = static_cast<const float*>(view.cpu_ptr);
                    double sum = 0.0;
                    for (int64_t i = 0; i < tensor.numel(); i++) {
                        sum += std::abs(static_cast<double>(data[i]));
                    }
                    return sum;
                }
            }
            // Fall back to explicit copy
            cpu_tensor = tensor.to(at::kCPU);
        } else {
            cpu_tensor = tensor;
        }
        
        // Compute sum of absolute values as checksum
        auto abs_tensor = cpu_tensor.abs();
        double checksum = abs_tensor.sum().item<double>();
        return checksum;
    } catch (const std::exception& e) {
        MCCL_WARN("GradientValidator::compute_checksum failed: %s", e.what());
        return 0.0;
    }
}

void GradientValidator::record_pre_allreduce(uint32_t seq, const at::Tensor& tensor) {
    if (!enabled_) return;
    
    double checksum = compute_checksum(tensor);
    
    std::lock_guard<std::mutex> lock(mu_);
    auto& record = records_[seq];
    record.pre_checksum = checksum;
    
    MCCL_DEBUG("GradientValidator: seq=%u pre_checksum=%.6e", seq, checksum);
}

void GradientValidator::record_post_allreduce(uint32_t seq, const at::Tensor& tensor, 
                                               int rank, int world_size) {
    if (!enabled_) return;
    
    double checksum = compute_checksum(tensor);
    
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(seq);
    if (it == records_.end()) {
        MCCL_WARN("GradientValidator: seq=%u post-allreduce without pre-allreduce", seq);
        return;
    }
    
    auto& record = it->second;
    record.post_checksum = checksum;
    record.validated = true;
    total_validations_++;
    
    // For SUM reduction, post_checksum should be approximately world_size * pre_checksum
    // (allowing for floating point error)
    double expected = record.pre_checksum * world_size;
    double relative_error = std::abs(checksum - expected) / (expected + 1e-10);
    
    if (relative_error > 0.01) {  // 1% tolerance
        record.mismatch = true;
        total_mismatches_++;
        MCCL_WARN("GradientValidator: seq=%u MISMATCH on rank %d: "
                  "pre=%.6e post=%.6e expected=%.6e rel_err=%.2f%%",
                  seq, rank, record.pre_checksum, checksum, expected, relative_error * 100.0);
    } else {
        MCCL_DEBUG("GradientValidator: seq=%u OK on rank %d: "
                   "pre=%.6e post=%.6e expected=%.6e rel_err=%.4f%%",
                   seq, rank, record.pre_checksum, checksum, expected, relative_error * 100.0);
    }
}

void GradientValidator::log_summary() const {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mu_);
    
    MCCL_INFO("=== Gradient Validation Summary ===");
    MCCL_INFO("  Total validations: %llu", (unsigned long long)total_validations_);
    MCCL_INFO("  Total mismatches:  %llu", (unsigned long long)total_mismatches_);
    
    if (total_validations_ > 0) {
        double mismatch_rate = 100.0 * total_mismatches_ / total_validations_;
        MCCL_INFO("  Mismatch rate:     %.2f%%", mismatch_rate);
        
        if (total_mismatches_ == 0) {
            MCCL_INFO("  ✓ All gradients synchronized correctly!");
        } else {
            MCCL_WARN("  ✗ Gradient synchronization issues detected!");
        }
    }
}

} // namespace mccl
