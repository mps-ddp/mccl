#pragma once

#include "transport/rdma/ibverbs_compat.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mccl {

/// Page-aligned buffer registered with one or more RDMA protection domains.
///
/// Mirrors MLX/JACCL's SharedBuffer: a single contiguous allocation that
/// can be registered against multiple PDs (one per device/connection).
/// The destructor deregisters all MRs then frees the backing memory.
class SharedBuffer {
public:
    SharedBuffer() = default;
    explicit SharedBuffer(size_t nbytes);
    ~SharedBuffer();

    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;
    SharedBuffer(SharedBuffer&& other) noexcept;
    SharedBuffer& operator=(SharedBuffer&& other) noexcept;

    /// Register this buffer with a protection domain.
    /// Returns the ibv_mr* on success, nullptr on failure.
    /// Access flags: LOCAL_WRITE | REMOTE_READ | REMOTE_WRITE.
    ibv_mr* register_with(ibv_pd* pd);

    /// Build an SGE covering [offset, offset+length) of this buffer,
    /// using the lkey from the given MR.
    ibv_sge to_sge(ibv_mr* mr, size_t offset, uint32_t length) const;

    /// Build an SGE covering the entire buffer.
    ibv_sge to_sge(ibv_mr* mr) const;

    void*  data()  const { return buf_; }
    size_t size()  const { return nbytes_; }

private:
    void cleanup();

    void*  buf_    = nullptr;
    size_t nbytes_ = 0;
    std::vector<ibv_mr*> registrations_;
};

} // namespace mccl
