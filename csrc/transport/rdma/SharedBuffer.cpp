#include "transport/rdma/SharedBuffer.hpp"
#include "transport/rdma/IbvWrapper.hpp"
#include "common/Logging.hpp"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <utility>

namespace mccl {

static size_t page_size() {
    static const size_t ps = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return ps;
}

SharedBuffer::SharedBuffer(size_t nbytes)
    : nbytes_(nbytes) {
    int rc = posix_memalign(&buf_, page_size(), nbytes_);
    if (rc != 0 || !buf_) {
        MCCL_ERROR("SharedBuffer: posix_memalign(%zu) failed: %d", nbytes_, rc);
        buf_ = nullptr;
        nbytes_ = 0;
        return;
    }
    std::memset(buf_, 0, nbytes_);
}

SharedBuffer::~SharedBuffer() {
    cleanup();
}

SharedBuffer::SharedBuffer(SharedBuffer&& other) noexcept
    : buf_(other.buf_),
      nbytes_(other.nbytes_),
      registrations_(std::move(other.registrations_)) {
    other.buf_ = nullptr;
    other.nbytes_ = 0;
}

SharedBuffer& SharedBuffer::operator=(SharedBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        buf_ = other.buf_;
        nbytes_ = other.nbytes_;
        registrations_ = std::move(other.registrations_);
        other.buf_ = nullptr;
        other.nbytes_ = 0;
    }
    return *this;
}

void SharedBuffer::cleanup() {
    auto* fns = ibv();
    if (fns) {
        for (ibv_mr* mr : registrations_) {
            if (mr) fns->dereg_mr(mr);
        }
    }
    registrations_.clear();
    if (buf_) {
        free(buf_);
        buf_ = nullptr;
    }
    nbytes_ = 0;
}

ibv_mr* SharedBuffer::register_with(ibv_pd* pd) {
    auto* fns = ibv();
    if (!fns || !buf_ || !pd) return nullptr;

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    ibv_mr* mr = fns->reg_mr(pd, buf_, nbytes_, access);
    if (!mr) {
        MCCL_ERROR("SharedBuffer: ibv_reg_mr failed for %zu bytes", nbytes_);
        return nullptr;
    }
    registrations_.push_back(mr);
    return mr;
}

ibv_sge SharedBuffer::to_sge(ibv_mr* mr, size_t offset, uint32_t length) const {
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(buf_) + offset);
    sge.length = length;
    sge.lkey   = mr->lkey;
    return sge;
}

ibv_sge SharedBuffer::to_sge(ibv_mr* mr) const {
    return to_sge(mr, 0, static_cast<uint32_t>(nbytes_));
}

} // namespace mccl
