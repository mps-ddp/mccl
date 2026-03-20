#include "runtime/MemoryPool.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <cstdlib>
#include <algorithm>

namespace mccl {

MemoryPool::MemoryPool(size_t alignment) : alignment_(alignment) {}

MemoryPool::~MemoryPool() {
    trim();
}

static size_t align_up(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

MemoryPool::Buffer MemoryPool::acquire(size_t nbytes) {
    nbytes = align_up(nbytes, alignment_);

    std::lock_guard<std::mutex> lock(mu_);

    // Find smallest buffer >= nbytes (best-fit)
    int best = -1;
    for (int i = 0; i < (int)pool_.size(); i++) {
        if (pool_[i].capacity >= nbytes) {
            if (best < 0 || pool_[i].capacity < pool_[best].capacity) {
                best = i;
            }
        }
    }

    if (best >= 0) {
        Buffer buf = pool_[best];
        pool_.erase(pool_.begin() + best);
        buf.used = nbytes;
        MCCL_TRACE("MemoryPool: reused %zu-byte buffer for %zu request",
                   buf.capacity, nbytes);
        return buf;
    }

    // Over-allocate by 25% to reduce future misses
    size_t alloc_size = nbytes + (nbytes >> 2);
    alloc_size = align_up(alloc_size, alignment_);

    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment_, alloc_size) != 0 || !ptr) {
        throw MCCLError("MemoryPool: allocation failed for " +
                        std::to_string(alloc_size) + " bytes");
    }

    total_allocated_ += alloc_size;
    MCCL_DEBUG("MemoryPool: allocated %zu bytes (total=%zu)",
               alloc_size, total_allocated_);

    return Buffer{ptr, alloc_size, nbytes};
}

void MemoryPool::release(Buffer buf) {
    if (!buf.data) return;

    std::lock_guard<std::mutex> lock(mu_);
    buf.used = 0;
    pool_.push_back(buf);
}

size_t MemoryPool::total_allocated() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_allocated_;
}

size_t MemoryPool::pooled_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pool_.size();
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& buf : pool_) {
        free(buf.data);
        total_allocated_ -= buf.capacity;
    }
    pool_.clear();
    MCCL_DEBUG("MemoryPool: trimmed, total_allocated=%zu", total_allocated_);
}

// ── PooledBuffer ────────────────────────────────────────────────────

PooledBuffer::PooledBuffer(MemoryPool& pool, size_t nbytes)
    : pool_(&pool), buf_(pool.acquire(nbytes)) {}

PooledBuffer::~PooledBuffer() {
    if (pool_ && buf_.data) {
        pool_->release(buf_);
    }
}

PooledBuffer::PooledBuffer(PooledBuffer&& other) noexcept
    : pool_(other.pool_), buf_(other.buf_) {
    other.pool_ = nullptr;
    other.buf_ = {};
}

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
        if (pool_ && buf_.data) pool_->release(buf_);
        pool_ = other.pool_;
        buf_ = other.buf_;
        other.pool_ = nullptr;
        other.buf_ = {};
    }
    return *this;
}

MemoryPool& staging_memory_pool() {
    static MemoryPool pool(16384);
    return pool;
}

} // namespace mccl
