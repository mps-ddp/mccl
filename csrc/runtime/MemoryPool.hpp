#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>
#include <memory>

namespace mccl {

/// Reusable staging buffer pool.
///
/// Allocates host-side buffers for CPU↔GPU staging and network I/O.
/// Buffers are returned to the pool instead of freed, reducing
/// allocation pressure during training loops.
class MemoryPool {
public:
    explicit MemoryPool(size_t alignment = 64);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    struct Buffer {
        void* data;
        size_t capacity;
        size_t used;
    };

    /// Acquire a buffer of at least `nbytes`. May return a larger cached buffer.
    Buffer acquire(size_t nbytes);

    /// Release a buffer back to the pool.
    void release(Buffer buf);

    /// Total bytes currently allocated (pooled + in-use).
    size_t total_allocated() const;

    /// Number of buffers currently in the pool.
    size_t pooled_count() const;

    /// Free all pooled buffers.
    void trim();

private:
    size_t alignment_;
    mutable std::mutex mu_;
    std::vector<Buffer> pool_;
    size_t total_allocated_ = 0;
};

/// RAII guard that auto-returns a buffer to the pool on scope exit.
class PooledBuffer {
public:
    PooledBuffer(MemoryPool& pool, size_t nbytes);
    ~PooledBuffer();

    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    PooledBuffer(PooledBuffer&& other) noexcept;
    PooledBuffer& operator=(PooledBuffer&& other) noexcept;

    void* data() const { return buf_.data; }
    size_t capacity() const { return buf_.capacity; }

    template<typename T>
    T* as() const { return static_cast<T*>(buf_.data); }

private:
    MemoryPool* pool_;
    MemoryPool::Buffer buf_;
};

/// Shared ownership wrapper for PooledBuffer to support cross-thread lifetime management.
/// Used when network phase and reduce phase execute on different engines.
using SharedPooledBuffer = std::shared_ptr<PooledBuffer>;

/// Global staging pool singleton.
MemoryPool& staging_memory_pool();

} // namespace mccl
