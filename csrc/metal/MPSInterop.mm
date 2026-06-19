#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>
#import <torch/mps.h>

#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "metal/MPSInterop.hpp"
#include "metal/EventSync.hpp"
#include "runtime/MCCLDeviceMutex.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include "common/TensorChecks.hpp"

namespace at::mps {
    static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
        return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
    }
}

namespace mccl {

namespace {

id<MTLDevice> cached_device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MCCL_CHECK(dev != nil, "No Metal device available");
    return dev;
}

id<MTLCommandQueue> cached_queue() {
    static id<MTLCommandQueue> q = [cached_device() newCommandQueue];
    MCCL_CHECK(q != nil, "Failed to create MTLCommandQueue");
    return q;
}

size_t metal_max_buffer_len() {
    static size_t max_len = static_cast<size_t>([cached_device() maxBufferLength]);
    return max_len;
}

constexpr size_t PAGE = 16384;

// Staging buffer — reused across calls to avoid repeated allocation.
// Thread-safety: ensure()/use is guarded by `mu` (held by the staging entry
// points below).  Concurrent collectives must NOT route through this pool —
// the pipelined ring paths use caller-owned buffers via blit_tensor_to_buffer
// / blit_buffer_to_tensor instead; the pool only backs the residual serial
// paths (legacy lock-step ring, 2-rank fallback, ensure_shared_storage).
struct StagingPool {
    std::mutex mu;
    void* ptr = nullptr;
    size_t capacity = 0;
    id<MTLBuffer> mtl_wrapper = nil;

    void* ensure(size_t nbytes, id<MTLDevice> device) {
        if (nbytes > capacity) {
            mtl_wrapper = nil;
            free(ptr);
            ptr = nullptr;

            capacity = (nbytes + PAGE - 1) & ~(PAGE - 1);
            int rc = posix_memalign(&ptr, PAGE, capacity);
            MCCL_CHECK(rc == 0 && ptr != nullptr, "Staging buffer allocation failed");

            size_t max_buf = metal_max_buffer_len();
            if (capacity <= max_buf) {
                mtl_wrapper = [device newBufferWithBytesNoCopy:ptr
                                                       length:capacity
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
                MCCL_CHECK(mtl_wrapper != nil, "Staging MTLBuffer creation failed");
            } else {
                MCCL_INFO("Staging pool %zu bytes exceeds maxBufferLength %zu; using chunked blits",
                          capacity, max_buf);
            }
            MCCL_DEBUG("Staging pool resized to %zu bytes (page-aligned)", capacity);
        }
        return ptr;
    }

    ~StagingPool() {
        mtl_wrapper = nil;
        free(ptr);
    }
};

StagingPool& staging_pool() {
    static StagingPool pool;
    return pool;
}

/// Blit from GPU buffer into CPU staging, handling buffers larger than maxBufferLength
/// by chunking into multiple blit commands with temporary MTLBuffer wrappers.
void check_command_buffer(id<MTLCommandBuffer> cmd, const char* context) {
    if (cmd.status == MTLCommandBufferStatusError) {
        NSError* err = cmd.error;
        MCCL_ERROR("%s: Metal command buffer error: %s (code %ld)",
                   context,
                   err ? [[err localizedDescription] UTF8String] : "unknown",
                   err ? (long)err.code : -1);
        MCCL_CHECK(false, std::string(context) + ": Metal command buffer failed");
    }
}

void chunked_blit_to_staging(id<MTLBuffer> src_buf, size_t src_offset,
                              void* dst, size_t nbytes) {
    StagingPool& pool = staging_pool();
    // Fast path is only valid when the caller's destination IS the pool;
    // otherwise (e.g. ensure_shared_storage passes a fresh allocation) the
    // data would land in the pool and `dst` would stay uninitialized.
    if (dst == pool.ptr && pool.mtl_wrapper && nbytes <= pool.capacity) {
        std::lock_guard<std::recursive_mutex> dev_lock(mccl_device_ops_mutex());
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:src_buf sourceOffset:src_offset
                        toBuffer:pool.mtl_wrapper destinationOffset:0
                            size:nbytes];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check_command_buffer(cmd, "chunked_blit_to_staging(fast)");
        }
        return;
    }

    size_t max_chunk = metal_max_buffer_len();
    size_t offset = 0;
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);

    MCCL_INFO("chunked_blit_to_staging: %zu bytes in chunks of %zu (maxBuf=%zu)",
              nbytes, max_chunk, max_chunk);

    // Encode every chunk blit into ONE command buffer with a single
    // commit + wait at the end.  Per-chunk commit/waitUntilCompleted turned
    // a multi-chunk transfer into N CPU<->GPU round trips.
    std::lock_guard<std::recursive_mutex> dev_lock(mccl_device_ops_mutex());
    @autoreleasepool {
        NSMutableArray<id<MTLBuffer>>* wrappers = [NSMutableArray array];
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

        while (offset < nbytes) {
            size_t chunk = std::min(max_chunk, nbytes - offset);
            size_t aligned_chunk = (chunk + PAGE - 1) & ~(PAGE - 1);

            id<MTLBuffer> chunk_mtl = [cached_device()
                newBufferWithBytesNoCopy:dst_bytes + offset
                length:aligned_chunk
                options:MTLResourceStorageModeShared
                deallocator:nil];
            MCCL_CHECK(chunk_mtl != nil,
                       "chunked_blit_to_staging: MTLBuffer wrap failed at offset " +
                       std::to_string(offset) + " chunk=" + std::to_string(aligned_chunk));
            [wrappers addObject:chunk_mtl];  // keep alive until the GPU is done

            [blit copyFromBuffer:src_buf sourceOffset:src_offset + offset
                        toBuffer:chunk_mtl destinationOffset:0
                            size:chunk];
            offset += chunk;
        }

        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        check_command_buffer(cmd, "chunked_blit_to_staging(chunk)");
    }
}

/// Blit from CPU staging into GPU buffer, chunked for large buffers.
void chunked_blit_from_staging(const void* src, size_t nbytes,
                                id<MTLBuffer> dst_buf, size_t dst_offset) {
    StagingPool& pool = staging_pool();
    if (src == pool.ptr && pool.mtl_wrapper && nbytes <= pool.capacity) {
        std::lock_guard<std::recursive_mutex> dev_lock(mccl_device_ops_mutex());
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:pool.mtl_wrapper sourceOffset:0
                        toBuffer:dst_buf destinationOffset:dst_offset
                            size:nbytes];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check_command_buffer(cmd, "chunked_blit_from_staging(fast)");
        }
        return;
    }

    size_t max_chunk = metal_max_buffer_len();
    size_t offset = 0;
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);

    // Single command buffer for all chunks (see chunked_blit_to_staging).
    std::lock_guard<std::recursive_mutex> dev_lock(mccl_device_ops_mutex());
    @autoreleasepool {
        NSMutableArray<id<MTLBuffer>>* wrappers = [NSMutableArray array];
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

        while (offset < nbytes) {
            size_t chunk = std::min(max_chunk, nbytes - offset);
            size_t aligned_chunk = (chunk + PAGE - 1) & ~(PAGE - 1);

            id<MTLBuffer> chunk_mtl = [cached_device()
                newBufferWithBytesNoCopy:const_cast<uint8_t*>(src_bytes + offset)
                length:aligned_chunk
                options:MTLResourceStorageModeShared
                deallocator:nil];
            MCCL_CHECK(chunk_mtl != nil,
                       "chunked_blit_from_staging: MTLBuffer wrap failed at offset " +
                       std::to_string(offset));
            [wrappers addObject:chunk_mtl];

            [blit copyFromBuffer:chunk_mtl sourceOffset:0
                        toBuffer:dst_buf destinationOffset:dst_offset + offset
                            size:chunk];
            offset += chunk;
        }

        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        check_command_buffer(cmd, "chunked_blit_from_staging(chunk)");
    }
}

} // anonymous namespace


bool tensor_cpu_accessible(const at::Tensor& tensor) {
    id<MTLBuffer> buffer = at::mps::getMTLBufferStorage(tensor);
    return buffer != nil && buffer.storageMode == MTLStorageModeShared;
}

at::Tensor ensure_shared_storage(const at::Tensor& tensor) {
    if (tensor.is_cpu() || tensor_cpu_accessible(tensor)) {
        return tensor;
    }

    size_t nbytes = static_cast<size_t>(tensor.numel()) * tensor.element_size();
    id<MTLBuffer> src_buf = at::mps::getMTLBufferStorage(tensor);
    size_t src_offset = static_cast<size_t>(tensor.storage_offset()) * tensor.element_size();

    size_t alloc_size = (nbytes + PAGE - 1) & ~(PAGE - 1);

    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, PAGE, alloc_size);
    MCCL_CHECK(rc == 0 && ptr != nullptr,
               "ensure_shared_storage: posix_memalign failed for " + std::to_string(alloc_size) + " bytes");

    chunked_blit_to_staging(src_buf, src_offset, ptr, nbytes);

    MCCL_DEBUG("ensure_shared_storage: blit %zu bytes from private to shared", nbytes);

    auto deleter = [](void* p) { free(p); };
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        at::DataPtr(ptr, ptr, deleter, c10::Device(c10::kCPU)),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    return at::empty({0}, tensor.options().device(at::kCPU))
        .set_(storage, 0, tensor.sizes(), tensor.strides());
}

void* get_mtl_device() {
    return (__bridge void*)cached_device();
}

void* get_mccl_command_queue() {
    return (__bridge void*)cached_queue();
}

MPSBufferView wrap_cpu_tensor_as_mps_buffer(const at::Tensor& tensor) {
    MCCL_CHECK(tensor.is_cpu(), "wrap_cpu_tensor_as_mps_buffer requires CPU tensor");
    MCCL_CHECK(tensor.is_contiguous(), "CPU tensor must be contiguous");

    void* data_ptr = tensor.data_ptr();
    size_t nbytes = tensor_nbytes(tensor);

    // newBufferWithBytesNoCopy requires a page-aligned pointer and a
    // page-multiple length.  A generic CPU tensor is only malloc-aligned, so
    // wrap the containing page range and report the intra-page offset.
    // (Reading up to the page boundary past the allocation is safe: heap
    // pages are mapped; the GPU never writes outside [byte_offset, +nbytes).)
    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);
    uintptr_t base = addr & ~static_cast<uintptr_t>(PAGE - 1);
    size_t byte_offset = static_cast<size_t>(addr - base);
    size_t wrap_len = (byte_offset + nbytes + PAGE - 1) & ~(PAGE - 1);

    // Cache wrappers so (a) the returned view's buffer stays alive after this
    // scope (ARC would otherwise release it and the view would dangle) and
    // (b) repeated collectives on the same tensor don't allocate per call.
    static std::mutex cache_mu;
    static std::unordered_map<uintptr_t, std::pair<size_t, id<MTLBuffer>>> cache;
    constexpr size_t MAX_CACHED_WRAPPERS = 256;

    id<MTLBuffer> buffer = nil;
    {
        std::lock_guard<std::mutex> lock(cache_mu);
        auto it = cache.find(base);
        if (it != cache.end() && it->second.first >= wrap_len) {
            buffer = it->second.second;
        }
    }

    if (buffer == nil) {
        buffer = [cached_device() newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                    length:wrap_len
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        MCCL_CHECK(buffer != nil, "Failed to wrap CPU tensor as MTLBuffer");
        std::lock_guard<std::mutex> lock(cache_mu);
        if (cache.size() >= MAX_CACHED_WRAPPERS) cache.clear();
        cache[base] = {wrap_len, buffer};
    }

    MCCL_TRACE("wrap_cpu_tensor: data_ptr=%p nbytes=%zu base=%p offset=%zu",
               data_ptr, nbytes, reinterpret_cast<void*>(base), byte_offset);

    return MPSBufferView{
        .mtl_buffer     = (__bridge void*)buffer,
        .byte_offset    = byte_offset,
        .nbytes         = nbytes,
        .cpu_accessible = true,
        .cpu_ptr        = data_ptr,
    };
}

MPSBufferView extract_mps_buffer(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    if (tensor.is_cpu()) {
        return wrap_cpu_tensor_as_mps_buffer(tensor);
    }

    id<MTLBuffer> buffer = at::mps::getMTLBufferStorage(tensor);
    MCCL_CHECK(buffer != nil, "getMTLBufferStorage returned nil");

    size_t storage_offset_bytes =
        static_cast<size_t>(tensor.storage_offset()) * tensor.element_size();
    size_t nbytes = tensor_nbytes(tensor);

    // Check CPU accessibility via storage mode
    bool cpu_ok = (buffer.storageMode == MTLStorageModeShared);
    void* cpu_ptr = cpu_ok ? (static_cast<uint8_t*>(buffer.contents) + storage_offset_bytes) : nullptr;

    MCCL_TRACE("extract_mps_buffer: offset=%zu nbytes=%zu cpu_ok=%d",
               storage_offset_bytes, nbytes, (int)cpu_ok);

    return MPSBufferView{
        .mtl_buffer     = (__bridge void*)buffer,
        .byte_offset    = storage_offset_bytes,
        .nbytes         = nbytes,
        .cpu_accessible = cpu_ok,
        .cpu_ptr        = cpu_ptr,
    };
}

void mps_stream_sync() {
    torch::mps::synchronize();
}

void mps_stream_sync_after_cpu_mps_buffer_write() {
    // CPU writes into MTLStorageModeShared buffers are coherent on Apple
    // unified memory: any GPU kernel enqueued AFTER the collective completes
    // observes the new data, and kernels enqueued BEFORE the collective were
    // already ordered via the pre-collective MPS sync.  The previous
    // implementation called torch::mps::synchronize() here — a full-stream
    // drain (including unrelated forward/backward kernels) per collective
    // that serialized DDP buckets and capped GPU utilization.
    //
    // MCCL_CPU_WRITE_SYNC=full restores the old blocking behavior for
    // debugging suspected coherence/ordering issues.
    static const bool full_sync = [] {
        auto* v = std::getenv("MCCL_CPU_WRITE_SYNC");
        return v && std::string(v) == "full";
    }();
    if (full_sync) {
        mps_stream_sync();
    }
}

void mccl_queue_drain() {
    std::lock_guard<std::recursive_mutex> lock(mccl_device_ops_mutex());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void mps_sync() {
    mps_stream_sync();
    mccl_queue_drain();
}

uint64_t mps_event_sync_nonblocking() {
    static const bool force_stream_sync = [] {
        auto* v = std::getenv("MCCL_EVENT_SYNC");
        if (v && (std::string(v) == "0" || std::string(v) == "false" ||
                  std::string(v) == "no")) {
            MCCL_WARN("MCCL_EVENT_SYNC=0: skipping global MPS stream sync; "
                       "collective staging uses MCCL-queue blit fence");
            return true;
        }
        return false;
    }();

    if (!force_stream_sync && event_sync_available()) {
        uint64_t val = next_mps_event_value();
        commit_mps_and_signal(val);
        return val;
    }
    // No torch::mps::synchronize() from engine threads — stage_for_send_collective
    // fences tensor bytes via blit+wait on the MCCL queue.
    return 0;
}

void mps_event_sync() {
    uint64_t val = mps_event_sync_nonblocking();
    if (val > 0) {
        wait_for_mps(val);
    }
}

StagingBuffer stage_for_send(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    // Engine threads must not torch::mps::synchronize() (thread-unsafe vs autograd
    // encode → objc_release SIGSEGV).  When event sync is available the caller has
    // already waited on the producer's mps_event (wait_for_mps), so the gradient is
    // GPU-complete and we only drain MCCL's own queue.  Legacy fallback (no shared
    // events) keeps the blocking stream sync.
    if (!event_sync_available()) {
        mps_stream_sync();
    }
    mccl_queue_drain();

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send: direct CPU path, %zu bytes", view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    MCCL_DEBUG("stage_for_send: blit fallback for %zu bytes", view.nbytes);

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    void* staging = pool.ensure(view.nbytes, cached_device());
    chunked_blit_to_staging(src_buf, view.byte_offset, staging, view.nbytes);

    return StagingBuffer{staging, view.nbytes};
}

StagingBuffer stage_for_send_nosync(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send_nosync: direct CPU path, %zu bytes", view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    MCCL_DEBUG("stage_for_send_nosync: blit fallback for %zu bytes", view.nbytes);

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    void* staging = pool.ensure(view.nbytes, cached_device());
    chunked_blit_to_staging(src_buf, view.byte_offset, staging, view.nbytes);

    return StagingBuffer{staging, view.nbytes};
}

StagingBuffer stage_for_send_collective(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    MPSBufferView view = extract_mps_buffer(tensor);

    // CPU metadata tensors (DDP verify int64, broadcast_object_list sizes):
    // read data_ptr directly — no MPS blit or shared StagingPool hop.
    if (tensor.is_cpu() && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send_collective: CPU tensor direct path, %zu bytes",
                   view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    // See stage_for_send: never torch::mps::synchronize() from engine threads when
    // shared events are available — the caller waited on the producer mps_event.
    if (!event_sync_available()) {
        mps_stream_sync();
    }
    mccl_queue_drain();

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    void* staging = pool.ensure(view.nbytes, cached_device());
    chunked_blit_to_staging(src_buf, view.byte_offset, staging, view.nbytes);
    mccl_queue_drain();
    MCCL_TRACE("stage_for_send_collective: blit+wait %zu bytes", view.nbytes);
    return StagingBuffer{staging, view.nbytes};
}

void blit_tensor_to_buffer(const at::Tensor& tensor, void* dst) {
    MPSBufferView view = extract_mps_buffer(tensor);
    // MPS tensors must blit from the device buffer; cpu_ptr may be stale for
    // Metal consumers (mirrors stage_for_send_collective send-side policy).
    if (tensor.is_cpu() && view.cpu_accessible && view.cpu_ptr) {
        memcpy(dst, view.cpu_ptr, view.nbytes);
        return;
    }
    MCCL_CHECK((reinterpret_cast<uintptr_t>(dst) & (PAGE - 1)) == 0,
               "blit_tensor_to_buffer: dst must be page-aligned");
    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    chunked_blit_to_staging(src_buf, view.byte_offset, dst, view.nbytes);
}

void blit_buffer_to_tensor(const void* src, const at::Tensor& tensor) {
    MPSBufferView view = extract_mps_buffer(tensor);
    // MPS tensors must blit into the device buffer; cpu_ptr memcpy leaves
    // Metal kernels reading stale GPU memory (breaks ring allreduce recv).
    if (tensor.is_cpu() && view.cpu_accessible && view.cpu_ptr) {
        memcpy(view.cpu_ptr, src, view.nbytes);
        return;
    }

    const void* blit_src = src;
    StagingPool& pool = staging_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    // Pooled wire buffers are 64-byte aligned; Metal blit requires PAGE.
    if ((reinterpret_cast<uintptr_t>(src) & (PAGE - 1)) != 0 || src != pool.ptr) {
        void* staging = pool.ensure(view.nbytes, cached_device());
        memcpy(staging, src, view.nbytes);
        blit_src = staging;
    }

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    chunked_blit_from_staging(blit_src, view.nbytes, dst_buf, view.byte_offset);
}

void unstage_from_recv(const at::Tensor& tensor, const void* src, size_t nbytes,
                       bool cpu_unified_stage) {
    check_single_tensor(tensor);
    MCCL_CHECK(nbytes == tensor_nbytes(tensor),
               "unstage size mismatch");

    MPSBufferView view = extract_mps_buffer(tensor);

    if (tensor.is_cpu() && view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("unstage_from_recv: direct memcpy path, %zu bytes", nbytes);
        memcpy(view.cpu_ptr, src, nbytes);
        return;
    }

    // fp32 unified MPS: memcpy into shared cpu_ptr for vDSP reduce (wire
    // staging).  Avoids blit+sync from ProgressEngine threads (can hang on
    // torch::mps::synchronize).  stage_for_send / copy_ callers sync later.
    if (cpu_unified_stage && view.cpu_accessible && view.cpu_ptr &&
        tensor.scalar_type() == at::kFloat) {
        MCCL_TRACE("unstage_from_recv: unified fp32 cpu staging, %zu bytes", nbytes);
        memcpy(view.cpu_ptr, src, nbytes);
        return;
    }

    MCCL_DEBUG("unstage_from_recv: blit path for %zu bytes", nbytes);

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;

    // Ensure staging pool has the data (may need to copy if src isn't the pool)
    StagingPool& pool = staging_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    const void* blit_src = src;
    if (src != pool.ptr) {
        void* staging = pool.ensure(nbytes, cached_device());
        memcpy(staging, src, nbytes);
        blit_src = staging;
    }

    chunked_blit_from_staging(blit_src, nbytes, dst_buf, view.byte_offset);
    mccl_queue_drain();
}

} // namespace mccl
