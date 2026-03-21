#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>
#import <torch/mps.h>

#include <cstdlib>
#include <algorithm>

#include "metal/MPSInterop.hpp"
#include "metal/EventSync.hpp"
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
// Thread-safety: one collective at a time per process (ProgressEngine serializes I/O).
struct StagingPool {
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
    if (pool.mtl_wrapper && nbytes <= pool.capacity) {
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

    while (offset < nbytes) {
        size_t chunk = std::min(max_chunk, nbytes - offset);
        size_t aligned_chunk = (chunk + PAGE - 1) & ~(PAGE - 1);

        @autoreleasepool {
            id<MTLBuffer> chunk_mtl = [cached_device()
                newBufferWithBytesNoCopy:dst_bytes + offset
                length:aligned_chunk
                options:MTLResourceStorageModeShared
                deallocator:nil];
            MCCL_CHECK(chunk_mtl != nil,
                       "chunked_blit_to_staging: MTLBuffer wrap failed at offset " +
                       std::to_string(offset) + " chunk=" + std::to_string(aligned_chunk));

            id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:src_buf sourceOffset:src_offset + offset
                        toBuffer:chunk_mtl destinationOffset:0
                            size:chunk];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check_command_buffer(cmd, "chunked_blit_to_staging(chunk)");
        }
        offset += chunk;
    }
}

/// Blit from CPU staging into GPU buffer, chunked for large buffers.
void chunked_blit_from_staging(const void* src, size_t nbytes,
                                id<MTLBuffer> dst_buf, size_t dst_offset) {
    StagingPool& pool = staging_pool();
    if (src == pool.ptr && pool.mtl_wrapper && nbytes <= pool.capacity) {
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

    while (offset < nbytes) {
        size_t chunk = std::min(max_chunk, nbytes - offset);
        size_t aligned_chunk = (chunk + PAGE - 1) & ~(PAGE - 1);

        @autoreleasepool {
            id<MTLBuffer> chunk_mtl = [cached_device()
                newBufferWithBytesNoCopy:const_cast<uint8_t*>(src_bytes + offset)
                length:aligned_chunk
                options:MTLResourceStorageModeShared
                deallocator:nil];
            MCCL_CHECK(chunk_mtl != nil,
                       "chunked_blit_from_staging: MTLBuffer wrap failed at offset " +
                       std::to_string(offset));

            id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:chunk_mtl sourceOffset:0
                        toBuffer:dst_buf destinationOffset:dst_offset + offset
                            size:chunk];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check_command_buffer(cmd, "chunked_blit_from_staging(chunk)");
        }
        offset += chunk;
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
    
    id<MTLDevice> device = cached_device();
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:data_ptr
                                                      length:nbytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    MCCL_CHECK(buffer != nil, "Failed to wrap CPU tensor as MTLBuffer");
    
    MCCL_TRACE("wrap_cpu_tensor: data_ptr=%p nbytes=%zu", data_ptr, nbytes);
    
    return MPSBufferView{
        .mtl_buffer     = (__bridge void*)buffer,
        .byte_offset    = 0,
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

void mccl_queue_drain() {
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
            MCCL_WARN("MCCL_EVENT_SYNC=0: MTLSharedEvent path disabled, "
                       "falling back to mps_stream_sync");
            return true;
        }
        return false;
    }();

    if (!force_stream_sync && event_sync_available()) {
        uint64_t val = next_event_value();
        commit_mps_and_signal(val);
        return val;
    } else {
        mps_stream_sync();
        return 0;
    }
}

void mps_event_sync() {
    uint64_t val = mps_event_sync_nonblocking();
    if (val > 0) {
        wait_for_mps(val);
    }
}

StagingBuffer stage_for_send(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    mps_stream_sync();
    mccl_queue_drain();

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send: direct CPU path, %zu bytes", view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    MCCL_DEBUG("stage_for_send: blit fallback for %zu bytes", view.nbytes);

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
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
    void* staging = pool.ensure(view.nbytes, cached_device());
    chunked_blit_to_staging(src_buf, view.byte_offset, staging, view.nbytes);

    return StagingBuffer{staging, view.nbytes};
}

void unstage_from_recv(const at::Tensor& tensor, const void* src, size_t nbytes) {
    check_single_tensor(tensor);
    MCCL_CHECK(nbytes == tensor_nbytes(tensor),
               "unstage size mismatch");

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("unstage_from_recv: direct memcpy path, %zu bytes", nbytes);
        memcpy(view.cpu_ptr, src, nbytes);
        return;
    }

    MCCL_DEBUG("unstage_from_recv: blit fallback for %zu bytes", nbytes);

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;

    // Ensure staging pool has the data (may need to copy if src isn't the pool)
    StagingPool& pool = staging_pool();
    const void* blit_src = src;
    if (src != pool.ptr) {
        void* staging = pool.ensure(nbytes, cached_device());
        memcpy(staging, src, nbytes);
        blit_src = staging;
    }

    chunked_blit_from_staging(blit_src, nbytes, dst_buf, view.byte_offset);
}

} // namespace mccl
