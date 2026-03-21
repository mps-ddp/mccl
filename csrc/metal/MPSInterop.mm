#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>
#import <torch/mps.h>

#include <cstdlib>

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

// Staging buffer — reused across calls to avoid repeated allocation.
// Thread-safety: only called from the progress engine's single thread.
struct StagingPool {
    void* ptr = nullptr;
    size_t capacity = 0;
    id<MTLBuffer> mtl_wrapper = nil;

    void* ensure(size_t nbytes, id<MTLDevice> device) {
        if (nbytes > capacity) {
            mtl_wrapper = nil;
            free(ptr);
            ptr = nullptr;

            size_t page = 16384;
            capacity = ((nbytes + (nbytes >> 2)) + page - 1) & ~(page - 1);
            int rc = posix_memalign(&ptr, page, capacity);
            MCCL_CHECK(rc == 0 && ptr != nullptr, "Staging buffer allocation failed");

            // Pre-create the MTLBuffer wrapper once (avoids per-call churn)
            mtl_wrapper = [device newBufferWithBytesNoCopy:ptr
                                                   length:capacity
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
            MCCL_CHECK(mtl_wrapper != nil, "Staging MTLBuffer creation failed");
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

} // anonymous namespace


bool tensor_cpu_accessible(const at::Tensor& tensor) {
    id<MTLBuffer> buffer = at::mps::getMTLBufferStorage(tensor);
    return buffer != nil && buffer.storageMode == MTLStorageModeShared;
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

void mps_event_sync() {
    if (event_sync_available()) {
        uint64_t val = next_event_value();
        commit_mps_and_signal(val);
        wait_for_mps(val);
    } else {
        mps_stream_sync();
    }
}

StagingBuffer stage_for_send(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    // Flush PyTorch MPS stream and drain any pending MCCL compute work
    mps_stream_sync();
    mccl_queue_drain();

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send: direct CPU path, %zu bytes", view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    // Fallback: blit to staging buffer
    MCCL_DEBUG("stage_for_send: blit fallback for %zu bytes", view.nbytes);

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
    void* staging = pool.ensure(view.nbytes, cached_device());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:src_buf
                sourceOffset:view.byte_offset
                    toBuffer:pool.mtl_wrapper
           destinationOffset:0
                        size:view.nbytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return StagingBuffer{staging, view.nbytes};
}

StagingBuffer stage_for_send_nosync(const at::Tensor& tensor) {
    check_single_tensor(tensor);

    MPSBufferView view = extract_mps_buffer(tensor);

    if (view.cpu_accessible && view.cpu_ptr) {
        MCCL_TRACE("stage_for_send_nosync: direct CPU path, %zu bytes", view.nbytes);
        return StagingBuffer{view.cpu_ptr, view.nbytes};
    }

    // Fallback: blit to staging buffer (still needs queue sync for the blit itself)
    MCCL_DEBUG("stage_for_send_nosync: blit fallback for %zu bytes", view.nbytes);

    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
    StagingPool& pool = staging_pool();
    void* staging = pool.ensure(view.nbytes, cached_device());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:src_buf
                sourceOffset:view.byte_offset
                    toBuffer:pool.mtl_wrapper
           destinationOffset:0
                        size:view.nbytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

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

    // Fallback: blit from staging into GPU buffer
    MCCL_DEBUG("unstage_from_recv: blit fallback for %zu bytes", nbytes);

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;

    // For unstage, we need to wrap the source pointer.
    // Use the staging pool if the pointer is our pool, otherwise wrap fresh.
    StagingPool& pool = staging_pool();
    id<MTLBuffer> src_mtl = nil;

    if (src == pool.ptr && pool.mtl_wrapper && nbytes <= pool.capacity) {
        src_mtl = pool.mtl_wrapper;
    } else {
        // Page-align check: newBufferWithBytesNoCopy requires page-aligned ptr
        // If not page-aligned, do a plain memcpy to the staging pool first
        uintptr_t addr = reinterpret_cast<uintptr_t>(src);
        if ((addr & 0x3FFF) != 0) {
            void* aligned = pool.ensure(nbytes, cached_device());
            memcpy(aligned, src, nbytes);
            src_mtl = pool.mtl_wrapper;
        } else {
            size_t page = 16384;
            size_t aligned_len = (nbytes + page - 1) & ~(page - 1);
            src_mtl = [cached_device()
                newBufferWithBytesNoCopy:const_cast<void*>(src)
                length:aligned_len
                options:MTLResourceStorageModeShared
                deallocator:nil];
            MCCL_CHECK(src_mtl != nil, "newBufferWithBytesNoCopy failed for unstage buffer");
        }
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [cached_queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:src_mtl
                sourceOffset:0
                    toBuffer:dst_buf
           destinationOffset:view.byte_offset
                        size:nbytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace mccl
