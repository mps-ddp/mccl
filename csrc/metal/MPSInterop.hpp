#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <cstdint>

namespace mccl {

struct MPSBufferView {
    void* mtl_buffer;       // id<MTLBuffer> — stored as void* for C++ header
    size_t byte_offset;
    size_t nbytes;
    bool cpu_accessible;
    void* cpu_ptr;          // nullable; non-null only if cpu_accessible
};

/// Extract the underlying Metal buffer from an MPS tensor.
/// Tensor MUST be contiguous and on MPS device.
/// Returns a view with buffer handle, offset, and CPU-accessibility info.
MPSBufferView extract_mps_buffer(const at::Tensor& tensor);

/// Synchronize MPS command queue — blocks until all enqueued MPS work completes.
/// Must be called before reading MPS tensor data from CPU / network.
void mps_sync();

/// Lightweight sync: only flushes the PyTorch MPS stream without draining
/// the MCCL command queue. Use at the start of a collective when no MCCL
/// Metal compute work is pending.
void mps_stream_sync();

/// Drain only the MCCL command queue (blocks until all committed MCCL
/// command buffers complete). Does NOT flush the PyTorch MPS stream.
void mccl_queue_drain();

/// Event-based MPS sync for compute-communication overlap.
/// Non-blocking: encode signal + commit on PyTorch's MPS command buffer.
/// Returns the event value to wait on (0 if fell back to blocking sync).
uint64_t mps_event_sync_nonblocking();

/// Blocking version: encode signal + commit, then wait for GPU completion.
/// Falls back to plain mps_stream_sync() if event sync is unavailable.
void mps_event_sync();

/// Stage MPS tensor to a CPU-accessible buffer for network send.
/// Returns a pinned host pointer and the byte count.
/// Caller does NOT own the memory — it is valid until next staging call.
struct StagingBuffer {
    void* data;
    size_t nbytes;
};

StagingBuffer stage_for_send(const at::Tensor& tensor);

/// Like stage_for_send but skips the internal mps_sync().
/// Caller MUST ensure all GPU work on the tensor is already flushed.
StagingBuffer stage_for_send_nosync(const at::Tensor& tensor);

/// Unstage received bytes back into an MPS tensor's buffer.
/// Handles the CPU→GPU direction (write-back after network receive).
void unstage_from_recv(const at::Tensor& tensor, const void* src, size_t nbytes);

/// Thread-safe variant of unstage_from_recv that never touches the global
/// StagingPool singleton.  For CPU-accessible tensors this is a plain memcpy.
/// For private-storage tensors the caller must supply a page-aligned staging
/// buffer (e.g. a PooledBuffer) that the blit encoder can wrap.
void unstage_from_recv_threadsafe(const at::Tensor& tensor, const void* src,
                                  size_t nbytes, void* staging_buf,
                                  size_t staging_capacity);

/// Returns true if the tensor's underlying MTLBuffer uses shared storage,
/// meaning the CPU can read/write it directly without blit staging.
/// Performs a lightweight runtime check (no sync, no copy).
bool tensor_cpu_accessible(const at::Tensor& tensor);

/// If the tensor uses private Metal storage, copy it into a new tensor
/// backed by shared (cpu_accessible) storage and return that. If already
/// shared, returns the original tensor with no copy. Caller must have
/// already synced MPS before calling (data must be committed to the buffer).
at::Tensor ensure_shared_storage(const at::Tensor& tensor);

/// Get the default MTLDevice as void* (id<MTLDevice>).
void* get_mtl_device();

/// Get or create a dedicated MTLCommandQueue for MCCL operations.
void* get_mccl_command_queue();

} // namespace mccl
