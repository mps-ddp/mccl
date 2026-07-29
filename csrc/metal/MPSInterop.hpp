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

/// CPU pointer for contiguous MPS (shared) or CPU tensors — allows complex64 for STFT.
/// Returns nullptr when MPS uses private storage (caller should fall back to copy_/cpu()).
void* shared_cpu_data_ptr(const at::Tensor& tensor);

/// Synchronize MPS command queue — blocks until all enqueued MPS work completes.
/// Must be called before reading MPS tensor data from CPU / network.
void mps_sync();

/// Lightweight sync: only flushes the PyTorch MPS stream without draining
/// the MCCL command queue. Use at the start of a collective when no MCCL
/// Metal compute work is pending.
void mps_stream_sync();

/// After CPU stores into an MPS tensor's shared MTLBuffer (cpu_ptr), ensure the
/// PyTorch default MPS stream observes those bytes before the next GPU kernel
/// reads the tensor. Call at end of CPU-finalized collectives on the engine
/// thread (not from Work::wait).
void mps_stream_sync_after_cpu_mps_buffer_write();

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

/// Collective send staging: always blit+wait on MCCL queue (safe under DDP overlap).
StagingBuffer stage_for_send_collective(const at::Tensor& tensor);

/// Unstage received bytes back into an MPS tensor's buffer.
/// cpu_unified_stage: memcpy into shared cpu_ptr for fp32 vDSP reduce (no GPU
/// blit); callers that need GPU-visible data (broadcast copy_) leave this false.
void unstage_from_recv(const at::Tensor& tensor, const void* src, size_t nbytes,
                       bool cpu_unified_stage = false);

/// Copy a (possibly private-storage) tensor's bytes into a caller-owned,
/// PAGE-ALIGNED host buffer (e.g. a PooledBuffer).  Shared storage: memcpy;
/// private: blit.  Unlike the StagingPool paths, the destination is owned by
/// the caller, so concurrent collectives never share staging memory.
void blit_tensor_to_buffer(const at::Tensor& tensor, void* dst);

/// Inverse: copy a caller-owned, PAGE-ALIGNED host buffer into a tensor.
void blit_buffer_to_tensor(const void* src, const at::Tensor& tensor);

/// Returns true if the tensor's underlying MTLBuffer uses shared storage,
/// meaning the CPU can read/write it directly without blit staging.
/// Performs a lightweight runtime check (no sync, no copy).
bool tensor_cpu_accessible(const at::Tensor& tensor);

/// Human-readable MTLStorageMode for an MPS tensor ("cpu", "private", "shared",
/// "managed", "none"). Used by Python tests to document why DDP grads cannot use
/// the wire cpu_ptr fast path.
std::string mps_storage_mode_string(const at::Tensor& tensor);

/// True when ``stage_for_send_collective`` blits instead of reading cpu_ptr.
bool collective_send_uses_blit(const at::Tensor& tensor);

/// True when ``stage_for_send`` blits instead of reading cpu_ptr.
bool stage_for_send_uses_blit(const at::Tensor& tensor);

/// Opt-in: ``MCCL_UNIFIED_COLLECTIVE=1`` — after producer MPS fence, send from
/// shared ``cpu_ptr`` and recv/reduce on unified buffer when safe (torch 2.12+).
bool unified_collective_enabled();

/// Unified send/recv + Metal reduce (not vDSP CPU reduce).
bool unified_metal_collective_path(const at::Tensor& tensor);

/// When unified, returns ``cpu_ptr`` for direct TCP recv; else nullptr.
void* tensor_wire_recv_ptr(const at::Tensor& tensor);

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
