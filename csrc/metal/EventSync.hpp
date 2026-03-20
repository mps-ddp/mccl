#pragma once

#include <cstdint>
#include <atomic>

namespace mccl {

/// Lightweight GPU/CPU synchronization built on MTLSharedEvent.
///
/// Provides fine-grained dependency tracking between PyTorch's MPS command
/// queue and MCCL's own command queue, replacing the heavyweight
/// mps_stream_sync() (which drains the entire MPS pipeline).
///
/// Typical flow for an allreduce:
///   1. commit_mps_and_signal(seq)  -- flush MPS work, signal when done
///   2. wait_for_mps(seq)           -- CPU blocks until gradients are ready
///   3. ... do allreduce (vDSP + network) ...
///   4. signal_mccl_done(seq)       -- mark reduced gradients as written
///   5. (next iteration) PyTorch reads gradients -- no sync needed for CPU
///      path because unified memory writes are immediately visible.

/// Initialize the shared event infrastructure. Called once from
/// metal_kernels_init() or ProcessGroupMCCL constructor.
void event_sync_init();

/// True after event_sync_init() succeeds (requires macOS 10.14+ and a
/// valid MTLDevice).
bool event_sync_available();

/// Flush PyTorch's MPS command stream and encode a signal with the given
/// value on the committed command buffer. The signal fires when all
/// previously-enqueued MPS GPU work completes.
///
/// This is non-blocking from the CPU's perspective -- it submits work to
/// the GPU and returns immediately.
void commit_mps_and_signal(uint64_t value);

/// Block the calling CPU thread until the shared event reaches >= value.
/// Uses polling on event.signaledValue with exponential back-off for low
/// latency without burning a core.
void wait_for_mps(uint64_t value);

/// Signal from MCCL's side that the reduced gradients have been written.
/// For the f32 CPU path this is a simple atomic store (unified memory is
/// coherent).
void signal_mccl_done(uint64_t value);

/// GPU-side signal variant: encodes a signal on MCCL's Metal command
/// queue so the GPU signals completion after shader-based reductions.
void signal_mccl_done_gpu(uint64_t value);

/// Block until MCCL's signal reaches >= value.  Used by the non-f32 path
/// where Metal shaders wrote the reduction result and the next MPS command
/// buffer needs to wait.
void wait_for_mccl(uint64_t value);

/// Monotonically increasing event counter. Each collective bumps this to
/// generate unique signal values.
uint64_t next_event_value();

} // namespace mccl
