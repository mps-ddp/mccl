#pragma once

#include <cstdint>
#include <atomic>

namespace mccl {

/// MPS / MCCL coordination helpers (MTLSharedEvent counters + bookkeeping).
///
/// ``commit_mps_and_signal`` encodes ``encodeSignalEvent`` on PyTorch's active
/// MPS command buffer (via ``torch::mps::get_dispatch_queue`` /
/// ``get_command_buffer``), then ``commit()``; ``wait_for_mps`` polls until the
/// GPU reaches that signal. This replaces a full ``torch::mps::synchronize()``
/// drain while still ordering MCCL reads after GPU-produced tensor data.
///
/// ``signal_mccl_done`` / ``wait_for_mccl`` exist for future shader paths;
/// collectives today still rely on op completion before returning.

/// Initialize the shared event infrastructure. Called once from
/// metal_kernels_init() or ProcessGroupMCCL constructor.
void event_sync_init();

/// True after event_sync_init() succeeds (requires macOS 10.14+ and a
/// valid MTLDevice).
bool event_sync_available();

/// Encode ``value`` on PyTorch's MPS command buffer and commit; GPU completion
/// is observed via ``wait_for_mps``.
void commit_mps_and_signal(uint64_t value);

/// Poll until ``mps_event.signaledValue >= value`` (used when another
/// producer signals the same event; same-thread use after commit is redundant).
void wait_for_mps(uint64_t value);

/// Signal from MCCL's side that the reduced gradients have been written.
/// For the f32 CPU path this is a simple atomic store (unified memory is
/// coherent).
void signal_mccl_done(uint64_t value);

/// GPU-side signal on MCCL's queue (reserved for shader-heavy paths; unused today).
void signal_mccl_done_gpu(uint64_t value);

/// Block until MCCL's signal reaches >= value.  Used by the non-f32 path
/// where Metal shaders wrote the reduction result and the next MPS command
/// buffer needs to wait.
void wait_for_mccl(uint64_t value);

/// Monotonically increasing event counter. Each collective bumps this to
/// generate unique signal values.
uint64_t next_event_value();

} // namespace mccl
