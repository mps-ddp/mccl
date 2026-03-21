#pragma once

#include <cstdint>
#include <atomic>

namespace mccl {

/// MPS / MCCL coordination helpers (MTLSharedEvent counters + bookkeeping).
///
/// PyTorch does not expose a safe way to encode completion on MPS command
/// buffers from outside the framework, so ``commit_mps_and_signal`` uses
/// ``torch::mps::synchronize()`` and then bumps ``signaledValue`` on the CPU.
/// That matches the actual ordering guarantee (full stream drain), not a
/// non-blocking GPU signal.
///
/// ``signal_mccl_done`` / ``wait_for_mccl`` exist for future shader paths;
/// collectives today still rely on op completion before returning.

/// Initialize the shared event infrastructure. Called once from
/// metal_kernels_init() or ProcessGroupMCCL constructor.
void event_sync_init();

/// True after event_sync_init() succeeds (requires macOS 10.14+ and a
/// valid MTLDevice).
bool event_sync_available();

/// Drain PyTorch's MPS stream (``torch::mps::synchronize``), then set
/// ``mps_event.signaledValue`` to ``value`` for bookkeeping / future use.
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
