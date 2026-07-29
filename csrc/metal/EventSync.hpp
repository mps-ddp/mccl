#pragma once

#include <cstdint>
#include <atomic>

namespace mccl {

/// MPS / MCCL coordination helpers (MTLSharedEvent counters + bookkeeping).
///
/// Each MTLSharedEvent has its own monotonic counter.  Sharing one counter
/// across mps_event, fence_event, and mccl_event caused wait_for_mps(N) to
/// hang when fence steps consumed N-1..N without encoding on mps_event.

void event_sync_init();
bool event_sync_available();

void commit_mps_and_signal(uint64_t value);
void wait_for_mps(uint64_t value);

/// CPU reduction path completion (atomic; does not mutate mccl_event).
void signal_mccl_done(uint64_t value);
void signal_mccl_done_gpu(uint64_t value);
void wait_for_mccl(uint64_t value);

/// NCCL ``ncclEndEvent_->block(currentStream)`` analogue: enqueue a wait on the
/// PyTorch MPS stream so subsequent MPS kernels cannot start until ``mccl_event``
/// reaches ``value`` (signaled after prior MCCL-queue work completes).
void block_mps_on_mccl(uint64_t value);

void signal_mccl_fence_gpu(uint64_t value);
void wait_for_mccl_fence(uint64_t value);

uint64_t next_mps_event_value();
uint64_t next_fence_event_value();
uint64_t next_mccl_event_value();

/// Deprecated: aliases next_mps_event_value(). Do not use for fence/mccl signals.
uint64_t next_event_value();

/// Engine thread: GPU+CPU release after collective writes land in output tensors.
/// Returns 0 when overlap/event sync is off (consumer uses legacy drain in Work::wait).
uint64_t publish_collective_release(bool overlap_comm);

} // namespace mccl
