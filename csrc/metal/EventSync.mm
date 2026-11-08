#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>
#import <torch/mps.h>

#include "metal/EventSync.hpp"
#include "metal/MPSInterop.hpp"
#include "runtime/MCCLDeviceMutex.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <atomic>
#include <thread>

namespace mccl {

namespace {

struct EventState {
    id<MTLSharedEvent> mps_event   = nil;
    id<MTLSharedEvent> mccl_event  = nil;
    id<MTLSharedEvent> fence_event = nil;
    id<MTLSharedEvent> p2p_event   = nil;
    id<MTLDevice>      device      = nil;
    id<MTLCommandQueue> mccl_queue = nil;
    std::atomic<uint64_t> mps_counter{0};
    std::atomic<uint64_t> fence_counter{0};
    std::atomic<uint64_t> mccl_counter{0};
    std::atomic<uint64_t> mccl_cpu_done{0};
    std::atomic<uint64_t> p2p_counter{0};
    std::mutex p2p_mu;
    id<MTLSharedEvent> a2a_event   = nil;  // created on first stream-ordered all_to_all
    std::atomic<uint64_t> a2a_counter{0};
    std::mutex a2a_mu;
    std::atomic<bool> initialized{false};
};

EventState& state() {
    static EventState s;
    return s;
}

void spin_wait_event(id<MTLSharedEvent> event, uint64_t target) {
    constexpr int FAST_SPINS = 200;
    constexpr auto TIMEOUT = std::chrono::seconds(30);

    for (int i = 0; i < FAST_SPINS; ++i) {
        if (event.signaledValue >= target) return;
    }

    auto deadline = std::chrono::steady_clock::now() + TIMEOUT;

    if (@available(macOS 12.0, *)) {
        while (event.signaledValue < target) {
            [event waitUntilSignaledValue:target timeoutMS:100];
            MCCL_CHECK(std::chrono::steady_clock::now() < deadline ||
                           event.signaledValue >= target,
                       "spin_wait_event timed out after 30s waiting for event value " +
                       std::to_string(target) + " (current=" +
                       std::to_string(event.signaledValue) + ")");
        }
        return;
    }

    auto delay = std::chrono::microseconds(10);
    constexpr auto max_delay = std::chrono::microseconds(500);
    while (event.signaledValue < target) {
        std::this_thread::sleep_for(delay);
        if (delay < max_delay) {
            delay *= 2;
        }
        MCCL_CHECK(std::chrono::steady_clock::now() < deadline,
                   "spin_wait_event timed out after 30s waiting for event value " +
                   std::to_string(target) + " (current=" +
                   std::to_string(event.signaledValue) + ")");
    }
}

} // anonymous namespace


void event_sync_init() {
    EventState& s = state();
    if (s.initialized.load(std::memory_order_acquire)) return;

    @autoreleasepool {
        s.device = (__bridge id<MTLDevice>)get_mtl_device();
        if (!s.device) {
            MCCL_WARN("EventSync: no Metal device, event sync disabled");
            return;
        }

        s.mccl_queue = (__bridge id<MTLCommandQueue>)get_mccl_command_queue();

        s.mps_event = [s.device newSharedEvent];
        if (!s.mps_event) {
            MCCL_WARN("EventSync: MTLSharedEvent creation failed (mps_event)");
            return;
        }

        s.mccl_event = [s.device newSharedEvent];
        if (!s.mccl_event) {
            MCCL_WARN("EventSync: MTLSharedEvent creation failed (mccl_event)");
            s.mps_event = nil;
            return;
        }

        s.fence_event = [s.device newSharedEvent];
        if (!s.fence_event) {
            MCCL_WARN("EventSync: MTLSharedEvent creation failed (fence_event)");
            s.mps_event = nil;
            s.mccl_event = nil;
            return;
        }

        s.p2p_event = [s.device newSharedEvent];
        if (!s.p2p_event) {
            MCCL_WARN("EventSync: MTLSharedEvent creation failed (p2p_event)");
            s.mps_event = nil;
            s.mccl_event = nil;
            s.fence_event = nil;
            return;
        }

        s.mps_counter.store(0);
        s.fence_counter.store(0);
        s.mccl_counter.store(0);
        s.mccl_cpu_done.store(0);
        s.p2p_counter.store(0);
        s.initialized.store(true, std::memory_order_release);
        MCCL_INFO("EventSync initialized (per-event MTLSharedEvent counters)");
    }
}

bool event_sync_available() {
    return state().initialized.load(std::memory_order_acquire);
}

void commit_mps_and_signal(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");

    dispatch_sync(
        (dispatch_queue_t)torch::mps::get_dispatch_queue(), ^{
            id<MTLCommandBuffer> cmd =
                (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
            [cmd encodeSignalEvent:s.mps_event value:value];
            torch::mps::commit();
        });
}

void wait_for_mps(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    spin_wait_event(s.mps_event, value);
}

void signal_mccl_done(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    uint64_t prev = s.mccl_cpu_done.load(std::memory_order_relaxed);
    while (value > prev &&
           !s.mccl_cpu_done.compare_exchange_weak(
               prev, value, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void signal_mccl_done_gpu(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");

    std::lock_guard<std::recursive_mutex> lock(mccl_device_ops_mutex());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [s.mccl_queue commandBuffer];
        cmd.label = @"mccl_signal_done";
        [cmd encodeSignalEvent:s.mccl_event value:value];
        [cmd commit];
    }
}

void wait_for_mccl(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    if (s.mccl_cpu_done.load(std::memory_order_acquire) >= value) {
        return;
    }
    spin_wait_event(s.mccl_event, value);
}

void block_mps_on_mccl(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    if (value == 0) {
        return;
    }
    dispatch_sync(
        (dispatch_queue_t)torch::mps::get_dispatch_queue(), ^{
            id<MTLCommandBuffer> cmd =
                (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
            [cmd encodeWaitForEvent:s.mccl_event value:value];
            torch::mps::commit();
        });
}

void signal_mccl_fence_gpu(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");

    std::lock_guard<std::recursive_mutex> lock(mccl_device_ops_mutex());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [s.mccl_queue commandBuffer];
        cmd.label = @"mccl_kernel_fence";
        [cmd encodeSignalEvent:s.fence_event value:value];
        [cmd commit];
    }
}

void wait_for_mccl_fence(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    spin_wait_event(s.fence_event, value);
}

uint64_t next_mps_event_value() {
    return state().mps_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t next_fence_event_value() {
    return state().fence_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t next_mccl_event_value() {
    return state().mccl_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t next_event_value() {
    return next_mps_event_value();
}

uint64_t publish_collective_release(bool overlap_comm) {
    if (!overlap_comm || !event_sync_available()) {
        return 0;
    }
    uint64_t v = next_mccl_event_value();
  // GPU signal: blits/reduces on MCCL queue are visible before the autograd thread
  // resumes PyTorch MPS encode (DDP Work::wait consumer fence — NCCL stream semantics).
    signal_mccl_done_gpu(v);
    signal_mccl_done(v);
    return v;
}

uint64_t next_p2p_event_value() {
    return state().p2p_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

void block_mps_on_p2p(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    if (value == 0) {
        return;
    }
    dispatch_sync(
        (dispatch_queue_t)torch::mps::get_dispatch_queue(), ^{
            id<MTLCommandBuffer> cmd =
                (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
            [cmd encodeWaitForEvent:s.p2p_event value:value];
        });
}

void signal_p2p_done_cpu(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    std::atomic_thread_fence(std::memory_order_release);
    std::lock_guard<std::mutex> lock(s.p2p_mu);
    if (s.p2p_event.signaledValue < value) {
        s.p2p_event.signaledValue = value;
    }
}

void wait_for_p2p(uint64_t value) {
    EventState& s = state();
    MCCL_CHECK(s.initialized, "EventSync not initialized");
    if (value == 0) return;
    spin_wait_event(s.p2p_event, value);
}

uint64_t next_a2a_event_value() {
    EventState& s = state();
    if (!s.initialized.load(std::memory_order_acquire)) return 0;
    {
        std::lock_guard<std::mutex> lock(s.a2a_mu);
        if (!s.a2a_event) {
            s.a2a_event = [s.device newSharedEvent];
            if (!s.a2a_event) {
                MCCL_WARN("EventSync: MTLSharedEvent creation failed (a2a_event)");
                return 0;
            }
        }
    }
    return s.a2a_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

void block_mps_on_a2a(uint64_t value) {
    EventState& s = state();
    if (value == 0 || !s.a2a_event) return;
    dispatch_sync(
        (dispatch_queue_t)torch::mps::get_dispatch_queue(), ^{
            id<MTLCommandBuffer> cmd =
                (id<MTLCommandBuffer>)torch::mps::get_command_buffer();
            [cmd encodeWaitForEvent:s.a2a_event value:value];
        });
}

void signal_a2a_done_cpu(uint64_t value) {
    EventState& s = state();
    if (value == 0 || !s.a2a_event) return;
    std::atomic_thread_fence(std::memory_order_release);
    std::lock_guard<std::mutex> lock(s.a2a_mu);
    if (s.a2a_event.signaledValue < value) {
        s.a2a_event.signaledValue = value;
    }
}

} // namespace mccl
