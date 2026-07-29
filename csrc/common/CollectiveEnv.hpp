#pragma once

#include <algorithm>
#include <cstdlib>

namespace mccl {

/// Upper bound on a single DDP bucket / ring payload (bytes).
/// Set ``MCCL_DEMUX_MAX_COLLECTIVE_BYTES`` to match ``DDP_BUCKET_MB`` on workers.
inline size_t demux_max_collective_bytes() {
    if (auto* v = std::getenv("MCCL_DEMUX_MAX_COLLECTIVE_BYTES")) {
        long long n = std::atoll(v);
        if (n > 0) {
            return static_cast<size_t>(n);
        }
    }
    return 64ULL << 20;
}

/// Hard ceiling on ``MCCL_COLLECTIVE_CONCURRENCY`` (override: ``MCCL_MAX_COLLECTIVE_CONCURRENCY``).
inline int collective_concurrency_max() {
    int cap = 8;
    if (auto* v = std::getenv("MCCL_MAX_COLLECTIVE_CONCURRENCY")) {
        cap = static_cast<int>(std::min(16L, std::max(1L, std::atol(v))));
    }
    return cap;
}

/// Parsed ``MCCL_COLLECTIVE_CONCURRENCY`` before bucket / demux safety cap.
inline int collective_concurrency_requested() {
    int requested = 2;
    if (auto* v = std::getenv("MCCL_COLLECTIVE_CONCURRENCY")) {
        requested = static_cast<int>(std::atol(v));
    }
    return std::max(1, std::min(requested, collective_concurrency_max()));
}

/// Total in-flight collective bytes budget for demux parking (override for lab tuning).
inline size_t demux_inflight_budget_bytes(int world_size) {
    if (auto* v = std::getenv("MCCL_DEMUX_INFLIGHT_BUDGET_BYTES")) {
        long long n = std::atoll(v);
        if (n > 0) {
            return static_cast<size_t>(n);
        }
    }
    // Conservative: ws=8 ENOBUFS at 64MiB×2; allow more concurrent small buckets.
    if (world_size >= 8) {
        return 128ULL << 20;
    }
    if (world_size >= 5) {
        return 192ULL << 20;
    }
    return 256ULL << 20;
}

/// Effective ``MCCL_COLLECTIVE_CONCURRENCY`` after bucket + world-size safety cap.
///
/// Overlapping *N* allreduces posts ~N × bucket bytes on the TCP demux.  Cap
/// concurrency from ``MCCL_DEMUX_INFLIGHT_BUDGET_BYTES / bucket`` so large buckets
/// stay at 1–2 and small buckets (≤16–25 MiB) can run 3–8 when requested.
inline int effective_collective_concurrency(int world_size) {
    const int requested = collective_concurrency_requested();
    if (requested <= 1 || world_size < 3) {
        return 1;
    }

    const size_t bucket = std::max(size_t(1), demux_max_collective_bytes());
    const size_t budget = demux_inflight_budget_bytes(world_size);
    const int by_budget = static_cast<int>(
        std::max(size_t(1), budget / bucket));

    return std::max(1, std::min(requested, by_budget));
}

}  // namespace mccl
