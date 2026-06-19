#pragma once

#include <algorithm>
#include <cstdlib>

namespace mccl {

/// Effective MCCL_COLLECTIVE_CONCURRENCY after world-size safety cap.
/// ws>=5: force 1 — concurrent large ring buckets exhaust socket buffers (ENOBUFS).
inline int effective_collective_concurrency(int world_size) {
    int threads = 2;
    if (auto* v = std::getenv("MCCL_COLLECTIVE_CONCURRENCY")) {
        threads = static_cast<int>(std::min(4L, std::max(1L, std::atol(v))));
    }
    if (world_size >= 5 && threads > 1) {
        threads = 1;
    }
    return threads;
}

}  // namespace mccl
