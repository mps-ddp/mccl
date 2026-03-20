#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mccl {

class Transport;

/// Passive liveness monitor for peer connections.
///
/// Periodically checks whether each peer's connection is still open.
/// Detects dead peers (crashed process, network partition) before collective
/// hangs occur. On peer death, fires a callback for abort propagation.
///
/// Note: this is a passive check (inspects socket state), not an active
/// heartbeat probe. A peer that is alive but unresponsive (e.g. stuck in
/// a long collective) will NOT be flagged. The Watchdog timer is the
/// safety net for hung collectives.
class HealthMonitor {
public:
    using PeerDeathCallback = std::function<void(int peer_rank)>;

    HealthMonitor(Transport* transport,
                  std::chrono::milliseconds interval,
                  PeerDeathCallback on_death);
    ~HealthMonitor();

    HealthMonitor(const HealthMonitor&) = delete;
    HealthMonitor& operator=(const HealthMonitor&) = delete;

    void start();
    void stop();

    bool is_peer_alive(int rank) const;

    /// Mark a peer as known-dead (e.g. after transport error).
    void mark_dead(int rank);

private:
    void heartbeat_loop();

    Transport* transport_;
    std::chrono::milliseconds interval_;
    PeerDeathCallback on_death_;

    mutable std::mutex mu_;
    std::vector<bool> alive_;
    int consecutive_failures_ = 0;

    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace mccl
