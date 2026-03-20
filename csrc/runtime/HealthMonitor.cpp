#include "runtime/HealthMonitor.hpp"
#include "transport/Transport.hpp"
#include "common/Logging.hpp"

namespace mccl {

HealthMonitor::HealthMonitor(Transport* transport,
                             std::chrono::milliseconds interval,
                             PeerDeathCallback on_death)
    : transport_(transport),
      interval_(interval),
      on_death_(std::move(on_death)),
      alive_(transport->world_size(), true) {}

HealthMonitor::~HealthMonitor() {
    stop();
}

void HealthMonitor::start() {
    if (running_.load()) return;
    running_ = true;
    thread_ = std::thread(&HealthMonitor::heartbeat_loop, this);
    MCCL_INFO("HealthMonitor started (interval=%lldms)",
              (long long)interval_.count());
}

void HealthMonitor::stop() {
    if (!running_.load()) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

bool HealthMonitor::is_peer_alive(int rank) const {
    std::lock_guard<std::mutex> lock(mu_);
    return alive_.at(rank);
}

void HealthMonitor::mark_dead(int rank) {
    bool was_alive;
    {
        std::lock_guard<std::mutex> lock(mu_);
        was_alive = alive_.at(rank);
        alive_[rank] = false;
    }
    if (was_alive) {
        MCCL_ERROR("HealthMonitor: rank %d marked dead", rank);
        if (on_death_) on_death_(rank);
    }
}

void HealthMonitor::heartbeat_loop() {
    // This loop checks the transport's local connection state flag, which is
    // set to false only when a send or recv operation fails.  It does NOT send
    // probe packets, so it cannot detect silent network partitions that occur
    // between collective operations.  The watchdog provides the timeout-based
    // safety net for in-progress collectives; this monitor catches the case
    // where is_peer_connected() returns false after a transport-level error.
    while (running_.load()) {
        std::this_thread::sleep_for(interval_);
        if (!running_.load()) break;

        for (int r = 0; r < transport_->world_size(); r++) {
            if (r == transport_->rank()) continue;

            bool peer_alive;
            {
                std::lock_guard<std::mutex> lock(mu_);
                peer_alive = alive_[r];
            }
            if (!peer_alive) continue;

            if (!transport_->is_peer_connected(r)) {
                mark_dead(r);
            }
        }
    }
}

} // namespace mccl
