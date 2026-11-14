#include "runtime/Watchdog.hpp"
#include "common/Logging.hpp"

namespace mccl {

Watchdog::Watchdog(std::chrono::milliseconds default_timeout,
                   AbortCallback on_abort)
    : default_timeout_(default_timeout), on_abort_(std::move(on_abort)) {}

Watchdog::~Watchdog() {
    stop();
}

void Watchdog::start() {
    if (running_.load()) return;
    running_ = true;
    thread_ = std::thread(&Watchdog::monitor_loop, this);
    MCCL_DEBUG("Watchdog started (default_timeout=%lldms)",
               (long long)default_timeout_.count());
}

void Watchdog::stop() {
    if (!running_.load()) return;
    {
        std::lock_guard<std::mutex> lock(stop_mu_);
        running_ = false;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    MCCL_DEBUG("Watchdog stopped");
}

void Watchdog::watch(uint32_t seq, const std::string& op_name) {
    watch(seq, op_name, default_timeout_);
}

void Watchdog::watch(uint32_t seq, const std::string& op_name,
                     std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(mu_);
    // Registered but NOT armed: the deadline clock starts at touch(), when
    // the op begins executing on an engine.  Arming at submission made
    // deep-but-healthy queues (DDP bucket backlogs) trip the watchdog purely
    // from queue wait.  A hung op ahead in the queue still aborts the group:
    // its own (armed) deadline fires and abort_all_inflight_works() errors
    // every registered work, including the still-queued ones.
    entries_[seq] = WatchEntry{seq, op_name,
                               std::chrono::steady_clock::time_point::max(),
                               timeout};
    MCCL_TRACE("Watchdog: watching seq=%u op=%s timeout=%lldms (armed at execute)",
               seq, op_name.c_str(), (long long)timeout.count());
}

void Watchdog::touch(uint32_t seq) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(seq);
    if (it != entries_.end()) {
        it->second.deadline = std::chrono::steady_clock::now() + it->second.timeout;
        MCCL_TRACE("Watchdog: armed seq=%u (execute start / phase boundary)", seq);
    }
}

void Watchdog::complete(uint32_t seq) {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.erase(seq);
    MCCL_TRACE("Watchdog: completed seq=%u", seq);
}

size_t Watchdog::active_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

void Watchdog::monitor_loop() {
    while (running_.load()) {
        {
            std::unique_lock<std::mutex> lock(stop_mu_);
            stop_cv_.wait_for(lock, std::chrono::milliseconds(500),
                              [this] { return !running_.load(); });
        }
        if (!running_.load()) break;

        auto now = std::chrono::steady_clock::now();
        std::vector<WatchEntry> expired;

        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto it = entries_.begin(); it != entries_.end(); ) {
                if (now >= it->second.deadline) {
                    expired.push_back(it->second);
                    it = entries_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (auto& entry : expired) {
            MCCL_FATAL("Watchdog: seq=%u op=%s TIMED OUT — aborting",
                       entry.seq, entry.op_name.c_str());
            if (on_abort_) {
                on_abort_(entry.seq,
                          "Collective " + entry.op_name + " seq=" +
                          std::to_string(entry.seq) + " exceeded deadline");
            }
        }
    }
}

} // namespace mccl
