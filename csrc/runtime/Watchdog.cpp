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
    running_ = false;
    if (thread_.joinable()) thread_.join();
    MCCL_DEBUG("Watchdog stopped");
}

void Watchdog::watch(uint32_t seq, const std::string& op_name) {
    watch(seq, op_name, default_timeout_);
}

void Watchdog::watch(uint32_t seq, const std::string& op_name,
                     std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(mu_);
    auto deadline = std::chrono::steady_clock::now() + timeout;
    entries_[seq] = WatchEntry{seq, op_name, deadline};
    MCCL_TRACE("Watchdog: watching seq=%u op=%s timeout=%lldms",
               seq, op_name.c_str(), (long long)timeout.count());
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
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

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
