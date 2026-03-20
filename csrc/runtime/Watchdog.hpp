#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mccl {

/// Watchdog timer that detects hung collectives.
///
/// Each in-flight collective registers a deadline. If the deadline passes
/// without the collective completing, the watchdog fires an abort callback.
/// This prevents silent hangs during training.
class Watchdog {
public:
    using AbortCallback = std::function<void(uint32_t seq, const std::string& msg)>;

    explicit Watchdog(std::chrono::milliseconds default_timeout,
                      AbortCallback on_abort);
    ~Watchdog();

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    void start();
    void stop();

    /// Register a collective with a deadline.
    void watch(uint32_t seq, const std::string& op_name);

    /// Register with a custom timeout.
    void watch(uint32_t seq, const std::string& op_name,
               std::chrono::milliseconds timeout);

    /// Mark a collective as complete — removes it from the watch list.
    void complete(uint32_t seq);

    /// Number of currently watched ops.
    size_t active_count() const;

private:
    void monitor_loop();

    struct WatchEntry {
        uint32_t seq;
        std::string op_name;
        std::chrono::steady_clock::time_point deadline;
    };

    std::chrono::milliseconds default_timeout_;
    AbortCallback on_abort_;

    mutable std::mutex mu_;
    std::unordered_map<uint32_t, WatchEntry> entries_;

    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace mccl
