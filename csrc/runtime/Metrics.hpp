#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mccl {

/// Per-collective timing record.
struct OpMetric {
    uint32_t seq;
    std::string op_name;
    size_t bytes;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double throughput_gbps() const {
        double elapsed_s = std::chrono::duration<double>(end - start).count();
        if (elapsed_s <= 0) return 0;
        return (bytes * 8.0) / (elapsed_s * 1e9);
    }
};

/// Lightweight metrics collector for MCCL operations.
///
/// Thread-safe. Records timing for each collective. Provides
/// aggregated stats for benchmarking and monitoring.
class Metrics {
public:
    Metrics();

    /// Start timing an operation.
    void op_start(uint32_t seq, const std::string& op_name, size_t bytes);

    /// End timing an operation.
    void op_end(uint32_t seq);

    /// Record a transport-level send/recv.
    void record_transport_bytes(size_t bytes, bool is_send);

    /// Record a transport error.
    void record_error();

    // ── Aggregated stats ────────────────────────────────────────────

    struct Summary {
        uint64_t total_ops;
        uint64_t total_bytes_sent;
        uint64_t total_bytes_recv;
        uint64_t total_errors;
        double avg_latency_ms;
        double p50_latency_ms;
        double p99_latency_ms;
        double peak_throughput_gbps;
    };

    Summary summarize() const;

    /// Dump stats to log at INFO level.
    void log_summary() const;

    /// Reset all counters.
    void reset();

    /// Get the last N completed op metrics.
    std::vector<OpMetric> recent_ops(size_t n = 100) const;

private:
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, OpMetric> inflight_;
    std::vector<OpMetric> completed_;
    size_t max_history_ = 10000;

    std::atomic<uint64_t> total_bytes_sent_{0};
    std::atomic<uint64_t> total_bytes_recv_{0};
    std::atomic<uint64_t> total_errors_{0};
};

} // namespace mccl
