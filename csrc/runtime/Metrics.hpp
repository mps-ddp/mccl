#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mccl {

/// Per-collective timing record with per-phase breakdown.
struct OpMetric {
    uint32_t seq;
    std::string op_name;
    size_t bytes;
    std::chrono::steady_clock::time_point start;          // When op was submitted
    std::chrono::steady_clock::time_point execute_start;  // When op.execute() began
    std::chrono::steady_clock::time_point end;            // When op completed

    double sync_ms = 0;
    double network_ms = 0;
    double reduce_ms = 0;

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double queue_wait_ms() const {
        if (execute_start == std::chrono::steady_clock::time_point{}) return 0;
        return std::chrono::duration<double, std::milli>(execute_start - start).count();
    }

    double execution_ms() const {
        if (execute_start == std::chrono::steady_clock::time_point{}) return elapsed_ms();
        return std::chrono::duration<double, std::milli>(end - execute_start).count();
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

    /// Record when execution actually begins (after queue wait).
    void op_execute_start(uint32_t seq);

    /// End timing an operation.
    void op_end(uint32_t seq);

    /// Record per-phase breakdown for an in-flight op.
    void record_phase(uint32_t seq, double sync_ms, double network_ms, double reduce_ms);

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
        double avg_sync_ms;
        double avg_network_ms;
        double avg_reduce_ms;
        double avg_queue_wait_ms;
        double avg_execution_ms;
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
