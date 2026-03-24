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
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    double sync_ms = 0;
    double network_ms = 0;
    double reduce_ms = 0;
    double queue_wait_ms = 0;
    double send_queue_wait_ms = 0;
    double recv_queue_wait_ms = 0;
    double send_ms = 0;
    double recv_ms = 0;
    double stage_ms = 0;
    double writeback_ms = 0;
    double backpressure_ms = 0;
    double pipeline_depth_sum = 0;
    uint64_t pipeline_depth_samples = 0;
    uint64_t max_pipeline_depth = 0;

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

    /// Record per-phase breakdown for an in-flight op.
    void record_phase(uint32_t seq, double sync_ms, double network_ms, double reduce_ms);
    /// Record engine queue wait for an in-flight op.
    void record_queue_wait(uint32_t seq, double queue_wait_ms);
    /// Record queue waits on ring send/recv worker queues.
    void record_ring_queue_wait(uint32_t seq, double send_queue_wait_ms, double recv_queue_wait_ms);
    /// Record ring send/recv wire execution times.
    void record_ring_io(uint32_t seq, double send_ms, double recv_ms);
    /// Record slot pipeline bookkeeping and staging costs.
    void record_pipeline(uint32_t seq, double stage_ms, double writeback_ms,
                         double backpressure_ms, uint64_t pipeline_depth,
                         bool sample_depth);

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
        uint64_t small_ops;
        uint64_t medium_ops;
        uint64_t large_ops;
        double avg_latency_ms;
        double avg_wall_ms;
        double p50_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double peak_throughput_gbps;
        double avg_sync_ms;
        double avg_network_ms;
        double avg_reduce_ms;
        double avg_queue_wait_ms;
        double p50_queue_wait_ms;
        double p95_queue_wait_ms;
        double p99_queue_wait_ms;
        double avg_send_queue_wait_ms;
        double avg_recv_queue_wait_ms;
        double avg_send_ms;
        double avg_recv_ms;
        double avg_stage_ms;
        double avg_writeback_ms;
        double avg_backpressure_ms;
        double p95_backpressure_ms;
        double p99_backpressure_ms;
        uint64_t pipeline_stall_count;
        double avg_pipeline_depth;
        uint64_t max_pipeline_depth;
        double avg_overlap_efficiency;
        double small_avg_wall_ms;
        double medium_avg_wall_ms;
        double large_avg_wall_ms;
        double small_p99_wall_ms;
        double medium_p99_wall_ms;
        double large_p99_wall_ms;
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
