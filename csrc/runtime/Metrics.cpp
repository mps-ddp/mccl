#include "runtime/Metrics.hpp"
#include "common/Logging.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace mccl {

Metrics::Metrics() = default;

void Metrics::op_start(uint32_t seq, const std::string& op_name, size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    OpMetric m;
    m.seq = seq;
    m.op_name = op_name;
    m.bytes = bytes;
    m.start = std::chrono::steady_clock::now();
    inflight_[seq] = m;
}

void Metrics::op_execute_start(uint32_t seq) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(seq);
    if (it != inflight_.end()) {
        it->second.execute_start = std::chrono::steady_clock::now();
    }
}

void Metrics::op_end(uint32_t seq) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(seq);
    if (it == inflight_.end()) {
        MCCL_WARN("Metrics::op_end: seq=%u not found in inflight table (double-complete or ordering bug)", seq);
        return;
    }

    it->second.end = std::chrono::steady_clock::now();
    completed_.push_back(it->second);
    inflight_.erase(it);

    if (completed_.size() > max_history_) {
        completed_.erase(completed_.begin(),
                         completed_.begin() + (completed_.size() - max_history_));
    }
}

void Metrics::record_phase(uint32_t seq, double sync_ms, double network_ms, double reduce_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = inflight_.find(seq);
    if (it == inflight_.end()) return;
    it->second.sync_ms += sync_ms;
    it->second.network_ms += network_ms;
    it->second.reduce_ms += reduce_ms;
}

void Metrics::record_transport_bytes(size_t bytes, bool is_send) {
    if (is_send) {
        total_bytes_sent_.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        total_bytes_recv_.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void Metrics::record_error() {
    total_errors_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::record_demux_zerocopy(size_t /*bytes*/) {
    demux_zerocopy_hits_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::record_demux_park(size_t /*bytes*/) {
    demux_park_hits_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::record_router_failure() {
    router_failures_.fetch_add(1, std::memory_order_relaxed);
    record_error();
}

void Metrics::record_credit_wait_ms(double ms) {
    if (ms <= 0) return;
    auto us = static_cast<uint64_t>(ms * 1000.0);
    total_credit_wait_us_.fetch_add(us, std::memory_order_relaxed);
}

void Metrics::record_demux_parked_bytes(size_t bytes) {
    uint64_t cur = demux_parked_bytes_peak_.load(std::memory_order_relaxed);
    while (bytes > cur &&
           !demux_parked_bytes_peak_.compare_exchange_weak(
               cur, bytes, std::memory_order_relaxed)) {}
}

Metrics::Summary Metrics::summarize() const {
    std::lock_guard<std::mutex> lock(mu_);
    Summary s{};
    s.total_ops = completed_.size();
    s.total_bytes_sent = total_bytes_sent_.load();
    s.total_bytes_recv = total_bytes_recv_.load();
    s.total_errors = total_errors_.load();

    if (completed_.empty()) {
        s.demux_zerocopy_hits = demux_zerocopy_hits_.load();
        s.demux_park_hits = demux_park_hits_.load();
        s.demux_parked_bytes_peak = demux_parked_bytes_peak_.load();
        s.router_failures = router_failures_.load();
        s.total_credit_wait_ms =
            static_cast<double>(total_credit_wait_us_.load()) / 1000.0;
        return s;
    }

    std::vector<double> latencies;
    latencies.reserve(completed_.size());
    double peak_tp = 0;

    for (auto& m : completed_) {
        double ms = m.elapsed_ms();
        latencies.push_back(ms);
        double tp = m.throughput_gbps();
        if (tp > peak_tp) peak_tp = tp;
    }

    std::sort(latencies.begin(), latencies.end());

    s.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                       latencies.size();

    size_t n = latencies.size();
    s.p50_latency_ms = latencies[n / 2];
    s.p99_latency_ms = latencies[std::min(n - 1, (size_t)(n * 0.99))];
    s.peak_throughput_gbps = peak_tp;

    double total_sync = 0, total_net = 0, total_reduce = 0;
    double total_queue_wait = 0, total_execution = 0;
    for (auto& m : completed_) {
        total_sync += m.sync_ms;
        total_net += m.network_ms;
        total_reduce += m.reduce_ms;
        total_queue_wait += m.queue_wait_ms();
        total_execution += m.execution_ms();
    }
    s.avg_sync_ms = total_sync / n;
    s.avg_network_ms = total_net / n;
    s.avg_reduce_ms = total_reduce / n;
    s.avg_queue_wait_ms = total_queue_wait / n;
    s.avg_execution_ms = total_execution / n;

    s.demux_zerocopy_hits = demux_zerocopy_hits_.load();
    s.demux_park_hits = demux_park_hits_.load();
    s.demux_parked_bytes_peak = demux_parked_bytes_peak_.load();
    s.router_failures = router_failures_.load();
    s.total_credit_wait_ms =
        static_cast<double>(total_credit_wait_us_.load()) / 1000.0;

    return s;
}

void Metrics::log_summary() const {
    auto s = summarize();
    MCCL_INFO("=== MCCL Metrics Summary ===");
    MCCL_INFO("  Total ops:        %llu", (unsigned long long)s.total_ops);
    MCCL_INFO("  Total sent:       %.2f MB",
              s.total_bytes_sent / (1024.0 * 1024.0));
    MCCL_INFO("  Total recv:       %.2f MB",
              s.total_bytes_recv / (1024.0 * 1024.0));
    MCCL_INFO("  Errors:           %llu", (unsigned long long)s.total_errors);
    MCCL_INFO("  Avg latency:      %.3f ms", s.avg_latency_ms);
    MCCL_INFO("  P50 latency:      %.3f ms", s.p50_latency_ms);
    MCCL_INFO("  P99 latency:      %.3f ms", s.p99_latency_ms);
    MCCL_INFO("  Peak throughput:  %.2f Gbps", s.peak_throughput_gbps);
    MCCL_INFO("  Avg queue wait:   %.3f ms", s.avg_queue_wait_ms);
    MCCL_INFO("  Avg execution:    %.3f ms", s.avg_execution_ms);
    MCCL_INFO("  Avg sync:         %.3f ms", s.avg_sync_ms);
    MCCL_INFO("  Avg network:      %.3f ms", s.avg_network_ms);
    MCCL_INFO("  Avg reduce:       %.3f ms", s.avg_reduce_ms);
    MCCL_INFO("  Demux zerocopy:   %llu hits", (unsigned long long)s.demux_zerocopy_hits);
    MCCL_INFO("  Demux park:       %llu hits (peak parked %llu bytes)",
              (unsigned long long)s.demux_park_hits,
              (unsigned long long)s.demux_parked_bytes_peak);
    MCCL_INFO("  Router failures:  %llu", (unsigned long long)s.router_failures);
    MCCL_INFO("  Credit wait:      %.3f ms total", s.total_credit_wait_ms);
    MCCL_INFO("============================");
}

void Metrics::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    inflight_.clear();
    completed_.clear();
    total_bytes_sent_.store(0, std::memory_order_seq_cst);
    total_bytes_recv_.store(0, std::memory_order_seq_cst);
    total_errors_.store(0, std::memory_order_seq_cst);
    demux_zerocopy_hits_.store(0, std::memory_order_seq_cst);
    demux_park_hits_.store(0, std::memory_order_seq_cst);
    demux_parked_bytes_peak_.store(0, std::memory_order_seq_cst);
    router_failures_.store(0, std::memory_order_seq_cst);
    total_credit_wait_us_.store(0, std::memory_order_seq_cst);
}

std::vector<OpMetric> Metrics::recent_ops(size_t n) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (completed_.size() <= n) return completed_;
    return std::vector<OpMetric>(completed_.end() - n, completed_.end());
}

} // namespace mccl
