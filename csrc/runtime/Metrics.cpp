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

Metrics::Summary Metrics::summarize() const {
    constexpr size_t kSmallBytes = 64 * 1024;
    constexpr size_t kMediumBytes = 1024 * 1024;

    std::lock_guard<std::mutex> lock(mu_);
    Summary s{};
    s.total_ops = completed_.size();
    s.total_bytes_sent = total_bytes_sent_.load();
    s.total_bytes_recv = total_bytes_recv_.load();
    s.total_errors = total_errors_.load();

    if (completed_.empty()) return s;

    std::vector<double> latencies;
    latencies.reserve(completed_.size());
    std::vector<double> small_latencies;
    std::vector<double> medium_latencies;
    std::vector<double> large_latencies;
    double peak_tp = 0;
    double total_overlap_eff = 0;

    for (auto& m : completed_) {
        double ms = m.elapsed_ms();
        latencies.push_back(ms);
        if (m.bytes <= kSmallBytes) {
            small_latencies.push_back(ms);
        } else if (m.bytes <= kMediumBytes) {
            medium_latencies.push_back(ms);
        } else {
            large_latencies.push_back(ms);
        }
        double tp = m.throughput_gbps();
        if (tp > peak_tp) peak_tp = tp;
        if (ms > 0) {
            total_overlap_eff += std::max(0.0, (m.network_ms + m.reduce_ms - ms) / ms);
        }
    }

    std::sort(latencies.begin(), latencies.end());
    std::sort(small_latencies.begin(), small_latencies.end());
    std::sort(medium_latencies.begin(), medium_latencies.end());
    std::sort(large_latencies.begin(), large_latencies.end());

    s.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                       latencies.size();
    s.avg_wall_ms = s.avg_latency_ms;

    size_t n = latencies.size();
    s.p50_latency_ms = latencies[n / 2];
    s.p99_latency_ms = latencies[std::min(n - 1, (size_t)(n * 0.99))];
    s.peak_throughput_gbps = peak_tp;

    double total_sync = 0, total_net = 0, total_reduce = 0;
    for (auto& m : completed_) {
        total_sync += m.sync_ms;
        total_net += m.network_ms;
        total_reduce += m.reduce_ms;
    }
    s.avg_sync_ms = total_sync / n;
    s.avg_network_ms = total_net / n;
    s.avg_reduce_ms = total_reduce / n;
    s.avg_overlap_efficiency = total_overlap_eff / n;
    s.small_ops = small_latencies.size();
    s.medium_ops = medium_latencies.size();
    s.large_ops = large_latencies.size();
    if (!small_latencies.empty()) {
        size_t sn = small_latencies.size();
        s.small_avg_wall_ms = std::accumulate(
            small_latencies.begin(), small_latencies.end(), 0.0) / sn;
        s.small_p99_wall_ms = small_latencies[std::min(sn - 1, (size_t)(sn * 0.99))];
    }
    if (!medium_latencies.empty()) {
        size_t mn = medium_latencies.size();
        s.medium_avg_wall_ms = std::accumulate(
            medium_latencies.begin(), medium_latencies.end(), 0.0) / mn;
        s.medium_p99_wall_ms = medium_latencies[std::min(mn - 1, (size_t)(mn * 0.99))];
    }
    if (!large_latencies.empty()) {
        size_t ln = large_latencies.size();
        s.large_avg_wall_ms = std::accumulate(
            large_latencies.begin(), large_latencies.end(), 0.0) / ln;
        s.large_p99_wall_ms = large_latencies[std::min(ln - 1, (size_t)(ln * 0.99))];
    }

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
    MCCL_INFO("  Avg sync:         %.3f ms", s.avg_sync_ms);
    MCCL_INFO("  Avg network:      %.3f ms", s.avg_network_ms);
    MCCL_INFO("  Avg reduce:       %.3f ms", s.avg_reduce_ms);
    MCCL_INFO("  Avg overlap eff:  %.3f", s.avg_overlap_efficiency);
    MCCL_INFO("  Small bucket:     ops=%llu avg=%.3fms p99=%.3fms",
              (unsigned long long)s.small_ops, s.small_avg_wall_ms, s.small_p99_wall_ms);
    MCCL_INFO("  Medium bucket:    ops=%llu avg=%.3fms p99=%.3fms",
              (unsigned long long)s.medium_ops, s.medium_avg_wall_ms, s.medium_p99_wall_ms);
    MCCL_INFO("  Large bucket:     ops=%llu avg=%.3fms p99=%.3fms",
              (unsigned long long)s.large_ops, s.large_avg_wall_ms, s.large_p99_wall_ms);
    MCCL_INFO("============================");
}

void Metrics::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    inflight_.clear();
    completed_.clear();
    total_bytes_sent_.store(0, std::memory_order_seq_cst);
    total_bytes_recv_.store(0, std::memory_order_seq_cst);
    total_errors_.store(0, std::memory_order_seq_cst);
}

std::vector<OpMetric> Metrics::recent_ops(size_t n) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (completed_.size() <= n) return completed_;
    return std::vector<OpMetric>(completed_.end() - n, completed_.end());
}

} // namespace mccl
