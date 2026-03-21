#pragma once

#include <torch/torch.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10/util/intrusive_ptr.h>

#include "backend/Options.hpp"
#include "backend/WorkMCCL.hpp"
#include "transport/Transport.hpp"
#include "transport/TcpTransport.hpp"
#include "runtime/ProgressEngine.hpp"
#include "runtime/Rendezvous.hpp"
#include "runtime/Watchdog.hpp"
#include "runtime/HealthMonitor.hpp"
#include "runtime/Metrics.hpp"
#include "runtime/MemoryPool.hpp"
#include "compression/Compression.hpp"

#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace mccl {

class ProcessGroupMCCL : public c10d::Backend {
public:
    ProcessGroupMCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int world_size,
        std::chrono::milliseconds timeout);

    ~ProcessGroupMCCL() override;

    const std::string getBackendName() const override {
        return "mccl";
    }

    // ── v1 collectives ──────────────────────────────────────────────

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override;

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> barrier(
        const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

    // ── v2 collectives ──────────────────────────────────────────────

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> send(
        std::vector<at::Tensor>& tensors,
        int dstRank,
        int tag) override;

    c10::intrusive_ptr<c10d::Work> recv(
        std::vector<at::Tensor>& tensors,
        int srcRank,
        int tag) override;

    // ── Metrics ─────────────────────────────────────────────────────

    Metrics& metrics() { return *metrics_; }
    Metrics::Summary get_metrics_summary() const { return metrics_->summarize(); }
    void log_metrics() const { metrics_->log_summary(); }
    void reset_metrics() { metrics_->reset(); }

private:
    void init_transport();
    void on_watchdog_abort(uint32_t seq, const std::string& msg);
    void on_peer_death(int peer_rank);

    /// Ensure a tensor is contiguous; clone if needed.
    at::Tensor ensure_contiguous(const at::Tensor& tensor);

    // Allreduce algorithm dispatch
    void allreduce_two_rank(at::Tensor& tensor, uint32_t seq,
                            c10d::ReduceOp::RedOpType op);
    void allreduce_ring(at::Tensor& tensor, uint32_t seq,
                        c10d::ReduceOp::RedOpType op);
    void allreduce_small(at::Tensor& tensor, uint32_t seq,
                         c10d::ReduceOp::RedOpType op);

    // Compressed send/recv helpers
    void compressed_send(int peer, OpType op, uint32_t seq, uint32_t tid,
                         const at::Tensor& tensor);
    void compressed_recv(int peer, OpType op, uint32_t seq, uint32_t tid,
                         const at::Tensor& tensor);

    // Work registry: tracks all in-flight Work objects so watchdog/health
    // callbacks can mark them as failed without waiting for the I/O to unblock.
    void register_work(uint32_t seq, c10::intrusive_ptr<WorkMCCL> work);
    void unregister_work(uint32_t seq);
    void abort_all_inflight_works(const std::string& reason);

    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;

    std::unique_ptr<Transport> transport_;
    std::unique_ptr<ProgressEngine> engine_;
    std::unique_ptr<Rendezvous> rendezvous_;
    std::unique_ptr<Watchdog> watchdog_;
    std::unique_ptr<HealthMonitor> health_;
    std::unique_ptr<Metrics> metrics_;
    std::unique_ptr<Compressor> compressor_;

    std::atomic<uint32_t> collective_seq_{0};
    bool transport_initialized_ = false;
    bool overlap_comm_ = true;

    mutable std::mutex work_registry_mu_;
    std::unordered_map<uint32_t, c10::weak_intrusive_ptr<WorkMCCL>> work_registry_;
};

c10::intrusive_ptr<c10d::Backend> createProcessGroupMCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size,
    const std::chrono::milliseconds& timeout);

void set_active_pg(ProcessGroupMCCL* pg);
void clear_active_pg_if(ProcessGroupMCCL* pg);

} // namespace mccl
