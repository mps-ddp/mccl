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

    /// Make an already-completed Work (no-op collectives, e.g. zero-size tensors).
    c10::intrusive_ptr<c10d::Work> make_completed_work(
        c10d::OpType op_type, const std::vector<at::Tensor>& tensors = {});

    // Allreduce algorithm dispatch
    void allreduce_two_rank(at::Tensor& tensor, uint32_t seq,
                            c10d::ReduceOp::RedOpType op);
    void allreduce_ring(at::Tensor& tensor, uint32_t seq,
                        c10d::ReduceOp::RedOpType op);
    void allreduce_ring_chunked(at::Tensor& tensor, uint32_t seq,
                                 c10d::ReduceOp::RedOpType op);
    /// Chunked ring with optional basic-ring fallback (MCCL_RING_FALLBACK_BASIC).
    void allreduce_ring_dispatch(at::Tensor& tensor, uint32_t seq,
                                   c10d::ReduceOp::RedOpType op);
    void allreduce_small(at::Tensor& tensor, uint32_t seq,
                         c10d::ReduceOp::RedOpType op);
    /// Recursive-doubling allreduce for small messages: 2 + log2(p) serial
    /// rounds instead of the 2(ws-1) of the rank-0 star.  CPU-reduce path
    /// (shared storage, no compression).
    void allreduce_tree_small(at::Tensor& tensor, uint32_t seq,
                              c10d::ReduceOp::RedOpType op);

    /// Pipelined ring broadcast (large payloads, ws >= 4): root streams
    /// slices around the ring; every rank forwards slice s while receiving
    /// s+1.  Root egress is S bytes instead of (ws-1)*S.
    void broadcast_ring_pipelined(at::Tensor& tensor, uint32_t seq, int root);

    /// Binomial tree broadcast (small payloads, ws >= 4): ceil(log2 ws)
    /// rounds instead of ws-1 serial root sends.
    void broadcast_tree_small(at::Tensor& tensor, uint32_t seq, int root);

    /// Root star broadcast (small payloads, ws >= 3): root sends to each peer
    /// with the same compressed send/recv ack pattern as broadcast_two_rank.
    void broadcast_star_small(at::Tensor& tensor, uint32_t seq, int root);

    /// Two-rank broadcast (ws == 2): serial send/recv on the ALLREDUCE wire
    /// path (verified for fp32 on TCP demux).
    void broadcast_two_rank(at::Tensor& tensor, uint32_t seq, int root);

    /// Rank-ordered star allgather for small payloads: each src sends to all
    /// dst > src over the full mesh (deadlock-free).  Avoids ring allgather
    /// on multi-node TCP where neighbor-only hops can leave slots zeroed.
    void allgather_star_small(std::vector<at::Tensor>& outputs,
                              const at::Tensor& input,
                              uint32_t seq, size_t nbytes);

    // Compressed send/recv helpers
    void compressed_send(int peer, OpType op, uint32_t seq, uint32_t tid,
                         const at::Tensor& tensor);
    void compressed_recv(int peer, OpType op, uint32_t seq, uint32_t tid,
                         const at::Tensor& tensor, bool cpu_unified_stage = false);

    /// First line of every collective's execute lambda: arms the watchdog
    /// deadline and stamps Metrics::op_execute_start with the COLLECTIVE seq
    /// (the engine's internal counter is a different namespace), so
    /// queue-wait vs execution time splits are attributed correctly.
    void begin_execute(uint32_t seq) {
        watchdog_->touch(seq);
        metrics_->op_execute_start(seq);
    }

    /// Engine thread: publish GPU release token for DDP/autograd Work::wait().
    void arm_work_release(const c10::intrusive_ptr<WorkMCCL>& work);

    /// Store barrier so every rank enters the collective before any wire I/O.
    /// Without this, a fast rank can send/complete before a slow rank posts recv.
    void rendezvous_collective_enter(uint32_t seq, const char* op);

    // Work registry: tracks all in-flight Work objects so watchdog/health
    // callbacks can mark them as failed without waiting for the I/O to unblock.
    void register_work(uint32_t seq, c10::intrusive_ptr<WorkMCCL> work);
    void unregister_work(uint32_t seq);
    void abort_all_inflight_works(const std::string& reason);

    /// Get the NetEngine for a specific peer rank. Used to route network operations
    /// to the correct per-peer engine for concurrent I/O.
    ProgressEngine& net_engine_for(int peer_rank);

    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;

    std::unique_ptr<Transport> transport_;
    std::unique_ptr<ProgressEngine> reduce_engine_;
    /// Executor pool for ws>=3 collectives: MCCL_COLLECTIVE_CONCURRENCY workers
    /// dequeue in submission order.  transport_collective_mu_ ensures only one
    /// collective uses TCP at a time per rank (ring, tree, broadcast, etc.).
    std::unique_ptr<ProgressEngine> collective_pool_;
    /// Serializes all collective_pool transport I/O on shared links.
    std::mutex transport_collective_mu_;
    std::vector<std::unique_ptr<ProgressEngine>> net_engines_;
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
