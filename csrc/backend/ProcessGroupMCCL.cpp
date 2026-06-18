#include "backend/ProcessGroupMCCL.hpp"
#include "metal/MetalKernels.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/EventSync.hpp"
#include "metal/AccelerateOps.hpp"
#include "transport/rdma/RdmaTransport.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include "common/TensorChecks.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace mccl {

namespace {

// MCCL_SYNC_MODE=coalesced was removed: it skipped the per-bucket MPS sync after
// the first collective on a thread, reading stale gradients and corrupting DDP.
// MCCL_ALLREDUCE_ALGO was dead code and is also removed.
void warn_removed_env_knobs() {
    static bool warned = [] {
        if (auto* v = std::getenv("MCCL_SYNC_MODE")) {
            if (std::string(v) == "coalesced" || std::string(v) == "fast") {
                MCCL_WARN("MCCL_SYNC_MODE=%s is no longer supported (it corrupted "
                          "DDP gradients); MCCL always syncs per collective", v);
            }
        }
        if (std::getenv("MCCL_ALLREDUCE_ALGO")) {
            MCCL_WARN("MCCL_ALLREDUCE_ALGO is ignored; use MCCL_RING_ALGO=basic|chunked");
        }
        return true;
    }();
    (void)warned;
}

bool use_chunked_ring_default() {
    static bool enabled = [] {
        auto* v = std::getenv("MCCL_RING_ALGO");
        if (v) {
            std::string s(v);
            if (s == "basic" || s == "ring" || s == "plain") return false;
            return true;
        }
        return true;  // chunked ring default at all world sizes
    }();
    return enabled;
}

bool ring_fallback_basic_enabled(int world_size) {
    (void)world_size;
    if (auto* v = std::getenv("MCCL_RING_FALLBACK_BASIC")) {
        std::string s(v);
        return !(s == "0" || s == "false" || s == "no" || s == "off");
    }
    return false;  // opt-in only
}

bool is_supported_reduce_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 ||
           dtype == at::kBool || dtype == at::kInt || dtype == at::kLong;
}

bool is_integral_reduce_dtype(at::ScalarType dtype) {
    return dtype == at::kBool || dtype == at::kInt || dtype == at::kLong;
}

// Kernel-fence pattern for the Metal reduce path (used by the ring and
// reduce_scatter loops): metal_reduce_op() commits its compute command buffer
// without waiting, so a chunk it touched must not be staged for a network
// send (or overwritten by incoming data) until its kernel completes.  The
// MCCL command queue is serial, so signaling the shared event after a kernel
// and waiting on that value also fences all earlier kernels.

// MCCL_RING_PIPELINE: streaming TX/RX ring pipeline (default on).  Set 0 to
// fall back to the lock-step per-step ring (debug escape hatch).
inline bool ring_pipeline_enabled() {
    static bool enabled = [] {
        auto* v = std::getenv("MCCL_RING_PIPELINE");
        if (!v) return true;
        std::string s(v);
        return !(s == "0" || s == "false" || s == "no" || s == "off");
    }();
    return enabled;
}

// Ring pipeline is for large chunks only.  Small allgather (DDP param-count
// int64 metadata, etc.) must use lock-step + pooled recv + unstage_from_recv.
inline bool ring_pipeline_for_message(size_t nbytes, size_t small_msg_threshold) {
    return ring_pipeline_enabled() && nbytes > small_msg_threshold;
}

// In-flight posted-receive depth for the RX side of ring pipelines.
inline int ring_pipeline_depth() {
    static int depth = [] {
        auto* v = std::getenv("MCCL_PIPELINE_DEPTH");
        long n = v ? std::atol(v) : 2;
        return static_cast<int>(std::min(8L, std::max(1L, n)));
    }();
    return depth;
}

// ── Credit-based flow control (NCCL-style) ──────────────────────────
//
// A sender's pipeline is gated by ITS OWN receive progress, not by the
// downstream rank's.  On a ring this means an upstream neighbor can run
// nearly the whole reduce-scatter phase while a slow rank (slower GPU,
// later bucket start) has consumed nothing — flooding it with up to a full
// bucket of unsolicited data that the demux must park.  Credits bound that:
// the receiver sends a 1-byte message (same seq, tid bit 31 set) after
// consuming each step; the sender may run at most `credit_window` steps
// ahead of the last credited step.  Credits ride the ordinary demux path —
// no protocol change — and flow strictly backward, so no cycles.
//
// Only engaged for chunks >= MCCL_CREDIT_MIN_CHUNK (default 1 MB): below
// that, the flood is bounded to a few MB anyway and the extra per-step
// message would cost latency on small rings.
constexpr uint32_t kCreditTidFlag = 0x80000000u;

inline size_t credit_min_chunk_bytes() {
    static size_t v = [] {
        auto* e = std::getenv("MCCL_CREDIT_MIN_CHUNK");
        long long n = e ? std::atoll(e) : (1LL << 20);
        return static_cast<size_t>(std::max(0LL, n));
    }();
    return v;
}

// Sender lead over the last credited step.  depth + 2 keeps the wire full
// (receiver posts `depth` ahead; the +2 covers credit round-trip latency)
// while bounding unsolicited data at a slow receiver to window x chunk.
inline int credit_window() {
    return ring_pipeline_depth() + 2;
}

inline std::chrono::milliseconds recv_wait_limit() {
    static auto limit = [] {
        if (auto* v = std::getenv("MCCL_RECV_TIMEOUT_MS")) {
            long long ms = std::atoll(v);
            if (ms > 0) return std::chrono::milliseconds(ms);
        }
        if (auto* v = std::getenv("MCCL_WATCHDOG_TIMEOUT_MS")) {
            long long ms = std::atoll(v);
            if (ms > 0) return std::chrono::milliseconds(ms);
        }
        return std::chrono::milliseconds(300'000);  // 5 min default
    }();
    return limit;
}

// ── Streaming ring pipeline ──────────────────────────────────────────
//
// NCCL-style execution of a ring schedule: a TX thread streams chunks to the
// right neighbor while the RX loop (caller thread) receives from the left
// and reduces/stores, with `depth` receives posted ahead.  Both directions
// of every link stay busy for the whole collective; reductions overlap both.
//
// Gating: the chunk sent at global step g is the chunk received at global
// step g - lookahead (verified schedules: lookahead 2 for the 2P chunked
// ring, 1 for the plain ring / reduce_scatter / allgather).  The first
// `lookahead` sends are the rank's own (already-ready) data.  RX opens
// send-gate g+lookahead after finishing recv-step g, passing along the
// Metal fence value the TX side must wait on before staging.

enum class RingRecvKind : uint8_t {
    REDUCE,       // accumulate into chunk via vDSP/Metal (reduce-scatter phase)
    COPY,         // incoming bytes ARE the final chunk data (allgather phase)
};

struct RingStep {
    int send_idx;        // chunk index to send this step (-1 = no send)
    int recv_idx;        // chunk index receiving this step (-1 = no recv)
    uint32_t send_tid;
    uint32_t recv_tid;
    RingRecvKind kind;
};

struct RingGates {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<uint8_t> open;
    std::vector<uint64_t> fence;   // MCCL fence value TX waits on (0 = none)
    bool failed = false;

    explicit RingGates(size_t n) : open(n, 0), fence(n, 0) {}

    void open_gate(size_t g, uint64_t fence_val) {
        if (g >= open.size()) return;
        {
            std::lock_guard<std::mutex> lock(mu);
            open[g] = 1;
            fence[g] = fence_val;
        }
        cv.notify_all();
    }
    void fail() {
        {
            std::lock_guard<std::mutex> lock(mu);
            failed = true;
        }
        cv.notify_all();
    }
    bool wait_gate(size_t g, uint64_t* fence_val,
                   Watchdog* watchdog = nullptr, uint32_t seq = 0) {
        const auto poll = std::chrono::milliseconds(2000);
        const auto limit = recv_wait_limit();
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mu);
        while (!failed && !open[g]) {
            if (watchdog) watchdog->touch(seq);
            if (cv.wait_for(lock, poll, [&] { return failed || open[g]; })) {
                break;
            }
            if (std::chrono::steady_clock::now() - start > limit) {
                failed = true;
                return false;
            }
        }
        if (failed) return false;
        *fence_val = fence[g];
        return true;
    }
};

// Keep-alive for `incoming` staging tensors consumed by ASYNC Metal reduce
// kernels.  metal_reduce_op() only commits; destroying the tensor lets
// PyTorch's MPS allocator recycle its buffer for the next step's
// empty_like, whose CPU fill then races the still-running kernel —
// silently corrupting partial sums (training still converges, just
// slower).  This was the corruption the old per-step queue drains were
// masking.  The destructor drains the MCCL queue before releasing the
// tensors, so even exception paths cannot free a buffer a kernel is
// reading.  (The normal path drains in the collective tail first, making
// the destructor's drain a no-op.)
struct IncomingKeepAlive {
    std::vector<at::Tensor> tensors;
    ~IncomingKeepAlive() {
        if (!tensors.empty()) {
            try {
                metal_sync_queue_only();
            } catch (...) {
                // Destructor must not throw; a failed drain here means the
                // device is already in a fatal state.
            }
        }
    }
};

/// Metal reduce with mandatory fence before chunk bytes may be sent or overwritten.
inline uint64_t reduce_chunk_metal_fenced(at::Tensor& dst, at::Tensor incoming,
                                          c10d::ReduceOp::RedOpType op,
                                          std::vector<at::Tensor>* keep_tensors) {
    uint64_t fence = metal_reduce_op_fenced(dst, incoming, op);
    if (keep_tensors) {
        keep_tensors->push_back(std::move(incoming));
    }
    return fence;
}

struct RingPipelineCtx {
    Transport* transport;
    Watchdog* watchdog;
    Metrics* metrics;
    uint32_t seq;
    int left;
    int right;
    OpType wire_op;
    c10d::ReduceOp::RedOpType red_op;
    bool use_cpu;   // fp32 vDSP reduce directly in unified memory
    std::vector<at::Tensor>* incoming_keep = nullptr;  // see IncomingKeepAlive
};

void run_ring_pipeline(const RingPipelineCtx& ctx,
                       std::vector<at::Tensor>& chunks,
                       const std::vector<RingStep>& plan,
                       int lookahead) {
    const size_t nsteps = plan.size();
    if (nsteps == 0) return;

    const bool use_event_fence = !ctx.use_cpu && event_sync_available();
    std::vector<uint64_t> chunk_pending(chunks.size(), 0);

    RingGates gates(nsteps);
    for (int g = 0; g < lookahead && g < static_cast<int>(nsteps); ++g) {
        gates.open_gate(static_cast<size_t>(g), 0);
    }

    auto chunk_bytes = [&](int idx) -> size_t {
        return static_cast<size_t>(chunks[idx].numel()) * chunks[idx].element_size();
    };

    // Credit flow control engages for large chunks only (see helpers above).
    // (std::max<size_t> explicitly: element_size() is int64_t, and the
    // mixed unsigned-long/unsigned-long-long product breaks deduction.)
    size_t max_chunk = 0;
    for (auto& c : chunks) {
        max_chunk = std::max<size_t>(
            max_chunk, static_cast<size_t>(c.numel()) *
                       static_cast<size_t>(c.element_size()));
    }
    const bool credits_on = max_chunk >= credit_min_chunk_bytes() &&
                            credit_min_chunk_bytes() > 0;
    const int cwin = credit_window();

    // Credits arrive from the RIGHT neighbor (the consumer of our sends):
    // one byte per step, tid = kCreditTidFlag | step.  Storage lives at
    // FUNCTION scope (not in the TX lambda): registered sinks reference it,
    // and on an early TX exit the error path below must be able to cancel
    // and drain them before this memory dies.
    std::vector<uint8_t> credit_bytes(credits_on ? nsteps : 0);
    std::vector<RecvTicket> credit_tickets(credits_on ? nsteps : 0);

    // ── TX: stream chunks to the right neighbor in schedule order ──
    std::exception_ptr tx_error;
    std::thread tx([&] {
        try {
            // Caller-owned staging for the private-storage fallback so
            // concurrent collectives never share the global StagingPool.
            std::unique_ptr<PooledBuffer> tx_staging;
            size_t tx_staging_size = 0;

            if (credits_on) {
                for (size_t g = 0; g + cwin < nsteps; ++g) {
                    credit_tickets[g] = ctx.transport->post_recv(
                        ctx.right, ctx.wire_op, ctx.seq,
                        kCreditTidFlag | static_cast<uint32_t>(g),
                        &credit_bytes[g], 1);
                }
            }

            for (size_t g = 0; g < nsteps; ++g) {
                const RingStep& st = plan[g];
                if (st.send_idx < 0) continue;

                size_t send_bytes = chunk_bytes(st.send_idx);

                uint64_t fence_val = 0;
                if (!gates.wait_gate(g, &fence_val, ctx.watchdog, ctx.seq)) {
                    MCCL_CHECK(false, "ring pipeline send gate wait failed at step " +
                               std::to_string(g) + " (seq=" +
                               std::to_string(ctx.seq) + ")");
                }
                if (fence_val > 0) wait_for_mccl_fence(fence_val);

                // Don't outrun the consumer: before sending step g, the right
                // neighbor must have consumed step g - cwin.  Bounds parked
                // bytes at a slow/late receiver to cwin x chunk.
                if (credits_on && g >= static_cast<size_t>(cwin)) {
                    MCCL_CHECK(ctx.transport->wait_recv(
                                   credit_tickets[g - cwin]),
                               "ring pipeline credit wait failed at step " +
                               std::to_string(g) + " (seq=" +
                               std::to_string(ctx.seq) + ")");
                }

                ctx.watchdog->touch(ctx.seq);

                if (send_bytes > 0) {
                    at::Tensor& send_chunk = chunks[st.send_idx];
                    if (!tx_staging || tx_staging_size < send_bytes) {
                        tx_staging = std::make_unique<PooledBuffer>(
                            staging_memory_pool(), send_bytes);
                        tx_staging_size = send_bytes;
                    }
                    blit_tensor_to_buffer(send_chunk, tx_staging->data());
                    const void* src = tx_staging->data();

                    MCCL_CHECK(ctx.transport->send_chunks(
                                   ctx.right, ctx.wire_op, ctx.seq, st.send_tid,
                                   src, send_bytes),
                               "ring pipeline send step " + std::to_string(g) +
                               " (seq=" + std::to_string(ctx.seq) + ") failed");
                    ctx.metrics->record_transport_bytes(send_bytes, true);
                }
                // send_bytes==0: no wire traffic; gate/credit protocol still ran.
            }
        } catch (...) {
            tx_error = std::current_exception();
            // Unblock the RX side: its posted receives will never be matched
            // consistently once our sends stopped.
            ctx.transport->cancel_recvs(ctx.seq, "ring pipeline TX failed");
        }
    });

    // ── RX: post `depth` receives ahead; reduce/store; open send gates ──
    const int depth = ring_pipeline_depth();
    std::vector<std::unique_ptr<PooledBuffer>> scratch(depth);
    std::vector<size_t> scratch_size(depth, 0);
    std::vector<RecvTicket> tickets(nsteps);
    std::vector<void*> recv_dst(nsteps, nullptr);
    std::vector<uint8_t> recv_inplace(nsteps, 0);

    // Phase accounting: wall time of the whole pipeline is the network
    // phase (TX/RX/reduce overlap by design, so phases exceed elapsed —
    // that is the point); reduce time is accumulated around the actual
    // reduction calls on the RX side.
    const auto pipe_t0 = std::chrono::steady_clock::now();
    double red_ms_accum = 0;

    std::exception_ptr rx_error;
    try {
        auto post_step = [&](size_t g) {
            const RingStep& st = plan[g];
            if (st.recv_idx < 0) return;
            size_t rbytes = chunk_bytes(st.recv_idx);
            if (rbytes == 0) return;

            void* dst = nullptr;
            if (st.kind == RingRecvKind::COPY) {
                // The incoming bytes are final data for this chunk.  Fence
                // any reduce kernel still pending on it BEFORE the reader
                // thread may write, then receive zero-copy into unified
                // memory.
                if (chunk_pending[st.recv_idx]) {
                    wait_for_mccl_fence(chunk_pending[st.recv_idx]);
                    chunk_pending[st.recv_idx] = 0;
                }
                MPSBufferView view = extract_mps_buffer(chunks[st.recv_idx]);
                if (view.cpu_accessible && view.cpu_ptr) {
                    dst = view.cpu_ptr;
                    recv_inplace[g] = 1;
                }
            }
            if (!dst) {
                int slot = static_cast<int>(g) % depth;
                if (!scratch[slot] || scratch_size[slot] < rbytes) {
                    scratch[slot] = std::make_unique<PooledBuffer>(
                        staging_memory_pool(), rbytes);
                    scratch_size[slot] = rbytes;
                }
                dst = scratch[slot]->data();
            }
            recv_dst[g] = dst;
            tickets[g] = ctx.transport->post_recv(
                ctx.left, ctx.wire_op, ctx.seq, st.recv_tid, dst, rbytes);
        };

        size_t next_post = 0;
        for (; next_post < nsteps && next_post < static_cast<size_t>(depth);
             ++next_post) {
            post_step(next_post);
        }

        for (size_t g = 0; g < nsteps; ++g) {
            const RingStep& st = plan[g];
            uint64_t gate_fence = 0;

            if (st.recv_idx >= 0 && chunk_bytes(st.recv_idx) > 0) {
                size_t rbytes = chunk_bytes(st.recv_idx);
                ctx.watchdog->touch(ctx.seq);
                MCCL_CHECK(tickets[g] && ctx.transport->wait_recv(tickets[g]),
                           "ring pipeline recv step " + std::to_string(g) +
                           " (seq=" + std::to_string(ctx.seq) + ") failed");
                ctx.metrics->record_transport_bytes(rbytes, false);

                at::Tensor& rchunk = chunks[st.recv_idx];
                if (st.kind == RingRecvKind::REDUCE) {
                    const auto red_t0 = std::chrono::steady_clock::now();
                    if (ctx.use_cpu) {
                        MPSBufferView view = extract_mps_buffer(rchunk);
                        cpu_reduce_op(static_cast<float*>(view.cpu_ptr),
                                      static_cast<const float*>(recv_dst[g]),
                                      rchunk.numel(), ctx.red_op);
                    } else {
                        at::Tensor incoming = torch::empty_like(rchunk);
                        unstage_from_recv(incoming, recv_dst[g], rbytes);
                        gate_fence = reduce_chunk_metal_fenced(
                            rchunk, std::move(incoming), ctx.red_op,
                            ctx.incoming_keep);
                        if (use_event_fence && gate_fence > 0) {
                            chunk_pending[st.recv_idx] = gate_fence;
                        }
                    }
                    red_ms_accum += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - red_t0).count();
                } else {
                    if (!recv_inplace[g]) {
                        // Private-storage fallback: blit caller-owned scratch
                        // into place (never the shared StagingPool).
                        blit_buffer_to_tensor(recv_dst[g], rchunk);
                    }
                }
            }

            gates.open_gate(g + static_cast<size_t>(lookahead), gate_fence);

            // Credit the LEFT neighbor: step g is consumed; it may now send
            // step g + cwin.  Sent for every step (even empty/recv-less
            // ones) so the sender's unconditional waits always resolve.
            if (credits_on && g + static_cast<size_t>(cwin) < nsteps) {
                uint8_t one = 1;
                MCCL_CHECK(ctx.transport->send_chunks(
                               ctx.left, ctx.wire_op, ctx.seq,
                               kCreditTidFlag | static_cast<uint32_t>(g),
                               &one, 1),
                           "ring pipeline credit send failed at step " +
                           std::to_string(g) + " (seq=" +
                           std::to_string(ctx.seq) + ")");
            }

            if (next_post < nsteps) {
                post_step(next_post);
                ++next_post;
            }
        }
    } catch (...) {
        rx_error = std::current_exception();
        gates.fail();
    }

    tx.join();

    if (rx_error || tx_error) {
        // Posted-but-unwaited receives (depth-ahead data, credit tickets)
        // still reference scratch/credit storage that dies with this frame.
        // Cancel them, then drain each ticket: wait_recv blocks until the
        // reader is not mid-copy, so destruction below is safe.  Waits
        // return instantly (cancelled or already complete) — never block on
        // the peer, which may itself be stuck.
        ctx.transport->cancel_recvs(ctx.seq, "ring pipeline aborted");
        for (auto& t : tickets) {
            if (!t) continue;
            try { ctx.transport->wait_recv(t); } catch (...) {}
        }
        for (auto& t : credit_tickets) {
            if (!t) continue;
            try { ctx.transport->wait_recv(t); } catch (...) {}
        }
    }

    if (rx_error) std::rethrow_exception(rx_error);
    if (tx_error) std::rethrow_exception(tx_error);

    const double net_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pipe_t0).count();
    ctx.metrics->record_phase(ctx.seq, 0, net_ms, red_ms_accum);
}

// MCCL_FP32_CPU_REDUCE: unset = Metal/staging for float32 (default, fewer full syncs).
// Set to 1/true/on/yes for CPU-side float32 reduce into the unified buffer (often faster
// allreduce bandwidth on UMA; more mps_stream_sync_after_cpu_mps_buffer_write tails).
inline bool fp32_cpu_reduce_enabled() {
    static bool enabled = [] {
        auto* v = std::getenv("MCCL_FP32_CPU_REDUCE");
        if (!v) return false;
        std::string s(v);
        return s == "1" || s == "true" || s == "on" || s == "yes";
    }();
    return enabled;
}

// Direct recv / staging: use shared cpu_ptr when allowed (see fp32_cpu_reduce_enabled for float32).
inline bool prefer_cpu_unified_buffer_path(const at::Tensor& tensor) {
    if (!tensor_cpu_accessible(tensor)) return false;
    if (tensor.scalar_type() == at::kFloat && !fp32_cpu_reduce_enabled()) return false;
    return true;
}

// Collective MPS ordering: per-tensor blit fence in stage_for_send_collective().
// No global torch::mps::synchronize() or commit_mps from engine threads.
inline uint64_t sync_mps_nonblocking(bool /*overlap*/) {
    return 0;
}

} // anonymous namespace

// ── Construction / destruction ──────────────────────────────────────

ProcessGroupMCCL::ProcessGroupMCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout)
    : c10d::Backend(rank, world_size),
      store_(store),
      timeout_(timeout) {

    refresh_log_level();
    warn_removed_env_knobs();
    MCCL_INFO("ProcessGroupMCCL creating: rank=%d world_size=%d timeout=%lldms",
              rank, world_size, (long long)timeout.count());

    metal_kernels_init();
    event_sync_init();

    metrics_ = std::make_unique<Metrics>();

    // Compression from env: MCCL_COMPRESSION=none|fp16|topk
    CompressionMode comp_mode = CompressionMode::NONE;
    if (auto* v = std::getenv("MCCL_COMPRESSION")) {
        std::string s(v);
        if (s == "fp16" || s == "FP16") comp_mode = CompressionMode::FP16;
        else if (s == "topk" || s == "TOPK") comp_mode = CompressionMode::TOPK;
    }
    double topk_ratio = 0.01;
    if (auto* v = std::getenv("MCCL_TOPK_RATIO")) topk_ratio = std::atof(v);
    compressor_ = make_compressor(comp_mode, topk_ratio);
    if (compressor_) {
        MCCL_INFO("Compression enabled: %s", compressor_->name().c_str());
    }

    if (auto* v = std::getenv("MCCL_OVERLAP_COMM")) {
        std::string s(v);
        overlap_comm_ = !(s == "0" || s == "false" || s == "no");
    }
    if (overlap_comm_ && !event_sync_available()) {
        MCCL_WARN("MCCL_OVERLAP_COMM requested but EventSync unavailable, falling back");
        overlap_comm_ = false;
    }

    size_t queue_depth = 1024;
    if (auto* v = std::getenv("MCCL_MAX_QUEUE_DEPTH"))
        queue_depth = static_cast<size_t>(std::atoll(v));

    // Create reduce engine
    reduce_engine_ = std::make_unique<ProgressEngine>(queue_depth, metrics_.get());
    reduce_engine_->start();

    // Collective executor pool for ws>=3 ring collectives: k workers start
    // collectives in seq order but run them concurrently (bucket overlap).
    int coll_threads = 2;
    if (auto* v = std::getenv("MCCL_COLLECTIVE_CONCURRENCY")) {
        coll_threads = static_cast<int>(std::min(4L, std::max(1L, std::atol(v))));
    }
    if (world_size >= 6 && coll_threads > 1) {
        MCCL_INFO("MCCL_COLLECTIVE_CONCURRENCY capped to 1 for world_size=%d", world_size);
        coll_threads = 1;
    }
    if (world_size >= 3) {
        collective_pool_ = std::make_unique<ProgressEngine>(
            queue_depth, metrics_.get(), coll_threads);
        collective_pool_->start();
    }

    // Create net engines (one per peer rank, excluding self)
    net_engines_.resize(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            net_engines_[i] = std::make_unique<ProgressEngine>(queue_depth, metrics_.get());
            net_engines_[i]->start();
        }
    }

    auto wd_timeout = timeout;
    if (auto* v = std::getenv("MCCL_WATCHDOG_TIMEOUT_MS"))
        wd_timeout = std::chrono::milliseconds(std::atoll(v));
    watchdog_ = std::make_unique<Watchdog>(
        wd_timeout,
        [this](uint32_t seq, const std::string& msg) {
            on_watchdog_abort(seq, msg);
        });
    watchdog_->start();

    init_transport();

    if (transport_) {
        auto hb_interval = transport_->config().heartbeat_interval;
        health_ = std::make_unique<HealthMonitor>(
            transport_.get(),
            hb_interval,
            [this](int peer) { on_peer_death(peer); });
        health_->start();
    }

    // ── Config dump ───────────────────────────────────────────────
    MCCL_INFO("=== MCCL Config (rank %d) ===", rank);
    MCCL_INFO("  world_size          = %d", world_size);
    MCCL_INFO("  timeout_ms          = %lld", (long long)timeout.count());
    MCCL_INFO("  watchdog_timeout_ms = %lld", (long long)wd_timeout.count());
    if (transport_) {
    MCCL_INFO("  transport           = %s", transport_->config().transport.c_str());
    MCCL_INFO("  listen_addr         = %s", transport_->config().listen_addr.c_str());
    MCCL_INFO("  port_base           = %u", (unsigned)transport_->config().port_base);
    MCCL_INFO("  ifname              = %s",
              transport_->config().ifname.empty() ? "(auto)" : transport_->config().ifname.c_str());
    MCCL_INFO("  chunk_bytes         = %zu", transport_->config().chunk_bytes);
    MCCL_INFO("  small_msg_threshold = %zu", transport_->config().small_msg_threshold);
    MCCL_INFO("  connect_timeout_ms  = %lld", (long long)transport_->config().connect_timeout.count());
    MCCL_INFO("  heartbeat_ms        = %lld", (long long)transport_->config().heartbeat_interval.count());
    } else {
    MCCL_INFO("  transport           = none (world_size == 1)");
    }
    MCCL_INFO("  max_queue_depth     = %zu", queue_depth);
    {
        const char* crc_env = std::getenv("MCCL_TRANSPORT_CRC");
        MCCL_INFO("  transport_crc       = %s", (crc_env && std::string(crc_env) == "1") ? "on" : "off");
    }
    {
        const char* fm = std::getenv("MCCL_FAST_MATH");
        MCCL_INFO("  fast_math           = %s", (!fm || std::string(fm) != "0") ? "on" : "off");
    }
    {
        const char* gt = std::getenv("MCCL_GPU_THRESHOLD");
        MCCL_INFO("  gpu_threshold       = %s", gt ? gt : "4096");
    }
    MCCL_INFO("  overlap_comm        = %s", overlap_comm_ ? "on" : "off");
    {
        const char* es = std::getenv("MCCL_EVENT_SYNC");
        bool es_off = es && (std::string(es) == "0" || std::string(es) == "false" ||
                             std::string(es) == "no");
        MCCL_INFO("  event_sync          = %s",
                   es_off ? "off (env)" :
                   event_sync_available() ? "on" : "off (unavailable)");
    }
    MCCL_INFO("  ring_algo           = %s",
              use_chunked_ring_default() ? "chunked" : "basic");
    MCCL_INFO("  ring_pipeline       = %s (depth=%d)",
              ring_pipeline_enabled() ? "on" : "off (lock-step)",
              ring_pipeline_depth());
    MCCL_INFO("  collective_pool     = %s",
              collective_pool_ ? "on" : "off (ws<3)");
    MCCL_INFO("  compression         = %s", compressor_ ? compressor_->name().c_str() : "none");
    if (compressor_ && comp_mode == CompressionMode::TOPK) {
        MCCL_INFO("  topk_ratio          = %.4f", topk_ratio);
    }
    MCCL_INFO("  log_level           = %s", level_str(global_log_level()));
    MCCL_INFO("==============================");

    MCCL_INFO("ProcessGroupMCCL rank=%d ready", rank);
}

ProgressEngine& ProcessGroupMCCL::net_engine_for(int peer_rank) {
    MCCL_CHECK(peer_rank >= 0 && peer_rank < getSize(),
               "net_engine_for: invalid peer_rank " + std::to_string(peer_rank));
    MCCL_CHECK(peer_rank != getRank(),
               "net_engine_for: cannot get engine for self rank " + std::to_string(peer_rank));
    MCCL_CHECK(net_engines_[peer_rank] != nullptr,
               "net_engine_for: engine for peer " + std::to_string(peer_rank) + " is null");
    return *net_engines_[peer_rank];
}

ProcessGroupMCCL::~ProcessGroupMCCL() {
    try {
        MCCL_INFO("ProcessGroupMCCL rank=%d shutting down", getRank());
        clear_active_pg_if(this);
        metrics_->log_summary();
        if (health_) health_->stop();
        if (watchdog_) watchdog_->stop();
        if (collective_pool_) collective_pool_->stop();
        if (reduce_engine_) reduce_engine_->stop();
        for (auto& engine : net_engines_) {
            if (engine) engine->stop();
        }
        if (transport_) transport_->shutdown();
    } catch (const std::exception& e) {
        MCCL_DEBUG("Exception during shutdown (suppressed): %s", e.what());
    } catch (...) {
        MCCL_DEBUG("Unknown exception during shutdown (suppressed)");
    }
}

void ProcessGroupMCCL::init_transport() {
    if (getSize() == 1) {
        // Single-rank group: every collective is a local no-op/copy; there is
        // no peer to talk to and TcpTransport requires world_size >= 2.
        MCCL_INFO("Rank %d: world_size == 1, transport disabled", getRank());
        transport_initialized_ = true;
        return;
    }

    TransportConfig cfg = TransportConfig::from_env();
    warn_if_mccl_port_overlaps_master(cfg);
    warn_if_master_addr_unresolvable();

    if (cfg.transport == "rdma" ||
        (cfg.transport == "auto" && RdmaTransport::is_available())) {
        MCCL_INFO("Rank %d: selecting RDMA transport (mode=%s)",
                  getRank(), cfg.transport.c_str());
        transport_ = std::make_unique<RdmaTransport>(getRank(), getSize(), cfg);
    } else {
        MCCL_INFO("Rank %d: selecting TCP transport (mode=%s)",
                  getRank(), cfg.transport.c_str());
        transport_ = std::make_unique<TcpTransport>(getRank(), getSize(), cfg);
    }

    rendezvous_ = std::make_unique<Rendezvous>(store_, getRank(), getSize(), timeout_);

    std::string my_endpoint = transport_->listen_endpoint();
    auto endpoints = rendezvous_->exchange_endpoints(my_endpoint);
    transport_->connect_all(endpoints);

    transport_initialized_ = true;
    MCCL_INFO("Rank %d: transport fully connected", getRank());
}

void ProcessGroupMCCL::register_work(uint32_t seq, c10::intrusive_ptr<WorkMCCL> work) {
    std::lock_guard<std::mutex> lock(work_registry_mu_);
    work_registry_.insert_or_assign(seq, c10::weak_intrusive_ptr<WorkMCCL>(work));
}

void ProcessGroupMCCL::unregister_work(uint32_t seq) {
    std::lock_guard<std::mutex> lock(work_registry_mu_);
    work_registry_.erase(seq);
}

void ProcessGroupMCCL::abort_all_inflight_works(const std::string& reason) {
    std::vector<c10::intrusive_ptr<WorkMCCL>> to_abort;
    {
        std::lock_guard<std::mutex> lock(work_registry_mu_);
        for (auto& [seq, weak] : work_registry_) {
            auto strong = weak.lock();
            if (strong) to_abort.push_back(std::move(strong));
        }
        work_registry_.clear();
    }
    auto err = std::make_exception_ptr(MCCLError(reason));
    for (auto& work : to_abort) {
        work->markError(err);
    }
    MCCL_ERROR("Rank %d: aborted %zu in-flight work(s): %s",
               getRank(), to_abort.size(), reason.c_str());
}

void ProcessGroupMCCL::on_watchdog_abort(uint32_t seq, const std::string& msg) {
    MCCL_ERROR("Rank %d: watchdog abort — %s", getRank(), msg.c_str());
    metrics_->record_error();
    abort_all_inflight_works("watchdog timeout: " + msg);
    if (transport_) {
        transport_->send_abort(seq, msg);
        transport_->shutdown();
    }
}

void ProcessGroupMCCL::on_peer_death(int peer_rank) {
    MCCL_ERROR("Rank %d: peer %d is dead", getRank(), peer_rank);
    metrics_->record_error();
    abort_all_inflight_works("peer " + std::to_string(peer_rank) + " died");
}

at::Tensor ProcessGroupMCCL::ensure_contiguous(const at::Tensor& tensor) {
    if (tensor.is_contiguous()) return tensor;
    MCCL_DEBUG("Cloning non-contiguous tensor to contiguous");
    return tensor.contiguous();
}

namespace {

// Collectives that write results in place must receive contiguous tensors.
// Silently cloning (the old behavior) would write results into the clone and
// drop them: the caller's tensor would never be updated.  Match NCCL and
// reject loudly instead.
void require_contiguous_output(const at::Tensor& tensor, const char* op_name) {
    MCCL_CHECK_TENSOR(tensor.is_contiguous(),
                      std::string("MCCL ") + op_name +
                      " writes results in place and requires a contiguous tensor. "
                      "Call .contiguous() and copy the result back yourself.");
}

// Reductions only support float dtypes (CPU vDSP + Metal kernels).  Reject at
// validation time instead of throwing from an engine thread mid-collective.
void require_reduce_dtype(const at::Tensor& tensor, const char* op_name) {
    auto dtype = tensor.scalar_type();
    MCCL_CHECK_TENSOR(
        is_supported_reduce_dtype(dtype),
        std::string("MCCL ") + op_name +
        " supports float32/float16/bfloat16/int/bool only, got: " +
        std::string(at::toString(dtype)));
}

} // anonymous namespace

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::make_completed_work(
    c10d::OpType op_type, const std::vector<at::Tensor>& tensors) {
    uint32_t seq = collective_seq_.fetch_add(1);
    auto work = c10::make_intrusive<WorkMCCL>(op_type, seq, tensors);
    work->markComplete();
    return work;
}

// TODO: compressed_send/compressed_recv use the legacy serial transport path
// (blocking stage_for_send with internal mps_sync, serial send_chunks/recv_chunks).
// When compression is enabled, the overlapped transport and nosync staging are
// bypassed. A future optimization would integrate compression with the
// send_recv_overlap path.

namespace {

// Stable per-tensor identity for stateful compression (TopK error feedback).
// Uses the original tensor's storage address + offset, which is stable across
// steps for DDP gradient buckets — NOT the staging-buffer address, which is
// shared between tensors.
inline uint64_t compression_stable_id(const at::Tensor& tensor) {
    return reinterpret_cast<uint64_t>(tensor.storage().data()) +
           static_cast<uint64_t>(tensor.storage_offset()) * tensor.element_size();
}

} // anonymous namespace

void ProcessGroupMCCL::compressed_send(int peer, OpType op, uint32_t seq,
                                       uint32_t tid, const at::Tensor& tensor) {
    MCCL_CHECK(tensor.scalar_type() != at::kBFloat16 || !compressor_,
               "BFloat16 tensors are not supported with compression enabled. "
               "Disable compression (MCCL_COMPRESSION=none) or use float32/float16.");

    StagingBuffer staged = stage_for_send(tensor);

    if (compressor_) {
        size_t max_comp = compressor_->max_compressed_size(staged.nbytes);
        PooledBuffer comp_buf(staging_memory_pool(), max_comp);

        size_t comp_size = compressor_->compress(
            staged.data, staged.nbytes,
            comp_buf.data(), max_comp, tensor.scalar_type(),
            compression_stable_id(tensor));

        // Two messages: a 4-byte size header, then EXACTLY comp_size bytes.
        // (Padding to max_compressed_size — the old scheme — transmitted the
        // full worst case regardless of ratio, negating TopK's savings.)
        uint32_t wire_size = static_cast<uint32_t>(comp_size);
        MCCL_CHECK(transport_->send_chunks(peer, op, seq, tid,
                                           &wire_size, sizeof(wire_size)),
                   "compressed_send size header failed");
        MCCL_CHECK(transport_->send_chunks(peer, op, seq, tid,
                                           comp_buf.data(), comp_size),
                   "compressed_send send_chunks failed");
        metrics_->record_transport_bytes(sizeof(wire_size) + comp_size, true);
    } else {
        MCCL_CHECK(transport_->send_chunks(peer, op, seq, tid, staged.data, staged.nbytes),
                   "compressed_send send_chunks (uncompressed) failed");
        metrics_->record_transport_bytes(staged.nbytes, true);
    }
}

void ProcessGroupMCCL::compressed_recv(int peer, OpType op, uint32_t seq,
                                       uint32_t tid, const at::Tensor& tensor) {
    MCCL_CHECK(tensor.scalar_type() != at::kBFloat16 || !compressor_,
               "BFloat16 tensors are not supported with compression enabled. "
               "Disable compression (MCCL_COMPRESSION=none) or use float32/float16.");

    size_t nbytes = tensor_nbytes(tensor);

    if (compressor_) {
        size_t max_comp = compressor_->max_compressed_size(nbytes);

        uint32_t wire_size = 0;
        MCCL_CHECK(transport_->recv_chunks(peer, op, seq, tid,
                                           &wire_size, sizeof(wire_size)),
                   "compressed_recv size header failed");
        MCCL_CHECK(wire_size > 0 && wire_size <= max_comp,
                   "Compressed payload size " + std::to_string(wire_size) +
                   " out of range (max " + std::to_string(max_comp) + ")");

        PooledBuffer comp_buf(staging_memory_pool(), wire_size);
        MCCL_CHECK(transport_->recv_chunks(peer, op, seq, tid,
                                           comp_buf.data(), wire_size),
                   "compressed_recv recv_chunks failed");

        PooledBuffer decomp_buf(staging_memory_pool(), nbytes);
        compressor_->decompress(
            comp_buf.data(), wire_size, decomp_buf.data(), nbytes,
            tensor.scalar_type());
        unstage_from_recv(tensor, decomp_buf.data(), nbytes);
        metrics_->record_transport_bytes(sizeof(wire_size) + wire_size, false);
    } else {
        PooledBuffer recv_buf(staging_memory_pool(), nbytes);
        MCCL_CHECK(transport_->recv_chunks(peer, op, seq, tid, recv_buf.data(), nbytes),
                   "compressed_recv recv_chunks (uncompressed) failed");
        unstage_from_recv(tensor, recv_buf.data(), nbytes);
        metrics_->record_transport_bytes(nbytes, false);
    }
}


// ── allreduce ───────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {

    MCCL_CHECK_TENSOR(tensors.size() == 1,
                      "MCCL allreduce expects exactly one tensor");

    at::Tensor& tensor = tensors[0];
    MCCL_CHECK_TENSOR(tensor.is_mps(),
                      "MCCL requires MPS tensors");

    require_contiguous_output(tensor, "allreduce");
    check_single_tensor(tensor);
    require_reduce_dtype(tensor, "allreduce");
    if (tensor.numel() == 0 || getSize() == 1) {
        // Zero elements, or single rank (reduction of one contribution is
        // the identity for SUM/AVG/MIN/MAX/PRODUCT): nothing to do.
        return make_completed_work(c10d::OpType::ALLREDUCE, tensors);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::ALLREDUCE, seq, std::vector<at::Tensor>{tensor});

    auto tensor_copy = tensor;
    auto work_ptr = work;
    int ws = getSize();
    size_t nbytes = tensor_nbytes(tensor);

    c10d::ReduceOp::RedOpType red_op = opts.reduceOp;
    const bool defer_mps_sync_to_engine = opts.asyncOp;

    register_work(seq, work);
    watchdog_->watch(seq, "allreduce");
    metrics_->op_start(seq, "allreduce", nbytes);

    auto sync_t0 = std::chrono::steady_clock::now();
    uint64_t sync_val = 0;
    if (!defer_mps_sync_to_engine) {
        sync_val = sync_mps_nonblocking(overlap_comm_);
    }
    auto sync_t1 = std::chrono::steady_clock::now();
    double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
    metrics_->record_phase(seq, sync_ms, 0, 0);

    if (defer_mps_sync_to_engine) {
        MCCL_DEBUG("allreduce seq=%u: asyncOp=true, MPS sync deferred to ProgressEngine", seq);
    }

    if (ws == 2) {
        // ── Two-rank path: ALL network I/O goes through net_engine_for(peer) ──
        // This serializes all sends/recvs to the same peer on one thread,
        // preventing protocol message interleaving. For large messages, the
        // reduce phase chains to reduce_engine for bucket overlap.
        int peer = 1 - getRank();

        if (nbytes > transport_->config().small_msg_threshold) {
            // Large message: split net/reduce across engines for bucket overlap
            auto shared_recv_buf = std::make_shared<PooledBuffer>(staging_memory_pool(), nbytes);

            net_engine_for(peer).submit(
                [this, tensor_copy, seq, red_op, sync_val, defer_mps_sync_to_engine,
                 peer, nbytes, shared_recv_buf]() mutable {
                    begin_execute(seq);  // watchdog arm + execute-start metric
                    StagingBuffer staged = stage_for_send_collective(tensor_copy);
                    auto net_t0 = std::chrono::steady_clock::now();
                    MCCL_CHECK(transport_->send_recv_overlap(
                        peer, OpType::ALLREDUCE, seq, 0, staged.data, nbytes,
                        peer, OpType::ALLREDUCE, seq, 0, shared_recv_buf->data(), nbytes),
                        "allreduce two_rank_split net phase failed");
                    auto net_t1 = std::chrono::steady_clock::now();

                    double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
                    metrics_->record_phase(seq, 0, net_ms, 0);
                    metrics_->record_transport_bytes(nbytes, true);
                    metrics_->record_transport_bytes(nbytes, false);
                    MCCL_INFO("allreduce seq=%u: algo=two_rank_split net=%.1fms nbytes=%zu",
                              seq, net_ms, nbytes);
                },
                [this, tensor_copy, seq, red_op, nbytes, shared_recv_buf, work_ptr]() mutable {
                    reduce_engine_->submit(
                        [this, tensor_copy, seq, red_op, nbytes, shared_recv_buf]() mutable {
                            watchdog_->touch(seq);  // re-arm for the reduce phase
                            auto red_t0 = std::chrono::steady_clock::now();
                            bool cpu_ok = prefer_cpu_unified_buffer_path(tensor_copy);
                            int64_t count = tensor_copy.numel();

                            if (cpu_ok) {
                                MPSBufferView view = extract_mps_buffer(tensor_copy);
                                float* dst = static_cast<float*>(view.cpu_ptr);
                                const float* src = static_cast<const float*>(shared_recv_buf->data());
                                if (red_op == c10d::ReduceOp::AVG) {
                                    cpu_accumulate_and_scale(dst, src, count, 0.5f);
                                } else {
                                    cpu_reduce_op(dst, src, count, red_op);
                                }
                            } else {
                                at::Tensor incoming = torch::empty_like(tensor_copy);
                                unstage_from_recv(incoming, shared_recv_buf->data(), nbytes);
                                if (red_op == c10d::ReduceOp::AVG) {
                                    metal_accumulate_and_scale(tensor_copy, incoming, 0.5);
                                } else {
                                    metal_reduce_op_fenced(tensor_copy, incoming, red_op);
                                }
                            }
                            if (cpu_ok) {
                                mps_stream_sync_after_cpu_mps_buffer_write();
                            }
                            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());

                            auto red_t1 = std::chrono::steady_clock::now();
                            double red_ms = std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
                            metrics_->record_phase(seq, 0, 0, red_ms);
                        },
                        [this, work_ptr, seq]() {
                            unregister_work(seq);
                            watchdog_->complete(seq);
                            metrics_->op_end(seq);
                            work_ptr->markComplete();
                        },
                        [this, work_ptr, seq](std::exception_ptr e) {
                            unregister_work(seq);
                            watchdog_->complete(seq);
                            metrics_->op_end(seq);
                            metrics_->record_error();
                            work_ptr->markError(e);
                        }
                    );
                },
                [this, work_ptr, seq](std::exception_ptr e) {
                    unregister_work(seq);
                    watchdog_->complete(seq);
                    metrics_->op_end(seq);
                    metrics_->record_error();
                    work_ptr->markError(e);
                }
            );
        } else {
            // Small message: run entirely on net_engine (no split needed, but
            // must use net_engine to avoid concurrent socket access with large ops)
            net_engine_for(peer).submit(
                [this, tensor_copy, seq, red_op, sync_val, defer_mps_sync_to_engine]() mutable {
                    begin_execute(seq);
                    allreduce_small(tensor_copy, seq, red_op);
                    MCCL_INFO("allreduce seq=%u: algo=small nbytes=%zu", seq, tensor_nbytes(tensor_copy));
                },
                [this, work_ptr, seq]() {
                    unregister_work(seq);
                    watchdog_->complete(seq);
                    metrics_->op_end(seq);
                    work_ptr->markComplete();
                },
                [this, work_ptr, seq](std::exception_ptr e) {
                    unregister_work(seq);
                    watchdog_->complete(seq);
                    metrics_->op_end(seq);
                    metrics_->record_error();
                    work_ptr->markError(e);
                }
            );
        }
    } else {
        // ── 3+ ranks: collective executor pool ──
        // Workers dequeue FIFO; transport_collective_mu_ serializes TCP I/O.
        collective_pool_->submit(
            [this, tensor_copy, seq, ws, nbytes, red_op, sync_val, defer_mps_sync_to_engine]() mutable {
                begin_execute(seq);
                std::lock_guard<std::mutex> transport_guard(transport_collective_mu_);
                const auto exec_t0 = std::chrono::steady_clock::now();
                const char* algo = "unknown";
                if (nbytes <= transport_->config().small_msg_threshold) {
                    algo = "tree_small";
                    allreduce_small(tensor_copy, seq, red_op);
                } else {
                    algo = use_chunked_ring_default() ? "ring_chunked" : "ring";
                    allreduce_ring_dispatch(tensor_copy, seq, red_op);
                }
                const double exec_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - exec_t0).count();
                const double gbps =
                    exec_ms > 0 ? (nbytes * 8.0) / (exec_ms * 1e6) : 0.0;
                MCCL_INFO("allreduce seq=%u: algo=%s nbytes=%zu exec=%.2fms (%.2f Gbps algbw)",
                          seq, algo, nbytes, exec_ms, gbps);
            },
            [this, work_ptr, seq]() {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                work_ptr->markComplete();
            },
            [this, work_ptr, seq](std::exception_ptr e) {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(e);
            }
        );
    }

    return work;
}


c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {

    MCCL_CHECK_TENSOR(!tensors.empty(), "allreduce_coalesced: empty tensor list");

    // Flatten all tensors into one contiguous buffer for a single collective op.
    std::vector<at::Tensor> flat_inputs;
    flat_inputs.reserve(tensors.size());
    for (auto& t : tensors) {
        MCCL_CHECK_TENSOR(t.is_mps(), "MCCL requires MPS tensors");
        require_reduce_dtype(t, "allreduce_coalesced");
        if (t.numel() == 0) continue;
        flat_inputs.push_back(t.flatten());
    }
    if (flat_inputs.empty() || getSize() == 1) {
        return make_completed_work(c10d::OpType::ALLREDUCE, tensors);
    }
    at::Tensor flat = at::cat(flat_inputs, 0);

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(flat);
    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::ALLREDUCE, seq, std::vector<at::Tensor>{flat});
    auto work_ptr = work;
    int ws = getSize();
    c10d::ReduceOp::RedOpType red_op = opts.reduceOp;

    register_work(seq, work);
    watchdog_->watch(seq, "allreduce_coalesced");
    metrics_->op_start(seq, "allreduce_coalesced", nbytes);

    auto sync_t0 = std::chrono::steady_clock::now();
    uint64_t sync_val = sync_mps_nonblocking(overlap_comm_);
    auto sync_t1 = std::chrono::steady_clock::now();
    metrics_->record_phase(seq, std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count(), 0, 0);

    // Capture the tensor list + flat buffer for the engine lambda
    auto tensors_copy = tensors;
    auto flat_copy = flat;

    ProgressEngine& coalesced_engine =
        (ws >= 3 && collective_pool_) ? *collective_pool_ : *reduce_engine_;
    coalesced_engine.submit(
        [this, flat_copy, tensors_copy, seq, ws, nbytes, red_op, sync_val]() mutable {
            begin_execute(seq);
            std::lock_guard<std::mutex> transport_guard(transport_collective_mu_);
            if (ws == 2) {
                allreduce_two_rank(flat_copy, seq, red_op);
            } else if (ws >= 3) {
                if (nbytes <= transport_->config().small_msg_threshold) {
                    allreduce_small(flat_copy, seq, red_op);
                } else {
                    allreduce_ring_dispatch(flat_copy, seq, red_op);
                }
            } else {
                allreduce_ring(flat_copy, seq, red_op);
            }

            // Scatter the reduced flat buffer back into the original tensors
            size_t offset = 0;
            for (auto& t : tensors_copy) {
                size_t t_nbytes = tensor_nbytes(t);
                auto src_slice = flat_copy.narrow(0, static_cast<int64_t>(offset / flat_copy.element_size()),
                                                  t.numel());
                t.view_as(src_slice).copy_(src_slice);
                offset += t_nbytes;
            }
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}


void ProcessGroupMCCL::allreduce_two_rank(at::Tensor& tensor, uint32_t seq,
                                           c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int peer = 1 - rank;
    bool cpu_ok = tensor_cpu_accessible(tensor);
    size_t nbytes = tensor_nbytes(tensor);
    int64_t count = tensor.numel();
    bool use_fast = (tensor.scalar_type() == at::kFloat) && !compressor_ &&
                    fp32_cpu_reduce_enabled();

    if (use_fast) {
        StagingBuffer staged = stage_for_send_collective(tensor);

        constexpr size_t RS_AG_THRESHOLD = 8 * 1024 * 1024;  // 8 MB
        constexpr size_t REDUCE_CHUNK = 2 * 1024 * 1024;   // 2 MB

        if (nbytes >= RS_AG_THRESHOLD && cpu_ok) {
            // Reduce-scatter + allgather: each rank reduces only its half, then
            // they exchange the reduced halves.  Halves reduction work and
            // overlaps recv with reduce using chunked pipelining.
            MPSBufferView view = extract_mps_buffer(tensor);
            float* base = static_cast<float*>(view.cpu_ptr);

            // Split on ELEMENT boundaries.  A raw byte split (nbytes / 2) is
            // misaligned for odd element counts: the second half would start
            // mid-float and the boundary element would be reduced by neither
            // rank, splicing garbage into the result.
            int64_t half_count = count / 2;
            // Rank 0 owns [0, half_count); rank 1 owns [half_count, count).
            int64_t my_count   = (rank == 0) ? half_count : (count - half_count);
            int64_t peer_count = count - my_count;
            size_t my_off   = (rank == 0) ? 0 : static_cast<size_t>(half_count) * sizeof(float);
            size_t peer_off = (rank == 0) ? static_cast<size_t>(half_count) * sizeof(float) : 0;
            size_t my_len   = static_cast<size_t>(my_count) * sizeof(float);
            size_t peer_len = static_cast<size_t>(peer_count) * sizeof(float);

            // Chunk counts per region (regions may differ by one element).
            size_t send_nchunks = (peer_len + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
            size_t recv_nchunks = (my_len + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
            PooledBuffer recv_buf(staging_memory_pool(), REDUCE_CHUNK);

            auto net_t0 = std::chrono::steady_clock::now();
            double red_ms_accum = 0;

            // ── Phase 1: reduce-scatter ──
            // Send peer's half, recv my half in chunks, reduce each chunk.
            std::atomic<bool> send_ok{false};
            std::thread send_thread([&]() {
                bool ok = true;
                for (size_t c = 0; c < send_nchunks && ok; ++c) {
                    size_t off = c * REDUCE_CHUNK;
                    size_t len = std::min(REDUCE_CHUNK, peer_len - off);
                    uint32_t tid = static_cast<uint32_t>(c + 1);
                    ok = transport_->send_chunks(
                        peer, OpType::ALLREDUCE, seq, tid,
                        static_cast<const uint8_t*>(staged.data) + peer_off + off, len);
                }
                send_ok.store(ok, std::memory_order_release);
            });

            // recv_chunks/MCCL_CHECK can throw; unwinding past a joinable
            // std::thread calls std::terminate and kills the process.
            // Catch, join, then rethrow.
            std::exception_ptr recv_ex;
            try {
                for (size_t c = 0; c < recv_nchunks; ++c) {
                    watchdog_->touch(seq);  // per-chunk progress re-arms the deadline
                    size_t off = c * REDUCE_CHUNK;
                    size_t len = std::min(REDUCE_CHUNK, my_len - off);
                    int64_t chunk_count = static_cast<int64_t>(len / sizeof(float));
                    uint32_t tid = static_cast<uint32_t>(c + 1);

                    MCCL_CHECK(transport_->recv_chunks(
                        peer, OpType::ALLREDUCE, seq, tid,
                        recv_buf.data(), len),
                        "allreduce_two_rank RS recv chunk failed");

                    auto rc0 = std::chrono::steady_clock::now();
                    float* chunk_dst = base + my_off / sizeof(float) + off / sizeof(float);
                    const float* chunk_src = static_cast<const float*>(recv_buf.data());
                    if (op == c10d::ReduceOp::AVG) {
                        cpu_accumulate_and_scale(chunk_dst, chunk_src, chunk_count, 0.5f);
                    } else {
                        cpu_reduce_op(chunk_dst, chunk_src, chunk_count, op);
                    }
                    auto rc1 = std::chrono::steady_clock::now();
                    red_ms_accum += std::chrono::duration<double, std::milli>(rc1 - rc0).count();
                }
            } catch (...) {
                recv_ex = std::current_exception();
            }

            send_thread.join();
            if (recv_ex) std::rethrow_exception(recv_ex);
            MCCL_CHECK(send_ok.load(std::memory_order_acquire),
                       "allreduce_two_rank RS send failed");

            // ── Phase 2: allgather ──
            // Exchange reduced halves so both ranks have the full result.
            // tid base must be identical on both ranks: derive it from both
            // region chunk counts (symmetric on both sides).
            uint32_t ag_base_tid = static_cast<uint32_t>(send_nchunks + recv_nchunks + 1);
            MCCL_CHECK(transport_->send_recv_overlap(
                peer, OpType::ALLREDUCE, seq, ag_base_tid,
                static_cast<const uint8_t*>(view.cpu_ptr) + my_off, my_len,
                peer, OpType::ALLREDUCE, seq, ag_base_tid,
                static_cast<uint8_t*>(view.cpu_ptr) + peer_off, peer_len),
                "allreduce_two_rank AG send_recv_overlap failed");

            auto net_t1 = std::chrono::steady_clock::now();
            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());

            double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
            double gbps = (nbytes * 2.0 * 8.0) / (net_ms / 1000.0) / 1e9;
            MCCL_INFO("allreduce_two_rank(RS+AG): %zu bytes (%zu RS chunks), "
                      "wall=%.1fms (%.2f Gbps), reduce=%.1fms",
                      nbytes, recv_nchunks, net_ms, gbps, red_ms_accum);
            metrics_->record_phase(seq, 0, net_ms, red_ms_accum);
        } else {
            // Original path: full send_recv_overlap then reduce.
            PooledBuffer recv_buf(staging_memory_pool(), nbytes);

            auto net_t0 = std::chrono::steady_clock::now();
            MCCL_CHECK(transport_->send_recv_overlap(
                peer, OpType::ALLREDUCE, seq, 0, staged.data, nbytes,
                peer, OpType::ALLREDUCE, seq, 0, recv_buf.data(), nbytes),
                "allreduce_two_rank send_recv_overlap failed");
            auto net_t1 = std::chrono::steady_clock::now();

            auto red_t0 = std::chrono::steady_clock::now();
            if (cpu_ok) {
                MPSBufferView view = extract_mps_buffer(tensor);
                float* dst = static_cast<float*>(view.cpu_ptr);
                const float* src = static_cast<const float*>(recv_buf.data());
                if (op == c10d::ReduceOp::AVG) {
                    cpu_accumulate_and_scale(dst, src, count, 0.5f);
                } else {
                    cpu_reduce_op(dst, src, count, op);
                }
            } else {
                float* dst = static_cast<float*>(staged.data);
                const float* src = static_cast<const float*>(recv_buf.data());
                if (op == c10d::ReduceOp::AVG) {
                    cpu_accumulate_and_scale(dst, src, count, 0.5f);
                } else {
                    cpu_reduce_op(dst, src, count, op);
                }
                unstage_from_recv(tensor, staged.data, nbytes);
            }
            auto red_t1 = std::chrono::steady_clock::now();

            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());

            double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
            double red_ms = std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
            double gbps = (nbytes * 2.0 * 8.0) / (net_ms / 1000.0) / 1e9;
            MCCL_INFO("allreduce_two_rank: %zu bytes, net=%.1fms (%.2f Gbps), reduce=%.1fms cpu_ok=%d",
                      nbytes, net_ms, gbps, red_ms, (int)cpu_ok);
            metrics_->record_phase(seq, 0, net_ms, red_ms);
        }

        metrics_->record_transport_bytes(nbytes, true);
        metrics_->record_transport_bytes(nbytes, false);
        mps_stream_sync_after_cpu_mps_buffer_write();
    } else {
        // f16/bf16 or compressed path: Metal pipeline
        at::Tensor recv_tensor = torch::empty_like(tensor);

        if (rank == 0) {
            compressed_send(peer, OpType::ALLREDUCE, seq, 0, tensor);
            compressed_recv(peer, OpType::ALLREDUCE, seq, 0, recv_tensor);
        } else {
            compressed_recv(peer, OpType::ALLREDUCE, seq, 0, recv_tensor);
            compressed_send(peer, OpType::ALLREDUCE, seq, 0, tensor);
        }

        if (op == c10d::ReduceOp::AVG) {
            metal_accumulate_and_scale(tensor, recv_tensor, 1.0 / 2.0);
        } else {
            metal_reduce_op_fenced(tensor, recv_tensor, op);
        }
    }
}

void ProcessGroupMCCL::allreduce_ring_chunked(at::Tensor& tensor, uint32_t seq,
                                               c10d::ReduceOp::RedOpType op) {
    // Gloo-style ring allreduce with 2P chunks for double buffering.
    // 4*P communication steps but only 2*S bytes on wire (vs P*S for basic ring).
    // Steps are serial (data dependencies), but the 2P chunking halves per-step
    // data volume, improving bandwidth utilization.
    int rank = getRank();
    int ws = getSize();
    size_t elem_size = tensor.element_size();
    int64_t total_elems = tensor.numel();

    int64_t chunk_elems = (total_elems + (2 * ws) - 1) / (2 * ws);
    bool use_cpu = (tensor.scalar_type() == at::kFloat) &&
                   prefer_cpu_unified_buffer_path(tensor);

    int left = (rank - 1 + ws) % ws;
    int right = (rank + 1) % ws;

    at::Tensor flat = tensor.flatten();
    std::vector<at::Tensor> chunks;
    for (int c = 0; c < 2 * ws; c++) {
        int64_t start = c * chunk_elems;
        int64_t len = std::min(chunk_elems, total_elems - start);
        if (len <= 0) {
            chunks.push_back(torch::empty(0, tensor.options()));
        } else {
            chunks.push_back(flat.narrow(0, start, len));
        }
    }

    // Staging tensors consumed by async Metal reduce kernels stay alive
    // until the tail drain (or the guard's own drain on error paths).
    IncomingKeepAlive incoming_keep;

    if (ring_pipeline_enabled()) {
        // Streaming TX/RX pipeline over the same (verified) 2P-chunk
        // schedule: both link directions and the reduce stay busy
        // concurrently instead of lock-stepping per step.
        std::vector<RingStep> plan;
        plan.reserve(4 * (ws - 1));
        for (int step = 0; step < 2 * (ws - 1); step++) {
            RingStep st;
            st.send_idx = (rank * 2 - step + 2 * ws) % (2 * ws);
            st.recv_idx = (rank * 2 - step - 2 + 2 * ws) % (2 * ws);
            st.send_tid = (static_cast<uint32_t>(step) << 16) | st.send_idx;
            st.recv_tid = (static_cast<uint32_t>(step) << 16) | st.recv_idx;
            st.kind = RingRecvKind::REDUCE;
            plan.push_back(st);
        }
        for (int step = 0; step < 2 * (ws - 1); step++) {
            uint32_t ag_step = static_cast<uint32_t>(2 * (ws - 1) + step);
            RingStep st;
            st.send_idx = (rank * 2 + 2 - step + 2 * ws) % (2 * ws);
            st.recv_idx = (rank * 2 - step + 2 * ws) % (2 * ws);
            st.send_tid = (ag_step << 16) | st.send_idx;
            st.recv_tid = (ag_step << 16) | st.recv_idx;
            st.kind = RingRecvKind::COPY;
            plan.push_back(st);
        }
        // After RS, rank r holds reduced chunks {2r+1, 2r+2}: the chunk sent
        // at global step g is the chunk received at step g-2, so lookahead=2
        // (the first two sends are the rank's own initial chunks).
        RingPipelineCtx ctx{transport_.get(), watchdog_.get(), metrics_.get(),
                            seq, left, right, OpType::ALLREDUCE, op, use_cpu,
                            &incoming_keep.tensors};
        run_ring_pipeline(ctx, chunks, plan, /*lookahead=*/2);
    } else {
    PooledBuffer recv_buf_pool(staging_memory_pool(), chunk_elems * elem_size);

    // Per-chunk kernel fences for the Metal reduce path.  metal_reduce_op()
    // commits without waiting; a chunk it touched must not be staged for a
    // network send (or overwritten by an incoming allgather chunk) until its
    // kernel completes.  The MCCL queue is serial, so waiting on a chunk's
    // event value also fences all kernels committed before it.
    const bool use_event_fence = !use_cpu && event_sync_available();
    std::vector<uint64_t> chunk_pending(2 * ws, 0);
    auto fence_chunk = [&](int idx) {
        if (use_event_fence && chunk_pending[idx]) {
            wait_for_mccl_fence(chunk_pending[idx]);
            chunk_pending[idx] = 0;
        }
    };
    auto arm_chunk = [&](int idx) {
        if (use_event_fence) {
            uint64_t v = next_fence_event_value();
            signal_mccl_fence_gpu(v);
            chunk_pending[idx] = v;
        } else if (!use_cpu) {
            // No event sync available: fall back to a blocking queue drain.
            metal_sync_queue_only();
        }
    };

    // ── Phase 1: Reduce-scatter (2*(ws-1) serial steps) ──
    for (int step = 0; step < 2 * (ws - 1); step++) {
        watchdog_->touch(seq);  // per-step progress re-arms the deadline
        int send_chunk_idx = (rank * 2 - step + 2 * ws) % (2 * ws);
        int recv_chunk_idx = (rank * 2 - step - 2 + 2 * ws) % (2 * ws);

        at::Tensor& send_chunk = chunks[send_chunk_idx];
        at::Tensor& recv_chunk = chunks[recv_chunk_idx];

        if (send_chunk.numel() == 0 && recv_chunk.numel() == 0) continue;

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_chunk_idx;
        uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_chunk_idx;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            fence_chunk(send_chunk_idx);
            staged = stage_for_send_collective(send_chunk);
        }

        MCCL_CHECK(transport_->send_recv_overlap(
            right, OpType::ALLREDUCE, seq, step_tid,
            staged.data, send_bytes,
            left, OpType::ALLREDUCE, seq, recv_tid,
            recv_buf_pool.data(), recv_bytes),
            "allreduce_ring_chunked reduce-scatter step " + std::to_string(step) + " failed");

        if (recv_bytes > 0) {
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                cpu_reduce_op(
                    static_cast<float*>(chunk_view.cpu_ptr),
                    static_cast<const float*>(recv_buf_pool.data()),
                    recv_chunk.numel(), op);
            } else {
                at::Tensor incoming = torch::empty_like(recv_chunk);
                unstage_from_recv(incoming, recv_buf_pool.data(), recv_bytes);
                uint64_t v = reduce_chunk_metal_fenced(
                    recv_chunk, std::move(incoming), op, &incoming_keep.tensors);
                if (use_event_fence && v > 0) {
                    chunk_pending[recv_chunk_idx] = v;
                }
            }
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    // ── Phase 2: Allgather (2*(ws-1) serial steps) ──
    // After reduce-scatter, rank r holds the fully reduced chunks
    // {2r+1, 2r+2}.  Step 0 sends 2r+2, step 1 sends 2r+1, and each
    // subsequent step forwards the chunk received two steps earlier:
    // send (2r+2-step), recv (2r-step).  (The previous schedule used
    // +step, which sent chunks that were never fully reduced and
    // overwrote reduced ones — silent corruption for ws >= 3.)
    for (int step = 0; step < 2 * (ws - 1); step++) {
        watchdog_->touch(seq);
        int send_chunk_idx = (rank * 2 + 2 - step + 2 * ws) % (2 * ws);
        int recv_chunk_idx = (rank * 2 - step + 2 * ws) % (2 * ws);

        at::Tensor& send_chunk = chunks[send_chunk_idx];
        at::Tensor& recv_chunk = chunks[recv_chunk_idx];

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        uint32_t ag_step = static_cast<uint32_t>(2 * (ws - 1) + step);
        uint32_t step_tid = (ag_step << 16) | send_chunk_idx;
        uint32_t recv_tid_ag = (ag_step << 16) | recv_chunk_idx;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            fence_chunk(send_chunk_idx);
            staged = stage_for_send_collective(send_chunk);
        }

        MCCL_CHECK(transport_->send_recv_overlap(
            right, OpType::ALLREDUCE, seq, step_tid,
            staged.data, send_bytes,
            left, OpType::ALLREDUCE, seq, recv_tid_ag,
            recv_buf_pool.data(), recv_bytes),
            "allreduce_ring_chunked allgather step " + std::to_string(step) + " failed");

        if (recv_bytes > 0) {
            // The incoming chunk replaces local data; make sure no reduce
            // kernel from the RS phase is still in flight on it.
            fence_chunk(recv_chunk_idx);
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                memcpy(chunk_view.cpu_ptr, recv_buf_pool.data(), recv_bytes);
            } else {
                unstage_from_recv(recv_chunk, recv_buf_pool.data(), recv_bytes);
            }
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }
    }  // end lock-step fallback

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr), total_elems, 1.0f / ws);
        }
        if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
        mps_stream_sync_after_cpu_mps_buffer_write();
    } else {
        if (op == c10d::ReduceOp::AVG) {
            metal_scale_inplace(tensor, 1.0 / ws);
        }
        metal_sync_queue_only();
    }
}

void ProcessGroupMCCL::allreduce_ring_dispatch(at::Tensor& tensor, uint32_t seq,
                                                c10d::ReduceOp::RedOpType op) {
    if (!use_chunked_ring_default()) {
        allreduce_ring(tensor, seq, op);
        return;
    }
    try {
        allreduce_ring_chunked(tensor, seq, op);
    } catch (...) {
        if (!ring_fallback_basic_enabled(getSize())) {
            throw;
        }
        MCCL_WARN("allreduce_ring_chunked failed (seq=%u rank=%d ws=%d nbytes=%zu); "
                  "retrying with basic ring (MCCL_RING_FALLBACK_BASIC)",
                  seq, getRank(), getSize(), tensor_nbytes(tensor));
        allreduce_ring(tensor, seq, op);
    }
}

void ProcessGroupMCCL::allreduce_ring(at::Tensor& tensor, uint32_t seq,
                                      c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int ws = getSize();
    size_t elem_size = tensor.element_size();
    int64_t total_elems = tensor.numel();
    int64_t chunk_elems = (total_elems + ws - 1) / ws;
    bool use_cpu = (tensor.scalar_type() == at::kFloat) &&
                   prefer_cpu_unified_buffer_path(tensor);

    int left = (rank - 1 + ws) % ws;
    int right = (rank + 1) % ws;

    at::Tensor flat = tensor.flatten();
    std::vector<at::Tensor> chunks;
    for (int c = 0; c < ws; c++) {
        int64_t start = c * chunk_elems;
        int64_t len = std::min(chunk_elems, total_elems - start);
        if (len <= 0) {
            chunks.push_back(torch::empty(0, tensor.options()));
        } else {
            chunks.push_back(flat.narrow(0, start, len));
        }
    }

    // Keep-alive for async-kernel staging tensors (see chunked ring).
    IncomingKeepAlive incoming_keep;

    if (ring_pipeline_enabled()) {
        // Streaming TX/RX pipeline over the plain-ring schedule.  The chunk
        // sent at step g is the chunk received at step g-1: lookahead=1.
        std::vector<RingStep> plan;
        plan.reserve(2 * (ws - 1));
        for (int step = 0; step < ws - 1; step++) {
            RingStep st;
            st.send_idx = (rank - step + ws) % ws;
            st.recv_idx = (rank - step - 1 + ws) % ws;
            st.send_tid = (static_cast<uint32_t>(step) << 16) | st.send_idx;
            st.recv_tid = (static_cast<uint32_t>(step) << 16) | st.recv_idx;
            st.kind = RingRecvKind::REDUCE;
            plan.push_back(st);
        }
        for (int step = 0; step < ws - 1; step++) {
            uint32_t ag_step = static_cast<uint32_t>(ws - 1 + step);
            RingStep st;
            st.send_idx = (rank - step + 1 + ws) % ws;
            st.recv_idx = (rank - step + ws) % ws;
            st.send_tid = (ag_step << 16) | st.send_idx;
            st.recv_tid = (ag_step << 16) | st.recv_idx;
            st.kind = RingRecvKind::COPY;
            plan.push_back(st);
        }
        RingPipelineCtx ctx{transport_.get(), watchdog_.get(), metrics_.get(),
                            seq, left, right, OpType::ALLREDUCE, op, use_cpu,
                            &incoming_keep.tensors};
        run_ring_pipeline(ctx, chunks, plan, /*lookahead=*/1);
    } else {
    PooledBuffer recv_buf_pool(staging_memory_pool(), chunk_elems * elem_size);

    // Kernel fence for the Metal reduce path (see allreduce_ring_chunked).
    // In the plain ring the chunk sent at step s+1 is exactly the chunk
    // reduced at step s, so without this fence the socket can transmit
    // pre-reduction bytes while the kernel is still in flight.
    const bool use_event_fence = !use_cpu && event_sync_available();
    std::vector<uint64_t> chunk_pending(ws, 0);
    auto fence_chunk = [&](int idx) {
        if (use_event_fence && chunk_pending[idx]) {
            wait_for_mccl_fence(chunk_pending[idx]);
            chunk_pending[idx] = 0;
        }
    };
    auto arm_chunk = [&](int idx) {
        if (use_event_fence) {
            uint64_t v = next_fence_event_value();
            signal_mccl_fence_gpu(v);
            chunk_pending[idx] = v;
        } else if (!use_cpu) {
            metal_sync_queue_only();
        }
    };

    // ── Reduce-scatter phase (serial steps -- data dependencies between steps) ──
    for (int step = 0; step < ws - 1; step++) {
        watchdog_->touch(seq);  // per-step progress re-arms the deadline
        int send_idx = (rank - step + ws) % ws;
        int recv_idx = (rank - step - 1 + ws) % ws;
        uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
        uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;

        at::Tensor& send_chunk = chunks[send_idx];
        at::Tensor& recv_chunk = chunks[recv_idx];

        if (send_chunk.numel() == 0 && recv_chunk.numel() == 0) continue;

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            fence_chunk(send_idx);
            staged = stage_for_send_collective(send_chunk);
        }

        MCCL_CHECK(transport_->send_recv_overlap(
            right, OpType::ALLREDUCE, seq, step_tid,
            staged.data, send_bytes,
            left, OpType::ALLREDUCE, seq, recv_tid,
            recv_buf_pool.data(), recv_bytes),
            "allreduce_ring reduce-scatter step " + std::to_string(step) + " failed");

        if (recv_bytes > 0) {
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                cpu_reduce_op(
                    static_cast<float*>(chunk_view.cpu_ptr),
                    static_cast<const float*>(recv_buf_pool.data()),
                    recv_chunk.numel(), op);
            } else {
                at::Tensor incoming = torch::empty_like(recv_chunk);
                unstage_from_recv(incoming, recv_buf_pool.data(), recv_bytes);
                uint64_t v = reduce_chunk_metal_fenced(
                    recv_chunk, std::move(incoming), op, &incoming_keep.tensors);
                if (use_event_fence && v > 0) {
                    chunk_pending[recv_idx] = v;
                }
            }
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    // ── Allgather phase (serial steps) ──
    for (int step = 0; step < ws - 1; step++) {
        watchdog_->touch(seq);
        int send_idx = (rank - step + 1 + ws) % ws;
        int recv_idx = (rank - step + ws) % ws;
        uint32_t ag_step = static_cast<uint32_t>(ws - 1 + step);
        uint32_t step_tid = (ag_step << 16) | send_idx;
        uint32_t recv_tid_ag = (ag_step << 16) | recv_idx;

        at::Tensor& send_chunk = chunks[send_idx];
        at::Tensor& recv_chunk = chunks[recv_idx];

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            fence_chunk(send_idx);
            staged = stage_for_send_collective(send_chunk);
        }

        MCCL_CHECK(transport_->send_recv_overlap(
            right, OpType::ALLREDUCE, seq, step_tid,
            staged.data, send_bytes,
            left, OpType::ALLREDUCE, seq, recv_tid_ag,
            recv_buf_pool.data(), recv_bytes),
            "allreduce_ring allgather step " + std::to_string(step) + " failed");

        if (recv_bytes > 0) {
            // The incoming chunk replaces local data; fence any reduce kernel
            // from the RS phase still in flight on it.
            fence_chunk(recv_idx);
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                memcpy(chunk_view.cpu_ptr, recv_buf_pool.data(), recv_bytes);
            } else {
                unstage_from_recv(recv_chunk, recv_buf_pool.data(), recv_bytes);
            }
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }
    }  // end lock-step fallback

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr),
                              total_elems, 1.0f / ws);
        }
        if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
        mps_stream_sync_after_cpu_mps_buffer_write();
    } else {
        if (op == c10d::ReduceOp::AVG) {
            metal_scale_inplace(tensor, 1.0 / ws);
        }
        metal_sync_queue_only();
    }
}


void ProcessGroupMCCL::allreduce_small(at::Tensor& tensor, uint32_t seq,
                                       c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int ws = getSize();

    if (ws == 2) {
        allreduce_two_rank(tensor, seq, op);
        return;
    }

    // Recursive-doubling tree: 2 + log2 rounds instead of the rank-0 star's
    // 2(ws-1) serial transactions (46 -> ~6 at ws=24).  Requires CPU-visible
    // storage (Apple Silicon shared MPS = always) and no compression.
    {
        auto dtype = tensor.scalar_type();
        bool tree_ok = !compressor_ && tensor_cpu_accessible(tensor) &&
                       (dtype == at::kFloat || dtype == at::kHalf ||
                        dtype == at::kBFloat16 || is_integral_reduce_dtype(dtype));
        if (tree_ok) {
            allreduce_tree_small(tensor, seq, op);
            return;
        }
    }

    size_t nbytes = tensor_nbytes(tensor);

    bool use_cpu = (tensor.scalar_type() == at::kFloat) && !compressor_ &&
                   prefer_cpu_unified_buffer_path(tensor);

    if (use_cpu) {
        MPSBufferView view = extract_mps_buffer(tensor);
        float* dst = static_cast<float*>(view.cpu_ptr);
        int64_t count = tensor.numel();

        if (rank == 0) {
            PooledBuffer recv_buf(staging_memory_pool(), nbytes);
            for (int peer = 1; peer < ws; peer++) {
                MCCL_CHECK(transport_->recv_chunks(peer, OpType::ALLREDUCE, seq, 0,
                                                   recv_buf.data(), nbytes),
                           "allreduce_small recv from rank " + std::to_string(peer) + " failed");
                cpu_reduce_op(dst,
                              static_cast<const float*>(recv_buf.data()),
                              count, op);
                metrics_->record_transport_bytes(nbytes, false);
            }

            if (op == c10d::ReduceOp::AVG) {
                cpu_scale_inplace(dst, count, 1.0f / ws);
            }

            StagingBuffer staged = stage_for_send_collective(tensor);
            for (int peer = 1; peer < ws; peer++) {
                MCCL_CHECK(transport_->send_chunks(peer, OpType::ALLREDUCE, seq, 1,
                                                   staged.data, nbytes),
                           "allreduce_small send to rank " + std::to_string(peer) + " failed");
                metrics_->record_transport_bytes(nbytes, true);
            }
            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
        } else {
            StagingBuffer staged = stage_for_send_collective(tensor);
            MCCL_CHECK(transport_->send_chunks(0, OpType::ALLREDUCE, seq, 0,
                                               staged.data, nbytes),
                       "allreduce_small send to rank 0 failed");
            metrics_->record_transport_bytes(nbytes, true);

            MCCL_CHECK(transport_->recv_chunks(0, OpType::ALLREDUCE, seq, 1,
                                               dst, nbytes),
                       "allreduce_small recv from rank 0 failed");
            metrics_->record_transport_bytes(nbytes, false);
            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
        }
        mps_stream_sync_after_cpu_mps_buffer_write();
    } else {
        // f16 or compressed path: existing Metal pipeline
        if (rank == 0) {
            // Pin staging tensors until the queue drain below — the reduce
            // kernels read them asynchronously, and letting them die mid-loop
            // lets the MPS allocator recycle their buffers into the next
            // iteration's empty_like while the kernel still reads them.
            IncomingKeepAlive incoming_keep;
            incoming_keep.tensors.reserve(ws - 1);
            for (int peer = 1; peer < ws; peer++) {
                at::Tensor incoming = torch::empty_like(tensor);
                compressed_recv(peer, OpType::ALLREDUCE, seq, 0, incoming);
                metal_reduce_op_fenced(tensor, incoming, op);
                incoming_keep.tensors.push_back(std::move(incoming));
            }

            if (op == c10d::ReduceOp::AVG) {
                metal_scale_inplace(tensor, 1.0 / ws);
            }
            metal_sync_queue_only();

            for (int peer = 1; peer < ws; peer++) {
                compressed_send(peer, OpType::ALLREDUCE, seq, 1, tensor);
            }
        } else {
            compressed_send(0, OpType::ALLREDUCE, seq, 0, tensor);

            at::Tensor result = torch::empty_like(tensor);
            compressed_recv(0, OpType::ALLREDUCE, seq, 1, result);
            tensor.copy_(result);
            metal_sync_queue_only();
        }
    }
}


void ProcessGroupMCCL::allreduce_tree_small(at::Tensor& tensor, uint32_t seq,
                                            c10d::ReduceOp::RedOpType op) {
    // Recursive doubling with non-power-of-2 folding.
    //   p = largest power of two <= ws; extra = ws - p.
    //   Round F (fold-in):  ranks >= p send their vector to (rank - p),
    //                       which reduces it in.
    //   Rounds 1..log2(p):  pairwise full-vector exchange + reduce among
    //                       ranks < p (partner = rank XOR 2^k).
    //   Round U (unfold):   ranks < extra send the final vector to rank + p.
    // All p participants compute identical parenthesizations modulo fp
    // commutativity (a+b == b+a bitwise), so results are bit-identical on
    // every rank.
    const int rank = getRank();
    const int ws = getSize();
    const size_t nbytes = tensor_nbytes(tensor);
    const int64_t count = tensor.numel();
    const auto dtype = tensor.scalar_type();

    int p = 1;
    while (p * 2 <= ws) p *= 2;
    const int extra = ws - p;

    MPSBufferView view = extract_mps_buffer(tensor);
    MCCL_CHECK(view.cpu_accessible && view.cpu_ptr,
               "allreduce_tree_small requires CPU-visible storage");
    uint8_t* mine = static_cast<uint8_t*>(view.cpu_ptr);

    PooledBuffer recv_buf(staging_memory_pool(), nbytes);

    auto reduce_in = [&](const void* src) {
        if (is_integral_reduce_dtype(dtype)) {
            cpu_reduce_op_integral(mine, src, count, dtype, op);
        } else if (dtype == at::kFloat) {
            cpu_reduce_op(reinterpret_cast<float*>(mine),
                          static_cast<const float*>(src), count, op);
        } else if (dtype == at::kHalf) {
            cpu_reduce_op_half(reinterpret_cast<c10::Half*>(mine),
                               static_cast<const c10::Half*>(src), count, op);
        } else {
            cpu_reduce_op_bf16(reinterpret_cast<c10::BFloat16*>(mine),
                               static_cast<const c10::BFloat16*>(src), count, op);
        }
    };
    auto scale_avg = [&] {
        if (op != c10d::ReduceOp::AVG) return;
        const float s = 1.0f / static_cast<float>(ws);
        if (is_integral_reduce_dtype(dtype)) {
            cpu_scale_inplace_integral(mine, count, dtype, s);
        } else if (dtype == at::kFloat) {
            cpu_scale_inplace(reinterpret_cast<float*>(mine), count, s);
        } else if (dtype == at::kHalf) {
            cpu_scale_inplace_half(reinterpret_cast<c10::Half*>(mine), count, s);
        } else {
            cpu_scale_inplace_bf16(reinterpret_cast<c10::BFloat16*>(mine), count, s);
        }
    };

    // tid layout: 0 = fold-in, 1..R = doubling rounds, R+1 = unfold.
    uint32_t round_tid = 0;
    int rounds = 0;
    for (int m = 1; m < p; m <<= 1) rounds++;
    const uint32_t final_tid = static_cast<uint32_t>(rounds) + 1;

    const auto tree_t0 = std::chrono::steady_clock::now();
    double red_ms_accum = 0;

    if (rank >= p) {
        // Fold into the partner, then wait for the finished result.
        MCCL_CHECK(transport_->send_chunks(rank - p, OpType::ALLREDUCE, seq,
                                           round_tid, mine, nbytes),
                   "tree allreduce fold-in send failed");
        metrics_->record_transport_bytes(nbytes, true);
        MCCL_CHECK(transport_->recv_chunks(rank - p, OpType::ALLREDUCE, seq,
                                           final_tid, mine, nbytes),
                   "tree allreduce unfold recv failed");
        metrics_->record_transport_bytes(nbytes, false);
    } else {
        if (rank < extra) {
            watchdog_->touch(seq);
            MCCL_CHECK(transport_->recv_chunks(rank + p, OpType::ALLREDUCE, seq,
                                               round_tid, recv_buf.data(), nbytes),
                       "tree allreduce fold-in recv failed");
            metrics_->record_transport_bytes(nbytes, false);
            const auto r0 = std::chrono::steady_clock::now();
            reduce_in(recv_buf.data());
            red_ms_accum += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - r0).count();
        }

        for (int mask = 1; mask < p; mask <<= 1) {
            const int partner = rank ^ mask;
            round_tid++;
            watchdog_->touch(seq);
            // Full-duplex exchange: post the receive, send, then wait.
            RecvTicket t = transport_->post_recv(
                partner, OpType::ALLREDUCE, seq, round_tid,
                recv_buf.data(), nbytes);
            MCCL_CHECK(transport_->send_chunks(partner, OpType::ALLREDUCE, seq,
                                               round_tid, mine, nbytes),
                       "tree allreduce round send failed");
            MCCL_CHECK(transport_->wait_recv(t),
                       "tree allreduce round recv failed");
            metrics_->record_transport_bytes(nbytes, true);
            metrics_->record_transport_bytes(nbytes, false);
            const auto r0 = std::chrono::steady_clock::now();
            reduce_in(recv_buf.data());
            red_ms_accum += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - r0).count();
        }

        scale_avg();

        if (rank < extra) {
            MCCL_CHECK(transport_->send_chunks(rank + p, OpType::ALLREDUCE, seq,
                                               final_tid, mine, nbytes),
                       "tree allreduce unfold send failed");
            metrics_->record_transport_bytes(nbytes, true);
        }
    }

    if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
    mps_stream_sync_after_cpu_mps_buffer_write();

    const double net_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tree_t0).count();
    metrics_->record_phase(seq, 0, net_ms, red_ms_accum);
}


void ProcessGroupMCCL::broadcast_ring_pipelined(at::Tensor& tensor,
                                                uint32_t seq, int root) {
    // Ring broadcast rooted at `root`: rank order root -> root+1 -> ... ->
    // root+ws-1 (mod ws).  The payload is split into slices; each rank
    // forwards slice s while slice s+1 is arriving (depth-ahead posted
    // receives), so per-link traffic is S bytes and the wire stays busy.
    const int rank = getRank();
    const int ws = getSize();
    const size_t nbytes = tensor_nbytes(tensor);
    const int relid = (rank - root + ws) % ws;
    const int next = (rank + 1) % ws;
    const int prev = (rank - 1 + ws) % ws;
    const bool is_root = (relid == 0);
    const bool is_tail = (relid == ws - 1);

    // Never recv/send into tensor storage over TCP (MPS unified cpu_ptr or CPU
    // data_ptr).  Same rule as broadcast_tree_small: one pooled wire buffer per
    // rank, blit in at root and blit out on non-root after the ring completes.
    auto wire_buf = std::make_unique<PooledBuffer>(staging_memory_pool(), nbytes);
    uint8_t* base = static_cast<uint8_t*>(wire_buf->data());
    if (is_root) {
        blit_tensor_to_buffer(tensor, base);
    }

    // Slice so the pipeline has parallelism without flooding tiny messages.
    size_t slice = std::min(transport_->config().chunk_bytes,
                            std::max<size_t>(256 * 1024, nbytes / 8));
    const size_t nslices = (nbytes + slice - 1) / slice;

    auto slice_len = [&](size_t s) {
        return std::min(slice, nbytes - s * slice);
    };

    // Credit flow control: every rank that SENDS (root + forwarders) may run
    // at most cwin slices ahead of what `next` has consumed; every rank that
    // RECEIVES credits `prev` per consumed slice.  Without this, the root
    // streams the entire payload into a slow successor's park buffer.
    const bool credits_on = slice >= credit_min_chunk_bytes() &&
                            credit_min_chunk_bytes() > 0 &&
                            nslices > static_cast<size_t>(credit_window());
    const int cwin = credit_window();
    const bool sender = !is_tail;          // root and middle ranks send
    std::vector<uint8_t> credit_bytes((credits_on && sender) ? nslices : 0);
    std::vector<RecvTicket> credit_tickets((credits_on && sender) ? nslices : 0);
    std::vector<RecvTicket> tickets(is_root ? 0 : nslices);

    const auto bc_t0 = std::chrono::steady_clock::now();

    std::exception_ptr err;
    try {
        if (credits_on && sender) {
            for (size_t s = 0; s + cwin < nslices; ++s) {
                credit_tickets[s] = transport_->post_recv(
                    next, OpType::BROADCAST, seq,
                    kCreditTidFlag | static_cast<uint32_t>(s),
                    &credit_bytes[s], 1);
            }
        }

        auto send_slice = [&](size_t s) {
            if (credits_on && s >= static_cast<size_t>(cwin)) {
                MCCL_CHECK(transport_->wait_recv(credit_tickets[s - cwin]),
                           "broadcast ring credit wait failed at slice " +
                           std::to_string(s));
            }
            MCCL_CHECK(transport_->send_chunks(
                           next, OpType::BROADCAST, seq,
                           static_cast<uint32_t>(s), base + s * slice,
                           slice_len(s)),
                       "broadcast ring send failed at slice " + std::to_string(s));
            metrics_->record_transport_bytes(slice_len(s), true);
        };
        auto send_credit = [&](size_t s) {
            if (!credits_on || s + static_cast<size_t>(cwin) >= nslices) return;
            uint8_t one = 1;
            MCCL_CHECK(transport_->send_chunks(
                           prev, OpType::BROADCAST, seq,
                           kCreditTidFlag | static_cast<uint32_t>(s), &one, 1),
                       "broadcast ring credit send failed at slice " +
                       std::to_string(s));
        };

        if (is_root) {
            for (size_t s = 0; s < nslices; ++s) {
                watchdog_->touch(seq);
                send_slice(s);
            }
        } else {
            const int depth = ring_pipeline_depth();
            size_t next_post = 0;
            for (; next_post < nslices && next_post < static_cast<size_t>(depth);
                 ++next_post) {
                tickets[next_post] = transport_->post_recv(
                    prev, OpType::BROADCAST, seq,
                    static_cast<uint32_t>(next_post),
                    base + next_post * slice, slice_len(next_post));
            }
            for (size_t s = 0; s < nslices; ++s) {
                watchdog_->touch(seq);
                MCCL_CHECK(transport_->wait_recv(tickets[s]),
                           "broadcast ring recv failed at slice " + std::to_string(s));
                metrics_->record_transport_bytes(slice_len(s), false);
                if (!is_tail) {
                    send_slice(s);
                }
                send_credit(s);
                if (next_post < nslices) {
                    tickets[next_post] = transport_->post_recv(
                        prev, OpType::BROADCAST, seq,
                        static_cast<uint32_t>(next_post),
                        base + next_post * slice, slice_len(next_post));
                    ++next_post;
                }
            }
        }
        if (!is_root) {
            unstage_from_recv(tensor, base, nbytes);
        }
    } catch (...) {
        err = std::current_exception();
    }

    if (err) {
        // Same lifetime rule as run_ring_pipeline: posted-but-unwaited sinks
        // reference `staged`/credit storage in this frame — cancel and drain
        // before unwinding.
        transport_->cancel_recvs(seq, "broadcast ring aborted");
        for (auto& t : tickets) {
            if (!t) continue;
            try { transport_->wait_recv(t); } catch (...) {}
        }
        for (auto& t : credit_tickets) {
            if (!t) continue;
            try { transport_->wait_recv(t); } catch (...) {}
        }
        std::rethrow_exception(err);
    }

    metrics_->record_phase(seq, 0,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bc_t0).count(), 0);
}


void ProcessGroupMCCL::broadcast_tree_small(at::Tensor& tensor,
                                            uint32_t seq, int root) {
    // Binomial tree on relative ids (relid = (rank - root) mod ws):
    // round k: ranks with relid < 2^k forward to relid + 2^k (if < ws);
    // each rank receives exactly once, in round floor(log2(relid)).
    const int rank = getRank();
    const int ws = getSize();
    const size_t nbytes = tensor_nbytes(tensor);
    const int relid = (rank - root + ws) % ws;

    // Never recv/send from tensor unified-memory cpu_ptr over TCP — use one
    // pooled wire buffer per rank, then copy into the output tensor on non-root.
    auto tree_buf = std::make_unique<PooledBuffer>(staging_memory_pool(), nbytes);
    uint8_t* base = static_cast<uint8_t*>(tree_buf->data());

    if (relid == 0) {
        StagingBuffer send_staged = stage_for_send_collective(tensor);
        memcpy(base, send_staged.data, send_staged.nbytes);
    }

    const auto bt_t0 = std::chrono::steady_clock::now();

    bool have_data = (relid == 0);
    for (int k = 0; (1 << k) < ws; ++k) {
        const int bit = 1 << k;
        watchdog_->touch(seq);
        if (!have_data && relid >= bit && relid < 2 * bit) {
            const int src = ((relid - bit) + root) % ws;
            MCCL_CHECK(transport_->recv_chunks(src, OpType::BROADCAST, seq,
                                               static_cast<uint32_t>(k),
                                               base, nbytes),
                       "broadcast tree recv failed at round " + std::to_string(k));
            metrics_->record_transport_bytes(nbytes, false);
            have_data = true;
        } else if (have_data && relid < bit && relid + bit < ws) {
            const int dst = ((relid + bit) + root) % ws;
            MCCL_CHECK(transport_->send_chunks(dst, OpType::BROADCAST, seq,
                                               static_cast<uint32_t>(k),
                                               base, nbytes),
                       "broadcast tree send failed at round " + std::to_string(k));
            metrics_->record_transport_bytes(nbytes, true);
        }
    }

    if (relid != 0) {
        unstage_from_recv(tensor, base, nbytes);
    }

    metrics_->record_phase(seq, 0,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bt_t0).count(), 0);
}


void ProcessGroupMCCL::allgather_star_small(std::vector<at::Tensor>& outputs,
                                            const at::Tensor& input,
                                            uint32_t seq, size_t nbytes) {
    // Deadlock-free star: for each src in [0, ws), rank src sends its input to
    // every dst > src; ranks dst > src recv into outputs[src].  Direct mesh
    // hops (no ring forwarding) — reliable for DDP init_sync int64 metadata on
    // multi-node TCP where lock-step ring could leave distant slots zeroed.
    const int rank = getRank();
    const int ws = getSize();

    outputs[rank].copy_(input);

    PooledBuffer wire(staging_memory_pool(), nbytes);
    const auto ag_t0 = std::chrono::steady_clock::now();

    for (int src = 0; src < ws; src++) {
        watchdog_->touch(seq);
        if (rank == src) {
            StagingBuffer staged = stage_for_send_collective(input);
            for (int dst = src + 1; dst < ws; dst++) {
                MCCL_CHECK(transport_->send_chunks(
                               dst, OpType::ALLGATHER, seq,
                               static_cast<uint32_t>(rank),
                               staged.data, nbytes),
                           "allgather star send to rank " + std::to_string(dst));
                metrics_->record_transport_bytes(nbytes, true);
            }
        } else if (rank > src) {
            MCCL_CHECK(transport_->recv_chunks(
                           src, OpType::ALLGATHER, seq,
                           static_cast<uint32_t>(src),
                           wire.data(), nbytes),
                       "allgather star recv from rank " + std::to_string(src));
            metrics_->record_transport_bytes(nbytes, false);
            unstage_from_recv(outputs[src], wire.data(), nbytes);
        }
    }

    metrics_->record_phase(seq, 0,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ag_t0).count(), 0);
}


// ── broadcast ───────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {

    MCCL_CHECK_TENSOR(tensors.size() == 1, "MCCL broadcast expects one tensor");
    at::Tensor& tensor = tensors[0];
    // Non-root ranks receive in place; require contiguity on all ranks for a
    // consistent contract (silently cloning would drop received data).
    require_contiguous_output(tensor, "broadcast");
    check_single_tensor(tensor);

    int root = static_cast<int>(opts.rootRank);
    MCCL_CHECK(root >= 0 && root < getSize(),
               "broadcast rootRank=" + std::to_string(root) +
               " out of range [0, " + std::to_string(getSize()) + ")");

    // world_size == 1 (or empty tensor): nothing to send.  The fan-out path
    // below would never complete at ws == 1 because the atomic send counter
    // starts at zero and markComplete only fired from send callbacks.
    if (getSize() == 1 || tensor.numel() == 0) {
        return make_completed_work(c10d::OpType::BROADCAST, tensors);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(tensor);

    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::BROADCAST, seq, std::vector<at::Tensor>{tensor});

    auto tensor_copy = tensor;
    auto work_ptr = work;
    int rank = getRank();
    int ws = getSize();

    register_work(seq, work);
    watchdog_->watch(seq, "broadcast");
    metrics_->op_start(seq, "broadcast", nbytes);

    uint64_t sync_val_bc = sync_mps_nonblocking(overlap_comm_);

    // ws >= 4: scale-aware algorithms on the collective pool.
    //   large: pipelined ring (root egress S bytes instead of (ws-1)*S);
    //   small: binomial tree (ceil(log2 ws) rounds instead of ws-1 sends).
    // ws <= 3 keeps the per-peer fan-out below (parallel across 1-2 links).
    if (ws >= 4 && collective_pool_) {
        const bool use_ring = nbytes > transport_->config().small_msg_threshold;
        collective_pool_->submit(
            [this, tensor_copy, seq, root, nbytes, use_ring, sync_val_bc]() mutable {
                begin_execute(seq);
                std::lock_guard<std::mutex> transport_guard(transport_collective_mu_);
                const auto exec_t0 = std::chrono::steady_clock::now();
                if (use_ring) {
                    broadcast_ring_pipelined(tensor_copy, seq, root);
                } else {
                    broadcast_tree_small(tensor_copy, seq, root);
                }
                // Bytes are recorded per slice/round inside the algorithms.
                if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
                mps_stream_sync_after_cpu_mps_buffer_write();
                const double exec_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - exec_t0).count();
                MCCL_INFO("broadcast seq=%u: algo=%s nbytes=%zu exec=%.2fms",
                          seq, use_ring ? "ring_pipelined" : "tree", nbytes, exec_ms);
            },
            [this, work_ptr, seq]() {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                work_ptr->markComplete();
            },
            [this, work_ptr, seq](std::exception_ptr e) {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(e);
            }
        );
        return work;
    }

    if (rank == root) {
        // Root: stage data then fan out sends to per-peer NetEngines.
        // Shared state tracks completion AND the first error: any failed
        // peer send must fail the whole broadcast.  (Previously a failed
        // send was swallowed whenever a later send succeeded last.)
        struct BcastFanoutState {
            explicit BcastFanoutState(int n) : remaining(n) {}
            std::atomic<int> remaining;
            std::mutex mu;
            std::exception_ptr first_error;
        };
        auto staged_buf = std::make_shared<PooledBuffer>(staging_memory_pool(), nbytes);
        int num_peers = ws - 1;
        auto state = std::make_shared<BcastFanoutState>(num_peers);

        auto finish_fanout = [this, work_ptr, seq, state]() {
            // Called by whichever send callback decrements remaining to zero.
            // Must not throw: a throw from a net engine's on_complete would be
            // rerouted to that op's on_error, double-decrementing the counter.
            std::exception_ptr err;
            {
                std::lock_guard<std::mutex> lock(state->mu);
                err = state->first_error;
            }
            if (err) {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(err);
                return;
            }
            try {
                reduce_engine_->submit(
                    [this]() {
                        if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
                    },
                    [this, work_ptr, seq]() {
                        unregister_work(seq);
                        watchdog_->complete(seq);
                        metrics_->op_end(seq);
                        work_ptr->markComplete();
                    },
                    [this, work_ptr, seq](std::exception_ptr e) {
                        unregister_work(seq);
                        watchdog_->complete(seq);
                        metrics_->op_end(seq);
                        metrics_->record_error();
                        work_ptr->markError(e);
                    }
                );
            } catch (...) {
                // reduce_engine_ shutting down: complete the work with the
                // error here so it cannot hang.
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(std::current_exception());
            }
        };

        reduce_engine_->submit(
            [this, tensor_copy, seq, sync_val_bc, staged_buf]() mutable {
                begin_execute(seq);
                StagingBuffer staged = stage_for_send_collective(tensor_copy);
                memcpy(staged_buf->data(), staged.data, staged.nbytes);
            },
            [this, staged_buf, seq, root, ws, nbytes, state, finish_fanout]() mutable {
                for (int peer = 0; peer < ws; peer++) {
                    if (peer == root) continue;
                    net_engine_for(peer).submit(
                        [this, peer, seq, staged_buf, nbytes]() mutable {
                            watchdog_->touch(seq);  // re-arm per peer-send phase
                            MCCL_CHECK(transport_->send_chunks(peer, OpType::BROADCAST, seq, 0,
                                                               staged_buf->data(), nbytes),
                                       "broadcast send to rank " + std::to_string(peer) + " failed");
                            metrics_->record_transport_bytes(nbytes, true);
                        },
                        [state, finish_fanout]() {
                            if (state->remaining.fetch_sub(1) == 1) {
                                finish_fanout();
                            }
                        },
                        [state, finish_fanout](std::exception_ptr e) {
                            MCCL_ERROR("broadcast send to peer failed");
                            {
                                std::lock_guard<std::mutex> lock(state->mu);
                                if (!state->first_error) state->first_error = e;
                            }
                            if (state->remaining.fetch_sub(1) == 1) {
                                finish_fanout();
                            }
                        }
                    );
                }
            },
            [this, work_ptr, seq](std::exception_ptr e) {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(e);
            }
        );
    } else {
        // Non-root receives from root using root's NetEngine
        net_engine_for(root).submit(
            [this, tensor_copy, root, seq, nbytes, sync_val_bc]() mutable {
                begin_execute(seq);
                PooledBuffer recv_buf(staging_memory_pool(), nbytes);
                MCCL_CHECK(transport_->recv_chunks(root, OpType::BROADCAST, seq, 0,
                                                   recv_buf.data(), nbytes),
                           "broadcast recv from root failed");
                unstage_from_recv(tensor_copy, recv_buf.data(), nbytes);
                metrics_->record_transport_bytes(nbytes, false);
                if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
                mps_stream_sync_after_cpu_mps_buffer_write();
            },
            [this, work_ptr, seq]() {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                work_ptr->markComplete();
            },
            [this, work_ptr, seq](std::exception_ptr e) {
                unregister_work(seq);
                watchdog_->complete(seq);
                metrics_->op_end(seq);
                metrics_->record_error();
                work_ptr->markError(e);
            }
        );
    }

    return work;
}


// ── barrier ─────────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::barrier(
    const c10d::BarrierOptions& opts) {

    if (getSize() == 1) {
        return make_completed_work(c10d::OpType::BARRIER);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    auto work = c10::make_intrusive<WorkMCCL>(c10d::OpType::BARRIER, seq);
    auto work_ptr = work;

    register_work(seq, work);
    watchdog_->watch(seq, "barrier");
    metrics_->op_start(seq, "barrier", 0);

    reduce_engine_->submit(
        [this, seq]() {
            begin_execute(seq);
            rendezvous_->barrier("collective_" + std::to_string(seq));
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}


// ── allgather ───────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {

    MCCL_CHECK_TENSOR(inputTensors.size() == 1, "MCCL allgather expects one input tensor");
    MCCL_CHECK_TENSOR(outputTensors.size() == 1, "MCCL allgather expects one output tensor list");
    MCCL_CHECK_TENSOR(
        static_cast<int>(outputTensors[0].size()) == getSize(),
        "Output tensor list size must equal world_size");

    at::Tensor input = ensure_contiguous(inputTensors[0]);
    check_single_tensor(input);

    auto& outputs = outputTensors[0];
    for (auto& t : outputs) {
        // Outputs are written in place; a silent clone would drop results.
        require_contiguous_output(t, "allgather");
        check_same_shape_dtype(input, t);
    }

    if (input.numel() == 0) {
        return make_completed_work(c10d::OpType::ALLGATHER, outputs);
    }
    if (getSize() == 1) {
        outputs[0].copy_(input);
        return make_completed_work(c10d::OpType::ALLGATHER, outputs);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(input);

    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::ALLGATHER, seq, outputs);

    auto input_copy = input;
    auto outputs_copy = outputs;
    auto work_ptr = work;
    int rank = getRank();
    int ws = getSize();

    register_work(seq, work);
    watchdog_->watch(seq, "allgather");
    metrics_->op_start(seq, "allgather", nbytes * ws);

    uint64_t sync_val_ag = sync_mps_nonblocking(overlap_comm_);

    ProgressEngine& ag_engine =
        (ws >= 3 && collective_pool_) ? *collective_pool_ : *reduce_engine_;
    ag_engine.submit(
        [this, input_copy, outputs_copy, seq, rank, ws, nbytes, sync_val_ag]() mutable {
            begin_execute(seq);
            std::lock_guard<std::mutex> transport_guard(transport_collective_mu_);
            bool use_cpu = (input_copy.scalar_type() == at::kFloat) &&
                           prefer_cpu_unified_buffer_path(input_copy);

            if (use_cpu) {
                MPSBufferView in_view = extract_mps_buffer(input_copy);
                MPSBufferView out_view = extract_mps_buffer(outputs_copy[rank]);
                memcpy(out_view.cpu_ptr, in_view.cpu_ptr, nbytes);
            } else {
                outputs_copy[rank].copy_(input_copy);
            }

            int left = (rank - 1 + ws) % ws;
            int right = (rank + 1) % ws;

            const size_t small_thresh = transport_->config().small_msg_threshold;
            if (nbytes <= small_thresh) {
                allgather_star_small(outputs_copy, input_copy, seq, nbytes);
            } else if (ring_pipeline_for_message(nbytes, small_thresh)) {
                // Streaming pipeline: forward chunk s while chunk s+1 arrives.
                // send(s) = (rank-s), recv(s) = (rank-s-1) = send(s+1):
                // lookahead 1; receives land zero-copy in the output tensors.
                std::vector<RingStep> plan;
                plan.reserve(ws - 1);
                for (int step = 0; step < ws - 1; step++) {
                    RingStep st;
                    st.send_idx = (rank - step + ws) % ws;
                    st.recv_idx = (rank - step - 1 + ws) % ws;
                    st.send_tid = (static_cast<uint32_t>(step) << 16) | st.send_idx;
                    st.recv_tid = (static_cast<uint32_t>(step) << 16) | st.recv_idx;
                    st.kind = RingRecvKind::COPY;
                    plan.push_back(st);
                }
                RingPipelineCtx ctx{transport_.get(), watchdog_.get(),
                                    metrics_.get(), seq, left, right,
                                    OpType::ALLGATHER,
                                    c10d::ReduceOp::SUM, use_cpu};
                run_ring_pipeline(ctx, outputs_copy, plan, /*lookahead=*/1);
            } else {
            PooledBuffer recv_buf_fallback(staging_memory_pool(), nbytes);

            for (int step = 0; step < ws - 1; step++) {
                watchdog_->touch(seq);  // per-step progress re-arms the deadline
                int send_idx = (rank - step + ws) % ws;
                int recv_idx = (rank - step - 1 + ws) % ws;
                uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
                uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;

                // nosync staging is safe here: allgather issues no async MCCL
                // kernels, the pre-collective MPS sync already ordered
                // PyTorch's writes, and unstage blits complete synchronously.
                StagingBuffer staged = stage_for_send_collective(outputs_copy[send_idx]);

                void* recv_dst = recv_buf_fallback.data();

                MCCL_CHECK(transport_->send_recv_overlap(
                    right, OpType::ALLGATHER, seq, step_tid,
                    staged.data, nbytes,
                    left, OpType::ALLGATHER, seq, recv_tid,
                    recv_dst, nbytes),
                    "allgather step " + std::to_string(step) + " failed");

                unstage_from_recv(outputs_copy[recv_idx], recv_dst, nbytes);
                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }
            }  // end lock-step fallback
            mps_stream_sync_after_cpu_mps_buffer_write();
            if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}


// ── reduce_scatter ──────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {

    MCCL_CHECK_TENSOR(outputTensors.size() == 1, "MCCL reduce_scatter expects one output");
    MCCL_CHECK_TENSOR(inputTensors.size() == 1, "MCCL reduce_scatter expects one input list");
    MCCL_CHECK_TENSOR(
        static_cast<int>(inputTensors[0].size()) == getSize(),
        "Input tensor list size must equal world_size");

    // Output is written in place; a silent clone would drop results.
    at::Tensor output = outputTensors[0];
    require_contiguous_output(output, "reduce_scatter");
    require_reduce_dtype(output, "reduce_scatter");
    auto& inputs = inputTensors[0];
    for (auto& t : inputs) {
        t = ensure_contiguous(t);
        check_single_tensor(t);
        check_same_shape_dtype(output, t);
    }

    if (output.numel() == 0) {
        return make_completed_work(c10d::OpType::REDUCE_SCATTER,
                                   std::vector<at::Tensor>{output});
    }
    if (getSize() == 1) {
        output.copy_(inputs[0]);
        return make_completed_work(c10d::OpType::REDUCE_SCATTER,
                                   std::vector<at::Tensor>{output});
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(output);

    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::REDUCE_SCATTER, seq, std::vector<at::Tensor>{output});

    auto output_copy = output;
    auto inputs_copy = inputs;
    auto work_ptr = work;
    int rank = getRank();
    int ws = getSize();

    c10d::ReduceOp::RedOpType rs_op = opts.reduceOp;

    register_work(seq, work);
    watchdog_->watch(seq, "reduce_scatter");
    metrics_->op_start(seq, "reduce_scatter", nbytes * ws);

    uint64_t sync_val_rs = sync_mps_nonblocking(overlap_comm_);

    ProgressEngine& rs_engine =
        (ws >= 3 && collective_pool_) ? *collective_pool_ : *reduce_engine_;
    rs_engine.submit(
        [this, output_copy, inputs_copy, seq, rank, ws, nbytes, rs_op, sync_val_rs]() mutable {
            begin_execute(seq);
            std::lock_guard<std::mutex> transport_guard(transport_collective_mu_);
            int left = (rank - 1 + ws) % ws;
            int right = (rank + 1) % ws;
            bool use_cpu = (inputs_copy[0].scalar_type() == at::kFloat) &&
                           prefer_cpu_unified_buffer_path(inputs_copy[0]);

            std::vector<at::Tensor> chunks = inputs_copy;

            // Keep-alive for async-kernel staging tensors; drained by the
            // metal_sync_queue_only in the tail below before destruction.
            IncomingKeepAlive incoming_keep;

            const size_t small_thresh_rs = transport_->config().small_msg_threshold;
            if (ring_pipeline_for_message(nbytes, small_thresh_rs)) {
                // Streaming pipeline: send(s) = (rank+1-s) = recv(s-1), so
                // lookahead 1; every received chunk is reduced into place
                // while the next is already on the wire.
                std::vector<RingStep> plan;
                plan.reserve(ws - 1);
                for (int step = 0; step < ws - 1; step++) {
                    RingStep st;
                    st.send_idx = (rank + 1 - step + ws) % ws;
                    st.recv_idx = (rank - step + ws) % ws;
                    st.send_tid = (static_cast<uint32_t>(step) << 16) | st.send_idx;
                    st.recv_tid = (static_cast<uint32_t>(step) << 16) | st.recv_idx;
                    st.kind = RingRecvKind::REDUCE;
                    plan.push_back(st);
                }
                RingPipelineCtx ctx{transport_.get(), watchdog_.get(),
                                    metrics_.get(), seq, left, right,
                                    OpType::REDUCE_SCATTER, rs_op, use_cpu,
                                    &incoming_keep.tensors};
                run_ring_pipeline(ctx, chunks, plan, /*lookahead=*/1);
            } else {
            PooledBuffer recv_buf(staging_memory_pool(), nbytes);

            // Kernel fence for the Metal reduce path: the chunk sent at step
            // s+1 is the chunk reduced at step s (send_idx_{s+1} == recv_idx_s),
            // so the reduce kernel must complete before that chunk is staged.
            // Event-based fencing replaces the previous per-step full queue
            // drain inside stage_for_send().
            const bool use_event_fence = !use_cpu && event_sync_available();
            std::vector<uint64_t> chunk_pending(ws, 0);
            auto fence_chunk = [&](int idx) {
                if (use_event_fence && chunk_pending[idx]) {
                    wait_for_mccl_fence(chunk_pending[idx]);
                    chunk_pending[idx] = 0;
                }
            };
            auto arm_chunk = [&](int idx) {
                if (use_event_fence) {
                    uint64_t v = next_fence_event_value();
                    signal_mccl_fence_gpu(v);
                    chunk_pending[idx] = v;
                } else if (!use_cpu) {
                    metal_sync_queue_only();
                }
            };

            for (int step = 0; step < ws - 1; step++) {
                watchdog_->touch(seq);  // per-step progress re-arms the deadline
                int send_idx = (rank + 1 - step + ws) % ws;
                int recv_idx = (rank - step + ws) % ws;
                uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
                uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;

                if (!use_cpu) fence_chunk(send_idx);
                StagingBuffer staged = stage_for_send_collective(chunks[send_idx]);

                MCCL_CHECK(transport_->send_recv_overlap(
                    right, OpType::REDUCE_SCATTER, seq, step_tid,
                    staged.data, nbytes,
                    left, OpType::REDUCE_SCATTER, seq, recv_tid,
                    recv_buf.data(), nbytes),
                    "reduce_scatter step " + std::to_string(step) + " failed");

                if (use_cpu) {
                    MPSBufferView chunk_view = extract_mps_buffer(chunks[recv_idx]);
                    cpu_reduce_op(
                        static_cast<float*>(chunk_view.cpu_ptr),
                        static_cast<const float*>(recv_buf.data()),
                        chunks[recv_idx].numel(), rs_op);
                } else {
                    at::Tensor incoming = torch::empty_like(chunks[recv_idx]);
                    unstage_from_recv(incoming, recv_buf.data(), nbytes);
                    reduce_chunk_metal_fenced(
                        chunks[recv_idx], std::move(incoming), rs_op,
                        &incoming_keep.tensors);
                }

                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }
            }  // end lock-step fallback

            int my_chunk = rank;
            if (use_cpu) {
                MPSBufferView src_view = extract_mps_buffer(chunks[my_chunk]);
                MPSBufferView dst_view = extract_mps_buffer(output_copy);
                memcpy(dst_view.cpu_ptr, src_view.cpu_ptr, nbytes);
                if (overlap_comm_) signal_mccl_done(next_mccl_event_value());
                mps_stream_sync_after_cpu_mps_buffer_write();
            } else {
                metal_sync_queue_only();
                output_copy.copy_(chunks[my_chunk]);
            }
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}


// ── Point-to-point send/recv ────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {

    MCCL_CHECK_TENSOR(tensors.size() == 1, "MCCL send expects one tensor");
    MCCL_CHECK(dstRank >= 0 && dstRank < getSize() && dstRank != getRank(),
               "send dstRank=" + std::to_string(dstRank) + " invalid (rank=" +
               std::to_string(getRank()) + " world=" + std::to_string(getSize()) + ")");

    at::Tensor tensor = ensure_contiguous(tensors[0]);
    check_single_tensor(tensor);
    if (tensor.numel() == 0) {
        // recv() no-ops on zero-size tensors; never put a 0-byte message on
        // the wire or the two sides would desynchronize.
        return make_completed_work(c10d::OpType::SEND);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(tensor);

    auto work = c10::make_intrusive<WorkMCCL>(c10d::OpType::SEND, seq);
    auto work_ptr = work;

    register_work(seq, work);
    watchdog_->watch(seq, "send");
    metrics_->op_start(seq, "send", nbytes);

    uint64_t sync_val_s = sync_mps_nonblocking(overlap_comm_);

    net_engine_for(dstRank).submit(
        [this, tensor, dstRank, seq, tag, nbytes, sync_val_s]() mutable {
            begin_execute(seq);
            StagingBuffer staged = stage_for_send_collective(tensor);
            MCCL_CHECK(transport_->send_chunks(dstRank, OpType::SEND, seq,
                                               static_cast<uint32_t>(tag),
                                               staged.data, staged.nbytes),
                       "send to rank " + std::to_string(dstRank) + " failed");
            metrics_->record_transport_bytes(nbytes, true);
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {

    MCCL_CHECK_TENSOR(tensors.size() == 1, "MCCL recv expects one tensor");
    MCCL_CHECK(srcRank >= 0 && srcRank < getSize() && srcRank != getRank(),
               "recv srcRank=" + std::to_string(srcRank) + " invalid (rank=" +
               std::to_string(getRank()) + " world=" + std::to_string(getSize()) + ")");

    at::Tensor tensor = tensors[0];
    // Received data is written in place; a silent clone would drop it.
    require_contiguous_output(tensor, "recv");
    check_single_tensor(tensor);
    if (tensor.numel() == 0) {
        return make_completed_work(c10d::OpType::RECV, tensors);
    }

    uint32_t seq = collective_seq_.fetch_add(1);
    size_t nbytes = tensor_nbytes(tensor);

    auto work = c10::make_intrusive<WorkMCCL>(
        c10d::OpType::RECV, seq, std::vector<at::Tensor>{tensor});
    auto work_ptr = work;

    register_work(seq, work);
    watchdog_->watch(seq, "recv");
    metrics_->op_start(seq, "recv", nbytes);

    net_engine_for(srcRank).submit(
        [this, tensor, srcRank, seq, tag, nbytes]() mutable {
            begin_execute(seq);
            bool use_cpu = prefer_cpu_unified_buffer_path(tensor);
            if (use_cpu) {
                MPSBufferView view = extract_mps_buffer(tensor);
                MCCL_CHECK(transport_->recv_chunks(srcRank, OpType::RECV, seq,
                                                   static_cast<uint32_t>(tag),
                                                   view.cpu_ptr, nbytes),
                           "recv from rank " + std::to_string(srcRank) + " failed");
            } else {
                PooledBuffer recv_buf(staging_memory_pool(), nbytes);
                MCCL_CHECK(transport_->recv_chunks(srcRank, OpType::RECV, seq,
                                                   static_cast<uint32_t>(tag),
                                                   recv_buf.data(), nbytes),
                           "recv from rank " + std::to_string(srcRank) + " failed");
                unstage_from_recv(tensor, recv_buf.data(), nbytes);
            }
            metrics_->record_transport_bytes(nbytes, false);
            if (use_cpu) {
                mps_stream_sync_after_cpu_mps_buffer_write();
            }
        },
        [this, work_ptr, seq]() {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            work_ptr->markComplete();
        },
        [this, work_ptr, seq](std::exception_ptr e) {
            unregister_work(seq);
            watchdog_->complete(seq);
            metrics_->op_end(seq);
            metrics_->record_error();
            work_ptr->markError(e);
        }
    );

    return work;
}


// ── Factory ─────────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Backend> createProcessGroupMCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size,
    const std::chrono::milliseconds& timeout) {
    return c10::make_intrusive<ProcessGroupMCCL>(store, rank, world_size, timeout);
}

} // namespace mccl
