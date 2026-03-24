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
#include <cstring>
#include <cstdlib>
#include <future>
#include <thread>

namespace mccl {

namespace {

enum class SyncMode {
    FULL,       // torch::mps::synchronize() every op (safest)
    COALESCED,  // sync once at start of a batch; skip for subsequent ops in same batch
};

SyncMode global_sync_mode() {
    static SyncMode mode = [] {
        bool requested_coalesced = false;
        auto* v = std::getenv("MCCL_SYNC_MODE");
        if (v) {
            std::string s(v);
            requested_coalesced = (s == "coalesced" || s == "fast");
        }
        if (requested_coalesced) {
            auto* unsafe = std::getenv("MCCL_ALLOW_UNSAFE_COALESCED_DDP");
            bool allow_unsafe = unsafe && (std::string(unsafe) == "1" ||
                                           std::string(unsafe) == "true" ||
                                           std::string(unsafe) == "yes");
            if (!allow_unsafe) {
                MCCL_WARN(
                    "MCCL_SYNC_MODE=coalesced requested but disabled for DDP safety. "
                    "Using FULL sync; set MCCL_ALLOW_UNSAFE_COALESCED_DDP=1 to override.");
                return SyncMode::FULL;
            }
            MCCL_WARN(
                "MCCL_SYNC_MODE=coalesced enabled with MCCL_ALLOW_UNSAFE_COALESCED_DDP=1. "
                "This is unsafe for bucketed DDP and may corrupt convergence.");
            return SyncMode::COALESCED;
        }
        // Default FULL: DDP gradient bucketing issues many allreduce calls per
        // backward; each must wait for that bucket's GPU work.  COALESCED skips
        // sync after the first op in a thread, which reads stale grads and
        // corrupts wire traffic (broken pipe on rank 2+ buckets).
        // Use MCCL_SYNC_MODE=coalesced only with a single batched collective
        // (e.g. one manual allreduce_coalesced), not with DDP hooks.
        return SyncMode::FULL;
    }();
    return mode;
}

thread_local bool tl_sync_done = false;

// Non-blocking: encode signal + commit, return event value (0 = already synced).
// The engine thread must call wait_for_mps(val) before reading tensor data.
inline uint64_t sync_mps_nonblocking(bool overlap) {
    if (global_sync_mode() == SyncMode::COALESCED && tl_sync_done) {
        return 0;
    }
    uint64_t val = 0;
    if (overlap) {
        val = mps_event_sync_nonblocking();
    } else {
        mps_stream_sync();
    }
    tl_sync_done = true;
    return val;
}

// Blocking: for callers that don't defer the wait to an engine thread.
// DEPRECATED: Use commit_mps_and_signal + deferred wait_for_mps in engine for better overlap.
inline void sync_mps_for_collective(bool overlap) {
    if (overlap && event_sync_available()) {
        uint64_t val = next_event_value();
        commit_mps_and_signal(val);
        wait_for_mps(val);
    } else {
        mps_stream_sync();
    }
}

inline void reset_sync_state() {
    tl_sync_done = false;
}

enum class RingPipelinePhase {
    REDUCE_SCATTER,
    ALLGATHER,
};

struct RingReduceResult {
    double reduce_ms = 0.0;
    double writeback_ms = 0.0;
};

struct RingPipelineSlot {
    RingPipelinePhase phase = RingPipelinePhase::REDUCE_SCATTER;
    int step = 0;
    int send_chunk_idx = 0;
    int recv_chunk_idx = 0;
    size_t recv_bytes = 0;
    at::Tensor recv_chunk;
    SharedPooledBuffer recv_buf;
    std::shared_ptr<std::future<void>> recv_future;
    std::future<RingReduceResult> reduce_future;
    bool recv_done = false;
    bool reduce_started = false;
};

inline std::shared_ptr<std::future<void>> make_ready_future() {
    auto promise = std::make_shared<std::promise<void>>();
    promise->set_value();
    return std::make_shared<std::future<void>>(promise->get_future());
}

template <typename T>
inline bool future_ready(std::future<T>& future) {
    if (!future.valid()) return false;
    return future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

inline bool future_ready(const std::shared_ptr<std::future<void>>& future) {
    if (!future) return true;
    return future->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
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
    if (auto* v = std::getenv("MCCL_RING_ASSERT_ORDER")) {
        std::string s(v);
        ring_assert_order_ = (s == "1" || s == "true" || s == "yes");
    }
    // Dedicated 3+ pipeline is the production default. Set the env to 0/no/false
    // only when rolling back to the legacy serial ring implementation.
    if (auto* v = std::getenv("MCCL_3PLUS_PIPELINE")) {
        std::string s(v);
        dedicated_3plus_pipeline_ = !(s == "0" || s == "false" || s == "no");
    }
    if (auto* v = std::getenv("MCCL_RING_PIPELINE_WINDOW")) {
        int parsed = std::atoi(v);
        ring_pipeline_window_ = std::max(1, parsed);
    } else {
        ring_pipeline_window_ = default_ring_pipeline_window(world_size);
    }

    size_t queue_depth = 1024;
    if (auto* v = std::getenv("MCCL_MAX_QUEUE_DEPTH"))
        queue_depth = static_cast<size_t>(std::atoll(v));
    
    // Create reduce engine
    reduce_engine_ = std::make_unique<ProgressEngine>(queue_depth);
    reduce_engine_->start();
    pipeline_reduce_engine_ = std::make_unique<ProgressEngine>(queue_depth);
    pipeline_reduce_engine_->start();
    
    // Create net engines (one per peer rank, excluding self)
    net_engines_.resize(world_size);
    pending_ring_sends_.resize(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            net_engines_[i] = std::make_unique<ProgressEngine>(queue_depth);
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

    auto hb_interval = transport_->config().heartbeat_interval;
    health_ = std::make_unique<HealthMonitor>(
        transport_.get(),
        hb_interval,
        [this](int peer) { on_peer_death(peer); });
    health_->start();

    // ── Config dump ───────────────────────────────────────────────
    MCCL_INFO("=== MCCL Config (rank %d) ===", rank);
    MCCL_INFO("  world_size          = %d", world_size);
    MCCL_INFO("  timeout_ms          = %lld", (long long)timeout.count());
    MCCL_INFO("  watchdog_timeout_ms = %lld", (long long)wd_timeout.count());
    MCCL_INFO("  transport           = %s", transport_->config().transport.c_str());
    MCCL_INFO("  listen_addr         = %s", transport_->config().listen_addr.c_str());
    MCCL_INFO("  port_base           = %u", (unsigned)transport_->config().port_base);
    MCCL_INFO("  ifname              = %s",
              transport_->config().ifname.empty() ? "(auto)" : transport_->config().ifname.c_str());
    MCCL_INFO("  chunk_bytes         = %zu", transport_->config().chunk_bytes);
    MCCL_INFO("  small_msg_threshold = %zu", transport_->config().small_msg_threshold);
    MCCL_INFO("  connect_timeout_ms  = %lld", (long long)transport_->config().connect_timeout.count());
    MCCL_INFO("  heartbeat_ms        = %lld", (long long)hb_interval.count());
    MCCL_INFO("  max_queue_depth     = %zu", queue_depth);
    MCCL_INFO("  pipeline_reduce_q   = %s", pipeline_reduce_engine_ ? "on" : "off");
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
    MCCL_INFO("  ring_assert_order   = %s", ring_assert_order_ ? "on" : "off");
    MCCL_INFO("  3plus_pipeline      = %s", dedicated_3plus_pipeline_ ? "on" : "off");
    MCCL_INFO("  ring_pipeline_win   = %d", ring_pipeline_window_);
    {
        const char* es = std::getenv("MCCL_EVENT_SYNC");
        bool es_off = es && (std::string(es) == "0" || std::string(es) == "false" ||
                             std::string(es) == "no");
        MCCL_INFO("  event_sync          = %s",
                   es_off ? "off (env)" :
                   event_sync_available() ? "on" : "off (unavailable)");
    }
    MCCL_INFO("  sync_mode           = %s",
              global_sync_mode() == SyncMode::COALESCED ? "coalesced" : "full");
    MCCL_INFO("  compression         = %s", compressor_ ? compressor_->name().c_str() : "none");
    if (compressor_ && comp_mode == CompressionMode::TOPK) {
        MCCL_INFO("  topk_ratio          = %.4f", topk_ratio);
    }
    MCCL_INFO("  log_level           = %s", level_str(global_log_level()));
    MCCL_INFO("==============================");

    MCCL_INFO("ProcessGroupMCCL rank=%d ready", rank);
}

int ProcessGroupMCCL::default_ring_pipeline_window(int world_size) const {
    if (world_size < 3 || !overlap_comm_) return 1;
    // Conservative adaptive default: keep low in-flight depth for small rings.
    if (world_size >= 10) return 4;
    if (world_size >= 5) return 3;
    return 2;
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

void ProcessGroupMCCL::drain_ring_send_futures() {
    std::vector<std::shared_ptr<std::future<void>>> waiters;
    {
        std::lock_guard<std::mutex> lock(ring_pipeline_mu_);
        for (auto& q : pending_ring_sends_) {
            while (!q.empty()) {
                waiters.push_back(q.front());
                q.pop_front();
            }
        }
    }
    for (auto& f : waiters) {
        if (f) f->get();
    }
}

void ProcessGroupMCCL::validate_ring_step_indices(
    int ws, int send_idx, int recv_idx, uint32_t step_tid, uint32_t recv_tid) const {
    if (ws < 3) return;
    MCCL_CHECK(send_idx >= 0 && send_idx < ws,
               "ring assert: send_idx out of range");
    MCCL_CHECK(recv_idx >= 0 && recv_idx < ws,
               "ring assert: recv_idx out of range");
    uint32_t send_tid_idx = (step_tid & 0xFFFFu);
    uint32_t recv_tid_idx = (recv_tid & 0xFFFFu);
    MCCL_CHECK(send_tid_idx == static_cast<uint32_t>(send_idx),
               "ring assert: send tid lower bits do not match send_idx");
    MCCL_CHECK(recv_tid_idx == static_cast<uint32_t>(recv_idx),
               "ring assert: recv tid lower bits do not match recv_idx");
    uint32_t send_step = (step_tid >> 16);
    uint32_t recv_step = (recv_tid >> 16);
    MCCL_CHECK(send_step == recv_step,
               "ring assert: send/recv step mismatch");
    if (ring_assert_order_) {
        MCCL_CHECK(send_step < static_cast<uint32_t>(4 * ws),
                   "ring assert: step index out of expected bound");
    }
}

void ProcessGroupMCCL::ring_send_recv(
    int send_peer, OpType op, uint32_t seq, uint32_t send_tid,
    const void* send_data, size_t send_nbytes,
    int recv_peer, uint32_t recv_tid,
    void* recv_data, size_t recv_nbytes) {

    if (send_nbytes == 0 && recv_nbytes == 0) return;

    if (send_nbytes == 0) {
        net_engine_for(recv_peer).submit_sync(
            [this, recv_peer, op, seq, recv_tid, recv_data, recv_nbytes]() {
                MCCL_CHECK(transport_->recv_chunks(recv_peer, op, seq, recv_tid,
                                                   recv_data, recv_nbytes),
                           "ring_send_recv recv failed");
            });
        return;
    }
    if (recv_nbytes == 0) {
        net_engine_for(send_peer).submit_sync(
            [this, send_peer, op, seq, send_tid, send_data, send_nbytes]() {
                MCCL_CHECK(transport_->send_chunks(send_peer, op, seq, send_tid,
                                                   send_data, send_nbytes),
                           "ring_send_recv send failed");
            });
        return;
    }

    // Same-peer full-duplex (ws=2 allgather/reduce_scatter): do NOT split
    // into two tasks on one single-threaded net engine, which can deadlock.
    // Use the transport's proven one-socket overlap path instead.
    if (send_peer == recv_peer) {
        MCCL_CHECK(transport_->send_recv_overlap(
            send_peer, op, seq, send_tid, send_data, send_nbytes,
            recv_peer, op, seq, recv_tid, recv_data, recv_nbytes),
            "ring_send_recv same-peer overlap failed");
        return;
    }

    auto send_promise = std::make_shared<std::promise<void>>();
    auto recv_promise = std::make_shared<std::promise<void>>();
    auto send_future = std::make_shared<std::future<void>>(send_promise->get_future());
    std::future<void> recv_future = recv_promise->get_future();

    net_engine_for(send_peer).submit(
        [this, send_peer, op, seq, send_tid, send_data, send_nbytes, send_promise]() {
            try {
                auto send_t0 = std::chrono::steady_clock::now();
                MCCL_CHECK(transport_->send_chunks(send_peer, op, seq, send_tid,
                                                   send_data, send_nbytes),
                           "ring_send_recv send failed");
                auto send_t1 = std::chrono::steady_clock::now();
                metrics_->record_ring_io(
                    seq,
                    std::chrono::duration<double, std::milli>(send_t1 - send_t0).count(),
                    0.0);
                send_promise->set_value();
            } catch (...) {
                send_promise->set_exception(std::current_exception());
            }
        },
        []() {},
        [](std::exception_ptr) {},
        [this, seq](double q_ms) {
            metrics_->record_ring_queue_wait(seq, q_ms, 0.0);
        }
    );

    net_engine_for(recv_peer).submit(
        [this, recv_peer, op, seq, recv_tid, recv_data, recv_nbytes, recv_promise]() {
            try {
                auto recv_t0 = std::chrono::steady_clock::now();
                MCCL_CHECK(transport_->recv_chunks(recv_peer, op, seq, recv_tid,
                                                   recv_data, recv_nbytes),
                           "ring_send_recv recv failed");
                auto recv_t1 = std::chrono::steady_clock::now();
                metrics_->record_ring_io(
                    seq,
                    0.0,
                    std::chrono::duration<double, std::milli>(recv_t1 - recv_t0).count());
                recv_promise->set_value();
            } catch (...) {
                recv_promise->set_exception(std::current_exception());
            }
        },
        []() {},
        [](std::exception_ptr) {},
        [this, seq](double q_ms) {
            metrics_->record_ring_queue_wait(seq, 0.0, q_ms);
        }
    );

    {
        std::shared_ptr<std::future<void>> to_wait;
        {
            std::lock_guard<std::mutex> lock(ring_pipeline_mu_);
            auto& q = pending_ring_sends_[send_peer];
            q.push_back(send_future);
            while (static_cast<int>(q.size()) > ring_pipeline_window_ - 1) {
                to_wait = q.front();
                q.pop_front();
                if (!to_wait) continue;
                break;
            }
        }
        if (to_wait) to_wait->get();
    }
    recv_future.get();
}

ProcessGroupMCCL::~ProcessGroupMCCL() {
    try {
        MCCL_INFO("ProcessGroupMCCL rank=%d shutting down", getRank());
        clear_active_pg_if(this);
        metrics_->log_summary();
        if (health_) health_->stop();
        if (watchdog_) watchdog_->stop();
        if (reduce_engine_) reduce_engine_->stop();
        if (pipeline_reduce_engine_) pipeline_reduce_engine_->stop();
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
    TransportConfig cfg = TransportConfig::from_env();
    warn_if_mccl_port_overlaps_master(cfg);

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

// TODO: compressed_send/compressed_recv use the legacy serial transport path
// (blocking stage_for_send with internal mps_sync, serial send_chunks/recv_chunks).
// When compression is enabled, the overlapped transport and nosync staging are
// bypassed. A future optimization would integrate compression with the
// send_recv_overlap path.

void ProcessGroupMCCL::compressed_send(int peer, OpType op, uint32_t seq,
                                       uint32_t tid, const at::Tensor& tensor) {
    MCCL_CHECK(tensor.scalar_type() != at::kBFloat16 || !compressor_,
               "BFloat16 tensors are not supported with compression enabled. "
               "Disable compression (MCCL_COMPRESSION=none) or use float32/float16.");

    StagingBuffer staged = stage_for_send(tensor);

    if (compressor_) {
        size_t max_comp = compressor_->max_compressed_size(staged.nbytes);
        size_t max_wire = sizeof(uint32_t) + max_comp;
        PooledBuffer comp_buf(staging_memory_pool(), max_wire);

        size_t comp_size = compressor_->compress(
            staged.data, staged.nbytes,
            static_cast<uint8_t*>(comp_buf.data()) + sizeof(uint32_t),
            max_comp, tensor.scalar_type());

        uint32_t wire_size = static_cast<uint32_t>(comp_size);
        memcpy(comp_buf.data(), &wire_size, sizeof(uint32_t));

        // Zero-pad to max_wire so receiver gets the exact byte count it expects.
        // The size prefix tells the receiver how many bytes are real payload.
        if (comp_size < max_comp) {
            memset(static_cast<uint8_t*>(comp_buf.data()) + sizeof(uint32_t) + comp_size,
                   0, max_comp - comp_size);
        }

        MCCL_CHECK(transport_->send_chunks(peer, op, seq, tid, comp_buf.data(), max_wire),
                   "compressed_send send_chunks failed");
        metrics_->record_transport_bytes(max_wire, true);
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
        size_t max_wire = sizeof(uint32_t) + max_comp;
        PooledBuffer comp_buf(staging_memory_pool(), max_wire);
        MCCL_CHECK(transport_->recv_chunks(peer, op, seq, tid, comp_buf.data(), max_wire),
                   "compressed_recv recv_chunks failed");

        uint32_t actual_comp_size;
        memcpy(&actual_comp_size, comp_buf.data(), sizeof(uint32_t));
        MCCL_CHECK(actual_comp_size <= max_comp,
                   "Compressed payload larger than max_compressed_size");

        PooledBuffer decomp_buf(staging_memory_pool(), nbytes);
        compressor_->decompress(
            static_cast<uint8_t*>(comp_buf.data()) + sizeof(uint32_t),
            actual_comp_size, decomp_buf.data(), nbytes,
            tensor.scalar_type());
        unstage_from_recv(tensor, decomp_buf.data(), nbytes);
        metrics_->record_transport_bytes(max_wire, false);
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

    tensor = ensure_contiguous(tensor);
    check_single_tensor(tensor);

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
                    if (defer_mps_sync_to_engine) {
                        if (overlap_comm_ && event_sync_available()) {
                            uint64_t v = next_event_value();
                            commit_mps_and_signal(v);
                            wait_for_mps(v);
                        } else {
                            mps_stream_sync();
                        }
                    } else if (sync_val > 0) {
                        wait_for_mps(sync_val);
                    }

                    StagingBuffer staged = stage_for_send_nosync(tensor_copy);
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
                            auto red_t0 = std::chrono::steady_clock::now();
                            bool cpu_ok = tensor_cpu_accessible(tensor_copy);
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
                                    metal_reduce_op(tensor_copy, incoming, red_op);
                                }
                                metal_sync();
                            }
                            if (overlap_comm_) signal_mccl_done(next_event_value());

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
                        },
                        [this, seq](double q_ms) {
                            metrics_->record_queue_wait(seq, q_ms);
                        }
                    );
                },
                [this, work_ptr, seq](std::exception_ptr e) {
                    unregister_work(seq);
                    watchdog_->complete(seq);
                    metrics_->op_end(seq);
                    metrics_->record_error();
                    work_ptr->markError(e);
                },
                [this, seq](double q_ms) {
                    metrics_->record_queue_wait(seq, q_ms);
                }
            );
        } else {
            // Small message: run entirely on net_engine (no split needed, but
            // must use net_engine to avoid concurrent socket access with large ops)
            net_engine_for(peer).submit(
                [this, tensor_copy, seq, red_op, sync_val, defer_mps_sync_to_engine]() mutable {
                    if (defer_mps_sync_to_engine) {
                        if (overlap_comm_ && event_sync_available()) {
                            uint64_t v = next_event_value();
                            commit_mps_and_signal(v);
                            wait_for_mps(v);
                        } else {
                            mps_stream_sync();
                        }
                    } else if (sync_val > 0) {
                        wait_for_mps(sync_val);
                    }
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
                },
                [this, seq](double q_ms) {
                    metrics_->record_queue_wait(seq, q_ms);
                }
            );
        }
    } else {
        // ── 3+ ranks: ring algorithms on reduce_engine ──
        reduce_engine_->submit(
            [this, tensor_copy, seq, ws, nbytes, red_op, sync_val, defer_mps_sync_to_engine]() mutable {
                if (defer_mps_sync_to_engine) {
                    if (overlap_comm_ && event_sync_available()) {
                        uint64_t v = next_event_value();
                        commit_mps_and_signal(v);
                        wait_for_mps(v);
                    } else {
                        mps_stream_sync();
                    }
                } else if (sync_val > 0) {
                    wait_for_mps(sync_val);
                }
                const char* algo = "unknown";
                if (nbytes <= transport_->config().small_msg_threshold) {
                    algo = "small";
                    allreduce_small(tensor_copy, seq, red_op);
                } else {
                    algo = "ring_chunked";
                    allreduce_ring_chunked(tensor_copy, seq, red_op);
                }
                MCCL_INFO("allreduce seq=%u: algo=%s nbytes=%zu", seq, algo, nbytes);
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
            },
            [this, seq](double q_ms) {
                metrics_->record_queue_wait(seq, q_ms);
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
        flat_inputs.push_back(t.flatten());
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

    reduce_engine_->submit(
        [this, flat_copy, tensors_copy, seq, ws, nbytes, red_op, sync_val]() mutable {
            if (sync_val > 0) wait_for_mps(sync_val);
            if (ws == 2) {
                allreduce_two_rank(flat_copy, seq, red_op);
            } else if (ws >= 3) {
                allreduce_ring_chunked(flat_copy, seq, red_op);
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
        },
        [this, seq](double q_ms) {
            metrics_->record_queue_wait(seq, q_ms);
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
    bool use_fast = (tensor.scalar_type() == at::kFloat) && !compressor_;

    if (use_fast) {
        StagingBuffer staged = stage_for_send_nosync(tensor);

        constexpr size_t RS_AG_THRESHOLD = 8 * 1024 * 1024;  // 8 MB
        constexpr size_t REDUCE_CHUNK = 2 * 1024 * 1024;   // 2 MB

        if (nbytes >= RS_AG_THRESHOLD && cpu_ok) {
            // Reduce-scatter + allgather: each rank reduces only its half, then
            // they exchange the reduced halves.  Halves reduction work and
            // overlaps recv with reduce using chunked pipelining.
            MPSBufferView view = extract_mps_buffer(tensor);
            float* base = static_cast<float*>(view.cpu_ptr);

            size_t half = nbytes / 2;
            size_t my_off  = rank * half;
            size_t peer_off = peer * half;
            int64_t half_count = static_cast<int64_t>(half / sizeof(float));

            size_t nchunks = (half + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
            PooledBuffer recv_buf(staging_memory_pool(), REDUCE_CHUNK);

            auto net_t0 = std::chrono::steady_clock::now();
            double red_ms_accum = 0;

            // ── Phase 1: reduce-scatter ──
            // Send peer's half, recv my half in chunks, reduce each chunk.
            std::atomic<bool> send_ok{false};
            std::thread send_thread([&]() {
                bool ok = true;
                for (size_t c = 0; c < nchunks && ok; ++c) {
                    size_t off = c * REDUCE_CHUNK;
                    size_t len = std::min(REDUCE_CHUNK, half - off);
                    uint32_t tid = static_cast<uint32_t>(c + 1);
                    ok = transport_->send_chunks(
                        peer, OpType::ALLREDUCE, seq, tid,
                        static_cast<const uint8_t*>(staged.data) + peer_off + off, len);
                }
                send_ok.store(ok, std::memory_order_release);
            });

            for (size_t c = 0; c < nchunks; ++c) {
                size_t off = c * REDUCE_CHUNK;
                size_t len = std::min(REDUCE_CHUNK, half - off);
                int64_t chunk_count = static_cast<int64_t>(len / sizeof(float));
                uint32_t tid = static_cast<uint32_t>(c + 1);

                MCCL_CHECK(transport_->recv_chunks(
                    peer, OpType::ALLREDUCE, seq, tid,
                    recv_buf.data(), len),
                    "allreduce_two_rank RS recv chunk failed");

                auto rc0 = std::chrono::steady_clock::now();
                float* chunk_dst = base + (my_off + off) / sizeof(float);
                const float* chunk_src = static_cast<const float*>(recv_buf.data());
                if (op == c10d::ReduceOp::AVG) {
                    cpu_accumulate_and_scale(chunk_dst, chunk_src, chunk_count, 0.5f);
                } else {
                    cpu_reduce_op(chunk_dst, chunk_src, chunk_count, op);
                }
                auto rc1 = std::chrono::steady_clock::now();
                red_ms_accum += std::chrono::duration<double, std::milli>(rc1 - rc0).count();
            }

            send_thread.join();
            MCCL_CHECK(send_ok.load(std::memory_order_acquire),
                       "allreduce_two_rank RS send failed");

            // ── Phase 2: allgather ──
            // Exchange reduced halves so both ranks have the full result.
            uint32_t ag_base_tid = static_cast<uint32_t>(nchunks + 1);
            MCCL_CHECK(transport_->send_recv_overlap(
                peer, OpType::ALLREDUCE, seq, ag_base_tid,
                static_cast<const uint8_t*>(view.cpu_ptr) + my_off, half,
                peer, OpType::ALLREDUCE, seq, ag_base_tid,
                static_cast<uint8_t*>(view.cpu_ptr) + peer_off, half),
                "allreduce_two_rank AG send_recv_overlap failed");

            auto net_t1 = std::chrono::steady_clock::now();
            if (overlap_comm_) signal_mccl_done(next_event_value());

            double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
            double gbps = (nbytes * 2.0 * 8.0) / (net_ms / 1000.0) / 1e9;
            MCCL_INFO("allreduce_two_rank(RS+AG): %zu bytes (%zu RS chunks), "
                      "wall=%.1fms (%.2f Gbps), reduce=%.1fms",
                      nbytes, nchunks, net_ms, gbps, red_ms_accum);
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

            if (overlap_comm_) signal_mccl_done(next_event_value());

            double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
            double red_ms = std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
            double gbps = (nbytes * 2.0 * 8.0) / (net_ms / 1000.0) / 1e9;
            MCCL_INFO("allreduce_two_rank: %zu bytes, net=%.1fms (%.2f Gbps), reduce=%.1fms cpu_ok=%d",
                      nbytes, net_ms, gbps, red_ms, (int)cpu_ok);
            metrics_->record_phase(seq, 0, net_ms, red_ms);
        }

        metrics_->record_transport_bytes(nbytes, true);
        metrics_->record_transport_bytes(nbytes, false);
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
            metal_reduce_op(tensor, recv_tensor, op);
        }
        metal_sync();
    }
}

void ProcessGroupMCCL::allreduce_ring_chunked(at::Tensor& tensor, uint32_t seq,
                                               c10d::ReduceOp::RedOpType op) {
    if (dedicated_3plus_pipeline_ && getSize() >= 3) {
        allreduce_ring_chunked_pipeline(tensor, seq, op);
        return;
    }
    allreduce_ring_chunked_serial(tensor, seq, op);
}

void ProcessGroupMCCL::allreduce_ring_chunked_pipeline(at::Tensor& tensor, uint32_t seq,
                                                       c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int ws = getSize();
    size_t elem_size = tensor.element_size();
    int64_t total_elems = tensor.numel();
    const int num_chunks = 2 * ws;
    const int total_steps = 2 * (ws - 1);
    const int slot_limit = std::max(1, ring_pipeline_window_);
    bool use_cpu = (tensor.scalar_type() == at::kFloat) && tensor_cpu_accessible(tensor);

    int left = (rank - 1 + ws) % ws;
    int right = (rank + 1) % ws;

    at::Tensor flat = tensor.flatten();
    std::vector<at::Tensor> chunks;
    chunks.reserve(num_chunks);
    int64_t chunk_elems = (total_elems + num_chunks - 1) / num_chunks;
    for (int c = 0; c < num_chunks; c++) {
        int64_t start = c * chunk_elems;
        int64_t len = std::min(chunk_elems, total_elems - start);
        if (len <= 0) {
            chunks.push_back(torch::empty(0, tensor.options()));
        } else {
            chunks.push_back(flat.narrow(0, start, len));
        }
    }

    double total_net_ms = 0.0;
    double total_reduce_ms = 0.0;
    double total_stage_ms = 0.0;
    double total_writeback_ms = 0.0;
    double total_backpressure_ms = 0.0;
    uint64_t max_pipeline_depth = 0;

    auto stable_stage_chunk = [&](const at::Tensor& chunk, void* dst, size_t nbytes) {
        if (nbytes == 0) return;
        auto stage_t0 = std::chrono::steady_clock::now();
        if (tensor_cpu_accessible(chunk)) {
            MPSBufferView view = extract_mps_buffer(chunk);
            std::memcpy(dst, view.cpu_ptr, nbytes);
        } else {
            StagingBuffer staged = stage_for_send_nosync(chunk);
            std::memcpy(dst, staged.data, nbytes);
        }
        auto stage_t1 = std::chrono::steady_clock::now();
        double stage_ms = std::chrono::duration<double, std::milli>(stage_t1 - stage_t0).count();
        total_stage_ms += stage_ms;
    };

    auto submit_send_async =
        [&](uint32_t send_tid, SharedPooledBuffer send_buf, size_t send_nbytes)
        -> std::shared_ptr<std::future<void>> {
            if (send_nbytes == 0) return make_ready_future();
            auto send_promise = std::make_shared<std::promise<void>>();
            auto send_future = std::make_shared<std::future<void>>(send_promise->get_future());
            net_engine_for(right).submit(
                [this, right, seq, send_tid, send_buf, send_nbytes, send_promise]() {
                    try {
                        auto send_t0 = std::chrono::steady_clock::now();
                        MCCL_CHECK(transport_->send_chunks(
                            right, OpType::ALLREDUCE, seq, send_tid,
                            send_buf->data(), send_nbytes),
                            "ring pipeline send failed");
                        auto send_t1 = std::chrono::steady_clock::now();
                        metrics_->record_transport_bytes(send_nbytes, true);
                        metrics_->record_ring_io(
                            seq,
                            std::chrono::duration<double, std::milli>(send_t1 - send_t0).count(),
                            0.0);
                        send_promise->set_value();
                    } catch (...) {
                        send_promise->set_exception(std::current_exception());
                    }
                },
                []() {},
                [](std::exception_ptr) {},
                [this, seq](double q_ms) {
                    metrics_->record_ring_queue_wait(seq, q_ms, 0.0);
                });
            return send_future;
        };

    auto submit_recv_async =
        [&](uint32_t recv_tid, SharedPooledBuffer recv_buf, size_t recv_nbytes)
        -> std::shared_ptr<std::future<void>> {
            if (recv_nbytes == 0) return make_ready_future();
            auto recv_promise = std::make_shared<std::promise<void>>();
            auto recv_future = std::make_shared<std::future<void>>(recv_promise->get_future());
            net_engine_for(left).submit(
                [this, left, seq, recv_tid, recv_buf, recv_nbytes, recv_promise]() {
                    try {
                        auto recv_t0 = std::chrono::steady_clock::now();
                        MCCL_CHECK(transport_->recv_chunks(
                            left, OpType::ALLREDUCE, seq, recv_tid,
                            recv_buf->data(), recv_nbytes),
                            "ring pipeline recv failed");
                        auto recv_t1 = std::chrono::steady_clock::now();
                        metrics_->record_transport_bytes(recv_nbytes, false);
                        metrics_->record_ring_io(
                            seq,
                            0.0,
                            std::chrono::duration<double, std::milli>(recv_t1 - recv_t0).count());
                        recv_promise->set_value();
                    } catch (...) {
                        recv_promise->set_exception(std::current_exception());
                    }
                },
                []() {},
                [](std::exception_ptr) {},
                [this, seq](double q_ms) {
                    metrics_->record_ring_queue_wait(seq, 0.0, q_ms);
                });
            return recv_future;
        };

    auto sample_pipeline_depth = [&](uint64_t depth) {
        max_pipeline_depth = std::max(max_pipeline_depth, depth);
        metrics_->record_pipeline(seq, 0.0, 0.0, 0.0, depth, true);
    };

    auto run_phase =
        [&](RingPipelinePhase phase,
            auto compute_indices) {
            std::vector<int> chunk_ready_step(num_chunks, 0);
            std::deque<RingPipelineSlot> inflight;
            std::deque<std::shared_ptr<std::future<void>>> pending_sends;

            auto complete_recv = [&](RingPipelineSlot& slot, bool block) -> bool {
                if (slot.recv_done) return true;
                if (!block && !future_ready(slot.recv_future)) return false;
                auto wait_t0 = std::chrono::steady_clock::now();
                slot.recv_future->get();
                auto wait_t1 = std::chrono::steady_clock::now();
                total_net_ms += std::chrono::duration<double, std::milli>(wait_t1 - wait_t0).count();
                slot.recv_done = true;
                return true;
            };

            auto drain_send_window = [&](bool force_all) {
                const size_t max_pending = force_all
                    ? 0
                    : static_cast<size_t>(std::max(0, slot_limit - 1));
                while (pending_sends.size() > max_pending) {
                    auto future = pending_sends.front();
                    pending_sends.pop_front();
                    auto wait_t0 = std::chrono::steady_clock::now();
                    future->get();
                    auto wait_t1 = std::chrono::steady_clock::now();
                    double wait_ms = std::chrono::duration<double, std::milli>(wait_t1 - wait_t0).count();
                    total_backpressure_ms += wait_ms;
                    total_net_ms += wait_ms;
                }
            };

            auto launch_reduce = [&](RingPipelineSlot& slot) {
                if (slot.reduce_started) return;
                slot.reduce_started = true;
                if (slot.recv_bytes == 0) {
                    slot.reduce_future = std::async(std::launch::deferred, [] {
                        return RingReduceResult{};
                    });
                    return;
                }
                if (use_cpu) {
                    at::Tensor recv_chunk = slot.recv_chunk;
                    SharedPooledBuffer recv_buf = slot.recv_buf;
                    slot.reduce_future = std::async(
                        std::launch::async,
                        [recv_chunk, recv_buf, recv_bytes = slot.recv_bytes, elem_size, op, phase]() {
                            RingReduceResult result;
                            if (phase == RingPipelinePhase::REDUCE_SCATTER) {
                                auto red_t0 = std::chrono::steady_clock::now();
                                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                                cpu_reduce_op(
                                    static_cast<float*>(chunk_view.cpu_ptr),
                                    static_cast<const float*>(recv_buf->data()),
                                    recv_chunk.numel(), op);
                                auto red_t1 = std::chrono::steady_clock::now();
                                result.reduce_ms = std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
                            } else {
                                auto wb_t0 = std::chrono::steady_clock::now();
                                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                                std::memcpy(chunk_view.cpu_ptr, recv_buf->data(), recv_bytes);
                                auto wb_t1 = std::chrono::steady_clock::now();
                                result.writeback_ms = std::chrono::duration<double, std::milli>(wb_t1 - wb_t0).count();
                            }
                            return result;
                        });
                } else {
                    auto promise = std::make_shared<std::promise<RingReduceResult>>();
                    slot.reduce_future = promise->get_future();
                    at::Tensor recv_chunk = slot.recv_chunk;
                    SharedPooledBuffer recv_buf = slot.recv_buf;
                    pipeline_reduce_engine_->submit(
                        [recv_chunk, recv_buf, recv_bytes = slot.recv_bytes, op, phase, promise]() mutable {
                            try {
                                RingReduceResult result;
                                if (recv_bytes > 0) {
                                    if (phase == RingPipelinePhase::REDUCE_SCATTER) {
                                        auto wb_t0 = std::chrono::steady_clock::now();
                                        at::Tensor incoming = torch::empty_like(recv_chunk);
                                        unstage_from_recv(incoming, recv_buf->data(), recv_bytes);
                                        auto wb_t1 = std::chrono::steady_clock::now();
                                        result.writeback_ms += std::chrono::duration<double, std::milli>(wb_t1 - wb_t0).count();

                                        auto red_t0 = std::chrono::steady_clock::now();
                                        metal_reduce_op(recv_chunk, incoming, op);
                                        auto red_t1 = std::chrono::steady_clock::now();
                                        result.reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
                                    } else {
                                        auto wb_t0 = std::chrono::steady_clock::now();
                                        unstage_from_recv(recv_chunk, recv_buf->data(), recv_bytes);
                                        auto wb_t1 = std::chrono::steady_clock::now();
                                        result.writeback_ms += std::chrono::duration<double, std::milli>(wb_t1 - wb_t0).count();
                                    }
                                }
                                promise->set_value(result);
                            } catch (...) {
                                promise->set_exception(std::current_exception());
                            }
                        },
                        []() {},
                        [promise](std::exception_ptr e) {
                            try {
                                promise->set_exception(e);
                            } catch (...) {}
                        },
                        [this, seq](double q_ms) {
                            metrics_->record_queue_wait(seq, q_ms);
                        });
                }
            };

            auto finish_front = [&](bool block) -> bool {
                if (inflight.empty()) return false;
                auto& slot = inflight.front();

                if (!complete_recv(slot, block)) return false;
                if (!slot.reduce_started) launch_reduce(slot);

                RingReduceResult reduce_result;
                if (!block && !future_ready(slot.reduce_future)) return false;
                reduce_result = slot.reduce_future.get();

                total_reduce_ms += reduce_result.reduce_ms;
                total_writeback_ms += reduce_result.writeback_ms;
                chunk_ready_step[slot.recv_chunk_idx] =
                    std::max(chunk_ready_step[slot.recv_chunk_idx], slot.step + 2);
                inflight.pop_front();
                return true;
            };

            auto kick_ready_reductions = [&]() {
                for (auto& slot : inflight) {
                    if (!complete_recv(slot, false)) continue;
                    launch_reduce(slot);
                }
            };

            for (int step = 0; step < total_steps;) {
                kick_ready_reductions();
                while (finish_front(false)) {}

                bool launched = false;
                while (step < total_steps && static_cast<int>(inflight.size()) < slot_limit) {
                    int send_chunk_idx = 0;
                    int recv_chunk_idx = 0;
                    uint32_t send_tid = 0;
                    uint32_t recv_tid = 0;
                    compute_indices(step, send_chunk_idx, recv_chunk_idx, send_tid, recv_tid);

                    if (step < chunk_ready_step[send_chunk_idx]) break;

                    at::Tensor& send_chunk = chunks[send_chunk_idx];
                    at::Tensor& recv_chunk = chunks[recv_chunk_idx];
                    size_t send_bytes = send_chunk.numel() * elem_size;
                    size_t recv_bytes = recv_chunk.numel() * elem_size;

                    if (send_bytes == 0 && recv_bytes == 0) {
                        chunk_ready_step[recv_chunk_idx] =
                            std::max(chunk_ready_step[recv_chunk_idx], step + 2);
                        step++;
                        launched = true;
                        continue;
                    }

                    if (send_bytes > 0) {
                        auto send_buf = std::make_shared<PooledBuffer>(staging_memory_pool(), send_bytes);
                        stable_stage_chunk(send_chunk, send_buf->data(), send_bytes);
                        auto send_future = submit_send_async(send_tid, send_buf, send_bytes);
                        pending_sends.push_back(send_future);
                        drain_send_window(false);
                    }

                    RingPipelineSlot slot;
                    slot.phase = phase;
                    slot.step = step;
                    slot.send_chunk_idx = send_chunk_idx;
                    slot.recv_chunk_idx = recv_chunk_idx;
                    slot.recv_bytes = recv_bytes;
                    slot.recv_chunk = recv_chunk;
                    if (recv_bytes > 0) {
                        slot.recv_buf = std::make_shared<PooledBuffer>(staging_memory_pool(), recv_bytes);
                    }
                    slot.recv_future = submit_recv_async(recv_tid, slot.recv_buf, recv_bytes);
                    inflight.push_back(std::move(slot));
                    sample_pipeline_depth(static_cast<uint64_t>(inflight.size() + pending_sends.size()));
                    launched = true;
                    step++;
                }

                kick_ready_reductions();

                if (!launched) {
                    auto wait_t0 = std::chrono::steady_clock::now();
                    bool progressed = finish_front(true);
                    auto wait_t1 = std::chrono::steady_clock::now();
                    double wait_ms = std::chrono::duration<double, std::milli>(wait_t1 - wait_t0).count();
                    total_backpressure_ms += wait_ms;
                    if (!progressed) break;
                }
            }

            while (finish_front(true)) {}
            drain_send_window(true);
        };

    auto compute_rs_indices =
        [&](int step, int& send_chunk_idx, int& recv_chunk_idx, uint32_t& send_tid, uint32_t& recv_tid) {
            send_chunk_idx = (rank * 2 - step + num_chunks) % num_chunks;
            recv_chunk_idx = (rank * 2 - step - 2 + num_chunks) % num_chunks;
            send_tid = (static_cast<uint32_t>(step) << 16) | static_cast<uint32_t>(send_chunk_idx);
            recv_tid = (static_cast<uint32_t>(step) << 16) | static_cast<uint32_t>(recv_chunk_idx);
            validate_ring_step_indices(num_chunks, send_chunk_idx, recv_chunk_idx, send_tid, recv_tid);
        };

    auto compute_ag_indices =
        [&](int step, int& send_chunk_idx, int& recv_chunk_idx, uint32_t& send_tid, uint32_t& recv_tid) {
            uint32_t ag_step = static_cast<uint32_t>(total_steps + step);
            send_chunk_idx = (rank * 2 + step + 2 + num_chunks) % num_chunks;
            recv_chunk_idx = (rank * 2 + step + num_chunks) % num_chunks;
            send_tid = (ag_step << 16) | static_cast<uint32_t>(send_chunk_idx);
            recv_tid = (ag_step << 16) | static_cast<uint32_t>(recv_chunk_idx);
            validate_ring_step_indices(num_chunks, send_chunk_idx, recv_chunk_idx, send_tid, recv_tid);
        };

    run_phase(
        RingPipelinePhase::REDUCE_SCATTER,
        compute_rs_indices);
    run_phase(
        RingPipelinePhase::ALLGATHER,
        compute_ag_indices);

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr), total_elems, 1.0f / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        if (overlap_comm_) signal_mccl_done(next_event_value());
    } else {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            metal_begin_batch("mccl_allreduce_ring_chunked_pipeline_avg");
            metal_scale_inplace(tensor, 1.0 / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        auto sync_t0 = std::chrono::steady_clock::now();
        metal_sync();
        auto sync_t1 = std::chrono::steady_clock::now();
        total_reduce_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
    }

    metrics_->record_pipeline(
        seq,
        total_stage_ms,
        total_writeback_ms,
        total_backpressure_ms,
        max_pipeline_depth,
        false);
    metrics_->record_phase(seq, 0.0, total_net_ms, total_reduce_ms);
}

void ProcessGroupMCCL::allreduce_ring_chunked_serial(at::Tensor& tensor, uint32_t seq,
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
    bool use_cpu = (tensor.scalar_type() == at::kFloat) && tensor_cpu_accessible(tensor);

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

    PooledBuffer recv_buf_pool(staging_memory_pool(), chunk_elems * elem_size);
    double total_net_ms = 0.0;
    double total_reduce_ms = 0.0;

    // ── Phase 1: Reduce-scatter (2*(ws-1) serial steps) ──
    for (int step = 0; step < 2 * (ws - 1); step++) {
        int send_chunk_idx = (rank * 2 - step + 2 * ws) % (2 * ws);
        int recv_chunk_idx = (rank * 2 - step - 2 + 2 * ws) % (2 * ws);

        at::Tensor& send_chunk = chunks[send_chunk_idx];
        at::Tensor& recv_chunk = chunks[recv_chunk_idx];

        if (send_chunk.numel() == 0 && recv_chunk.numel() == 0) continue;

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_chunk_idx;
        uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_chunk_idx;
        validate_ring_step_indices(2 * ws, send_chunk_idx, recv_chunk_idx, step_tid, recv_tid);

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            staged = stage_for_send_nosync(send_chunk);
        }

        auto net_t0 = std::chrono::steady_clock::now();
        ring_send_recv(right, OpType::ALLREDUCE, seq, step_tid,
                       staged.data, send_bytes,
                       left, recv_tid,
                       recv_buf_pool.data(), recv_bytes);
        auto net_t1 = std::chrono::steady_clock::now();
        total_net_ms += std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();

        if (recv_bytes > 0) {
            auto red_t0 = std::chrono::steady_clock::now();
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                cpu_reduce_op(
                    static_cast<float*>(chunk_view.cpu_ptr),
                    static_cast<const float*>(recv_buf_pool.data()),
                    recv_chunk.numel(), op);
            } else {
                at::Tensor incoming = torch::empty_like(recv_chunk);
                unstage_from_recv(incoming, recv_buf_pool.data(), recv_bytes);
                metal_reduce_op(recv_chunk, incoming, op);
            }
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    // ── Phase 2: Allgather (2*(ws-1) serial steps) ──
    for (int step = 0; step < 2 * (ws - 1); step++) {
        int send_chunk_idx = (rank * 2 + step + 2 + 2 * ws) % (2 * ws);
        int recv_chunk_idx = (rank * 2 + step + 2 * ws) % (2 * ws);

        at::Tensor& send_chunk = chunks[send_chunk_idx];
        at::Tensor& recv_chunk = chunks[recv_chunk_idx];

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        uint32_t ag_step = static_cast<uint32_t>(2 * (ws - 1) + step);
        uint32_t step_tid = (ag_step << 16) | send_chunk_idx;
        uint32_t recv_tid_ag = (ag_step << 16) | recv_chunk_idx;
        validate_ring_step_indices(2 * ws, send_chunk_idx, recv_chunk_idx, step_tid, recv_tid_ag);

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            staged = stage_for_send_nosync(send_chunk);
        }

        auto net_t0 = std::chrono::steady_clock::now();
        ring_send_recv(right, OpType::ALLREDUCE, seq, step_tid,
                       staged.data, send_bytes,
                       left, recv_tid_ag,
                       recv_buf_pool.data(), recv_bytes);
        auto net_t1 = std::chrono::steady_clock::now();
        total_net_ms += std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();

        if (recv_bytes > 0) {
            auto red_t0 = std::chrono::steady_clock::now();
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                memcpy(chunk_view.cpu_ptr, recv_buf_pool.data(), recv_bytes);
            } else {
                unstage_from_recv(recv_chunk, recv_buf_pool.data(), recv_bytes);
            }
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr), total_elems, 1.0f / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        if (overlap_comm_) signal_mccl_done(next_event_value());
    } else {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            metal_begin_batch("mccl_allreduce_ring_chunked_avg");
            metal_scale_inplace(tensor, 1.0 / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        auto sync_t0 = std::chrono::steady_clock::now();
        metal_sync();
        auto sync_t1 = std::chrono::steady_clock::now();
        total_reduce_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
    }

    drain_ring_send_futures();
    metrics_->record_phase(seq, 0, total_net_ms, total_reduce_ms);
}

void ProcessGroupMCCL::allreduce_ring(at::Tensor& tensor, uint32_t seq,
                                      c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int ws = getSize();
    size_t elem_size = tensor.element_size();
    int64_t total_elems = tensor.numel();
    int64_t chunk_elems = (total_elems + ws - 1) / ws;
    bool use_cpu = (tensor.scalar_type() == at::kFloat) && tensor_cpu_accessible(tensor);

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

    PooledBuffer recv_buf_pool(staging_memory_pool(), chunk_elems * elem_size);
    double total_net_ms = 0.0;
    double total_reduce_ms = 0.0;

    // ── Reduce-scatter phase (serial steps -- data dependencies between steps) ──
    for (int step = 0; step < ws - 1; step++) {
        int send_idx = (rank - step + ws) % ws;
        int recv_idx = (rank - step - 1 + ws) % ws;
        uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
        uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;
        validate_ring_step_indices(ws, send_idx, recv_idx, step_tid, recv_tid);

        at::Tensor& send_chunk = chunks[send_idx];
        at::Tensor& recv_chunk = chunks[recv_idx];

        if (send_chunk.numel() == 0 && recv_chunk.numel() == 0) continue;

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            staged = stage_for_send_nosync(send_chunk);
        }

        auto net_t0 = std::chrono::steady_clock::now();
        ring_send_recv(right, OpType::ALLREDUCE, seq, step_tid,
                       staged.data, send_bytes,
                       left, recv_tid,
                       recv_buf_pool.data(), recv_bytes);
        auto net_t1 = std::chrono::steady_clock::now();
        total_net_ms += std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();

        if (recv_bytes > 0) {
            auto red_t0 = std::chrono::steady_clock::now();
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                cpu_reduce_op(
                    static_cast<float*>(chunk_view.cpu_ptr),
                    static_cast<const float*>(recv_buf_pool.data()),
                    recv_chunk.numel(), op);
            } else {
                at::Tensor incoming = torch::empty_like(recv_chunk);
                unstage_from_recv(incoming, recv_buf_pool.data(), recv_bytes);
                metal_reduce_op(recv_chunk, incoming, op);
            }
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    // ── Allgather phase (serial steps) ──
    for (int step = 0; step < ws - 1; step++) {
        int send_idx = (rank - step + 1 + ws) % ws;
        int recv_idx = (rank - step + ws) % ws;
        uint32_t ag_step = static_cast<uint32_t>(ws - 1 + step);
        uint32_t step_tid = (ag_step << 16) | send_idx;
        uint32_t recv_tid_ag = (ag_step << 16) | recv_idx;
        validate_ring_step_indices(ws, send_idx, recv_idx, step_tid, recv_tid_ag);

        at::Tensor& send_chunk = chunks[send_idx];
        at::Tensor& recv_chunk = chunks[recv_idx];

        size_t send_bytes = send_chunk.numel() * elem_size;
        size_t recv_bytes = recv_chunk.numel() * elem_size;

        StagingBuffer staged = {nullptr, 0};
        if (send_bytes > 0) {
            staged = stage_for_send_nosync(send_chunk);
        }

        auto net_t0 = std::chrono::steady_clock::now();
        ring_send_recv(right, OpType::ALLREDUCE, seq, step_tid,
                       staged.data, send_bytes,
                       left, recv_tid_ag,
                       recv_buf_pool.data(), recv_bytes);
        auto net_t1 = std::chrono::steady_clock::now();
        total_net_ms += std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();

        if (recv_bytes > 0) {
            auto red_t0 = std::chrono::steady_clock::now();
            if (use_cpu) {
                MPSBufferView chunk_view = extract_mps_buffer(recv_chunk);
                memcpy(chunk_view.cpu_ptr, recv_buf_pool.data(), recv_bytes);
            } else {
                unstage_from_recv(recv_chunk, recv_buf_pool.data(), recv_bytes);
            }
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr),
                              total_elems, 1.0f / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        if (overlap_comm_) signal_mccl_done(next_event_value());
    } else {
        if (op == c10d::ReduceOp::AVG) {
            auto red_t0 = std::chrono::steady_clock::now();
            metal_begin_batch("mccl_allreduce_ring_avg");
            metal_scale_inplace(tensor, 1.0 / ws);
            auto red_t1 = std::chrono::steady_clock::now();
            total_reduce_ms += std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        }
        auto sync_t0 = std::chrono::steady_clock::now();
        metal_sync();
        auto sync_t1 = std::chrono::steady_clock::now();
        total_reduce_ms += std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
    }

    drain_ring_send_futures();
    metrics_->record_phase(seq, 0, total_net_ms, total_reduce_ms);
}


void ProcessGroupMCCL::allreduce_small(at::Tensor& tensor, uint32_t seq,
                                       c10d::ReduceOp::RedOpType op) {
    int rank = getRank();
    int ws = getSize();

    if (ws == 2) {
        allreduce_two_rank(tensor, seq, op);
        return;
    }

    size_t nbytes = tensor_nbytes(tensor);

    bool use_cpu = (tensor.scalar_type() == at::kFloat) && !compressor_ && tensor_cpu_accessible(tensor);

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

            StagingBuffer staged = stage_for_send_nosync(tensor);
            for (int peer = 1; peer < ws; peer++) {
                MCCL_CHECK(transport_->send_chunks(peer, OpType::ALLREDUCE, seq, 1,
                                                   staged.data, nbytes),
                           "allreduce_small send to rank " + std::to_string(peer) + " failed");
                metrics_->record_transport_bytes(nbytes, true);
            }
            if (overlap_comm_) signal_mccl_done(next_event_value());
        } else {
            StagingBuffer staged = stage_for_send_nosync(tensor);
            MCCL_CHECK(transport_->send_chunks(0, OpType::ALLREDUCE, seq, 0,
                                               staged.data, nbytes),
                       "allreduce_small send to rank 0 failed");
            metrics_->record_transport_bytes(nbytes, true);

            MCCL_CHECK(transport_->recv_chunks(0, OpType::ALLREDUCE, seq, 1,
                                               dst, nbytes),
                       "allreduce_small recv from rank 0 failed");
            metrics_->record_transport_bytes(nbytes, false);
            if (overlap_comm_) signal_mccl_done(next_event_value());
        }
    } else {
        // f16 or compressed path: existing Metal pipeline
        if (rank == 0) {
            metal_begin_batch("mccl_allreduce_small_gpu");
            for (int peer = 1; peer < ws; peer++) {
                at::Tensor incoming = torch::empty_like(tensor);
                compressed_recv(peer, OpType::ALLREDUCE, seq, 0, incoming);
                metal_reduce_op(tensor, incoming, op);
            }

            if (op == c10d::ReduceOp::AVG) {
                metal_scale_inplace(tensor, 1.0 / ws);
            }
            metal_sync();

            for (int peer = 1; peer < ws; peer++) {
                compressed_send(peer, OpType::ALLREDUCE, seq, 1, tensor);
            }
        } else {
            compressed_send(0, OpType::ALLREDUCE, seq, 0, tensor);

            at::Tensor result = torch::empty_like(tensor);
            compressed_recv(0, OpType::ALLREDUCE, seq, 1, result);
            tensor.copy_(result);
            metal_sync();
        }
    }
}


// ── broadcast ───────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {

    MCCL_CHECK_TENSOR(tensors.size() == 1, "MCCL broadcast expects one tensor");
    at::Tensor& tensor = tensors[0];
    tensor = ensure_contiguous(tensor);
    check_single_tensor(tensor);

    int root = static_cast<int>(opts.rootRank);
    MCCL_CHECK(root >= 0 && root < getSize(),
               "broadcast rootRank=" + std::to_string(root) +
               " out of range [0, " + std::to_string(getSize()) + ")");

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

    if (rank == root) {
        // Root: stage data then fan out sends to per-peer NetEngines.
        // An atomic counter tracks completion; the last send to finish
        // submits markComplete to reduce_engine.
        auto staged_buf = std::make_shared<PooledBuffer>(staging_memory_pool(), nbytes);
        int num_peers = ws - 1;
        auto sends_remaining = std::make_shared<std::atomic<int>>(num_peers);

        reduce_engine_->submit(
            [this, tensor_copy, sync_val_bc, staged_buf]() mutable {
                if (sync_val_bc > 0) wait_for_mps(sync_val_bc);
                StagingBuffer staged = stage_for_send_nosync(tensor_copy);
                memcpy(staged_buf->data(), staged.data, staged.nbytes);
            },
            [this, staged_buf, seq, root, ws, nbytes, sends_remaining, work_ptr]() mutable {
                for (int peer = 0; peer < ws; peer++) {
                    if (peer == root) continue;
                    net_engine_for(peer).submit(
                        [this, peer, seq, staged_buf, nbytes]() mutable {
                            MCCL_CHECK(transport_->send_chunks(peer, OpType::BROADCAST, seq, 0,
                                                               staged_buf->data(), nbytes),
                                       "broadcast send to rank " + std::to_string(peer) + " failed");
                            metrics_->record_transport_bytes(nbytes, true);
                        },
                        [this, sends_remaining, work_ptr, seq]() {
                            if (sends_remaining->fetch_sub(1) == 1) {
                                reduce_engine_->submit(
                                    [this]() {
                                        if (overlap_comm_) signal_mccl_done(next_event_value());
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
                        },
                        [this, work_ptr, seq, sends_remaining](std::exception_ptr e) {
                            MCCL_ERROR("broadcast send to peer failed");
                            if (sends_remaining->fetch_sub(1) == 1) {
                                unregister_work(seq);
                                watchdog_->complete(seq);
                                metrics_->op_end(seq);
                                metrics_->record_error();
                                work_ptr->markError(e);
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
                if (sync_val_bc > 0) wait_for_mps(sync_val_bc);
                bool use_cpu = tensor_cpu_accessible(tensor_copy);
                if (use_cpu) {
                    MPSBufferView view = extract_mps_buffer(tensor_copy);
                    MCCL_CHECK(transport_->recv_chunks(root, OpType::BROADCAST, seq, 0,
                                                       view.cpu_ptr, nbytes),
                               "broadcast recv from root failed");
                } else {
                    PooledBuffer recv_buf(staging_memory_pool(), nbytes);
                    MCCL_CHECK(transport_->recv_chunks(root, OpType::BROADCAST, seq, 0,
                                                       recv_buf.data(), nbytes),
                               "broadcast recv from root failed");
                    unstage_from_recv(tensor_copy, recv_buf.data(), nbytes);
                }
                metrics_->record_transport_bytes(nbytes, false);
                if (overlap_comm_) signal_mccl_done(next_event_value());
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

    uint32_t seq = collective_seq_.fetch_add(1);
    auto work = c10::make_intrusive<WorkMCCL>(c10d::OpType::BARRIER, seq);
    auto work_ptr = work;

    register_work(seq, work);
    watchdog_->watch(seq, "barrier");
    metrics_->op_start(seq, "barrier", 0);

    // Drain all per-peer net engines before submitting to reduce_engine_.
    //
    // The 2-rank large-message allreduce splits work across two engines:
    //   1. net_engine_for(peer)  — network I/O
    //   2. reduce_engine_        — local reduction, enqueued from net on_complete
    //
    // On slow TCP the net phase may still be in-flight when barrier() is
    // called.  If we submit the store barrier directly to reduce_engine_
    // it can land there *before* the net on_complete has had a chance to
    // enqueue the local reduction, so the barrier completes while the last
    // allreduce result has not yet been written back to the model tensor.
    //
    // Fix: flush every net engine with a synchronous no-op.  This blocks
    // the caller until all pending net-engine on_complete callbacks have
    // fired (and therefore any reduce_engine_ submissions they make have
    // been enqueued).  Only then do we enqueue the store barrier on
    // reduce_engine_, which will execute after those reductions complete.
    for (auto& engine : net_engines_) {
        if (engine && engine->running()) {
            engine->submit_sync([]() {});
        }
    }

    reduce_engine_->submit(
        [this, seq]() {
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
        t = ensure_contiguous(t);
        check_same_shape_dtype(input, t);
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

    reduce_engine_->submit(
        [this, input_copy, outputs_copy, seq, rank, ws, nbytes, sync_val_ag]() mutable {
            if (sync_val_ag > 0) wait_for_mps(sync_val_ag);
            bool use_cpu = (input_copy.scalar_type() == at::kFloat) && tensor_cpu_accessible(input_copy);

            if (use_cpu) {
                MPSBufferView in_view = extract_mps_buffer(input_copy);
                MPSBufferView out_view = extract_mps_buffer(outputs_copy[rank]);
                memcpy(out_view.cpu_ptr, in_view.cpu_ptr, nbytes);
            } else {
                outputs_copy[rank].copy_(input_copy);
            }

            int left = (rank - 1 + ws) % ws;
            int right = (rank + 1) % ws;

            PooledBuffer recv_buf_fallback(staging_memory_pool(), use_cpu ? 0 : nbytes);

            for (int step = 0; step < ws - 1; step++) {
                int send_idx = (rank - step + ws) % ws;
                int recv_idx = (rank - step - 1 + ws) % ws;
                uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
                uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;
                validate_ring_step_indices(ws, send_idx, recv_idx, step_tid, recv_tid);

                StagingBuffer staged = use_cpu
                    ? stage_for_send_nosync(outputs_copy[send_idx])
                    : stage_for_send(outputs_copy[send_idx]);

                void* recv_dst;
                if (use_cpu) {
                    MPSBufferView view = extract_mps_buffer(outputs_copy[recv_idx]);
                    recv_dst = view.cpu_ptr;
                } else {
                    recv_dst = recv_buf_fallback.data();
                }

                ring_send_recv(right, OpType::ALLGATHER, seq, step_tid,
                               staged.data, nbytes,
                               left, recv_tid,
                               recv_dst, nbytes);

                if (!use_cpu) {
                    unstage_from_recv(outputs_copy[recv_idx], recv_dst, nbytes);
                }
                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }
            drain_ring_send_futures();
            if (overlap_comm_) signal_mccl_done(next_event_value());
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

    at::Tensor output = ensure_contiguous(outputTensors[0]);
    auto& inputs = inputTensors[0];
    for (auto& t : inputs) {
        t = ensure_contiguous(t);
        check_single_tensor(t);
        check_same_shape_dtype(output, t);
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

    reduce_engine_->submit(
        [this, output_copy, inputs_copy, seq, rank, ws, nbytes, rs_op, sync_val_rs]() mutable {
            if (sync_val_rs > 0) wait_for_mps(sync_val_rs);
            int left = (rank - 1 + ws) % ws;
            int right = (rank + 1) % ws;
            bool use_cpu = (inputs_copy[0].scalar_type() == at::kFloat) && tensor_cpu_accessible(inputs_copy[0]);

            std::vector<at::Tensor> chunks = inputs_copy;
            PooledBuffer recv_buf(staging_memory_pool(), nbytes);

            for (int step = 0; step < ws - 1; step++) {
                int send_idx = (rank + 1 - step + ws) % ws;
                int recv_idx = (rank - step + ws) % ws;
                uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
                uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;
                validate_ring_step_indices(ws, send_idx, recv_idx, step_tid, recv_tid);

                StagingBuffer staged = use_cpu
                    ? stage_for_send_nosync(chunks[send_idx])
                    : stage_for_send(chunks[send_idx]);

                ring_send_recv(right, OpType::REDUCE_SCATTER, seq, step_tid,
                               staged.data, nbytes,
                               left, recv_tid,
                               recv_buf.data(), nbytes);

                if (use_cpu) {
                    MPSBufferView chunk_view = extract_mps_buffer(chunks[recv_idx]);
                    cpu_reduce_op(
                        static_cast<float*>(chunk_view.cpu_ptr),
                        static_cast<const float*>(recv_buf.data()),
                        chunks[recv_idx].numel(), rs_op);
                } else {
                    at::Tensor incoming = torch::empty_like(chunks[recv_idx]);
                    unstage_from_recv(incoming, recv_buf.data(), nbytes);
                    metal_reduce_op(chunks[recv_idx], incoming, rs_op);
                }

                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }
            drain_ring_send_futures();

            int my_chunk = rank;
            if (use_cpu) {
                MPSBufferView src_view = extract_mps_buffer(chunks[my_chunk]);
                MPSBufferView dst_view = extract_mps_buffer(output_copy);
                memcpy(dst_view.cpu_ptr, src_view.cpu_ptr, nbytes);
                if (overlap_comm_) signal_mccl_done(next_event_value());
            } else {
                metal_sync();
                output_copy.copy_(chunks[my_chunk]);
                // Use event sync for better overlap
                if (overlap_comm_ && event_sync_available()) {
                    uint64_t val = next_event_value();
                    commit_mps_and_signal(val);
                    wait_for_mps(val);
                } else {
                    mps_stream_sync();
                }
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
            if (sync_val_s > 0) wait_for_mps(sync_val_s);
            StagingBuffer staged = stage_for_send_nosync(tensor);
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

    at::Tensor tensor = ensure_contiguous(tensors[0]);
    check_single_tensor(tensor);

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
            bool use_cpu = tensor_cpu_accessible(tensor);
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
