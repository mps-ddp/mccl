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

namespace mccl {

namespace {

enum class SyncMode {
    FULL,       // torch::mps::synchronize() every op (safest)
    COALESCED,  // sync once at start of a batch; skip for subsequent ops in same batch
};

SyncMode global_sync_mode() {
    static SyncMode mode = [] {
        auto* v = std::getenv("MCCL_SYNC_MODE");
        if (v) {
            std::string s(v);
            if (s == "coalesced" || s == "fast") return SyncMode::COALESCED;
        }
        return SyncMode::FULL;
    }();
    return mode;
}

thread_local bool tl_sync_done = false;

inline void sync_mps_for_collective(bool overlap) {
    if (global_sync_mode() == SyncMode::COALESCED && tl_sync_done) {
        return;
    }
    if (overlap) {
        mps_event_sync();
    } else {
        mps_stream_sync();
    }
    tl_sync_done = true;
}

inline void reset_sync_state() {
    tl_sync_done = false;
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

    size_t queue_depth = 1024;
    if (auto* v = std::getenv("MCCL_MAX_QUEUE_DEPTH"))
        queue_depth = static_cast<size_t>(std::atoll(v));
    engine_ = std::make_unique<ProgressEngine>(queue_depth);
    engine_->start();

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

ProcessGroupMCCL::~ProcessGroupMCCL() {
    try {
        MCCL_INFO("ProcessGroupMCCL rank=%d shutting down", getRank());
        clear_active_pg_if(this);
        metrics_->log_summary();
        if (health_) health_->stop();
        if (watchdog_) watchdog_->stop();
        if (engine_) engine_->stop();
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

    register_work(seq, work);
    watchdog_->watch(seq, "allreduce");
    metrics_->op_start(seq, "allreduce", nbytes);

    auto sync_t0 = std::chrono::steady_clock::now();
    sync_mps_for_collective(overlap_comm_);
    auto sync_t1 = std::chrono::steady_clock::now();
    double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();
    metrics_->record_phase(seq, sync_ms, 0, 0);

    engine_->submit(
        [this, tensor_copy, seq, ws, nbytes, red_op]() mutable {
            if (nbytes <= transport_->config().small_msg_threshold) {
                allreduce_small(tensor_copy, seq, red_op);
            } else if (ws == 2) {
                allreduce_two_rank(tensor_copy, seq, red_op);
            } else {
                allreduce_ring(tensor_copy, seq, red_op);
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
    sync_mps_for_collective(overlap_comm_);
    auto sync_t1 = std::chrono::steady_clock::now();
    metrics_->record_phase(seq, std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count(), 0, 0);

    // Capture the tensor list + flat buffer for the engine lambda
    auto tensors_copy = tensors;
    auto flat_copy = flat;

    engine_->submit(
        [this, flat_copy, tensors_copy, seq, ws, nbytes, red_op]() mutable {
            if (ws == 2) {
                allreduce_two_rank(flat_copy, seq, red_op);
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
    bool use_cpu = (tensor.scalar_type() == at::kFloat) && tensor_cpu_accessible(tensor);

    if (use_cpu && !compressor_) {
        MPSBufferView view = extract_mps_buffer(tensor);
        StagingBuffer staged = stage_for_send_nosync(tensor);
        size_t nbytes = view.nbytes;
        int64_t count = tensor.numel();

        PooledBuffer recv_buf(staging_memory_pool(), nbytes);

        auto net_t0 = std::chrono::steady_clock::now();
        MCCL_CHECK(transport_->send_recv_overlap(
            peer, OpType::ALLREDUCE, seq, 0, staged.data, nbytes,
            peer, OpType::ALLREDUCE, seq, 0, recv_buf.data(), nbytes),
            "allreduce_two_rank send_recv_overlap failed");
        auto net_t1 = std::chrono::steady_clock::now();

        float* dst = static_cast<float*>(view.cpu_ptr);
        const float* src = static_cast<const float*>(recv_buf.data());

        auto red_t0 = std::chrono::steady_clock::now();
        if (op == c10d::ReduceOp::AVG) {
            cpu_accumulate_and_scale(dst, src, count, 0.5f);
        } else {
            cpu_reduce_op(dst, src, count, op);
        }
        auto red_t1 = std::chrono::steady_clock::now();

        if (overlap_comm_) signal_mccl_done(next_event_value());

        double net_ms = std::chrono::duration<double, std::milli>(net_t1 - net_t0).count();
        double red_ms = std::chrono::duration<double, std::milli>(red_t1 - red_t0).count();
        metrics_->record_phase(seq, 0, net_ms, red_ms);

        metrics_->record_transport_bytes(nbytes, true);
        metrics_->record_transport_bytes(nbytes, false);
    } else {
        // f16 or compressed path: existing Metal pipeline
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

    // ── Reduce-scatter phase ──
    for (int step = 0; step < ws - 1; step++) {
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
            staged = stage_for_send_nosync(send_chunk);
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
                metal_reduce_op(recv_chunk, incoming, op);
            }
        }

        metrics_->record_transport_bytes(send_bytes, true);
        metrics_->record_transport_bytes(recv_bytes, false);
    }

    // ── Allgather phase ──
    for (int step = 0; step < ws - 1; step++) {
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
            staged = stage_for_send_nosync(send_chunk);
        }

        MCCL_CHECK(transport_->send_recv_overlap(
            right, OpType::ALLREDUCE, seq, step_tid,
            staged.data, send_bytes,
            left, OpType::ALLREDUCE, seq, recv_tid_ag,
            recv_buf_pool.data(), recv_bytes),
            "allreduce_ring allgather step " + std::to_string(step) + " failed");

        if (recv_bytes > 0) {
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

    if (use_cpu) {
        if (op == c10d::ReduceOp::AVG) {
            MPSBufferView view = extract_mps_buffer(tensor);
            cpu_scale_inplace(static_cast<float*>(view.cpu_ptr),
                              total_elems, 1.0f / ws);
        }
        if (overlap_comm_) signal_mccl_done(next_event_value());
    } else {
        if (op == c10d::ReduceOp::AVG) {
            metal_begin_batch("mccl_allreduce_ring_avg");
            metal_scale_inplace(tensor, 1.0 / ws);
        }
        metal_sync();
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

    sync_mps_for_collective(overlap_comm_);

    engine_->submit(
        [this, tensor_copy, root, seq, rank, ws]() mutable {
            if (rank == root) {
                StagingBuffer staged = stage_for_send_nosync(tensor_copy);
                for (int peer = 0; peer < ws; peer++) {
                    if (peer == root) continue;
                    MCCL_CHECK(transport_->send_chunks(peer, OpType::BROADCAST, seq, 0,
                                                       staged.data, staged.nbytes),
                               "broadcast send to rank " + std::to_string(peer) + " failed");
                    metrics_->record_transport_bytes(staged.nbytes, true);
                }
            } else {
                size_t nbytes = tensor_nbytes(tensor_copy);
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
            }
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


// ── barrier ─────────────────────────────────────────────────────────

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::barrier(
    const c10d::BarrierOptions& opts) {

    uint32_t seq = collective_seq_.fetch_add(1);
    auto work = c10::make_intrusive<WorkMCCL>(c10d::OpType::BARRIER, seq);
    auto work_ptr = work;

    register_work(seq, work);
    watchdog_->watch(seq, "barrier");
    metrics_->op_start(seq, "barrier", 0);

    engine_->submit(
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

    sync_mps_for_collective(overlap_comm_);

    engine_->submit(
        [this, input_copy, outputs_copy, seq, rank, ws, nbytes]() mutable {
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

            // For CPU-accessible tensors, recv directly into output tensor memory
            PooledBuffer recv_buf_fallback(staging_memory_pool(), use_cpu ? 0 : nbytes);

            for (int step = 0; step < ws - 1; step++) {
                int send_idx = (rank - step + ws) % ws;
                int recv_idx = (rank - step - 1 + ws) % ws;
                uint32_t step_tid = (static_cast<uint32_t>(step) << 16) | send_idx;
                uint32_t recv_tid = (static_cast<uint32_t>(step) << 16) | recv_idx;

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

                MCCL_CHECK(transport_->send_recv_overlap(
                    right, OpType::ALLGATHER, seq, step_tid,
                    staged.data, nbytes,
                    left, OpType::ALLGATHER, seq, recv_tid,
                    recv_dst, nbytes),
                    "allgather step " + std::to_string(step) + " failed");

                if (!use_cpu) {
                    unstage_from_recv(outputs_copy[recv_idx], recv_dst, nbytes);
                }
                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }
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

    sync_mps_for_collective(overlap_comm_);

    engine_->submit(
        [this, output_copy, inputs_copy, seq, rank, ws, nbytes, rs_op]() mutable {
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

                StagingBuffer staged = use_cpu
                    ? stage_for_send_nosync(chunks[send_idx])
                    : stage_for_send(chunks[send_idx]);

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
                    metal_reduce_op(chunks[recv_idx], incoming, rs_op);
                }

                metrics_->record_transport_bytes(nbytes, true);
                metrics_->record_transport_bytes(nbytes, false);
            }

            int my_chunk = rank;
            if (use_cpu) {
                MPSBufferView src_view = extract_mps_buffer(chunks[my_chunk]);
                MPSBufferView dst_view = extract_mps_buffer(output_copy);
                memcpy(dst_view.cpu_ptr, src_view.cpu_ptr, nbytes);
                if (overlap_comm_) signal_mccl_done(next_event_value());
            } else {
                metal_sync();
                output_copy.copy_(chunks[my_chunk]);
                sync_mps_for_collective(overlap_comm_);
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

    sync_mps_for_collective(overlap_comm_);

    engine_->submit(
        [this, tensor, dstRank, seq, tag, nbytes]() mutable {
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

    engine_->submit(
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
