#include "transport/rdma/RdmaTransport.hpp"
#include "transport/rdma/IbvWrapper.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>

namespace mccl {

// ── Static availability check ───────────────────────────────────────

bool RdmaTransport::is_available() {
    return ibv_is_available();
}

// ── Construction / destruction ──────────────────────────────────────

RdmaTransport::RdmaTransport(int rank, int world_size, const TransportConfig& config)
    : tcp_(std::make_unique<TcpTransport>(rank, world_size, config)) {
    MCCL_INFO("RdmaTransport: rank=%d world_size=%d", rank, world_size);
}

RdmaTransport::~RdmaTransport() {
    shutdown();
}

void RdmaTransport::connect_all(const std::vector<std::string>& endpoints) {
    tcp_->connect_all(endpoints);
    try_init_rdma();

    if (rdma_active_.load(std::memory_order_acquire)) {
        MCCL_INFO("RdmaTransport rank %d: RDMA active on all peers", rank());
    } else {
        MCCL_INFO("RdmaTransport rank %d: falling back to TCP", rank());
    }
}

// ── RDMA initialization via TCP side channel ────────────────────────

void RdmaTransport::try_init_rdma() {
    auto* fns = ibv();
    if (!fns) return;

    int ws = world_size();
    int my_rank = rank();

    int num_devices = 0;
    ibv_device** dev_list = fns->get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        MCCL_INFO("RdmaTransport: no RDMA devices");
        if (dev_list) fns->free_device_list(dev_list);
        return;
    }

    connections_.resize(ws);
    peer_bufs_.resize(ws);
    rdma_mu_.resize(ws);

    for (int i = 0; i < ws; i++) {
        rdma_mu_[i] = std::make_unique<std::mutex>();
    }

    bool all_ok = true;
    for (int peer = 0; peer < ws; peer++) {
        if (peer == my_rank) continue;

        if (!connections_[peer].open(dev_list[0])) {
            MCCL_WARN("RdmaTransport: failed to open device for peer %d", peer);
            all_ok = false;
            break;
        }
        if (!connections_[peer].init()) {
            MCCL_WARN("RdmaTransport: QP INIT failed for peer %d", peer);
            all_ok = false;
            break;
        }

        peer_bufs_[peer].send_buf = SharedBuffer(kRdmaBufSize);
        peer_bufs_[peer].recv_buf = SharedBuffer(kRdmaBufSize);

        if (!peer_bufs_[peer].send_buf.data() || !peer_bufs_[peer].recv_buf.data()) {
            MCCL_WARN("RdmaTransport: buffer allocation failed for peer %d", peer);
            all_ok = false;
            break;
        }

        peer_bufs_[peer].send_mr = peer_bufs_[peer].send_buf.register_with(
            connections_[peer].pd());
        peer_bufs_[peer].recv_mr = peer_bufs_[peer].recv_buf.register_with(
            connections_[peer].pd());

        if (!peer_bufs_[peer].send_mr || !peer_bufs_[peer].recv_mr) {
            MCCL_WARN("RdmaTransport: MR registration failed for peer %d", peer);
            all_ok = false;
            break;
        }
    }

    fns->free_device_list(dev_list);

    if (!all_ok) {
        peer_bufs_.clear();
        connections_.clear();
        return;
    }

    // Exchange QP metadata over TCP side channel.
    // Each rank sends its local_destination to every peer, and receives theirs.
    std::vector<RdmaDestination> local_dests(ws);
    std::vector<RdmaDestination> remote_dests(ws);

    for (int peer = 0; peer < ws; peer++) {
        if (peer == my_rank) continue;
        local_dests[peer] = connections_[peer].local_destination();
    }

    // Pairwise exchange: for each peer, lower rank sends first.
    for (int peer = 0; peer < ws; peer++) {
        if (peer == my_rank) continue;

        uint8_t send_buf[RdmaDestination::WIRE_SIZE];
        uint8_t recv_buf[RdmaDestination::WIRE_SIZE];
        local_dests[peer].serialize(send_buf);

        MessageHeader hdr{};
        hdr.protocol_version = MCCL_PROTOCOL_VERSION;
        hdr.op_type = static_cast<uint8_t>(OpType::HANDSHAKE);
        hdr.payload_bytes = RdmaDestination::WIRE_SIZE;
        MessageHeader reply{};

        if (my_rank < peer) {
            if (!tcp_->send_msg(peer, hdr, send_buf, RdmaDestination::WIRE_SIZE) ||
                !tcp_->recv_msg(peer, reply, recv_buf, RdmaDestination::WIRE_SIZE)) {
                MCCL_WARN("RdmaTransport: QP metadata exchange failed with peer %d", peer);
                peer_bufs_.clear();
                connections_.clear();
                return;
            }
        } else {
            if (!tcp_->recv_msg(peer, reply, recv_buf, RdmaDestination::WIRE_SIZE) ||
                !tcp_->send_msg(peer, hdr, send_buf, RdmaDestination::WIRE_SIZE)) {
                MCCL_WARN("RdmaTransport: QP metadata exchange failed with peer %d", peer);
                peer_bufs_.clear();
                connections_.clear();
                return;
            }
        }

        remote_dests[peer] = RdmaDestination::deserialize(recv_buf);
    }

    // Activate QPs: INIT → RTR → RTS
    for (int peer = 0; peer < ws; peer++) {
        if (peer == my_rank) continue;
        if (!connections_[peer].activate(remote_dests[peer])) {
            MCCL_WARN("RdmaTransport: QP activation failed for peer %d", peer);
            peer_bufs_.clear();
            connections_.clear();
            return;
        }
    }

    rdma_active_.store(true, std::memory_order_release);
    MCCL_INFO("RdmaTransport: all %d QPs activated", ws - 1);
}

// ── CQ polling helper ───────────────────────────────────────────────

bool RdmaTransport::poll_completion(RdmaConnection& conn, uint64_t expected_wr_id) {
    ibv_wc wc[kCqPollBatchSize];
    constexpr int kMaxEmptyPolls = 1000000;
    int empty_polls = 0;

    while (true) {
        int n = conn.poll(kCqPollBatchSize, wc);
        if (n < 0) {
            MCCL_ERROR("RdmaTransport: CQ poll error");
            return false;
        }
        if (n == 0) {
            if (++empty_polls > kMaxEmptyPolls) {
                MCCL_ERROR("RdmaTransport: poll_completion timed out waiting for wr_id=%llu",
                           (unsigned long long)expected_wr_id);
                return false;
            }
            std::this_thread::yield();
            continue;
        }
        empty_polls = 0;
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                MCCL_ERROR("RdmaTransport: WC error status=%d wr_id=%llu",
                           wc[i].status, (unsigned long long)wc[i].wr_id);
                return false;
            }
            if (wc[i].wr_id == expected_wr_id) {
                return true;
            }
            MCCL_WARN("RdmaTransport: discarding unexpected completion wr_id=%llu (expected %llu)",
                       (unsigned long long)wc[i].wr_id, (unsigned long long)expected_wr_id);
        }
    }
}

// ── RDMA data path ──────────────────────────────────────────────────

bool RdmaTransport::rdma_send(int peer_rank, const void* data, size_t nbytes) {
    auto& bufs = peer_bufs_[peer_rank];
    auto& conn = connections_[peer_rank];
    size_t offset = 0;
    uint64_t seq = 0;

    while (offset < nbytes) {
        size_t chunk = std::min(nbytes - offset, kRdmaBufSize);
        std::memcpy(bufs.send_buf.data(),
                    static_cast<const uint8_t*>(data) + offset,
                    chunk);

        uint64_t wr_id = (seq << 1) | 0;  // bit 0 = 0 for send
        ibv_sge sge = bufs.send_buf.to_sge(bufs.send_mr, 0,
                                            static_cast<uint32_t>(chunk));
        if (!conn.post_send(sge, wr_id)) return false;
        if (!poll_completion(conn, wr_id)) return false;

        offset += chunk;
        seq++;
    }
    return true;
}

bool RdmaTransport::rdma_recv(int peer_rank, void* data, size_t nbytes) {
    auto& bufs = peer_bufs_[peer_rank];
    auto& conn = connections_[peer_rank];
    size_t offset = 0;
    uint64_t seq = 0;

    while (offset < nbytes) {
        size_t chunk = std::min(nbytes - offset, kRdmaBufSize);

        uint64_t wr_id = (seq << 1) | 1;  // bit 0 = 1 for recv
        ibv_sge sge = bufs.recv_buf.to_sge(bufs.recv_mr, 0,
                                            static_cast<uint32_t>(chunk));
        if (!conn.post_recv(sge, wr_id)) return false;
        if (!poll_completion(conn, wr_id)) return false;

        std::memcpy(static_cast<uint8_t*>(data) + offset,
                    bufs.recv_buf.data(), chunk);

        offset += chunk;
        seq++;
    }
    return true;
}

// ── Transport interface ─────────────────────────────────────────────

bool RdmaTransport::send_chunks(int peer_rank, OpType op, uint32_t seq,
                                uint32_t tensor_id, const void* data,
                                size_t nbytes) {
    if (rdma_active_.load(std::memory_order_acquire) &&
        peer_rank < static_cast<int>(connections_.size()) &&
        connections_[peer_rank].ok()) {
        std::lock_guard<std::mutex> lock(*rdma_mu_[peer_rank]);
        if (rdma_send(peer_rank, data, nbytes)) return true;
        MCCL_WARN("RdmaTransport: RDMA send failed for peer %d, marking connection down",
                   peer_rank);
        connections_[peer_rank].destroy();
    }
    return tcp_->send_chunks(peer_rank, op, seq, tensor_id, data, nbytes);
}

bool RdmaTransport::recv_chunks(int peer_rank, OpType op, uint32_t seq,
                                uint32_t tensor_id, void* data,
                                size_t nbytes) {
    if (rdma_active_.load(std::memory_order_acquire) &&
        peer_rank < static_cast<int>(connections_.size()) &&
        connections_[peer_rank].ok()) {
        std::lock_guard<std::mutex> lock(*rdma_mu_[peer_rank]);
        if (rdma_recv(peer_rank, data, nbytes)) return true;
        MCCL_WARN("RdmaTransport: RDMA recv failed for peer %d, marking connection down",
                   peer_rank);
        connections_[peer_rank].destroy();
    }
    return tcp_->recv_chunks(peer_rank, op, seq, tensor_id, data, nbytes);
}

bool RdmaTransport::send_recv_overlap(
    int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
    const void* send_data, size_t send_nbytes,
    int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
    void* recv_data, size_t recv_nbytes) {

    if (!rdma_active_.load(std::memory_order_acquire)) {
        return tcp_->send_recv_overlap(
            send_peer, send_op, send_seq, send_tid, send_data, send_nbytes,
            recv_peer, recv_op, recv_seq, recv_tid, recv_data, recv_nbytes);
    }

    bool send_rdma = send_peer < static_cast<int>(connections_.size()) &&
                     connections_[send_peer].ok();
    bool recv_rdma = recv_peer < static_cast<int>(connections_.size()) &&
                     connections_[recv_peer].ok();

    if (!send_rdma || !recv_rdma) {
        return tcp_->send_recv_overlap(
            send_peer, send_op, send_seq, send_tid, send_data, send_nbytes,
            recv_peer, recv_op, recv_seq, recv_tid, recv_data, recv_nbytes);
    }

    // Lock both peer mutexes in consistent order to avoid deadlock.
    int first = std::min(send_peer, recv_peer);
    int second = std::max(send_peer, recv_peer);
    std::lock_guard<std::mutex> lock1(*rdma_mu_[first]);
    std::unique_lock<std::mutex> lock2(*rdma_mu_[second], std::defer_lock);
    if (send_peer != recv_peer) lock2.lock();

    auto& send_bufs = peer_bufs_[send_peer];
    auto& recv_bufs = peer_bufs_[recv_peer];
    auto& send_conn = connections_[send_peer];
    auto& recv_conn = connections_[recv_peer];

    size_t send_off = 0, recv_off = 0;
    uint64_t send_seq_id = 0, recv_seq_id = 0;

    while (send_off < send_nbytes || recv_off < recv_nbytes) {
        bool recv_posted = false;
        size_t recv_chunk = 0;
        uint64_t recv_wr = 0;
        if (recv_off < recv_nbytes) {
            recv_chunk = std::min(recv_nbytes - recv_off, kRdmaBufSize);
            recv_wr = (recv_seq_id << 1) | 1;
            ibv_sge rsge = recv_bufs.recv_buf.to_sge(
                recv_bufs.recv_mr, 0, static_cast<uint32_t>(recv_chunk));
            if (recv_conn.post_recv(rsge, recv_wr)) {
                recv_posted = true;
            }
        }

        bool send_posted = false;
        size_t send_chunk = 0;
        uint64_t send_wr = 0;
        if (send_off < send_nbytes) {
            send_chunk = std::min(send_nbytes - send_off, kRdmaBufSize);
            std::memcpy(send_bufs.send_buf.data(),
                        static_cast<const uint8_t*>(send_data) + send_off,
                        send_chunk);
            send_wr = (send_seq_id << 1) | 0;
            ibv_sge ssge = send_bufs.send_buf.to_sge(
                send_bufs.send_mr, 0, static_cast<uint32_t>(send_chunk));
            if (send_conn.post_send(ssge, send_wr)) {
                send_posted = true;
            }
        }

        if (send_posted) {
            if (!poll_completion(send_conn, send_wr)) {
                MCCL_WARN("RdmaTransport: overlap send poll failed, marking connection down");
                send_conn.destroy();
                return tcp_->send_recv_overlap(
                    send_peer, send_op, send_seq, send_tid, send_data, send_nbytes,
                    recv_peer, recv_op, recv_seq, recv_tid, recv_data, recv_nbytes);
            }
            send_off += send_chunk;
            send_seq_id++;
        }

        if (recv_posted) {
            if (!poll_completion(recv_conn, recv_wr)) {
                MCCL_WARN("RdmaTransport: overlap recv poll failed, marking connection down");
                recv_conn.destroy();
                return tcp_->send_recv_overlap(
                    send_peer, send_op, send_seq, send_tid, send_data, send_nbytes,
                    recv_peer, recv_op, recv_seq, recv_tid, recv_data, recv_nbytes);
            }
            std::memcpy(static_cast<uint8_t*>(recv_data) + recv_off,
                        recv_bufs.recv_buf.data(), recv_chunk);
            recv_off += recv_chunk;
            recv_seq_id++;
        }

        if (!send_posted && !recv_posted) {
            // Both post operations failed without a prior destroy() — the QP
            // returned an ibv error (e.g. work queue full).  We cannot continue
            // the RDMA transfer; fall back to TCP to complete the operation.
            MCCL_WARN("RdmaTransport: both post_send and post_recv failed, "
                      "falling back to TCP for remainder of transfer");
            return tcp_->send_recv_overlap(
                send_peer, send_op, send_seq, send_tid, send_data, send_nbytes,
                recv_peer, recv_op, recv_seq, recv_tid, recv_data, recv_nbytes);
        }
    }

    return true;
}

void RdmaTransport::send_abort(uint32_t seq, const std::string& reason) {
    tcp_->send_abort(seq, reason);
}

bool RdmaTransport::is_peer_connected(int peer_rank) const {
    return tcp_->is_peer_connected(peer_rank);
}

int RdmaTransport::rank() const { return tcp_->rank(); }
int RdmaTransport::world_size() const { return tcp_->world_size(); }
const TransportConfig& RdmaTransport::config() const { return tcp_->config(); }
std::string RdmaTransport::listen_endpoint() const { return tcp_->listen_endpoint(); }

void RdmaTransport::shutdown() {
    rdma_active_.store(false, std::memory_order_release);
    peer_bufs_.clear();
    connections_.clear();
    if (tcp_) tcp_->shutdown();
}

} // namespace mccl
