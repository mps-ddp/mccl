#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>

#include "transport/Transport.hpp"
#include "transport/Connection.hpp"
#include "transport/Protocol.hpp"

namespace mccl {

struct TransportConfig {
    std::string transport = "auto";  // "auto", "tcp", "rdma"
    std::string listen_addr = "0.0.0.0";
    uint16_t port_base = 29600;
    std::string ifname;              // MCCL_IFNAME — advisory interface hint
    size_t chunk_bytes = 16 * 1024 * 1024;  // 16 MB default (increased from 4MB for Gloo parity); >=16 MB if MCCL_LINK_PROFILE=thunderbolt
    // Algorithm selection for world_size >= 3: at or below threshold uses star (rank-0);
    // above uses plain ring by default; MCCL_RING_ALGO=chunked|ring_chunked|fast for chunked ring.
    size_t small_msg_threshold = 262144;
    std::chrono::milliseconds connect_timeout{30000};
    std::chrono::milliseconds heartbeat_interval{5000};

    static TransportConfig from_env();
};

/// Warn if MCCL listen base port matches MASTER_PORT (TCP store vs MCCL rank-0 collision).
void warn_if_mccl_port_overlaps_master(const TransportConfig& cfg);

class TcpTransport : public Transport {
public:
    TcpTransport(int rank, int world_size, const TransportConfig& config);
    ~TcpTransport() override;

    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;

    void connect_all(const std::vector<std::string>& endpoints) override;

    /// Defer reader-thread startup past connect_all (RdmaTransport needs the
    /// raw socket for its QP metadata exchange before the demux owns reads).
    void set_auto_start_readers(bool v) { auto_start_readers_ = v; }

    /// Spawn one reader thread per peer connection.  After this, ALL reads
    /// from peer sockets happen on reader threads; recv_msg must not be used.
    void start_readers();

    /// Send a message header + payload to a peer rank (acquires mutex).
    /// Used by abort and the RDMA QP side-channel exchange.
    bool send_msg(int peer_rank, const MessageHeader& header,
                  const void* payload, size_t payload_len);

    /// Receive a message header + payload from a peer rank (acquires mutex).
    /// ONLY valid before start_readers() (RDMA QP exchange).
    bool recv_msg(int peer_rank, MessageHeader& header,
                  void* payload, size_t max_payload);

    bool send_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, const void* data, size_t nbytes) override;

    bool recv_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, void* data, size_t nbytes) override;

    RecvTicket post_recv(int peer_rank, OpType op, uint32_t seq,
                         uint32_t tid, void* data, size_t nbytes) override;

    bool wait_recv(const RecvTicket& ticket) override;

    /// Fail every posted receive belonging to collective `seq` (local abort:
    /// e.g. the TX side of a pipelined ring failed and the RX side must not
    /// block forever).  Safe against in-flight reader copies.
    void fail_recvs_for_seq(uint32_t seq, const std::string& reason);

    void cancel_recvs(uint32_t seq, const std::string& reason) override {
        fail_recvs_for_seq(seq, reason);
    }

    void send_abort(uint32_t seq, const std::string& reason) override;

    bool is_peer_connected(int peer_rank) const override;

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    const TransportConfig& config() const override { return config_; }

    std::string listen_endpoint() const override;

    void shutdown() override;

private:
    Connection& conn_for(int peer_rank);
    std::mutex& send_mu_for(int peer_rank);
    std::mutex& recv_mu_for(int peer_rank);

    /// Internal send/recv without mutex — caller must hold the lock.
    bool send_msg_locked(int peer_rank, const MessageHeader& header,
                         const void* payload, size_t payload_len);
    bool recv_msg_locked(int peer_rank, MessageHeader& header,
                         void* payload, size_t max_payload);

    // ── Demultiplexed receive path ────────────────────────────────────
    //
    // One reader thread per peer socket parses framed messages and routes
    // each to the posted receive registered under its key:
    //   collectives: key = (seq << 32) | tid   (seq is globally unique)
    //   p2p (SEND):  key = tid                 (sender seq unknowable)
    // Messages arriving before their receive is posted are parked (bounded).
    struct ParkedMsg {
        MessageHeader hdr;
        std::vector<uint8_t> payload;
    };
    struct PeerRouter {
        std::thread reader;
        std::mutex mu;  // guards everything below
        std::unordered_map<uint64_t, std::deque<RecvTicket>> sinks;
        std::unordered_map<uint64_t, std::deque<RecvTicket>> p2p_sinks;
        std::unordered_map<uint64_t, std::deque<ParkedMsg>> parked;
        std::unordered_map<uint64_t, std::deque<ParkedMsg>> p2p_parked;
        size_t parked_bytes = 0;
        std::exception_ptr failed;   // router-fatal: EOF, ABORT, protocol error
        bool conn_closed = false;
    };

    static uint64_t collective_key(uint32_t seq, uint32_t tid) {
        return (static_cast<uint64_t>(seq) << 32) | tid;
    }

    void reader_loop(int peer);
    /// Deliver one message (header already read) to a sink or park it.
    /// Returns false if the router must stop (fatal error already recorded).
    bool route_message(int peer, const MessageHeader& hdr);
    /// Fail all sinks and future posts on this router.
    void fail_router(int peer, std::exception_ptr err, bool conn_closed);
    /// Complete `received += len` bookkeeping; returns true when sink done.
    static bool sink_account_locked(PostedRecv& s, const MessageHeader& hdr);

    int rank_;
    int world_size_;
    TransportConfig config_;
    int listen_fd_ = -1;
    std::atomic<bool> shut_down_{false};
    std::atomic<bool> readers_started_{false};
    bool auto_start_readers_ = true;
    bool crc_enabled_ = false;
    // Bound on buffered not-yet-posted messages per peer.  Sized for
    // concurrent collectives: a peer that starts bucket N+1 earlier than us
    // may stream up to pipeline_lookahead x ring_chunk unsolicited bytes.
    size_t park_limit_bytes_ = 256ULL << 20;  // MCCL_DEMUX_PARK_BYTES

    std::vector<Connection> peers_;
    std::vector<std::unique_ptr<PeerRouter>> routers_;

    std::vector<std::unique_ptr<std::mutex>> send_mu_;
    std::vector<std::unique_ptr<std::mutex>> recv_mu_;
};

} // namespace mccl
