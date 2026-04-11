#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <atomic>

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

    /// Send a message header + payload to a peer rank (acquires mutex).
    /// Used by heartbeat/abort paths that may race with the progress engine.
    bool send_msg(int peer_rank, const MessageHeader& header,
                  const void* payload, size_t payload_len);

    /// Receive a message header + payload from a peer rank (acquires mutex).
    bool recv_msg(int peer_rank, MessageHeader& header,
                  void* payload, size_t max_payload);

    bool send_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, const void* data, size_t nbytes) override;

    bool recv_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, void* data, size_t nbytes) override;

    /// poll()-based concurrent send + recv (full-duplex on same socket).
    bool send_recv_overlap(
        int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
        const void* send_data, size_t send_nbytes,
        int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
        void* recv_data, size_t recv_nbytes) override;

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

    int rank_;
    int world_size_;
    TransportConfig config_;
    int listen_fd_ = -1;
    std::atomic<bool> shut_down_{false};
    bool crc_enabled_ = false;

    std::vector<Connection> peers_;

    std::vector<std::unique_ptr<std::mutex>> send_mu_;
    std::vector<std::unique_ptr<std::mutex>> recv_mu_;
};

} // namespace mccl
