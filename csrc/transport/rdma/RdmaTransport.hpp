#pragma once

#include "transport/Transport.hpp"
#include "transport/TcpTransport.hpp"
#include "transport/rdma/RdmaConnection.hpp"
#include "transport/rdma/SharedBuffer.hpp"

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

namespace mccl {

/// RDMA transport for macOS 26.2+ with Thunderbolt 5.
///
/// Uses Apple's libibverbs API (librdma.dylib) loaded at runtime via dlopen.
/// When RDMA is available and QP setup succeeds, the data path bypasses TCP
/// entirely for sub-10µs latency and ~80 Gbps bandwidth.
///
/// Falls back to TCP transparently when:
///   - librdma.dylib is absent (macOS < 26.2)
///   - No RDMA devices found (no Thunderbolt 5 link)
///   - QP exchange or initialization fails
///   - MCCL_TRANSPORT=tcp is set explicitly
///
/// The TCP side channel (existing TcpTransport connections) is always
/// established first and used for:
///   1. QP metadata exchange (RdmaDestination structs)
///   2. Control messages (abort, heartbeat)
///   3. Full fallback when rdma_active_ is false
class RdmaTransport : public Transport {
public:
    RdmaTransport(int rank, int world_size, const TransportConfig& config);
    ~RdmaTransport() override;

    void connect_all(const std::vector<std::string>& endpoints) override;

    bool send_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, const void* data, size_t nbytes) override;

    bool recv_chunks(int peer_rank, OpType op, uint32_t seq,
                     uint32_t tensor_id, void* data, size_t nbytes) override;

    bool send_recv_overlap(
        int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
        const void* send_data, size_t send_nbytes,
        int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
        void* recv_data, size_t recv_nbytes) override;

    void send_abort(uint32_t seq, const std::string& reason) override;

    bool is_peer_connected(int peer_rank) const override;

    int rank() const override;
    int world_size() const override;
    const TransportConfig& config() const override;
    std::string listen_endpoint() const override;

    void shutdown() override;

    /// Check if RDMA is available on this machine (dlopen + device probe).
    static bool is_available();

private:
    static constexpr size_t kRdmaBufSize    = 512 * 1024;  // 512 KB per buffer
    static constexpr int    kPipelineDepth   = 2;
    static constexpr int    kCqPollBatchSize = 8;

    /// Try to set up RDMA connections using the TCP side channel for QP
    /// metadata exchange. Sets rdma_active_ = true on success.
    void try_init_rdma();

    /// Poll a connection's CQ until a specific wr_id completes.
    bool poll_completion(RdmaConnection& conn, uint64_t expected_wr_id);

    /// RDMA data path implementations.
    bool rdma_send(int peer_rank, const void* data, size_t nbytes);
    bool rdma_recv(int peer_rank, void* data, size_t nbytes);

    std::unique_ptr<TcpTransport> tcp_;
    std::atomic<bool> rdma_active_{false};

    std::vector<RdmaConnection> connections_;

    struct PeerBuffers {
        SharedBuffer send_buf;
        SharedBuffer recv_buf;
        ibv_mr* send_mr = nullptr;
        ibv_mr* recv_mr = nullptr;
    };
    std::vector<PeerBuffers> peer_bufs_;

    std::vector<std::unique_ptr<std::mutex>> rdma_mu_;
};

} // namespace mccl
