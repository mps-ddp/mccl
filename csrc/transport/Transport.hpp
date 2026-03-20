#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "transport/Protocol.hpp"

namespace mccl {

struct TransportConfig;

/// Abstract transport interface.
///
/// Separates the collective logic from the wire protocol, allowing
/// TCP (v1) and RDMA (v2) transports to coexist behind the same API.
class Transport {
public:
    virtual ~Transport() = default;

    virtual void connect_all(const std::vector<std::string>& endpoints) = 0;

    virtual bool send_chunks(int peer_rank, OpType op, uint32_t seq,
                             uint32_t tensor_id, const void* data, size_t nbytes) = 0;

    virtual bool recv_chunks(int peer_rank, OpType op, uint32_t seq,
                             uint32_t tensor_id, void* data, size_t nbytes) = 0;

    /// Concurrent send + recv. Default implementation is serial fallback;
    /// TcpTransport overrides with poll()-based overlap, RdmaTransport
    /// with native RDMA pipelining.
    virtual bool send_recv_overlap(
        int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
        const void* send_data, size_t send_nbytes,
        int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
        void* recv_data, size_t recv_nbytes) {
        bool ok = send_chunks(send_peer, send_op, send_seq, send_tid,
                              send_data, send_nbytes);
        ok = recv_chunks(recv_peer, recv_op, recv_seq, recv_tid,
                         recv_data, recv_nbytes) && ok;
        return ok;
    }

    virtual void send_abort(uint32_t seq, const std::string& reason) = 0;

    virtual bool is_peer_connected(int peer_rank) const = 0;

    virtual int rank() const = 0;
    virtual int world_size() const = 0;
    virtual const TransportConfig& config() const = 0;
    virtual std::string listen_endpoint() const = 0;

    virtual void shutdown() = 0;
};

} // namespace mccl
