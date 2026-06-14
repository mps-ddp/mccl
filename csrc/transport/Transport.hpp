#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <exception>

#include "transport/Protocol.hpp"

namespace mccl {

struct TransportConfig;

/// A posted (asynchronous) receive.
///
/// Created by Transport::post_recv and completed by the transport's
/// per-connection reader thread (TCP demux) or lazily inside wait_recv
/// (RDMA).  The caller owns `data` and must keep it alive until wait_recv
/// returns; the transport keeps the PostedRecv struct itself alive via
/// shared_ptr.
struct PostedRecv {
    // Immutable after post.
    int peer = -1;
    OpType op{};
    uint32_t seq = 0;
    uint32_t tid = 0;
    uint8_t* data = nullptr;
    size_t nbytes = 0;
    int kind = 0;  // 0 = TCP demux sink, 1 = RDMA lazy (executed in wait_recv)

    // Completion state, guarded by mu.
    std::mutex mu;
    std::condition_variable cv;
    size_t received = 0;
    uint32_t next_chunk = 0;     // expected chunk_index of the next message
    bool done = false;
    bool conn_closed = false;    // EOF/closed: wait_recv returns false
    bool reader_busy = false;    // reader mid-copy into `data`; defer wakeup
    std::exception_ptr error;    // protocol/ABORT errors: wait_recv rethrows
};

using RecvTicket = std::shared_ptr<PostedRecv>;

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

    /// Post an asynchronous receive for (op, seq, tid) from `peer`.
    /// Returns immediately; complete with wait_recv.  Multiple posts may be
    /// outstanding to the same peer with different (seq, tid); pipelined
    /// collectives rely on this.  P2P (op == SEND) matches on tid only.
    virtual RecvTicket post_recv(int peer_rank, OpType op, uint32_t seq,
                                 uint32_t tid, void* data, size_t nbytes) = 0;

    /// Block until a posted receive completes.  Returns true on success,
    /// false if the connection closed; rethrows protocol/ABORT errors.
    virtual bool wait_recv(const RecvTicket& ticket) = 0;

    /// Fail every posted receive belonging to collective `seq` (local abort:
    /// e.g. a pipeline's TX side failed and its RX side must not block).
    virtual void cancel_recvs(uint32_t seq, const std::string& reason) {
        (void)seq; (void)reason;
    }

    /// Concurrent send + recv (full duplex).  With demultiplexed receives
    /// this is simply post_recv + blocking send + wait.
    virtual bool send_recv_overlap(
        int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
        const void* send_data, size_t send_nbytes,
        int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
        void* recv_data, size_t recv_nbytes) {
        if (recv_nbytes == 0) {
            return send_nbytes == 0 ||
                   send_chunks(send_peer, send_op, send_seq, send_tid,
                               send_data, send_nbytes);
        }
        RecvTicket t = post_recv(recv_peer, recv_op, recv_seq, recv_tid,
                                 recv_data, recv_nbytes);
        bool sent = (send_nbytes == 0) ||
                    send_chunks(send_peer, send_op, send_seq, send_tid,
                                send_data, send_nbytes);
        bool received = wait_recv(t);
        return sent && received;
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
