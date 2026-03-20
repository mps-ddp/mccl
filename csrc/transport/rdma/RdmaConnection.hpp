#pragma once

#include "transport/rdma/ibverbs_compat.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace mccl {

struct IbvFunctions;

/// Metadata exchanged over the TCP side channel so that each pair of ranks
/// can transition their queue pairs from INIT → RTR → RTS.
struct RdmaDestination {
    uint16_t  local_id;
    uint32_t  qp_number;
    uint32_t  packet_seq_number;
    ibv_gid   gid;

    static constexpr size_t WIRE_SIZE = 2 + 4 + 4 + 16;  // 26 bytes

    void serialize(uint8_t* buf) const {
        std::memcpy(buf,      &local_id,          2);
        std::memcpy(buf + 2,  &qp_number,         4);
        std::memcpy(buf + 6,  &packet_seq_number, 4);
        std::memcpy(buf + 10, &gid,              16);
    }

    static RdmaDestination deserialize(const uint8_t* buf) {
        RdmaDestination d{};
        std::memcpy(&d.local_id,          buf,      2);
        std::memcpy(&d.qp_number,         buf + 2,  4);
        std::memcpy(&d.packet_seq_number, buf + 6,  4);
        std::memcpy(&d.gid,              buf + 10, 16);
        return d;
    }
};

/// Per-peer RDMA connection managing a single RC (Reliable Connected) queue pair.
///
/// RC provides ordered, reliable delivery with hardware-level retry on NAK and
/// RNR (Receiver Not Ready), eliminating silent message drops that UC would
/// produce if a recv WR is not posted before the sender fires.
///
/// Lifecycle: construct → open() (RESET) → init() (→INIT) →
/// activate(remote_dest) (→RTR→RTS) → post_send / post_recv / poll.
class RdmaConnection {
public:
    RdmaConnection();
    ~RdmaConnection();

    RdmaConnection(const RdmaConnection&) = delete;
    RdmaConnection& operator=(const RdmaConnection&) = delete;
    RdmaConnection(RdmaConnection&& other) noexcept;
    RdmaConnection& operator=(RdmaConnection&& other) noexcept;

    /// Open a device, allocate PD + CQ, create the QP in RESET state.
    /// Returns false on any failure.
    bool open(ibv_device* device, int cq_depth = 128);

    /// Transition QP: RESET → INIT. Must be called before activate().
    bool init(uint8_t port = 1);

    /// Transition QP: INIT → RTR → RTS using remote peer's destination info.
    bool activate(const RdmaDestination& remote, uint8_t port = 1);

    /// Get this side's destination metadata (for exchange over TCP).
    RdmaDestination local_destination(uint8_t port = 1) const;

    /// Post a send work request (IBV_WR_SEND, signaled).
    bool post_send(ibv_sge& sge, uint64_t wr_id);

    /// Post a receive work request.
    bool post_recv(ibv_sge& sge, uint64_t wr_id);

    /// Poll the CQ for up to max_wc completions. Returns count (≥0),
    /// or -1 on error.
    int poll(int max_wc, ibv_wc* wc_out);

    ibv_pd* pd()  const { return pd_; }
    ibv_qp* qp()  const { return qp_; }
    bool    ok()  const { return qp_ != nullptr; }

    void destroy();

private:
    const IbvFunctions* fns_ = nullptr;
    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
    ibv_qp*      qp_  = nullptr;
};

} // namespace mccl
