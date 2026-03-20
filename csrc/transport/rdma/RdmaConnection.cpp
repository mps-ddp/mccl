#include "transport/rdma/RdmaConnection.hpp"
#include "transport/rdma/IbvWrapper.hpp"
#include "common/Logging.hpp"

#include <cstring>
#include <utility>

namespace mccl {

RdmaConnection::RdmaConnection() = default;

RdmaConnection::~RdmaConnection() {
    destroy();
}

RdmaConnection::RdmaConnection(RdmaConnection&& other) noexcept
    : fns_(other.fns_), ctx_(other.ctx_), pd_(other.pd_),
      cq_(other.cq_), qp_(other.qp_) {
    other.fns_ = nullptr;
    other.ctx_ = nullptr;
    other.pd_  = nullptr;
    other.cq_  = nullptr;
    other.qp_  = nullptr;
}

RdmaConnection& RdmaConnection::operator=(RdmaConnection&& other) noexcept {
    if (this != &other) {
        destroy();
        fns_ = other.fns_;  ctx_ = other.ctx_;
        pd_  = other.pd_;   cq_  = other.cq_;
        qp_  = other.qp_;
        other.fns_ = nullptr; other.ctx_ = nullptr;
        other.pd_  = nullptr; other.cq_  = nullptr;
        other.qp_  = nullptr;
    }
    return *this;
}

void RdmaConnection::destroy() {
    if (!fns_) return;
    if (qp_)  { fns_->destroy_qp(qp_);   qp_  = nullptr; }
    if (cq_)  { fns_->destroy_cq(cq_);   cq_  = nullptr; }
    if (pd_)  { fns_->dealloc_pd(pd_);   pd_  = nullptr; }
    if (ctx_) { fns_->close_device(ctx_); ctx_ = nullptr; }
    fns_ = nullptr;
}

bool RdmaConnection::open(ibv_device* device, int cq_depth) {
    fns_ = ibv();
    if (!fns_ || !device) return false;

    ctx_ = fns_->open_device(device);
    if (!ctx_) {
        MCCL_ERROR("RdmaConnection: ibv_open_device failed");
        return false;
    }

    pd_ = fns_->alloc_pd(ctx_);
    if (!pd_) {
        MCCL_ERROR("RdmaConnection: ibv_alloc_pd failed");
        destroy();
        return false;
    }

    int actual_cq_depth = cq_depth * 2;
    cq_ = fns_->create_cq(ctx_, actual_cq_depth, nullptr, nullptr, 0);
    if (!cq_) {
        MCCL_ERROR("RdmaConnection: ibv_create_cq(%d) failed", actual_cq_depth);
        destroy();
        return false;
    }

    ibv_qp_init_attr init_attr{};
    init_attr.send_cq = cq_;
    init_attr.recv_cq = cq_;
    init_attr.qp_type = IBV_QPT_RC;   // RC for reliable ordered delivery + RNR retry
    init_attr.sq_sig_all = 0;
    init_attr.cap.max_send_wr  = cq_depth;
    init_attr.cap.max_recv_wr  = cq_depth;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 0;

    qp_ = fns_->create_qp(pd_, &init_attr);
    if (!qp_) {
        MCCL_ERROR("RdmaConnection: ibv_create_qp (RC) failed");
        destroy();
        return false;
    }

    MCCL_DEBUG("RdmaConnection: QP %u created (RC, CQ depth %d)", qp_->qp_num, cq_depth);
    return true;
}

bool RdmaConnection::init(uint8_t port) {
    if (!fns_ || !qp_) return false;

    ibv_qp_attr attr{};
    attr.qp_state   = IBV_QPS_INIT;
    attr.port_num   = port;
    attr.pkey_index = 0;
    // RC requires REMOTE_READ for atomic ops; LOCAL_WRITE for recv buffers.
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                           | IBV_ACCESS_REMOTE_READ;

    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (fns_->modify_qp(qp_, &attr, mask) != 0) {
        MCCL_ERROR("RdmaConnection: QP → INIT failed");
        return false;
    }
    return true;
}

bool RdmaConnection::activate(const RdmaDestination& remote, uint8_t port) {
    if (!fns_ || !qp_) return false;

    // INIT → RTR
    {
        ibv_qp_attr attr{};
        attr.qp_state    = IBV_QPS_RTR;
        attr.path_mtu    = IBV_MTU_4096;
        attr.dest_qp_num = remote.qp_number;
        attr.rq_psn      = remote.packet_seq_number;
        // RC requires max_dest_rd_atomic and min_rnr_timer for RTR.
        attr.max_dest_rd_atomic = 1;
        // min_rnr_timer = 12 → ~0.64 ms before RNR NAK retried by remote.
        attr.min_rnr_timer = 12;

        attr.ah_attr.dlid       = remote.local_id;
        attr.ah_attr.sl         = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num   = port;
        attr.ah_attr.is_global  = 1;
        attr.ah_attr.grh.dgid   = remote.gid;
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.hop_limit  = 1;

        int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                   IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        if (fns_->modify_qp(qp_, &attr, mask) != 0) {
            MCCL_ERROR("RdmaConnection: QP → RTR failed");
            return false;
        }
    }

    // RTR → RTS
    {
        ibv_qp_attr attr{};
        attr.qp_state  = IBV_QPS_RTS;
        attr.sq_psn    = 0;
        // retry_cnt: retransmit up to 7 times on timeout/NAK.
        // rnr_retry:  7 = infinite RNR retry (hardware handles recv-not-ready).
        // timeout:   14 ~ 67 ms local ACK timeout (2^(timeout-1) * 4.096 us).
        attr.timeout      = 14;
        attr.retry_cnt    = 7;
        attr.rnr_retry    = 7;
        attr.max_rd_atomic = 1;

        int mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                   IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
        if (fns_->modify_qp(qp_, &attr, mask) != 0) {
            MCCL_ERROR("RdmaConnection: QP → RTS failed");
            return false;
        }
    }

    MCCL_DEBUG("RdmaConnection: QP %u activated (remote QP %u, LID %u)",
               qp_->qp_num, remote.qp_number, remote.local_id);
    return true;
}

RdmaDestination RdmaConnection::local_destination(uint8_t port) const {
    RdmaDestination dest{};
    if (!fns_ || !qp_ || !ctx_) return dest;

    ibv_port_attr port_attr{};
    fns_->query_port(ctx_, port, &port_attr);
    dest.local_id = port_attr.lid;
    dest.qp_number = qp_->qp_num;
    dest.packet_seq_number = 0;

    fns_->query_gid(ctx_, port, 0, &dest.gid);
    return dest;
}

bool RdmaConnection::post_send(ibv_sge& sge, uint64_t wr_id) {
    if (!fns_ || !qp_) return false;

    ibv_send_wr wr{};
    wr.wr_id    = wr_id;
    wr.next     = nullptr;
    wr.sg_list  = &sge;
    wr.num_sge  = 1;
    wr.opcode   = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr* bad_wr = nullptr;
    int rc = fns_->post_send(qp_, &wr, &bad_wr);
    if (rc != 0) {
        MCCL_ERROR("RdmaConnection: ibv_post_send failed: %d", rc);
        return false;
    }
    return true;
}

bool RdmaConnection::post_recv(ibv_sge& sge, uint64_t wr_id) {
    if (!fns_ || !qp_) return false;

    ibv_recv_wr wr{};
    wr.wr_id   = wr_id;
    wr.next    = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* bad_wr = nullptr;
    int rc = fns_->post_recv(qp_, &wr, &bad_wr);
    if (rc != 0) {
        MCCL_ERROR("RdmaConnection: ibv_post_recv failed: %d", rc);
        return false;
    }
    return true;
}

int RdmaConnection::poll(int max_wc, ibv_wc* wc_out) {
    if (!fns_ || !cq_) return -1;
    return fns_->poll_cq(cq_, max_wc, wc_out);
}

} // namespace mccl
