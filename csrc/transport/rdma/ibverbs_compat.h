/*
 * ibverbs_compat.h — Minimal vendored libibverbs types for MCCL.
 *
 * On macOS 26.2+ with Xcode 26.2+, the real <infiniband/verbs.h> exists
 * in the SDK and we use it directly.  On older systems (macOS 15, 14, …)
 * we define only the struct layouts and enum constants that MCCL needs so
 * the code compiles everywhere.  Actual function symbols are resolved at
 * runtime via dlsym — no link-time dependency on librdma.dylib.
 *
 * Struct layouts are ABI-compatible with Apple's librdma.dylib as shipped
 * in macOS 26.2 (based on the standard libibverbs ABI).
 */

#pragma once

#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#else

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque / forward types ─────────────────────────────────────────── */

struct ibv_device {
    char name[64];
    char dev_name[64];
    uint64_t node_guid;
    /* opaque tail — we never inspect past here */
};

struct ibv_context {
    struct ibv_device* device;
    /* opaque */
};

struct ibv_pd {
    struct ibv_context* context;
    uint32_t handle;
};

struct ibv_comp_channel {
    struct ibv_context* context;
    int fd;
    int refcnt;
};

struct ibv_cq {
    struct ibv_context* context;
    uint32_t handle;
    int cqe;  /* requested CQ depth */
};

/* ── Memory registration ────────────────────────────────────────────── */

enum ibv_access_flags {
    IBV_ACCESS_LOCAL_WRITE  = 1 << 0,
    IBV_ACCESS_REMOTE_WRITE = 1 << 1,
    IBV_ACCESS_REMOTE_READ  = 1 << 2,
};

struct ibv_mr {
    struct ibv_context* context;
    struct ibv_pd*      pd;
    void*               addr;
    size_t              length;
    uint32_t            handle;
    uint32_t            lkey;
    uint32_t            rkey;
};

/* ── Scatter / gather ───────────────────────────────────────────────── */

struct ibv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

/* ── Queue pair types ───────────────────────────────────────────────── */

enum ibv_qp_type {
    IBV_QPT_RC  = 2,
    IBV_QPT_UC  = 3,
    IBV_QPT_UD  = 4,
};

enum ibv_qp_state {
    IBV_QPS_RESET = 0,
    IBV_QPS_INIT  = 1,
    IBV_QPS_RTR   = 2,
    IBV_QPS_RTS   = 3,
    IBV_QPS_SQD   = 4,
    IBV_QPS_SQE   = 5,
    IBV_QPS_ERR   = 6,
};

enum ibv_qp_attr_mask {
    IBV_QP_STATE              = 1 <<  0,
    IBV_QP_CUR_STATE          = 1 <<  1,
    IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2,
    IBV_QP_ACCESS_FLAGS       = 1 <<  3,
    IBV_QP_PKEY_INDEX         = 1 <<  4,
    IBV_QP_PORT               = 1 <<  5,
    IBV_QP_QKEY              = 1 <<  6,
    IBV_QP_AV                = 1 <<  7,
    IBV_QP_PATH_MTU          = 1 <<  8,
    IBV_QP_TIMEOUT           = 1 <<  9,
    IBV_QP_RETRY_CNT         = 1 << 10,
    IBV_QP_RNR_RETRY         = 1 << 11,
    IBV_QP_RQ_PSN            = 1 << 12,
    IBV_QP_MAX_QP_RD_ATOMIC  = 1 << 13,
    IBV_QP_ALT_PATH          = 1 << 14,
    IBV_QP_MIN_RNR_TIMER     = 1 << 15,
    IBV_QP_SQ_PSN            = 1 << 16,
    IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
    IBV_QP_PATH_MIG_STATE    = 1 << 18,
    IBV_QP_CAP               = 1 << 19,
    IBV_QP_DEST_QPN          = 1 << 20,
};

enum ibv_mtu {
    IBV_MTU_256  = 1,
    IBV_MTU_512  = 2,
    IBV_MTU_1024 = 3,
    IBV_MTU_2048 = 4,
    IBV_MTU_4096 = 5,
};

/* ── GID (128-bit) ──────────────────────────────────────────────────── */

union ibv_gid {
    uint8_t  raw[16];
    struct {
        uint64_t subnet_prefix;
        uint64_t interface_id;
    } global;
};

/* ── Address handle / global route ──────────────────────────────────── */

struct ibv_global_route {
    union ibv_gid dgid;
    uint32_t flow_label;
    uint8_t  sgid_index;
    uint8_t  hop_limit;
    uint8_t  traffic_class;
};

struct ibv_ah_attr {
    struct ibv_global_route grh;
    uint16_t dlid;
    uint8_t  sl;
    uint8_t  src_path_bits;
    uint8_t  static_rate;
    uint8_t  is_global;
    uint8_t  port_num;
};

/* ── QP capabilities ────────────────────────────────────────────────── */

struct ibv_qp_cap {
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
    void*             qp_context;
    struct ibv_cq*    send_cq;
    struct ibv_cq*    recv_cq;
    void*             srq;  /* ibv_srq* — unused by MCCL */
    struct ibv_qp_cap cap;
    enum ibv_qp_type  qp_type;
    int               sq_sig_all;
};

struct ibv_qp {
    struct ibv_context*  context;
    void*                qp_context;
    struct ibv_pd*       pd;
    struct ibv_cq*       send_cq;
    struct ibv_cq*       recv_cq;
    uint32_t             qp_num;
    enum ibv_qp_state    state;
    enum ibv_qp_type     qp_type;
};

struct ibv_qp_attr {
    enum ibv_qp_state qp_state;
    enum ibv_qp_state cur_qp_state;
    enum ibv_mtu      path_mtu;
    uint32_t          qp_access_flags;
    uint16_t          pkey_index;
    uint16_t          alt_pkey_index;
    uint8_t           en_sqd_async_notify;
    uint8_t           sq_draining;
    uint8_t           max_rd_atomic;
    uint8_t           max_dest_rd_atomic;
    uint8_t           min_rnr_timer;
    uint8_t           port_num;
    uint8_t           timeout;
    uint8_t           retry_cnt;
    uint8_t           rnr_retry;
    uint8_t           alt_port_num;
    uint8_t           alt_timeout;
    uint32_t          sq_psn;
    uint32_t          rq_psn;
    uint32_t          dest_qp_num;
    struct ibv_qp_cap cap;
    struct ibv_ah_attr ah_attr;
    struct ibv_ah_attr alt_ah_attr;
};

/* ── Work requests ──────────────────────────────────────────────────── */

enum ibv_wr_opcode {
    IBV_WR_RDMA_WRITE          = 0,
    IBV_WR_RDMA_WRITE_WITH_IMM = 1,
    IBV_WR_SEND                = 2,
    IBV_WR_SEND_WITH_IMM       = 3,
    IBV_WR_RDMA_READ           = 4,
};

enum ibv_send_flags {
    IBV_SEND_FENCE    = 1 << 0,
    IBV_SEND_SIGNALED = 1 << 1,
    IBV_SEND_SOLICITED = 1 << 2,
    IBV_SEND_INLINE   = 1 << 3,
};

struct ibv_send_wr {
    uint64_t             wr_id;
    struct ibv_send_wr*  next;
    struct ibv_sge*      sg_list;
    int                  num_sge;
    enum ibv_wr_opcode   opcode;
    unsigned int         send_flags;
    uint32_t             imm_data;
    union {
        struct { uint64_t remote_addr; uint32_t rkey; } rdma;
        struct { uint64_t remote_addr; uint64_t rkey; uint64_t compare_add; uint64_t swap; } atomic;
        struct { void* ah; uint32_t remote_qpn; uint32_t remote_qkey; } ud;
    } wr;
};

struct ibv_recv_wr {
    uint64_t             wr_id;
    struct ibv_recv_wr*  next;
    struct ibv_sge*      sg_list;
    int                  num_sge;
};

/* ── Work completion ────────────────────────────────────────────────── */

enum ibv_wc_status {
    IBV_WC_SUCCESS          = 0,
    IBV_WC_LOC_LEN_ERR      = 1,
    IBV_WC_LOC_QP_OP_ERR    = 2,
    IBV_WC_LOC_PROT_ERR     = 4,
    IBV_WC_WR_FLUSH_ERR     = 5,
    IBV_WC_REM_ACCESS_ERR   = 12,
    IBV_WC_GENERAL_ERR      = 22,
};

enum ibv_wc_opcode {
    IBV_WC_SEND         = 0,
    IBV_WC_RDMA_WRITE   = 1,
    IBV_WC_RDMA_READ    = 2,
    IBV_WC_RECV         = 128,
};

struct ibv_wc {
    uint64_t            wr_id;
    enum ibv_wc_status  status;
    enum ibv_wc_opcode  opcode;
    uint32_t            vendor_err;
    uint32_t            byte_len;
    uint32_t            imm_data;
    uint32_t            qp_num;
    uint32_t            src_qp;
    unsigned int        wc_flags;
    uint16_t            pkey_index;
    uint16_t            slid;
    uint8_t             sl;
    uint8_t             dlid_path_bits;
};

/* ── Port attributes ────────────────────────────────────────────────── */

enum ibv_port_state {
    IBV_PORT_NOP         = 0,
    IBV_PORT_DOWN        = 1,
    IBV_PORT_INIT        = 2,
    IBV_PORT_ARMED       = 3,
    IBV_PORT_ACTIVE      = 4,
    IBV_PORT_ACTIVE_DEFER = 5,
};

struct ibv_port_attr {
    enum ibv_port_state state;
    enum ibv_mtu        max_mtu;
    enum ibv_mtu        active_mtu;
    int                 gid_tbl_len;
    uint32_t            port_cap_flags;
    uint32_t            max_msg_sz;
    uint32_t            bad_pkey_cntr;
    uint32_t            qkey_viol_cntr;
    uint16_t            pkey_tbl_len;
    uint16_t            lid;
    uint16_t            sm_lid;
    uint8_t             lmc;
    uint8_t             max_vl_num;
    uint8_t             sm_sl;
    uint8_t             subnet_timeout;
    uint8_t             init_type_reply;
    uint8_t             active_width;
    uint8_t             active_speed;
    uint8_t             phys_state;
};

#ifdef __cplusplus
}
#endif

#endif /* !__has_include(<infiniband/verbs.h>) */
