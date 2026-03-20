#pragma once

#include "transport/rdma/ibverbs_compat.h"

namespace mccl {

struct IbvFunctions {
    ibv_device** (*get_device_list)(int* num_devices);
    void         (*free_device_list)(ibv_device** list);
    ibv_context* (*open_device)(ibv_device* device);
    int          (*close_device)(ibv_context* context);
    ibv_pd*      (*alloc_pd)(ibv_context* context);
    int          (*dealloc_pd)(ibv_pd* pd);
    ibv_cq*      (*create_cq)(ibv_context* context, int cqe,
                               void* cq_context, ibv_comp_channel* channel,
                               int comp_vector);
    int          (*destroy_cq)(ibv_cq* cq);
    ibv_qp*      (*create_qp)(ibv_pd* pd, ibv_qp_init_attr* qp_init_attr);
    int          (*destroy_qp)(ibv_qp* qp);
    int          (*modify_qp)(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask);
    ibv_mr*      (*reg_mr)(ibv_pd* pd, void* addr, size_t length, int access);
    int          (*dereg_mr)(ibv_mr* mr);
    int          (*post_send)(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr);
    int          (*post_recv)(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr);
    int          (*poll_cq)(ibv_cq* cq, int num_entries, ibv_wc* wc);
    int          (*query_port)(ibv_context* context, uint8_t port_num,
                               ibv_port_attr* port_attr);
    int          (*query_gid)(ibv_context* context, uint8_t port_num,
                              int index, ibv_gid* gid);
};

/// Thread-safe singleton accessor. Returns nullptr if librdma.dylib is
/// absent or failed to load, or if no RDMA devices are present.
const IbvFunctions* ibv();

/// Quick availability check (calls ibv() internally).
bool ibv_is_available();

} // namespace mccl
