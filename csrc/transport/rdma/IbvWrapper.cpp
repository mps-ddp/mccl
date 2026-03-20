#include "transport/rdma/IbvWrapper.hpp"
#include "common/Logging.hpp"

#include <dlfcn.h>
#include <mutex>

namespace mccl {

namespace {

template <typename T>
bool load_sym(void* handle, const char* name, T& out) {
    out = reinterpret_cast<T>(dlsym(handle, name));
    if (!out) {
        MCCL_DEBUG("IbvWrapper: dlsym(%s) failed: %s", name, dlerror());
        return false;
    }
    return true;
}

struct IbvState {
    bool attempted = false;
    bool available = false;
    void* lib_handle = nullptr;
    IbvFunctions fns{};
};

IbvState g_state;
std::mutex g_mu;

bool try_load() {
    dlerror();  // clear

    g_state.lib_handle = dlopen("librdma.dylib", RTLD_NOW);
    if (!g_state.lib_handle) {
        MCCL_INFO("IbvWrapper: dlopen(librdma.dylib) failed: %s", dlerror());
        return false;
    }

    auto& f = g_state.fns;
    bool ok = true;
    ok = ok && load_sym(g_state.lib_handle, "ibv_get_device_list",  f.get_device_list);
    ok = ok && load_sym(g_state.lib_handle, "ibv_free_device_list", f.free_device_list);
    ok = ok && load_sym(g_state.lib_handle, "ibv_open_device",      f.open_device);
    ok = ok && load_sym(g_state.lib_handle, "ibv_close_device",     f.close_device);
    ok = ok && load_sym(g_state.lib_handle, "ibv_alloc_pd",         f.alloc_pd);
    ok = ok && load_sym(g_state.lib_handle, "ibv_dealloc_pd",       f.dealloc_pd);
    ok = ok && load_sym(g_state.lib_handle, "ibv_create_cq",        f.create_cq);
    ok = ok && load_sym(g_state.lib_handle, "ibv_destroy_cq",       f.destroy_cq);
    ok = ok && load_sym(g_state.lib_handle, "ibv_create_qp",        f.create_qp);
    ok = ok && load_sym(g_state.lib_handle, "ibv_destroy_qp",       f.destroy_qp);
    ok = ok && load_sym(g_state.lib_handle, "ibv_modify_qp",        f.modify_qp);
    ok = ok && load_sym(g_state.lib_handle, "ibv_reg_mr",           f.reg_mr);
    ok = ok && load_sym(g_state.lib_handle, "ibv_dereg_mr",         f.dereg_mr);
    ok = ok && load_sym(g_state.lib_handle, "ibv_post_send",        f.post_send);
    ok = ok && load_sym(g_state.lib_handle, "ibv_post_recv",        f.post_recv);
    ok = ok && load_sym(g_state.lib_handle, "ibv_poll_cq",          f.poll_cq);
    ok = ok && load_sym(g_state.lib_handle, "ibv_query_port",       f.query_port);
    ok = ok && load_sym(g_state.lib_handle, "ibv_query_gid",        f.query_gid);

    if (!ok) {
        MCCL_WARN("IbvWrapper: some ibv symbols missing — RDMA disabled");
        dlclose(g_state.lib_handle);
        g_state.lib_handle = nullptr;
        return false;
    }

    int num_devices = 0;
    ibv_device** dev_list = f.get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        MCCL_INFO("IbvWrapper: librdma loaded but no RDMA devices found");
        if (dev_list) f.free_device_list(dev_list);
        dlclose(g_state.lib_handle);
        g_state.lib_handle = nullptr;
        return false;
    }
    f.free_device_list(dev_list);

    MCCL_INFO("IbvWrapper: librdma.dylib loaded — %d device(s) available", num_devices);
    return true;
}

} // anonymous namespace

const IbvFunctions* ibv() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_state.attempted) {
        g_state.attempted = true;
        g_state.available = try_load();
    }
    return g_state.available ? &g_state.fns : nullptr;
}

bool ibv_is_available() {
    return ibv() != nullptr;
}

} // namespace mccl
