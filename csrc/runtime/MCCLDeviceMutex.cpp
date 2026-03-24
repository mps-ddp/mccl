#include "runtime/MCCLDeviceMutex.hpp"

namespace mccl {

std::recursive_mutex& mccl_device_ops_mutex() {
    static std::recursive_mutex m;
    return m;
}

} // namespace mccl
