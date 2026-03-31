#pragma once

#define MCCL_VERSION_MAJOR 0
#define MCCL_VERSION_MINOR 3
#define MCCL_VERSION_PATCH 3

#define MCCL_PROTOCOL_VERSION 3

#define MCCL_VERSION_STRING "0.3.3"

namespace mccl {

struct Version {
    static constexpr int major = MCCL_VERSION_MAJOR;
    static constexpr int minor = MCCL_VERSION_MINOR;
    static constexpr int patch = MCCL_VERSION_PATCH;
    static constexpr int protocol = MCCL_PROTOCOL_VERSION;
};

} // namespace mccl
