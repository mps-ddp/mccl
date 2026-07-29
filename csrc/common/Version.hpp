#pragma once

#define MCCL_VERSION_MAJOR 0
#define MCCL_VERSION_MINOR 5
#define MCCL_VERSION_PATCH 0

// v4: compressed payloads framed as [4-byte size][exact payload]; chunked-
//     ring allgather schedule corrected.
// v5: demultiplexed receive path — ALL payloads are chunk-framed
//     (<= chunk_bytes per message); the v4 single-message overlap framing
//     is gone, and ring traffic from concurrent collectives interleaves.
//     Mixed-version jobs must be rejected at handshake.
#define MCCL_PROTOCOL_VERSION 5

#define MCCL_VERSION_STRING "0.5.0"

namespace mccl {

struct Version {
    static constexpr int major = MCCL_VERSION_MAJOR;
    static constexpr int minor = MCCL_VERSION_MINOR;
    static constexpr int patch = MCCL_VERSION_PATCH;
    static constexpr int protocol = MCCL_PROTOCOL_VERSION;
};

} // namespace mccl
