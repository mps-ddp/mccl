#pragma once

#define MCCL_VERSION_MAJOR 6
#define MCCL_VERSION_MINOR 8
#define MCCL_VERSION_PATCH 0

// v4: compressed payloads framed as [4-byte size][exact payload]; chunked-
//     ring allgather schedule corrected.
// v5: demultiplexed receive path — ALL payloads are chunk-framed
//     (<= chunk_bytes per message); the v4 single-message overlap framing
//     is gone, and ring traffic from concurrent collectives interleaves.
//     Mixed-version jobs must be rejected at handshake.
// v6: small allreduces at world_size >= 3 default to recursive doubling,
//     and star / broadcast / allgather / reduce_scatter acks are one 4-byte
//     word (v5 sent a tensor-shaped ack, tensor-sized for 2-rank broadcast).
#define MCCL_PROTOCOL_VERSION 6

#define MCCL_VERSION_STRING "6.8.0"

namespace mccl {

struct Version {
    static constexpr int major = MCCL_VERSION_MAJOR;
    static constexpr int minor = MCCL_VERSION_MINOR;
    static constexpr int patch = MCCL_VERSION_PATCH;
    static constexpr int protocol = MCCL_PROTOCOL_VERSION;
};

} // namespace mccl
