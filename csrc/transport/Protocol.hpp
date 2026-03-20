#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include "common/Version.hpp"

#if defined(__aarch64__)
#include <arm_acle.h>
#endif

namespace mccl {

enum class OpType : uint8_t {
    ALLREDUCE      = 1,
    BROADCAST      = 2,
    BARRIER        = 3,
    ALLGATHER      = 4,
    REDUCE_SCATTER = 5,
    SEND           = 6,
    RECV           = 7,
    HANDSHAKE      = 10,
    HEARTBEAT      = 11,
    ABORT          = 255,
};

enum class MsgFlags : uint8_t {
    NONE       = 0,
    LAST_CHUNK = 1 << 0,
    ERROR      = 1 << 1,
    ABORT      = 1 << 2,
};

inline MsgFlags operator|(MsgFlags a, MsgFlags b) {
    return static_cast<MsgFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline bool has_flag(MsgFlags flags, MsgFlags test) {
    return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(test)) != 0;
}

struct __attribute__((packed)) MessageHeader {
    uint16_t protocol_version;
    uint8_t  op_type;        // OpType
    uint8_t  flags;          // MsgFlags
    uint32_t seq_num;
    uint32_t tensor_id;
    uint32_t chunk_index;
    uint32_t payload_bytes;
    uint32_t checksum;       // CRC32 of payload

    static constexpr size_t WIRE_SIZE = 24;

    void encode(uint8_t* buf) const {
        memcpy(buf, this, WIRE_SIZE);
    }

    static MessageHeader decode(const uint8_t* buf) {
        MessageHeader h;
        memcpy(&h, buf, WIRE_SIZE);
        return h;
    }

    bool version_ok() const {
        return protocol_version == MCCL_PROTOCOL_VERSION;
    }
};

static_assert(sizeof(MessageHeader) == MessageHeader::WIRE_SIZE,
              "MessageHeader must be exactly WIRE_SIZE bytes");

/// CRC32 (ISO 3309) for payload integrity.
/// Uses ARM CRC32 hardware instructions when available (Apple Silicon),
/// with a table-based fallback.
namespace detail {

inline const uint32_t* crc32_table() {
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
                c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)));
            t[i] = c;
        }
        return t;
    }();
    return table.data();
}

} // namespace detail

inline uint32_t crc32_compute(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);

#if defined(__aarch64__) && __has_builtin(__builtin_arm_crc32d)
    // ARM CRC32 hardware path: ~30 GB/s on Apple Silicon.
    // __crc32d/w/h/b use the ISO 3309 polynomial (not Castagnoli).
    uint32_t crc = 0xFFFFFFFF;

    // 8 bytes per cycle
    while (len >= 8) {
        uint64_t val;
        memcpy(&val, p, 8);
        crc = __crc32d(crc, val);
        p += 8;
        len -= 8;
    }
    if (len >= 4) {
        uint32_t val;
        memcpy(&val, p, 4);
        crc = __crc32w(crc, val);
        p += 4;
        len -= 4;
    }
    if (len >= 2) {
        uint16_t val;
        memcpy(&val, p, 2);
        crc = __crc32h(crc, val);
        p += 2;
        len -= 2;
    }
    if (len >= 1) {
        crc = __crc32b(crc, *p);
    }

    return ~crc;
#else
    // Table-based fallback for non-ARM platforms
    const uint32_t* table = detail::crc32_table();
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    }

    return ~crc;
#endif
}

struct __attribute__((packed)) HandshakePayload {
    uint16_t protocol_version;
    int32_t  rank;
    int32_t  world_size;
    char     hostname[64];

    static constexpr size_t WIRE_SIZE = 74;

    void encode(uint8_t* buf) const {
        memcpy(buf, this, WIRE_SIZE);
    }
    static HandshakePayload decode(const uint8_t* buf) {
        HandshakePayload h;
        memcpy(&h, buf, WIRE_SIZE);
        return h;
    }
};

static_assert(sizeof(HandshakePayload) == HandshakePayload::WIRE_SIZE,
              "HandshakePayload must be exactly WIRE_SIZE bytes (check packing)");

} // namespace mccl
