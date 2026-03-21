#include "transport/TcpTransport.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include "common/Version.hpp"

#include <cstdlib>
#include <cstring>
#include <thread>
#include <algorithm>
#include <sstream>
#include <unistd.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>

namespace mccl {

namespace {

/// Scan network interfaces for a Thunderbolt bridge.
/// Returns the IP address string if found, empty string otherwise.
/// Looks for interfaces named "bridge*" or "en*" with link-local 169.254.x.x
/// addresses, which is the typical Thunderbolt bridge configuration on macOS.
std::string detect_thunderbolt_bridge() {
    struct ifaddrs* iflist = nullptr;
    if (getifaddrs(&iflist) != 0) return "";

    std::string best_addr;
    std::string best_ifname;

    for (struct ifaddrs* ifa = iflist; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;

        std::string name(ifa->ifa_name);
        auto* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        uint32_t ip = ntohl(sin->sin_addr.s_addr);

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sin->sin_addr, ip_str, sizeof(ip_str));

        // Thunderbolt bridge: "bridge*" with link-local 169.254.x.x
        bool is_bridge = (name.find("bridge") == 0);
        bool is_link_local = ((ip >> 16) == 0xA9FE); // 169.254.x.x

        if (is_bridge && is_link_local) {
            best_addr = ip_str;
            best_ifname = name;
            break;
        }

        // Also check Thunderbolt Ethernet adapters (en* with 169.254.x.x)
        if (name.find("en") == 0 && is_link_local && best_addr.empty()) {
            best_addr = ip_str;
            best_ifname = name;
        }
    }

    freeifaddrs(iflist);

    if (!best_addr.empty()) {
        MCCL_INFO("Auto-detected Thunderbolt bridge: %s on %s",
                  best_addr.c_str(), best_ifname.c_str());
    }

    return best_addr;
}

/// Resolve the IPv4 address of a hostname or dotted-quad string.
/// Returns host-order uint32, or 0 on failure.
uint32_t resolve_ipv4(const char* host) {
    struct in_addr addr{};
    if (inet_pton(AF_INET, host, &addr) == 1)
        return ntohl(addr.s_addr);

    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, nullptr, &hints, &res) != 0 || !res)
        return 0;

    auto* sin = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
    uint32_t ip = ntohl(sin->sin_addr.s_addr);
    freeaddrinfo(res);
    return ip;
}

/// Pick the best local IPv4 address to publish as our MCCL endpoint.
///
/// Strategy: prefer the interface whose subnet contains MASTER_ADDR (already
/// known reachable between all nodes). Falls back to the first non-loopback
/// interface if no subnet match is found.
///
/// out_ifname receives the chosen interface name; out_subnet_match is true
/// when the result was chosen because it shares a subnet with MASTER_ADDR.
std::string resolve_best_local_addr(std::string& out_ifname, bool& out_subnet_match) {
    out_subnet_match = false;

    const char* master_env = std::getenv("MASTER_ADDR");
    uint32_t master_ip = master_env ? resolve_ipv4(master_env) : 0;

    struct ifaddrs* iflist = nullptr;
    if (getifaddrs(&iflist) != 0) return "";

    std::string subnet_match_addr;
    std::string subnet_match_ifname;
    std::string fallback_addr;
    std::string fallback_ifname;

    for (struct ifaddrs* ifa = iflist; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;

        auto* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        uint32_t local_ip = ntohl(sin->sin_addr.s_addr);

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sin->sin_addr, ip_str, sizeof(ip_str));
        std::string name(ifa->ifa_name);

        if (fallback_addr.empty()) {
            fallback_addr = ip_str;
            fallback_ifname = name;
        }

        if (master_ip != 0 && ifa->ifa_netmask) {
            auto* mask_sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_netmask);
            uint32_t mask = ntohl(mask_sin->sin_addr.s_addr);

            if ((local_ip & mask) == (master_ip & mask)) {
                bool is_link_local = ((local_ip >> 16) == 0xA9FE);
                if (subnet_match_addr.empty() || !is_link_local) {
                    subnet_match_addr = ip_str;
                    subnet_match_ifname = name;
                }
            }
        }
    }

    freeifaddrs(iflist);

    if (!subnet_match_addr.empty()) {
        out_ifname = subnet_match_ifname;
        out_subnet_match = true;
        return subnet_match_addr;
    }

    out_ifname = fallback_ifname;
    return fallback_addr;
}

} // anonymous namespace

TransportConfig TransportConfig::from_env() {
    TransportConfig cfg;

    if (auto* v = std::getenv("MCCL_TRANSPORT"))     cfg.transport = v;
    if (auto* v = std::getenv("MCCL_LISTEN_ADDR"))   cfg.listen_addr = v;
    if (auto* v = std::getenv("MCCL_PORT_BASE"))     cfg.port_base = static_cast<uint16_t>(std::atoi(v));
    if (auto* v = std::getenv("MCCL_IFNAME"))        cfg.ifname = v;
    if (auto* v = std::getenv("MCCL_CHUNK_BYTES"))   cfg.chunk_bytes = static_cast<size_t>(std::atoll(v));
    if (auto* v = std::getenv("MCCL_SMALL_MSG_THRESHOLD"))
        cfg.small_msg_threshold = static_cast<size_t>(std::atoll(v));
    if (auto* v = std::getenv("MCCL_CONNECT_TIMEOUT_MS"))
        cfg.connect_timeout = std::chrono::milliseconds(std::atoll(v));
    if (auto* v = std::getenv("MCCL_HEARTBEAT_INTERVAL_MS"))
        cfg.heartbeat_interval = std::chrono::milliseconds(std::atoll(v));

    // Auto-detect Thunderbolt bridge if no explicit listen address set
    if (cfg.listen_addr == "0.0.0.0" && cfg.ifname.empty()) {
        std::string tb_addr = detect_thunderbolt_bridge();
        if (!tb_addr.empty()) {
            cfg.listen_addr = tb_addr;
            MCCL_INFO("Using Thunderbolt bridge address: %s", tb_addr.c_str());
        }
    }

    cfg.chunk_bytes = std::max(cfg.chunk_bytes, size_t(4096));
    cfg.small_msg_threshold = std::max(cfg.small_msg_threshold, size_t(256));

    return cfg;
}

void warn_if_mccl_port_overlaps_master(const TransportConfig& cfg) {
    const char* mp = std::getenv("MASTER_PORT");
    if (!mp) return;
    int master_port = std::atoi(mp);
    if (master_port <= 0 || master_port > 65535) return;
    if (static_cast<int>(cfg.port_base) != master_port) return;
    MCCL_WARN(
        "MCCL_PORT_BASE (%u) equals MASTER_PORT (%d): PyTorch's TCP store and MCCL rank 0 "
        "must not share the same port. Set MCCL_PORT_BASE away from MASTER_PORT on all nodes "
        "(e.g. export MCCL_PORT_BASE=$((MASTER_PORT+100))).",
        (unsigned)cfg.port_base, master_port);
}


TcpTransport::TcpTransport(int rank, int world_size, const TransportConfig& config)
    : rank_(rank), world_size_(world_size), config_(config),
      peers_(world_size) {

    MCCL_CHECK(rank >= 0 && rank < world_size, "Invalid rank");
    MCCL_CHECK(world_size >= 2, "world_size must be >= 2");

    if (auto* v = std::getenv("MCCL_TRANSPORT_CRC")) {
        crc_enabled_ = (std::string(v) == "1" || std::string(v) == "true");
    }

    send_mu_.resize(world_size);
    recv_mu_.resize(world_size);
    for (int i = 0; i < world_size; i++) {
        send_mu_[i] = std::make_unique<std::mutex>();
        recv_mu_[i] = std::make_unique<std::mutex>();
    }

    uint16_t my_port = config_.port_base + static_cast<uint16_t>(rank);
    listen_fd_ = create_listen_socket(config_.listen_addr, my_port);
    MCCL_CHECK(listen_fd_ >= 0, "Failed to create listen socket");

    MCCL_INFO("Rank %d: transport initialized, listening on %s:%u (crc=%s)",
              rank_, config_.listen_addr.c_str(), my_port,
              crc_enabled_ ? "on" : "off");
}

TcpTransport::~TcpTransport() {
    shutdown();
}

std::string TcpTransport::listen_endpoint() const {
    uint16_t port = config_.port_base + static_cast<uint16_t>(rank_);
    std::string addr = config_.listen_addr;

    // 0.0.0.0 is valid for binding (accept from any interface) but cannot
    // be published to remote ranks — they'd connect to themselves.
    // Prefer the interface on the same subnet as MASTER_ADDR so multi-node
    // works without manual MCCL_LISTEN_ADDR.
    if (addr == "0.0.0.0") {
        std::string ifname;
        bool subnet_match = false;
        std::string resolved = resolve_best_local_addr(ifname, subnet_match);
        if (!resolved.empty()) {
            addr = resolved;
            MCCL_INFO("Rank %d: resolved listen address 0.0.0.0 → %s (%s, %s)",
                       rank_, addr.c_str(), ifname.c_str(),
                       subnet_match ? "subnet matches MASTER_ADDR" : "fallback — no subnet match for MASTER_ADDR");
        }
    }

    return addr + ":" + std::to_string(port);
}

Connection& TcpTransport::conn_for(int peer_rank) {
    return peers_[peer_rank];
}

std::mutex& TcpTransport::send_mu_for(int peer_rank) {
    return *send_mu_[peer_rank];
}

std::mutex& TcpTransport::recv_mu_for(int peer_rank) {
    return *recv_mu_[peer_rank];
}

void TcpTransport::connect_all(const std::vector<std::string>& endpoints) {
    MCCL_CHECK(static_cast<int>(endpoints.size()) == world_size_,
               "endpoints size mismatch");

    // Run outbound connects and inbound accepts concurrently to avoid
    // deadlock when world_size >= 3. Without concurrency, rank 0 blocks
    // on its outbound handshake with rank 1, while rank 1 blocks on its
    // outbound to rank 2, and rank 2 waits for inbound from rank 0.

    std::exception_ptr outbound_error;

    std::thread outbound_thread([&]() {
        try {
            for (int peer = rank_ + 1; peer < world_size_; peer++) {
                auto colon = endpoints[peer].find(':');
                MCCL_CHECK(colon != std::string::npos,
                           "Invalid endpoint format: " + endpoints[peer]);
                std::string host = endpoints[peer].substr(0, colon);
                uint16_t port = static_cast<uint16_t>(
                    std::atoi(endpoints[peer].substr(colon + 1).c_str()));

                MCCL_INFO("Rank %d: connecting to rank %d at %s:%u",
                          rank_, peer, host.c_str(), port);

                bool connected = false;
                for (int attempt = 0; attempt < 30; attempt++) {
                    if (peers_[peer].connect(host, port, config_.connect_timeout)) {
                        connected = true;
                        break;
                    }
                    MCCL_WARN("Rank %d: connect to rank %d attempt %d failed, retrying...",
                              rank_, peer, attempt + 1);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                MCCL_CHECK(connected, "Failed to connect to rank " + std::to_string(peer));
                peers_[peer].set_peer_rank(peer);

                HandshakePayload hs{};
                hs.protocol_version = MCCL_PROTOCOL_VERSION;
                hs.rank = rank_;
                hs.world_size = world_size_;
                gethostname(hs.hostname, sizeof(hs.hostname));

                uint8_t buf[HandshakePayload::WIRE_SIZE];
                hs.encode(buf);
                MCCL_CHECK(peers_[peer].send_all(buf, sizeof(buf)),
                           "Handshake send failed");

                uint8_t ack_buf[HandshakePayload::WIRE_SIZE];
                MCCL_CHECK(peers_[peer].recv_all(ack_buf, sizeof(ack_buf)),
                           "Handshake ACK recv failed");
                HandshakePayload ack = HandshakePayload::decode(ack_buf);
                MCCL_CHECK(ack.protocol_version == MCCL_PROTOCOL_VERSION,
                           "Handshake ACK protocol version mismatch");
            }
        } catch (...) {
            outbound_error = std::current_exception();
        }
    });

    // Accept inbound connections from all lower ranks concurrently.
    int num_inbound = rank_;
    std::vector<Connection> pending(num_inbound);

    for (int i = 0; i < num_inbound; i++) {
        MCCL_INFO("Rank %d: accepting inbound connection %d/%d", rank_, i + 1, num_inbound);
        MCCL_CHECK(
            pending[i].accept_from(listen_fd_, config_.connect_timeout),
            "Failed to accept inbound connection " + std::to_string(i)
        );
    }

    for (int i = 0; i < num_inbound; i++) {
        uint8_t buf[HandshakePayload::WIRE_SIZE];
        MCCL_CHECK(pending[i].recv_all(buf, sizeof(buf)),
                   "Handshake recv failed on inbound connection");
        HandshakePayload hs = HandshakePayload::decode(buf);
        MCCL_CHECK(hs.protocol_version == MCCL_PROTOCOL_VERSION,
                   "Protocol version mismatch");
        int peer = hs.rank;
        MCCL_CHECK(peer >= 0 && peer < rank_,
                   "Unexpected handshake rank " + std::to_string(peer) +
                   " (expected < " + std::to_string(rank_) + ")");
        MCCL_CHECK(!peers_[peer].is_alive(),
                   "Duplicate connection from rank " + std::to_string(peer));

        peers_[peer] = std::move(pending[i]);
        peers_[peer].set_peer_rank(peer);

        HandshakePayload ack{};
        ack.protocol_version = MCCL_PROTOCOL_VERSION;
        ack.rank = rank_;
        ack.world_size = world_size_;
        gethostname(ack.hostname, sizeof(ack.hostname));

        uint8_t ack_buf[HandshakePayload::WIRE_SIZE];
        ack.encode(ack_buf);
        MCCL_CHECK(peers_[peer].send_all(ack_buf, sizeof(ack_buf)),
                   "Handshake ACK send failed");
    }

    outbound_thread.join();
    if (outbound_error) std::rethrow_exception(outbound_error);

    MCCL_INFO("Rank %d: all %d peers connected (bidirectional handshake complete)",
              rank_, world_size_ - 1);
}

bool TcpTransport::is_peer_connected(int peer_rank) const {
    if (peer_rank < 0 || peer_rank >= world_size_ || peer_rank == rank_)
        return false;
    return peers_[peer_rank].is_alive();
}

// ── Internal lockless send/recv (caller holds mutex) ────────────────

bool TcpTransport::send_msg_locked(int peer_rank, const MessageHeader& header,
                                   const void* payload, size_t payload_len) {
    Connection& conn = conn_for(peer_rank);
    if (!conn.is_alive()) {
        MCCL_ERROR("send_msg_locked: connection to rank %d is dead", peer_rank);
        return false;
    }

    uint8_t hdr_buf[MessageHeader::WIRE_SIZE];
    header.encode(hdr_buf);

    return conn.send_header_payload(hdr_buf, MessageHeader::WIRE_SIZE,
                                    payload, payload_len);
}

bool TcpTransport::recv_msg_locked(int peer_rank, MessageHeader& header,
                                   void* payload, size_t max_payload) {
    Connection& conn = conn_for(peer_rank);
    if (!conn.is_alive()) {
        MCCL_ERROR("recv_msg_locked: connection from rank %d is dead", peer_rank);
        return false;
    }

    uint8_t hdr_buf[MessageHeader::WIRE_SIZE];
    if (!conn.recv_all(hdr_buf, MessageHeader::WIRE_SIZE)) return false;

    header = MessageHeader::decode(hdr_buf);

    if (!header.version_ok()) {
        throw ProtocolError("Received protocol version " +
                            std::to_string(header.protocol_version) +
                            ", expected " + std::to_string(MCCL_PROTOCOL_VERSION));
    }

    if (header.op_type == static_cast<uint8_t>(OpType::ABORT)) {
        throw MCCLError("Received ABORT from rank " + std::to_string(peer_rank));
    }

    if (header.payload_bytes > 0) {
        MCCL_CHECK(header.payload_bytes <= max_payload,
                   "Payload too large: " + std::to_string(header.payload_bytes) +
                   " > " + std::to_string(max_payload));
        if (!conn.recv_all(payload, header.payload_bytes)) return false;

        if (crc_enabled_ && header.checksum != 0) {
            uint32_t crc = crc32_compute(payload, header.payload_bytes);
            if (crc != header.checksum) {
                throw ProtocolError("CRC mismatch: expected " +
                                    std::to_string(header.checksum) +
                                    ", got " + std::to_string(crc));
            }
        }
    }

    return true;
}

// ── Public send_msg/recv_msg (with mutex, for heartbeat/abort) ──────

bool TcpTransport::send_msg(int peer_rank, const MessageHeader& header,
                            const void* payload, size_t payload_len) {
    std::lock_guard<std::mutex> lock(send_mu_for(peer_rank));
    return send_msg_locked(peer_rank, header, payload, payload_len);
}

bool TcpTransport::recv_msg(int peer_rank, MessageHeader& header,
                            void* payload, size_t max_payload) {
    std::lock_guard<std::mutex> lock(recv_mu_for(peer_rank));
    return recv_msg_locked(peer_rank, header, payload, max_payload);
}

// ── Bulk send/recv for collective data path ─────────────────────────

bool TcpTransport::send_chunks(int peer_rank, OpType op, uint32_t seq,
                               uint32_t tensor_id, const void* data, size_t nbytes) {
    std::lock_guard<std::mutex> lock(send_mu_for(peer_rank));

    MCCL_CHECK(nbytes <= static_cast<size_t>(UINT32_MAX),
               "Payload too large for TCP header (" + std::to_string(nbytes) +
               " bytes, max " + std::to_string(UINT32_MAX) + ")");

    if (!crc_enabled_) {
        MessageHeader hdr{};
        hdr.protocol_version = MCCL_PROTOCOL_VERSION;
        hdr.op_type = static_cast<uint8_t>(op);
        hdr.flags = static_cast<uint8_t>(MsgFlags::LAST_CHUNK);
        hdr.seq_num = seq;
        hdr.tensor_id = tensor_id;
        hdr.chunk_index = 0;
        hdr.payload_bytes = static_cast<uint32_t>(nbytes);
        hdr.checksum = 0;

        return send_msg_locked(peer_rank, hdr, data, nbytes);
    }

    // Chunked path with per-chunk CRC (when MCCL_TRANSPORT_CRC=1)
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t offset = 0;
    uint32_t chunk_idx = 0;

    while (offset < nbytes) {
        size_t chunk_len = std::min(config_.chunk_bytes, nbytes - offset);
        bool is_last = (offset + chunk_len >= nbytes);

        MessageHeader hdr{};
        hdr.protocol_version = MCCL_PROTOCOL_VERSION;
        hdr.op_type = static_cast<uint8_t>(op);
        hdr.flags = is_last ? static_cast<uint8_t>(MsgFlags::LAST_CHUNK)
                            : static_cast<uint8_t>(MsgFlags::NONE);
        hdr.seq_num = seq;
        hdr.tensor_id = tensor_id;
        hdr.chunk_index = chunk_idx;
        hdr.payload_bytes = static_cast<uint32_t>(chunk_len);
        hdr.checksum = crc32_compute(p + offset, chunk_len);

        if (!send_msg_locked(peer_rank, hdr, p + offset, chunk_len)) {
            return false;
        }

        offset += chunk_len;
        chunk_idx++;
    }
    return true;
}

bool TcpTransport::recv_chunks(int peer_rank, OpType op, uint32_t seq,
                               uint32_t tensor_id, void* data, size_t nbytes) {
    std::lock_guard<std::mutex> lock(recv_mu_for(peer_rank));

    // Point-to-point operations (SEND/RECV) match on tensor_id (the user tag)
    // rather than on seq_num.  The sender's per-rank sequence counter cannot be
    // predicted by the receiver when workloads are asymmetric between ranks.
    const bool is_p2p = (op == OpType::SEND || op == OpType::RECV);

    uint8_t* p = static_cast<uint8_t*>(data);
    size_t received = 0;

    while (received < nbytes) {
        size_t remaining = nbytes - received;
        size_t max_chunk = crc_enabled_
            ? std::min(config_.chunk_bytes, remaining)
            : remaining;

        MessageHeader hdr{};
        if (!recv_msg_locked(peer_rank, hdr, p + received, max_chunk)) {
            return false;
        }

        // For p2p: sender always writes OpType::SEND; accept either direction.
        if (is_p2p) {
            MCCL_CHECK(hdr.op_type == static_cast<uint8_t>(OpType::SEND),
                       "Expected SEND optype for p2p recv, got " +
                       std::to_string(static_cast<int>(hdr.op_type)));
            MCCL_CHECK(hdr.tensor_id == tensor_id,
                       "Tag mismatch in p2p recv: expected " +
                       std::to_string(tensor_id) + " got " +
                       std::to_string(hdr.tensor_id));
        } else {
            MCCL_CHECK(hdr.op_type == static_cast<uint8_t>(op),
                       "Unexpected op type in recv_chunks");
            MCCL_CHECK(hdr.seq_num == seq, "Sequence number mismatch: expected " +
                       std::to_string(seq) + " got " + std::to_string(hdr.seq_num));
        }

        received += hdr.payload_bytes;

        if (has_flag(static_cast<MsgFlags>(hdr.flags), MsgFlags::LAST_CHUNK)) {
            break;
        }
    }

    MCCL_CHECK(received == nbytes, "Short receive: got " +
               std::to_string(received) + " expected " + std::to_string(nbytes));
    return true;
}

// ── Overlapped send+recv via poll() ─────────────────────────────────

bool TcpTransport::send_recv_overlap(
    int send_peer, OpType send_op, uint32_t send_seq, uint32_t send_tid,
    const void* send_data, size_t send_nbytes,
    int recv_peer, OpType recv_op, uint32_t recv_seq, uint32_t recv_tid,
    void* recv_data, size_t recv_nbytes) {

    MCCL_CHECK(send_nbytes <= static_cast<size_t>(UINT32_MAX),
               "send_recv_overlap: send payload too large");
    MCCL_CHECK(recv_nbytes <= static_cast<size_t>(UINT32_MAX),
               "send_recv_overlap: recv payload too large");

    if (send_nbytes == 0 && recv_nbytes == 0) return true;

    if (send_nbytes == 0) {
        return recv_chunks(recv_peer, recv_op, recv_seq, recv_tid,
                           recv_data, recv_nbytes);
    }
    if (recv_nbytes == 0) {
        return send_chunks(send_peer, send_op, send_seq, send_tid,
                           send_data, send_nbytes);
    }

    // For very large payloads, fall back to threaded blocking send+recv.
    // The poll loop can handle multi-GB transfers on macOS but above 8GB we
    // switch to the proven blocking path with a background send thread.
    constexpr size_t OVERLAP_THRESHOLD = 1ULL << 33; // 8 GB
    if (send_nbytes > OVERLAP_THRESHOLD || recv_nbytes > OVERLAP_THRESHOLD) {
        MCCL_INFO("send_recv_overlap: payload %zu/%zu exceeds 8GB, using threaded blocking fallback",
                  send_nbytes, recv_nbytes);

        std::atomic<bool> send_ok{false};
        std::thread send_thread([&]() {
            send_ok = send_chunks(send_peer, send_op, send_seq, send_tid,
                                  send_data, send_nbytes);
        });

        bool recv_ok = recv_chunks(recv_peer, recv_op, recv_seq, recv_tid,
                                   recv_data, recv_nbytes);
        send_thread.join();

        return send_ok.load() && recv_ok;
    }

    // Lock both directions. When send_peer == recv_peer, these are
    // still different mutexes (send_mu vs recv_mu).
    std::lock_guard<std::mutex> send_lock(send_mu_for(send_peer));
    std::lock_guard<std::mutex> recv_lock(recv_mu_for(recv_peer));

    Connection& send_conn = conn_for(send_peer);
    Connection& recv_conn = conn_for(recv_peer);

    if (!send_conn.is_alive() || !recv_conn.is_alive()) {
        MCCL_ERROR("send_recv_overlap: dead connection (send_peer=%d recv_peer=%d)",
                   send_peer, recv_peer);
        return false;
    }

    // Build send header
    MessageHeader send_hdr{};
    send_hdr.protocol_version = MCCL_PROTOCOL_VERSION;
    send_hdr.op_type = static_cast<uint8_t>(send_op);
    send_hdr.flags = static_cast<uint8_t>(MsgFlags::LAST_CHUNK);
    send_hdr.seq_num = send_seq;
    send_hdr.tensor_id = send_tid;
    send_hdr.chunk_index = 0;
    send_hdr.payload_bytes = static_cast<uint32_t>(send_nbytes);
    send_hdr.checksum = crc_enabled_ ? crc32_compute(send_data, send_nbytes) : 0;

    uint8_t send_hdr_buf[MessageHeader::WIRE_SIZE];
    send_hdr.encode(send_hdr_buf);

    // Send-side state: header then payload
    const uint8_t* send_bufs[2] = {
        send_hdr_buf,
        static_cast<const uint8_t*>(send_data)
    };
    size_t send_lens[2] = { MessageHeader::WIRE_SIZE, send_nbytes };
    int send_phase = 0;
    size_t send_off = 0;

    // Recv-side state: header then payload
    uint8_t recv_hdr_buf[MessageHeader::WIRE_SIZE];
    uint8_t* recv_bufs[2] = {
        recv_hdr_buf,
        static_cast<uint8_t*>(recv_data)
    };
    size_t recv_lens[2] = { MessageHeader::WIRE_SIZE, recv_nbytes };
    int recv_phase = 0;
    size_t recv_off = 0;

    bool send_done = false;
    bool recv_done = false;

    int send_fd = send_conn.fd();
    int recv_fd = recv_conn.fd();

    send_conn.set_nonblocking();
    if (recv_fd != send_fd) {
        recv_conn.set_nonblocking();
    }

    size_t total_sent = 0, total_recvd = 0;
    size_t send_total = MessageHeader::WIRE_SIZE + send_nbytes;
    size_t recv_total = MessageHeader::WIRE_SIZE + recv_nbytes;

    while (!send_done || !recv_done) {
        struct pollfd pfds[2];
        int nfds = 0;
        int send_pfd_idx = -1, recv_pfd_idx = -1;

        if (!send_done) {
            send_pfd_idx = nfds;
            pfds[nfds].fd = send_fd;
            pfds[nfds].events = POLLOUT;
            pfds[nfds].revents = 0;
            nfds++;
        }
        if (!recv_done) {
            if (recv_fd == send_fd && send_pfd_idx >= 0) {
                pfds[send_pfd_idx].events |= POLLIN;
                recv_pfd_idx = send_pfd_idx;
            } else {
                recv_pfd_idx = nfds;
                pfds[nfds].fd = recv_fd;
                pfds[nfds].events = POLLIN;
                pfds[nfds].revents = 0;
                nfds++;
            }
        }

        int ready = ::poll(pfds, nfds, 60000);
        if (ready < 0) {
            if (errno == EINTR) continue;
            MCCL_ERROR("send_recv_overlap: poll() failed: %s", strerror(errno));
            send_conn.set_blocking();
            if (recv_fd != send_fd) recv_conn.set_blocking();
            return false;
        }
        if (ready == 0) {
            MCCL_ERROR("send_recv_overlap: poll() timed out");
            send_conn.set_blocking();
            if (recv_fd != send_fd) recv_conn.set_blocking();
            return false;
        }

        // Check for errors only on sockets whose direction is still active.
        // A completed direction may report POLLHUP (remote closed send half)
        // which is benign -- only flag errors on in-progress transfers.
        if (!send_done && send_pfd_idx >= 0 &&
            (pfds[send_pfd_idx].revents & (POLLERR | POLLNVAL))) {
            MCCL_ERROR("send_recv_overlap: send socket error fd=%d revents=0x%x",
                       send_fd, pfds[send_pfd_idx].revents);
            send_conn.set_blocking();
            if (recv_fd != send_fd) recv_conn.set_blocking();
            return false;
        }
        if (!recv_done && recv_pfd_idx >= 0 &&
            (pfds[recv_pfd_idx].revents & (POLLERR | POLLNVAL))) {
            MCCL_ERROR("send_recv_overlap: recv socket error fd=%d revents=0x%x",
                       recv_fd, pfds[recv_pfd_idx].revents);
            send_conn.set_blocking();
            if (recv_fd != send_fd) recv_conn.set_blocking();
            return false;
        }

        // Drive send
        if (!send_done && send_pfd_idx >= 0 &&
            (pfds[send_pfd_idx].revents & POLLOUT)) {
            while (send_phase < 2) {
                size_t remain = send_lens[send_phase] - send_off;
                ssize_t n = send_conn.try_send(
                    send_bufs[send_phase] + send_off, remain);
                if (n < 0) {
                    MCCL_ERROR("send_recv_overlap: send failed");
                    send_conn.set_blocking();
                    if (recv_fd != send_fd) recv_conn.set_blocking();
                    return false;
                }
                if (n == 0) break; // EAGAIN
                send_off += static_cast<size_t>(n);
                if (send_off >= send_lens[send_phase]) {
                    send_phase++;
                    send_off = 0;
                }
            }
            if (send_phase >= 2) send_done = true;
        }

        // Drive recv
        if (!recv_done && recv_pfd_idx >= 0 &&
            (pfds[recv_pfd_idx].revents & POLLIN)) {
            while (recv_phase < 2) {
                size_t remain = recv_lens[recv_phase] - recv_off;
                ssize_t n = recv_conn.try_recv(
                    recv_bufs[recv_phase] + recv_off, remain);
                if (n < 0) {
                    MCCL_ERROR("send_recv_overlap: recv failed (phase=%d, recvd %zu/%zu bytes total, peer=%d)",
                               recv_phase, recv_off + (recv_phase > 0 ? recv_lens[0] : 0),
                               recv_total, recv_peer);
                    send_conn.set_blocking();
                    if (recv_fd != send_fd) recv_conn.set_blocking();
                    return false;
                }
                if (n == 0) break; // EAGAIN
                recv_off += static_cast<size_t>(n);
                if (recv_off >= recv_lens[recv_phase]) {
                    recv_phase++;
                    recv_off = 0;
                }
            }
            if (recv_phase >= 2) recv_done = true;
        }
    }

    // Restore blocking mode
    send_conn.set_blocking();
    if (recv_fd != send_fd) {
        recv_conn.set_blocking();
    }

    MCCL_DEBUG("send_recv_overlap: sent %zu recv %zu bytes (send_peer=%d recv_peer=%d)",
               send_total, recv_total, send_peer, recv_peer);

    // Validate received header
    MessageHeader recv_hdr = MessageHeader::decode(recv_hdr_buf);

    if (!recv_hdr.version_ok()) {
        throw ProtocolError("send_recv_overlap: protocol version mismatch");
    }
    if (recv_hdr.op_type == static_cast<uint8_t>(OpType::ABORT)) {
        throw MCCLError("Received ABORT from rank " + std::to_string(recv_peer));
    }
    MCCL_CHECK(recv_hdr.op_type == static_cast<uint8_t>(recv_op),
               "send_recv_overlap: unexpected op type");
    MCCL_CHECK(recv_hdr.seq_num == recv_seq,
               "send_recv_overlap: sequence mismatch: expected " +
               std::to_string(recv_seq) + " got " + std::to_string(recv_hdr.seq_num));
    MCCL_CHECK(recv_hdr.payload_bytes == recv_nbytes,
               "send_recv_overlap: payload size mismatch");

    if (crc_enabled_ && recv_hdr.checksum != 0) {
        uint32_t crc = crc32_compute(recv_data, recv_nbytes);
        if (crc != recv_hdr.checksum) {
            throw ProtocolError("send_recv_overlap: CRC mismatch");
        }
    }

    return true;
}

void TcpTransport::send_abort(uint32_t seq, const std::string& reason) {
    MCCL_ERROR("Rank %d: sending ABORT (seq=%u reason=%s)",
               rank_, seq, reason.c_str());

    for (int peer = 0; peer < world_size_; peer++) {
        if (peer == rank_) continue;
        if (!is_peer_connected(peer)) continue;

        MessageHeader hdr{};
        hdr.protocol_version = MCCL_PROTOCOL_VERSION;
        hdr.op_type = static_cast<uint8_t>(OpType::ABORT);
        hdr.flags = static_cast<uint8_t>(MsgFlags::ABORT);
        hdr.seq_num = seq;
        hdr.payload_bytes = 0;
        hdr.checksum = 0;

        // Best-effort — don't throw on failure during abort
        try {
            send_msg(peer, hdr, nullptr, 0);
        } catch (...) {
            MCCL_WARN("Rank %d: failed to send ABORT to rank %d", rank_, peer);
        }
    }
}

void TcpTransport::shutdown() {
    if (shut_down_.exchange(true)) return;

    for (auto& c : peers_) c.close();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    MCCL_INFO("Rank %d: transport shut down", rank_);
}

} // namespace mccl
