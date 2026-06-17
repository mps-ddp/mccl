#include "transport/TcpTransport.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include "common/Version.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <algorithm>
#include <sstream>
#include <unistd.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

namespace mccl {

namespace {

/// Scan network interfaces for a Thunderbolt bridge.
/// Returns the IP address string if found, empty string otherwise.
/// Looks for interfaces named "bridge*" or "en*" with link-local 169.254.x.x
/// addresses, which is the typical Thunderbolt bridge configuration on macOS.
/// If ``out_ifname`` is non-null, writes the chosen interface name.
std::string detect_thunderbolt_bridge(std::string* out_ifname = nullptr) {
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
        if (out_ifname) *out_ifname = best_ifname;
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

    // MASTER_ADDR on 169.254.x.x (typical Thunderbolt IP between two Macs):
    // always publish the TB bridge address so a multi-homed machine does not
    // advertise Wi‑Fi/LAN to peers that only route via the bridge.
    if (master_ip != 0 && ((master_ip >> 16) == 0xA9FE)) {
        std::string tb_if;
        std::string tb_addr = detect_thunderbolt_bridge(&tb_if);
        if (!tb_addr.empty()) {
            out_ifname = tb_if;
            out_subnet_match = true;
            MCCL_INFO("Link-local MASTER_ADDR: publishing Thunderbolt endpoint %s on %s",
                      tb_addr.c_str(), tb_if.c_str());
            return tb_addr;
        }
    }

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

    const bool chunk_bytes_explicit = (std::getenv("MCCL_CHUNK_BYTES") != nullptr);

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

    // Production-oriented defaults for direct Thunderbolt IP (high bandwidth,
    // multi-GB messages). Opt-in via MCCL_LINK_PROFILE=thunderbolt.
    if (!chunk_bytes_explicit) {
        const char* prof = std::getenv("MCCL_LINK_PROFILE");
        if (prof && std::string(prof) == "thunderbolt") {
            cfg.chunk_bytes = std::max(cfg.chunk_bytes, size_t(16) * 1024 * 1024);
            MCCL_INFO("MCCL_LINK_PROFILE=thunderbolt: using chunk_bytes=%zu (set "
                      "MCCL_CHUNK_BYTES to override)",
                      cfg.chunk_bytes);
        }
    }

    return cfg;
}

void warn_if_master_addr_unresolvable() {
    const char* master = std::getenv("MASTER_ADDR");
    if (!master || master[0] == '\0') return;
    if (resolve_ipv4(master) != 0) return;  // resolves fine (numeric or DNS/mDNS)

    // PyTorch's TCPStore resolves MASTER_ADDR before MCCL ever runs; it
    // probes IPv6 first ("The IPv6 network addresses of (...) cannot be
    // retrieved (gai error: 8)") and fails hard if IPv4 cannot resolve
    // either.  On macOS a bare computer name often does not resolve —
    // mDNS needs the ".local" suffix and the ComputerName/HostName/
    // LocalHostName values can disagree (scutil --get HostName).
    MCCL_WARN(
        "MASTER_ADDR='%s' does not resolve to an IPv4 address on this host. "
        "PyTorch's store will likely fail with 'gai error: 8 - nodename nor "
        "servname provided'. Use a NUMERIC IP instead: 127.0.0.1 for "
        "single-node, the Thunderbolt bridge IP (169.254.x.x) or LAN IP for "
        "multi-node. Verify with: dscacheutil -q host -a name %s",
        master, master);
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

    // Scale the small-message threshold with world size when not pinned by
    // env: the recursive-doubling tree costs ~log2(N) full-payload rounds
    // per rank vs the ring's 2(N-1) latency-bound steps, so the tree wins
    // for progressively larger payloads as N grows (latency dominates the
    // ring at 2(N-1) hops).  256 KiB at N<=4, scaling to 1.5 MiB at N=24.
    if (!std::getenv("MCCL_SMALL_MSG_THRESHOLD") && world_size > 4) {
        size_t scaled = config_.small_msg_threshold *
                        (static_cast<size_t>(world_size) / 4);
        config_.small_msg_threshold =
            std::min<size_t>(scaled, 2 * 1024 * 1024);
        MCCL_INFO("Rank %d: small_msg_threshold auto-scaled to %zu for "
                  "world_size=%d (set MCCL_SMALL_MSG_THRESHOLD to override)",
                  rank_, config_.small_msg_threshold, world_size);
    }

    if (auto* v = std::getenv("MCCL_TRANSPORT_CRC")) {
        crc_enabled_ = (std::string(v) == "1" || std::string(v) == "true");
    }
    if (auto* v = std::getenv("MCCL_DEMUX_PARK_BYTES")) {
        park_limit_bytes_ = static_cast<size_t>(std::atoll(v));
    } else {
        // Scale demux park with world size and in-flight collectives so a fast
        // rank does not trip park_limit_bytes_ when concurrency>1 at ws>=8.
        int concurrency = 2;
        if (auto* cv = std::getenv("MCCL_COLLECTIVE_CONCURRENCY")) {
            concurrency = static_cast<int>(
                std::min(4L, std::max(1L, std::atol(cv))));
        }
        size_t scaled = (512ULL << 20) *
                        static_cast<size_t>(std::max(1, world_size / 4)) *
                        static_cast<size_t>(concurrency);
        park_limit_bytes_ = std::min<size_t>(scaled, 2ULL << 30);
        MCCL_INFO("Rank %d: demux park_limit auto-scaled to %zu bytes "
                  "(world_size=%d concurrency=%d; set MCCL_DEMUX_PARK_BYTES to override)",
                  rank_, park_limit_bytes_, world_size_, concurrency);
    }

    if (static_cast<uint32_t>(config_.port_base) +
            static_cast<uint32_t>(world_size_) > 65535u) {
        MCCL_WARN("MCCL_PORT_BASE %u + world_size %d exceeds 65535 — "
                  "ensure firewall allows MCCL_PORT_BASE .. +world_size-1",
                  (unsigned)config_.port_base, world_size_);
    }

    send_mu_.resize(world_size);
    recv_mu_.resize(world_size);
    routers_.resize(world_size);
    for (int i = 0; i < world_size; i++) {
        send_mu_[i] = std::make_unique<std::mutex>();
        recv_mu_[i] = std::make_unique<std::mutex>();
        if (i != rank) routers_[i] = std::make_unique<PeerRouter>();
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
                hs.hostname[sizeof(hs.hostname) - 1] = '\0';  // gethostname may truncate without NUL

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
                MCCL_CHECK(ack.world_size == world_size_,
                           "Handshake ACK world_size mismatch: peer reports " +
                           std::to_string(ack.world_size) + ", expected " +
                           std::to_string(world_size_) +
                           " (two jobs sharing MCCL_PORT_BASE?)");
                MCCL_CHECK(ack.rank == peer,
                           "Handshake ACK rank mismatch: expected " +
                           std::to_string(peer) + " got " + std::to_string(ack.rank));
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
        MCCL_CHECK(hs.world_size == world_size_,
                   "Handshake world_size mismatch: peer reports " +
                   std::to_string(hs.world_size) + ", expected " +
                   std::to_string(world_size_) +
                   " (two jobs sharing MCCL_PORT_BASE?)");
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
        ack.hostname[sizeof(ack.hostname) - 1] = '\0';

        uint8_t ack_buf[HandshakePayload::WIRE_SIZE];
        ack.encode(ack_buf);
        MCCL_CHECK(peers_[peer].send_all(ack_buf, sizeof(ack_buf)),
                   "Handshake ACK send failed");
    }

    outbound_thread.join();
    if (outbound_error) std::rethrow_exception(outbound_error);

    MCCL_INFO("Rank %d: all %d peers connected (bidirectional handshake complete)",
              rank_, world_size_ - 1);

    // Hand socket reads over to the demux readers.  RdmaTransport defers
    // this until after its QP metadata exchange (which uses recv_msg).
    if (auto_start_readers_) {
        start_readers();
    }
}

// ── Demultiplexed receive path ───────────────────────────────────────

void TcpTransport::start_readers() {
    if (readers_started_.exchange(true)) return;
    int spawned = 0;
    for (int p = 0; p < world_size_; p++) {
        if (p == rank_ || !routers_[p]) continue;
        if (!peers_[p].is_alive()) continue;
        routers_[p]->reader = std::thread(&TcpTransport::reader_loop, this, p);
        spawned++;
    }
    MCCL_INFO("Rank %d: demux readers started (%d peers, park_limit=%zu bytes)",
              rank_, spawned, park_limit_bytes_);
}

/// Account one delivered message against a sink.  Caller holds s.mu (or the
/// sink is not yet published).  Returns true when the sink is complete.
/// Sets s.error on protocol violations (short message, bad chunk order).
bool TcpTransport::sink_account_locked(PostedRecv& s, const MessageHeader& hdr) {
    if (hdr.chunk_index != s.next_chunk) {
        s.error = std::make_exception_ptr(ProtocolError(
            "chunk order violation: expected chunk " + std::to_string(s.next_chunk) +
            " got " + std::to_string(hdr.chunk_index) +
            " (seq=" + std::to_string(hdr.seq_num) +
            " tid=" + std::to_string(hdr.tensor_id) + ")"));
        s.done = true;
        return true;
    }
    s.next_chunk++;
    s.received += hdr.payload_bytes;

    const bool last = has_flag(static_cast<MsgFlags>(hdr.flags), MsgFlags::LAST_CHUNK);
    if (s.received == s.nbytes) {
        if (!last) {
            // Sender framed more data than we expected for this op.
            s.error = std::make_exception_ptr(ProtocolError(
                "message complete without LAST_CHUNK (seq=" +
                std::to_string(hdr.seq_num) + " tid=" +
                std::to_string(hdr.tensor_id) + ")"));
        }
        s.done = true;
        return true;
    }
    if (last) {
        s.error = std::make_exception_ptr(ProtocolError(
            "short message: got " + std::to_string(s.received) + " of " +
            std::to_string(s.nbytes) + " bytes (seq=" + std::to_string(hdr.seq_num) +
            " tid=" + std::to_string(hdr.tensor_id) + ")"));
        s.done = true;
        return true;
    }
    return false;
}

void TcpTransport::fail_router(int peer, std::exception_ptr err, bool conn_closed) {
    PeerRouter& rt = *routers_[peer];
    std::vector<RecvTicket> victims;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        if (!rt.failed) rt.failed = err;
        rt.conn_closed = rt.conn_closed || conn_closed;
        for (auto& [key, dq] : rt.sinks) {
            for (auto& s : dq) victims.push_back(s);
        }
        for (auto& [key, dq] : rt.p2p_sinks) {
            for (auto& s : dq) victims.push_back(s);
        }
        rt.sinks.clear();
        rt.p2p_sinks.clear();
        rt.parked.clear();
        rt.p2p_parked.clear();
        rt.parked_bytes = 0;
    }
    for (auto& s : victims) {
        std::lock_guard<std::mutex> slock(s->mu);
        if (!s->done) {
            if (conn_closed) {
                s->conn_closed = true;
            } else {
                s->error = err;
            }
            s->done = true;
        }
        s->cv.notify_all();
    }
    if (!victims.empty()) {
        MCCL_ERROR("Rank %d: demux router for peer %d failed %zu pending recv(s)%s",
                   rank_, peer, victims.size(), conn_closed ? " (connection closed)" : "");
    }
}

bool TcpTransport::route_message(int peer, const MessageHeader& hdr) {
    PeerRouter& rt = *routers_[peer];
    Connection& conn = peers_[peer];

    const bool is_p2p = (hdr.op_type == static_cast<uint8_t>(OpType::SEND));
    const uint64_t key = is_p2p ? static_cast<uint64_t>(hdr.tensor_id)
                                : collective_key(hdr.seq_num, hdr.tensor_id);

    // Fast path: a receive is already posted — read payload straight into
    // the caller's buffer (zero copy).
    RecvTicket sink;
    size_t dst_offset = 0;
    bool overflow = false;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto& map = is_p2p ? rt.p2p_sinks : rt.sinks;
        auto it = map.find(key);
        if (it != map.end() && !it->second.empty()) {
            sink = it->second.front();
            std::lock_guard<std::mutex> slock(sink->mu);
            if (sink->received + hdr.payload_bytes > sink->nbytes) {
                overflow = true;  // fail_router below, outside both locks
            } else {
                dst_offset = sink->received;
                sink->reader_busy = true;  // pin caller's buffer during the copy
            }
        }
    }
    if (overflow) {
        fail_router(peer, std::make_exception_ptr(ProtocolError(
            "payload overflows posted recv (seq=" + std::to_string(hdr.seq_num) +
            " tid=" + std::to_string(hdr.tensor_id) + ")")), false);
        return false;
    }

    if (sink) {
        bool ok = true;
        if (hdr.payload_bytes > 0) {
            ok = conn.recv_all(sink->data + dst_offset, hdr.payload_bytes);
        }
        bool crc_ok = true;
        if (ok && crc_enabled_ && hdr.checksum != 0 && hdr.payload_bytes > 0) {
            crc_ok = (crc32_compute(sink->data + dst_offset, hdr.payload_bytes) ==
                      hdr.checksum);
        }

        bool completed = false;
        {
            std::lock_guard<std::mutex> slock(sink->mu);
            sink->reader_busy = false;
            if (ok && crc_ok) {
                completed = sink_account_locked(*sink, hdr);
            }
            sink->cv.notify_all();
        }
        if (!ok) {
            fail_router(peer, std::make_exception_ptr(MCCLError(
                "connection to rank " + std::to_string(peer) +
                " closed mid-message")), true);
            return false;
        }
        if (!crc_ok) {
            fail_router(peer, std::make_exception_ptr(ProtocolError(
                "CRC mismatch from rank " + std::to_string(peer))), false);
            return false;
        }
        if (completed) {
            std::lock_guard<std::mutex> lock(rt.mu);
            auto& map = is_p2p ? rt.p2p_sinks : rt.sinks;
            auto it = map.find(key);
            if (it != map.end() && !it->second.empty() &&
                it->second.front() == sink) {
                it->second.pop_front();
                if (it->second.empty()) map.erase(it);
            }
        }
        return true;
    }

    // Slow path: no receive posted yet — park a copy (bounded).
    ParkedMsg pm;
    pm.hdr = hdr;
    pm.payload.resize(hdr.payload_bytes);
    if (hdr.payload_bytes > 0 &&
        !conn.recv_all(pm.payload.data(), hdr.payload_bytes)) {
        fail_router(peer, std::make_exception_ptr(MCCLError(
            "connection to rank " + std::to_string(peer) +
            " closed mid-message")), true);
        return false;
    }
    if (crc_enabled_ && hdr.checksum != 0 && hdr.payload_bytes > 0 &&
        crc32_compute(pm.payload.data(), hdr.payload_bytes) != hdr.checksum) {
        fail_router(peer, std::make_exception_ptr(ProtocolError(
            "CRC mismatch from rank " + std::to_string(peer))), false);
        return false;
    }

    bool park_overflow = false;
    RecvTicket late_sink;
    bool late_completed = false;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        // Re-check the sink map: a post_recv may have registered between our
        // first lookup and now (it found no parked data then, so it would
        // wait forever if we parked this message).  Deliver directly under
        // rt.mu — safe: this thread is the only reader for the connection,
        // so no concurrent zero-copy fill can be in flight for this sink.
        auto& map = is_p2p ? rt.p2p_sinks : rt.sinks;
        auto sit = map.find(key);
        if (sit != map.end() && !sit->second.empty()) {
            late_sink = sit->second.front();
            std::lock_guard<std::mutex> slock(late_sink->mu);
            if (late_sink->received + hdr.payload_bytes > late_sink->nbytes) {
                park_overflow = true;  // protocol violation; fail below
            } else {
                if (hdr.payload_bytes > 0) {
                    memcpy(late_sink->data + late_sink->received,
                           pm.payload.data(), hdr.payload_bytes);
                }
                late_completed = sink_account_locked(*late_sink, hdr);
                late_sink->cv.notify_all();
            }
            if (late_completed) {
                sit->second.pop_front();
                if (sit->second.empty()) map.erase(sit);
            }
        } else {
            rt.parked_bytes += pm.payload.size();
            if (rt.parked_bytes > park_limit_bytes_) {
                park_overflow = true;
            } else {
                auto& pk = is_p2p ? rt.p2p_parked : rt.parked;
                pk[key].push_back(std::move(pm));
            }
        }
    }
    if (park_overflow) {
        fail_router(peer, std::make_exception_ptr(MCCLError(
            "demux park limit exceeded or payload overflow (seq=" +
            std::to_string(hdr.seq_num) + " tid=" +
            std::to_string(hdr.tensor_id) + "): receiver desynchronized "
            "from sender, or concurrent collectives outpacing this rank — "
            "raise MCCL_DEMUX_PARK_BYTES or lower MCCL_COLLECTIVE_CONCURRENCY")),
            false);
        return false;
    }
    return true;
}

void TcpTransport::reader_loop(int peer) {
    MCCL_DEBUG("Rank %d: reader for peer %d started", rank_, peer);
    std::vector<uint8_t> scratch;

    while (true) {
        uint8_t hdr_buf[MessageHeader::WIRE_SIZE];
        if (!peers_[peer].recv_all(hdr_buf, MessageHeader::WIRE_SIZE)) {
            // EOF or socket shutdown.  Quiet during intentional shutdown.
            if (!shut_down_.load()) {
                MCCL_WARN("Rank %d: connection from rank %d closed", rank_, peer);
            }
            fail_router(peer, std::make_exception_ptr(MCCLError(
                "connection to rank " + std::to_string(peer) + " closed")), true);
            return;
        }

        MessageHeader hdr = MessageHeader::decode(hdr_buf);

        if (!hdr.version_ok()) {
            fail_router(peer, std::make_exception_ptr(ProtocolError(
                "Received protocol version " + std::to_string(hdr.protocol_version) +
                ", expected " + std::to_string(MCCL_PROTOCOL_VERSION))), false);
            return;
        }
        if (hdr.op_type == static_cast<uint8_t>(OpType::ABORT)) {
            fail_router(peer, std::make_exception_ptr(MCCLError(
                "Received ABORT from rank " + std::to_string(peer))), false);
            return;
        }
        if (hdr.op_type == static_cast<uint8_t>(OpType::HEARTBEAT)) {
            if (hdr.payload_bytes > 0) {
                scratch.resize(hdr.payload_bytes);
                if (!peers_[peer].recv_all(scratch.data(), hdr.payload_bytes)) {
                    fail_router(peer, std::make_exception_ptr(MCCLError(
                        "connection to rank " + std::to_string(peer) + " closed")), true);
                    return;
                }
            }
            continue;
        }

        if (!route_message(peer, hdr)) {
            return;  // fatal already recorded by route_message
        }
    }
}

RecvTicket TcpTransport::post_recv(int peer_rank, OpType op, uint32_t seq,
                                   uint32_t tid, void* data, size_t nbytes) {
    auto t = std::make_shared<PostedRecv>();
    t->peer = peer_rank;
    t->op = op;
    t->seq = seq;
    t->tid = tid;
    t->data = static_cast<uint8_t*>(data);
    t->nbytes = nbytes;
    t->kind = 0;

    if (nbytes == 0) {
        // No empty messages on the wire by protocol invariant.
        t->done = true;
        return t;
    }

    MCCL_CHECK(readers_started_.load(),
               "post_recv before demux readers started");
    MCCL_CHECK(peer_rank >= 0 && peer_rank < world_size_ && peer_rank != rank_,
               "post_recv: invalid peer " + std::to_string(peer_rank));

    PeerRouter& rt = *routers_[peer_rank];
    const bool is_p2p = (op == OpType::SEND || op == OpType::RECV);
    const uint64_t key = is_p2p ? static_cast<uint64_t>(tid)
                                : collective_key(seq, tid);

    std::lock_guard<std::mutex> lock(rt.mu);

    if (rt.failed || rt.conn_closed) {
        if (rt.conn_closed) {
            t->conn_closed = true;
        } else {
            t->error = rt.failed;
        }
        t->done = true;
        return t;
    }

    // Drain any messages that arrived before this post (sender raced ahead).
    // The ticket is not yet published, so no lock on t->mu is needed.
    auto& pk = is_p2p ? rt.p2p_parked : rt.parked;
    auto pit = pk.find(key);
    if (pit != pk.end()) {
        auto& dq = pit->second;
        while (!dq.empty() && !t->done) {
            ParkedMsg& pm = dq.front();
            if (t->received + pm.hdr.payload_bytes > t->nbytes) {
                t->error = std::make_exception_ptr(ProtocolError(
                    "parked payload overflows posted recv (seq=" +
                    std::to_string(seq) + " tid=" + std::to_string(tid) + ")"));
                t->done = true;
                break;
            }
            if (pm.hdr.payload_bytes > 0) {
                memcpy(t->data + t->received, pm.payload.data(),
                       pm.hdr.payload_bytes);
            }
            rt.parked_bytes -= pm.payload.size();
            sink_account_locked(*t, pm.hdr);  // sets done when complete
            dq.pop_front();
        }
        if (dq.empty()) pk.erase(pit);
    }

    if (!t->done) {
        auto& map = is_p2p ? rt.p2p_sinks : rt.sinks;
        map[key].push_back(t);
    }
    return t;
}

bool TcpTransport::wait_recv(const RecvTicket& ticket) {
    MCCL_CHECK(ticket != nullptr, "wait_recv: null ticket");
    std::unique_lock<std::mutex> lk(ticket->mu);
    ticket->cv.wait(lk, [&] {
        return (ticket->done || ticket->conn_closed) && !ticket->reader_busy;
    });
    if (ticket->error) std::rethrow_exception(ticket->error);
    if (ticket->conn_closed && ticket->received < ticket->nbytes) return false;
    return true;
}

void TcpTransport::fail_recvs_for_seq(uint32_t seq, const std::string& reason) {
    auto err = std::make_exception_ptr(MCCLError(
        "receive cancelled (seq=" + std::to_string(seq) + "): " + reason));
    for (int p = 0; p < world_size_; p++) {
        if (p == rank_ || !routers_[p]) continue;
        PeerRouter& rt = *routers_[p];
        std::vector<RecvTicket> victims;
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            for (auto it = rt.sinks.begin(); it != rt.sinks.end();) {
                if (static_cast<uint32_t>(it->first >> 32) == seq) {
                    for (auto& s : it->second) victims.push_back(s);
                    it = rt.sinks.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& s : victims) {
            std::lock_guard<std::mutex> slock(s->mu);
            if (!s->done) {
                s->error = err;
                s->done = true;
            }
            s->cv.notify_all();
        }
    }
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

    // No total-size cap: only the per-chunk payload must fit the u32 header
    // field, and chunks are bounded by config_.chunk_bytes below.

    // Always chunk to config_.chunk_bytes. A single writev with hundreds of MB
    // (default when CRC was off) hits ENOBUFS on macOS TCP during DDP broadcast.
    if (nbytes == 0) {
        MessageHeader hdr{};
        hdr.protocol_version = MCCL_PROTOCOL_VERSION;
        hdr.op_type = static_cast<uint8_t>(op);
        hdr.flags = static_cast<uint8_t>(MsgFlags::LAST_CHUNK);
        hdr.seq_num = seq;
        hdr.tensor_id = tensor_id;
        hdr.chunk_index = 0;
        hdr.payload_bytes = 0;
        hdr.checksum = 0;
        return send_msg_locked(peer_rank, hdr, data, 0);
    }

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
        hdr.checksum =
            crc_enabled_ ? crc32_compute(p + offset, chunk_len) : 0;

        if (!send_msg_locked(peer_rank, hdr, p + offset, chunk_len)) {
            return false;
        }

        offset += chunk_len;
        ++chunk_idx;
    }
    return true;
}

bool TcpTransport::recv_chunks(int peer_rank, OpType op, uint32_t seq,
                               uint32_t tensor_id, void* data, size_t nbytes) {
    // Demultiplexed: register a sink for (op, seq, tid) and block until the
    // reader thread fills it.  Routing replaces the old in-line matching;
    // op/seq/tid mismatches now surface as parked-data timeouts or chunk
    // order errors instead of consuming another op's message.
    RecvTicket t = post_recv(peer_rank, op, seq, tensor_id, data, nbytes);
    return wait_recv(t);
}

// send_recv_overlap: the poll()-based duplex loop (and its threaded
// fallback) is gone.  The Transport base-class implementation — post_recv,
// blocking send_chunks, wait_recv — is full duplex by construction: the
// peer's reader thread always drains incoming traffic regardless of what
// its application threads are doing, so a blocking send can never deadlock
// against a pending receive.  All payloads are chunk-framed (<= chunk_bytes
// per message), which also removes the old 4 GB single-message limit.

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

    // Three-step teardown so reader threads never race fd reuse:
    // 1. shutdown(2) every socket — wakes readers blocked in recv()
    //    (fds stay valid, so an in-flight recv targets the right socket);
    // 2. join the readers (they fail their pending sinks and exit);
    // 3. close the fds.
    for (auto& c : peers_) c.shutdown_socket();
    for (auto& rt : routers_) {
        if (rt && rt->reader.joinable()) rt->reader.join();
    }
    for (auto& c : peers_) c.close();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    MCCL_INFO("Rank %d: transport shut down", rank_);
}

} // namespace mccl
