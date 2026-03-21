#include "transport/Connection.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <climits>
#include <cstring>
#include <algorithm>

namespace mccl {

Connection::Connection() = default;

Connection::~Connection() {
    close();
}

Connection::Connection(Connection&& other) noexcept
    : fd_(other.fd_), peer_rank_(other.peer_rank_),
      alive_(other.alive_.load()) {
    other.fd_ = -1;
    other.alive_ = false;
}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        peer_rank_ = other.peer_rank_;
        alive_ = other.alive_.load();
        other.fd_ = -1;
        other.alive_ = false;
    }
    return *this;
}

void Connection::configure_socket() {
    if (fd_ < 0) return;

    int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));

    // Default to 16MB socket buffers for high-bandwidth links (Thunderbolt,
    // 10GbE+).  macOS default (~128KB) is too small and causes TCP ramp-up
    // delays on multi-GB transfers.  Override with MCCL_SOCK_BUFSIZE=0 to
    // let the kernel auto-tune, or set a specific size in bytes.
    {
        int bufsize = 16 * 1024 * 1024;
        if (auto* v = std::getenv("MCCL_SOCK_BUFSIZE")) {
            bufsize = std::atoi(v);
        } else {
            const char* prof = std::getenv("MCCL_LINK_PROFILE");
            if (prof && std::strcmp(prof, "thunderbolt") == 0)
                bufsize = 32 * 1024 * 1024;
        }
        if (bufsize > 0) {
            setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
            setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
        }
    }

#if defined(__APPLE__)
    // TCP_NOTSENT_LOWAT: higher value for better throughput on big transfers;
    // configurable for latency-sensitive small-message workloads.
    int lowat = 131072;
    if (auto* v = std::getenv("MCCL_TCP_LOWAT")) {
        lowat = std::atoi(v);
    }
    setsockopt(fd_, IPPROTO_TCP, 0x201, &lowat, sizeof(lowat));
#endif

    setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
}

bool Connection::connect(const std::string& host, uint16_t port,
                         std::chrono::milliseconds timeout) {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        MCCL_ERROR("socket() failed: %s", strerror(errno));
        return false;
    }

    // Non-blocking connect with timeout
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    int ret = ::connect(fd_, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0 && errno != EINPROGRESS) {
        MCCL_ERROR("connect() to %s:%u failed: %s",
                   host.c_str(), port, strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    if (ret < 0) {
        // Wait for connection to complete
        struct pollfd pfd{};
        pfd.fd = fd_;
        pfd.events = POLLOUT;

        int poll_ret = poll(&pfd, 1, static_cast<int>(timeout.count()));
        if (poll_ret <= 0) {
            MCCL_ERROR("connect() to %s:%u timed out", host.c_str(), port);
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd_, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
            MCCL_ERROR("connect() to %s:%u async error: %s",
                       host.c_str(), port, strerror(err));
            ::close(fd_);
            fd_ = -1;
            return false;
        }
    }

    // Restore blocking mode
    fcntl(fd_, F_SETFL, flags);
    configure_socket();
    alive_ = true;

    MCCL_DEBUG("Connected to %s:%u (fd=%d)", host.c_str(), port, fd_);
    return true;
}

bool Connection::accept_from(int listen_fd, std::chrono::milliseconds timeout) {
    struct pollfd pfd{};
    pfd.fd = listen_fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, static_cast<int>(timeout.count()));
    if (ret <= 0) {
        MCCL_WARN("accept timed out after %lld ms", (long long)timeout.count());
        return false;
    }

    struct sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    fd_ = ::accept(listen_fd, (struct sockaddr*)&peer, &peer_len);
    if (fd_ < 0) {
        MCCL_ERROR("accept() failed: %s", strerror(errno));
        return false;
    }

    configure_socket();
    alive_ = true;

    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
    MCCL_DEBUG("Accepted connection from %s:%u (fd=%d)",
               ip, ntohs(peer.sin_port), fd_);
    return true;
}

bool Connection::send_all(const void* data, size_t len) {
    if (!alive_ || fd_ < 0) return false;

    // macOS limits individual send() to ~2GB; cap each call.
    constexpr size_t MAX_SEND = 1ULL << 30; // 1GB per syscall

    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        size_t chunk = std::min(MAX_SEND, len - sent);
        ssize_t n = ::send(fd_, p + sent, chunk, 0);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            MCCL_ERROR("send failed after %zu/%zu bytes: %s",
                       sent, len, strerror(errno));
            alive_ = false;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool Connection::recv_all(void* data, size_t len) {
    if (!alive_ || fd_ < 0) return false;

    constexpr size_t MAX_RECV = 1ULL << 30; // 1GB per syscall

    uint8_t* p = static_cast<uint8_t*>(data);
    size_t recvd = 0;
    while (recvd < len) {
        size_t chunk = std::min(MAX_RECV, len - recvd);
        ssize_t n = ::recv(fd_, p + recvd, chunk, 0);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            if (n == 0) {
                MCCL_WARN("peer closed connection after %zu/%zu bytes",
                          recvd, len);
            } else {
                MCCL_ERROR("recv failed after %zu/%zu bytes: %s",
                           recvd, len, strerror(errno));
            }
            alive_ = false;
            return false;
        }
        recvd += static_cast<size_t>(n);
    }
    return true;
}

bool Connection::send_header_payload(const void* header, size_t hdr_len,
                                     const void* payload, size_t payload_len) {
    if (!alive_ || fd_ < 0) return false;

    if (!payload || payload_len == 0) {
        return send_all(header, hdr_len);
    }

    size_t total = hdr_len + payload_len;

    // macOS writev fails with EINVAL when total iov length exceeds INT32_MAX (~2GB).
    // For large payloads, send header and payload separately.
    if (total > static_cast<size_t>(INT32_MAX)) {
        if (!send_all(header, hdr_len)) return false;
        return send_all(payload, payload_len);
    }

    struct iovec iov[2];
    iov[0].iov_base = const_cast<void*>(header);
    iov[0].iov_len = hdr_len;
    iov[1].iov_base = const_cast<void*>(payload);
    iov[1].iov_len = payload_len;

    size_t sent = 0;
    int iov_idx = 0;

    while (sent < total) {
        ssize_t n = ::writev(fd_, &iov[iov_idx], 2 - iov_idx);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            MCCL_ERROR("writev failed after %zu/%zu bytes: %s",
                       sent, total, strerror(errno));
            alive_ = false;
            return false;
        }
        sent += static_cast<size_t>(n);

        size_t consumed = static_cast<size_t>(n);
        while (iov_idx < 2 && consumed >= iov[iov_idx].iov_len) {
            consumed -= iov[iov_idx].iov_len;
            iov_idx++;
        }
        if (iov_idx < 2 && consumed > 0) {
            iov[iov_idx].iov_base = static_cast<uint8_t*>(iov[iov_idx].iov_base) + consumed;
            iov[iov_idx].iov_len -= consumed;
        }
    }
    return true;
}

bool Connection::send_heartbeat() {
    uint8_t beat = 0xFF;
    return send_all(&beat, 1);
}

void Connection::set_nonblocking() {
    if (fd_ < 0) return;
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
}

void Connection::set_blocking() {
    if (fd_ < 0) return;
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);
}

ssize_t Connection::try_send(const void* data, size_t len) {
    if (!alive_ || fd_ < 0) {
        MCCL_ERROR("try_send: connection dead (alive=%d fd=%d peer=%d)",
                   (int)alive_.load(), fd_, peer_rank_);
        return -1;
    }
    ssize_t n = ::send(fd_, data, len, MSG_DONTWAIT);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        if (errno == EINTR) return 0;
        MCCL_ERROR("try_send: errno=%d (%s) fd=%d peer=%d len=%zu",
                   errno, strerror(errno), fd_, peer_rank_, len);
        alive_ = false;
        return -1;
    }
    return n;
}

ssize_t Connection::try_recv(void* data, size_t len) {
    if (!alive_ || fd_ < 0) {
        MCCL_ERROR("try_recv: connection dead (alive=%d fd=%d peer=%d)",
                   (int)alive_.load(), fd_, peer_rank_);
        return -1;
    }
    ssize_t n = ::recv(fd_, data, len, MSG_DONTWAIT);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        if (errno == EINTR) return 0;
        MCCL_ERROR("try_recv: errno=%d (%s) fd=%d peer=%d len=%zu",
                   errno, strerror(errno), fd_, peer_rank_, len);
        alive_ = false;
        return -1;
    }
    if (n == 0) {
        MCCL_ERROR("try_recv: peer closed connection (fd=%d peer=%d)", fd_, peer_rank_);
        alive_ = false;
        return -1;
    }
    return n;
}

bool Connection::is_alive() const {
    return alive_.load();
}

void Connection::close() {
    if (fd_ >= 0) {
        MCCL_TRACE("Closing connection fd=%d", fd_);
        ::shutdown(fd_, SHUT_RDWR);
        ::close(fd_);
        fd_ = -1;
    }
    alive_ = false;
}


int create_listen_socket(const std::string& addr, uint16_t port, int backlog) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        MCCL_ERROR("socket() failed: %s", strerror(errno));
        return -1;
    }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (addr.empty() || addr == "0.0.0.0") {
        sa.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, addr.c_str(), &sa.sin_addr);
    }

    if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        MCCL_ERROR("bind(%s:%u) failed: %s", addr.c_str(), port, strerror(errno));
        ::close(fd);
        return -1;
    }

    if (listen(fd, backlog) < 0) {
        MCCL_ERROR("listen() failed: %s", strerror(errno));
        ::close(fd);
        return -1;
    }

    MCCL_INFO("Listening on %s:%u (fd=%d)", addr.c_str(), port, fd);
    return fd;
}

} // namespace mccl
