#pragma once

#include <string>
#include <cstdint>
#include <chrono>
#include <atomic>

namespace mccl {

class Connection {
public:
    Connection();
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;

    /// Connect to a remote endpoint.
    bool connect(const std::string& host, uint16_t port,
                 std::chrono::milliseconds timeout);

    /// Accept a connection on a listening socket.
    /// `listen_fd` must already be bound and listening.
    bool accept_from(int listen_fd, std::chrono::milliseconds timeout);

    /// Send exactly `len` bytes. Returns true on success.
    bool send_all(const void* data, size_t len);

    /// Send header + payload in a single writev syscall.
    /// Reduces syscall overhead for the common message send pattern.
    bool send_header_payload(const void* header, size_t hdr_len,
                             const void* payload, size_t payload_len);

    /// Receive exactly `len` bytes. Returns true on success.
    bool recv_all(void* data, size_t len);

    /// Check if connection is alive.
    bool is_alive() const;

    /// Half-teardown: ::shutdown(fd, SHUT_RDWR) WITHOUT closing the fd.
    /// Wakes a reader thread blocked in recv() so it can exit; the fd stays
    /// valid (no fd-reuse race) until close() is called after the join.
    void shutdown_socket();

    /// Graceful close.
    void close();

    int fd() const { return fd_; }
    int peer_rank() const { return peer_rank_; }
    void set_peer_rank(int rank) { peer_rank_ = rank; }

private:
    int fd_ = -1;
    int peer_rank_ = -1;
    std::atomic<bool> alive_{false};

    void configure_socket();
};

/// Create a listening socket bound to addr:port with SO_REUSEADDR.
/// Returns the file descriptor, or -1 on failure.
int create_listen_socket(const std::string& addr, uint16_t port, int backlog = 64);

} // namespace mccl
