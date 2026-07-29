#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <atomic>
#include <cstdint>

namespace mccl {

// Forward declaration
class Metrics;

/// A single unit of work submitted to the progress engine.
struct EngineOp {
    uint32_t seq_num;
    std::function<void()> execute;
    std::function<void()> on_complete;
    std::function<void(std::exception_ptr)> on_error;
};

/// Bounded progress engine with one or more worker threads.  Tensor
/// collectives call MPS sync on the caller thread, then ``submit`` transport
/// work here (e.g. barrier, allreduce I/O).  With num_threads > 1, ops still
/// START in submission order (FIFO dequeue) but run concurrently — used by
/// the collective executor pool for bucket-level overlap.
class ProgressEngine {
public:
    explicit ProgressEngine(size_t max_queue_depth = 1024, Metrics* metrics = nullptr,
                            int num_threads = 1);
    ~ProgressEngine();

    ProgressEngine(const ProgressEngine&) = delete;
    ProgressEngine& operator=(const ProgressEngine&) = delete;

    /// Start the engine thread.
    void start();

    /// Submit an operation. Blocks if the queue is at capacity.
    /// Returns the sequence number assigned.
    uint32_t submit(std::function<void()> execute,
                    std::function<void()> on_complete,
                    std::function<void(std::exception_ptr)> on_error);

    /// Drain the queue and stop the engine thread.
    void stop();

    /// True if the engine is running.
    bool running() const { return running_.load(); }

    /// Number of ops currently in the queue.
    size_t queue_depth() const;

    /// Monotonically increasing sequence counter.
    uint32_t next_seq() const { return seq_counter_.load(); }

private:
    void worker_loop();

    size_t max_depth_;
    int num_threads_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<uint32_t> seq_counter_{0};

    std::deque<EngineOp> queue_;
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;

    std::vector<std::thread> threads_;
    Metrics* metrics_; // Not owned
};

} // namespace mccl
