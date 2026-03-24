#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdint>

namespace mccl {

/// Lightweight bounded thread pool for parallel network I/O operations.
/// Replaces per-peer single-threaded ProgressEngine to eliminate queue bottleneck.
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 4, size_t max_queue_depth = 4096);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Start the thread pool.
    void start();

    /// Submit a task and return a future. Blocks if queue is full.
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(mu_);
            not_full_.wait(lock, [this] {
                return queue_.size() < max_depth_ || stop_requested_.load();
            });
            
            if (stop_requested_) {
                throw std::runtime_error("ThreadPool shutting down, cannot submit");
            }
            
            queue_.emplace_back([task]() { (*task)(); });
        }
        not_empty_.notify_one();
        
        return result;
    }

    /// Submit a task without returning a future (fire-and-forget).
    void submit_detached(std::function<void()> task);

    /// Stop the thread pool and drain all pending tasks.
    void stop();

    /// True if the thread pool is running.
    bool running() const { return running_.load(); }

    /// Number of tasks currently in the queue.
    size_t queue_depth() const;

private:
    void worker_loop();

    size_t num_threads_;
    size_t max_depth_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    std::deque<std::function<void()>> queue_;
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;

    std::vector<std::thread> threads_;
};

} // namespace mccl
