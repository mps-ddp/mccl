#include "runtime/ThreadPool.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include <algorithm>

namespace mccl {

ThreadPool::ThreadPool(size_t num_threads, size_t max_queue_depth)
    : num_threads_(num_threads), max_depth_(max_queue_depth) {
    MCCL_CHECK(num_threads > 0, "ThreadPool requires at least 1 thread");
    MCCL_CHECK(max_queue_depth > 0, "ThreadPool requires positive max_queue_depth");
}

ThreadPool::~ThreadPool() {
    if (running_.load()) {
        stop();
    }
}

void ThreadPool::start() {
    if (running_.load()) {
        MCCL_WARN("ThreadPool already running");
        return;
    }

    running_.store(true);
    stop_requested_.store(false);

    threads_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; i++) {
        threads_.emplace_back(&ThreadPool::worker_loop, this);
    }

    MCCL_INFO("ThreadPool started with %zu threads (max_depth=%zu)", num_threads_, max_depth_);
}

void ThreadPool::submit_detached(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [this] {
            return queue_.size() < max_depth_ || stop_requested_.load();
        });

        if (stop_requested_) {
            throw std::runtime_error("ThreadPool shutting down, cannot submit");
        }

        queue_.push_back(std::move(task));
    }
    not_empty_.notify_one();
}

void ThreadPool::stop() {
    if (!running_.load()) {
        return;
    }

    MCCL_DEBUG("ThreadPool stopping...");
    stop_requested_.store(true);

    not_empty_.notify_all();
    not_full_.notify_all();

    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    threads_.clear();
    running_.store(false);
    MCCL_DEBUG("ThreadPool stopped");
}

size_t ThreadPool::queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mu_);
            not_empty_.wait(lock, [this] {
                return !queue_.empty() || stop_requested_.load();
            });

            if (stop_requested_ && queue_.empty()) {
                break;
            }

            if (queue_.empty()) continue;

            task = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();

        try {
            task();
        } catch (const std::exception& e) {
            MCCL_WARN("ThreadPool task threw exception: %s", e.what());
        } catch (...) {
            MCCL_WARN("ThreadPool task threw unknown exception");
        }
    }
}

} // namespace mccl
