#include "runtime/ProgressEngine.hpp"
#include "runtime/Metrics.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <utility>

namespace mccl {

ProgressEngine::ProgressEngine(size_t max_queue_depth, Metrics* metrics,
                               int num_threads)
    : max_depth_(max_queue_depth), num_threads_(num_threads), metrics_(metrics) {
    MCCL_CHECK(max_depth_ > 0, "max_queue_depth must be > 0");
    MCCL_CHECK(num_threads_ >= 1, "num_threads must be >= 1");
}

ProgressEngine::~ProgressEngine() {
    stop();
}

void ProgressEngine::start() {
    if (running_.load()) return;

    stop_requested_ = false;
    running_ = true;
    threads_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
        threads_.emplace_back([this] {
#if defined(__APPLE__)
            pthread_setname_np("mccl_engine");
#endif
            worker_loop();
        });
    }

    MCCL_INFO("ProgressEngine started (max_depth=%zu threads=%d)",
              max_depth_, num_threads_);
}

uint32_t ProgressEngine::submit(std::function<void()> execute,
                                std::function<void()> on_complete,
                                std::function<void(std::exception_ptr)> on_error) {
    MCCL_CHECK(running_.load(), "ProgressEngine is not running");

    uint32_t seq = seq_counter_.fetch_add(1);

    EngineOp op;
    op.seq_num = seq;
    op.execute = std::move(execute);
    op.on_complete = std::move(on_complete);
    op.on_error = std::move(on_error);

    {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [this] {
            return queue_.size() < max_depth_ || stop_requested_.load();
        });

        if (stop_requested_) {
            throw MCCLError("ProgressEngine shutting down, cannot submit");
        }

        queue_.push_back(std::move(op));
    }
    not_empty_.notify_one();

    MCCL_TRACE("Submitted op seq=%u (queue_depth=%zu)", seq, queue_depth());
    return seq;
}

void ProgressEngine::stop() {
    if (!running_.load()) return;

    MCCL_INFO("ProgressEngine stopping...");

    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_requested_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();

    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    running_ = false;

    // A submit() that passed the running_ check concurrently with shutdown
    // may have enqueued after the worker exited.  Fail those ops so their
    // Work objects complete (with an error) instead of hanging forever.
    std::deque<EngineOp> orphans;
    {
        std::lock_guard<std::mutex> lock(mu_);
        orphans.swap(queue_);
    }
    if (!orphans.empty()) {
        MCCL_WARN("ProgressEngine: failing %zu op(s) queued during shutdown",
                  orphans.size());
        auto err = std::make_exception_ptr(
            MCCLError("ProgressEngine stopped before op executed"));
        for (auto& op : orphans) {
            try {
                if (op.on_error) op.on_error(err);
            } catch (...) {
                MCCL_ERROR("ProgressEngine: on_error threw during shutdown drain");
            }
        }
    }

    MCCL_INFO("ProgressEngine stopped");
}

size_t ProgressEngine::queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

void ProgressEngine::worker_loop() {
    MCCL_DEBUG("ProgressEngine worker thread started");

    while (true) {
        EngineOp op;

        {
            std::unique_lock<std::mutex> lock(mu_);
            not_empty_.wait(lock, [this] {
                return !queue_.empty() || stop_requested_.load();
            });

            if (stop_requested_ && queue_.empty()) {
                break;
            }

            if (queue_.empty()) continue;

            // FIFO dequeue: with multiple workers, ops still START in
            // submission order (collective schedules stay rank-aligned).
            op = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();

        MCCL_TRACE("Executing op seq=%u", op.seq_num);

        // NOTE: execute-start metrics are recorded by the collective's own
        // execute lambda (ProcessGroupMCCL::begin_execute) — the engine's
        // internal seq_counter_ is NOT the collective seq, so stamping
        // metrics here would hit the wrong (or no) inflight entry.

        bool exec_ok = false;
        std::exception_ptr exec_ex;
        try {
            op.execute();
            exec_ok = true;
        } catch (const std::exception& e) {
            exec_ex = std::current_exception();
            MCCL_ERROR("Op seq=%u execute() failed: %s", op.seq_num, e.what());
        } catch (...) {
            exec_ex = std::current_exception();
            MCCL_ERROR("Op seq=%u execute() failed with non-standard exception", op.seq_num);
        }

        if (exec_ok) {
            std::exception_ptr complete_ex;
            try {
                if (op.on_complete) op.on_complete();
                MCCL_TRACE("Op seq=%u completed", op.seq_num);
            } catch (const std::exception& e) {
                complete_ex = std::current_exception();
                MCCL_ERROR("Op seq=%u on_complete() threw: %s — routing to on_error",
                           op.seq_num, e.what());
            } catch (...) {
                complete_ex = std::current_exception();
                MCCL_ERROR("Op seq=%u on_complete() threw non-standard exception — routing to on_error",
                           op.seq_num);
            }
            if (complete_ex) {
                try {
                    if (op.on_error) op.on_error(complete_ex);
                } catch (...) {
                    MCCL_ERROR("Op seq=%u on_error() also threw after on_complete() failure", op.seq_num);
                }
            }
        } else {
            try {
                if (op.on_error) {
                    op.on_error(exec_ex);
                }
            } catch (...) {
                MCCL_ERROR("Op seq=%u on_error() threw (swallowing to keep engine alive)", op.seq_num);
            }
        }
    }

    MCCL_DEBUG("ProgressEngine worker thread exiting");
}

} // namespace mccl
