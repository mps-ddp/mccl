#include "runtime/ProgressEngine.hpp"
#include "runtime/Metrics.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <utility>

namespace mccl {

ProgressEngine::ProgressEngine(size_t max_queue_depth, Metrics* metrics)
    : max_depth_(max_queue_depth), metrics_(metrics) {
    MCCL_CHECK(max_depth_ > 0, "max_queue_depth must be > 0");
}

ProgressEngine::~ProgressEngine() {
    stop();
}

void ProgressEngine::start() {
    if (running_.load()) return;

    stop_requested_ = false;
    running_ = true;
    thread_ = std::thread(&ProgressEngine::worker_loop, this);

#if defined(__APPLE__)
    pthread_setname_np("mccl_engine");
#endif

    MCCL_INFO("ProgressEngine started (max_depth=%zu)", max_depth_);
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
    not_empty_.notify_one();
    not_full_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }
    running_ = false;

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

            op = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();

        MCCL_TRACE("Executing op seq=%u", op.seq_num);

        // Record when execution actually starts (after queue wait)
        if (metrics_) {
            metrics_->op_execute_start(op.seq_num);
        }

        bool exec_ok = false;
        std::exception_ptr exec_ex;
        try {
            op.execute();
            exec_ok = true;
        } catch (...) {
            exec_ex = std::current_exception();
            MCCL_ERROR("Op seq=%u execute() failed with exception", op.seq_num);
        }

        if (exec_ok) {
            std::exception_ptr complete_ex;
            try {
                if (op.on_complete) op.on_complete();
                MCCL_TRACE("Op seq=%u completed", op.seq_num);
            } catch (...) {
                complete_ex = std::current_exception();
                MCCL_ERROR("Op seq=%u on_complete() threw — routing to on_error", op.seq_num);
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
