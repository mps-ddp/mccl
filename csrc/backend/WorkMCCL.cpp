#include "backend/WorkMCCL.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

namespace mccl {

WorkMCCL::WorkMCCL(c10d::OpType opType, uint32_t seq,
                   std::vector<at::Tensor> outputTensors)
    : c10d::Work(-1, opType), seq_(seq),
      outputs_(std::move(outputTensors)) {

    if (!outputs_.empty()) {
        future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
    }
}

WorkMCCL::~WorkMCCL() = default;

bool WorkMCCL::wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (completed_) {
        if (exception_) std::rethrow_exception(exception_);
        return success_;
    }

    if (timeout == kNoTimeout) {
        cv_.wait(lock, [this] { return completed_; });
    } else {
        bool done = cv_.wait_for(lock, timeout, [this] { return completed_; });
        if (!done) {
            throw TimeoutError(
                "WorkMCCL::wait timed out after " +
                std::to_string(timeout.count()) + "ms on seq=" +
                std::to_string(seq_));
        }
    }

    if (exception_) std::rethrow_exception(exception_);
    return success_;
}

bool WorkMCCL::isCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}

bool WorkMCCL::isSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return success_;
}

std::exception_ptr WorkMCCL::exception() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exception_;
}

std::vector<at::Tensor> WorkMCCL::result() {
    std::lock_guard<std::mutex> lock(mutex_);
    return outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkMCCL::getFuture() {
    return future_;
}

void WorkMCCL::finishWorkMCCLFuture() {
    if (future_) {
        if (exception_) {
            future_->setError(exception_);
        } else {
            c10::IValue val(outputs_);
            future_->markCompleted(val);
        }
    }
}

void WorkMCCL::markComplete() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completed_) return;
        completed_ = true;
        success_ = true;
        finishWorkMCCLFuture();
    }
    cv_.notify_all();
    MCCL_TRACE("WorkMCCL seq=%u completed successfully", seq_);
}

void WorkMCCL::markError(std::exception_ptr err) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completed_) return;
        completed_ = true;
        success_ = false;
        exception_ = std::move(err);
        finishWorkMCCLFuture();
    }
    cv_.notify_all();
    MCCL_ERROR("WorkMCCL seq=%u failed with error", seq_);
}

} // namespace mccl
