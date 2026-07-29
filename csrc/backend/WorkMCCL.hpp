#pragma once

#include <torch/torch.h>
#include <c10d/Work.hpp>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/ivalue.h>

#include <condition_variable>
#include <exception>
#include <atomic>
#include <mutex>
#include <vector>

namespace mccl {

class WorkMCCL : public c10d::Work {
public:
    WorkMCCL(c10d::OpType opType, uint32_t seq,
             std::vector<at::Tensor> outputTensors = {});
    ~WorkMCCL() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    bool isCompleted() override;
    bool isSuccess() const override;
    std::exception_ptr exception() const override;
    std::vector<at::Tensor> result() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    void markComplete();
    void markError(std::exception_ptr err);

    /// Set by the engine thread before markComplete (overlap consumer release token).
    void set_release_token(uint64_t token);

    /// Hold async-reduce staging tensors until ``wait()`` (NCCL TensorShelf).
    void stash_tensors(std::vector<at::Tensor> tensors);

    uint32_t seq() const { return seq_; }

private:
    void finishWorkMCCLFuture();
    /// Autograd / DDP thread: wait until collective output is visible on MPS.
    void wait_consumer_release();

    uint32_t seq_;
    std::atomic<uint64_t> release_token_{0};
    std::atomic<bool> release_waited_{false};
    bool completed_ = false;
    bool success_ = false;
    std::exception_ptr exception_;
    std::vector<at::Tensor> outputs_;
    std::vector<at::Tensor> stashed_tensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // mutable so that const methods (isSuccess, exception) can acquire the lock
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace mccl
