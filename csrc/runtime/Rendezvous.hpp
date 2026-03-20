#pragma once

#include <torch/torch.h>
#include <c10d/Store.hpp>
#include <string>
#include <vector>
#include <chrono>

namespace mccl {

/// Store-based rendezvous: exchange transport endpoints among all ranks.
///
/// Each rank publishes its "host:port" string under a well-known key,
/// then reads all other ranks' endpoints.
/// Uses the Store provided by torch.distributed.init_process_group.
class Rendezvous {
public:
    Rendezvous(c10::intrusive_ptr<c10d::Store> store,
               int rank, int world_size,
               std::chrono::milliseconds timeout);

    /// Publish this rank's endpoint and collect all endpoints.
    /// Returns a vector of size world_size, indexed by rank.
    std::vector<std::string> exchange_endpoints(const std::string& my_endpoint);

    /// Store-backed barrier — all ranks must call before any can proceed.
    void barrier(const std::string& tag = "mccl_barrier");

private:
    c10::intrusive_ptr<c10d::Store> store_;
    int rank_;
    int world_size_;
    std::chrono::milliseconds timeout_;

    static std::string endpoint_key(int rank);
    static std::string barrier_key(const std::string& tag, int rank);
};

} // namespace mccl
