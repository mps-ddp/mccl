#pragma once

#include <torch/torch.h>
#include <c10d/Store.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <unordered_map>

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
    /// Reusable: each call with the same tag uses a fresh epoch, so repeated
    /// barriers do not pass instantly on stale keys from a previous call.
    void barrier(const std::string& tag = "mccl_barrier");

private:
    c10::intrusive_ptr<c10d::Store> store_;
    int rank_;
    int world_size_;
    std::chrono::milliseconds timeout_;

    // Per-tag use counter.  All ranks call collectives (and thus barriers)
    // in the same order, so the local count matches across ranks.
    std::mutex barrier_mu_;
    std::unordered_map<std::string, uint64_t> barrier_epochs_;

    static std::string endpoint_key(int rank);
    static std::string barrier_key(const std::string& tag, uint64_t epoch, int rank);
};

} // namespace mccl
