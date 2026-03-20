#include "runtime/Rendezvous.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

namespace mccl {

Rendezvous::Rendezvous(c10::intrusive_ptr<c10d::Store> store,
                       int rank, int world_size,
                       std::chrono::milliseconds timeout)
    : store_(std::move(store)), rank_(rank), world_size_(world_size),
      timeout_(timeout) {
    MCCL_CHECK(store_ != nullptr, "Store must not be null");
}

std::string Rendezvous::endpoint_key(int rank) {
    return "mccl/endpoint/" + std::to_string(rank);
}

std::string Rendezvous::barrier_key(const std::string& tag, int rank) {
    return "mccl/barrier/" + tag + "/" + std::to_string(rank);
}

std::vector<std::string> Rendezvous::exchange_endpoints(const std::string& my_endpoint) {
    // Publish our endpoint
    std::string key = endpoint_key(rank_);
    std::vector<uint8_t> val(my_endpoint.begin(), my_endpoint.end());
    store_->set(key, val);
    MCCL_INFO("Rank %d: published endpoint '%s'", rank_, my_endpoint.c_str());

    // Collect all endpoints
    std::vector<std::string> endpoints(world_size_);
    endpoints[rank_] = my_endpoint;

    for (int r = 0; r < world_size_; r++) {
        if (r == rank_) continue;

        std::string peer_key = endpoint_key(r);
        MCCL_DEBUG("Rank %d: waiting for endpoint from rank %d", rank_, r);

        try {
            store_->wait({peer_key}, timeout_);
            auto peer_val = store_->get(peer_key);
            endpoints[r] = std::string(peer_val.begin(), peer_val.end());
            MCCL_INFO("Rank %d: got endpoint for rank %d = '%s'",
                      rank_, r, endpoints[r].c_str());
        } catch (const std::exception& e) {
            throw TimeoutError(
                "Rank " + std::to_string(rank_) +
                " timed out waiting for endpoint from rank " +
                std::to_string(r) + ": " + e.what()
            );
        }
    }

    return endpoints;
}

void Rendezvous::barrier(const std::string& tag) {
    std::string key = barrier_key(tag, rank_);
    std::vector<uint8_t> val = {1};
    store_->set(key, val);

    for (int r = 0; r < world_size_; r++) {
        if (r == rank_) continue;
        std::string peer_key = barrier_key(tag, r);
        try {
            store_->wait({peer_key}, timeout_);
            store_->get(peer_key);
        } catch (const std::exception& e) {
            throw TimeoutError(
                "Barrier '" + tag + "': rank " + std::to_string(rank_) +
                " timed out waiting for rank " + std::to_string(r) +
                ": " + e.what()
            );
        }
    }

    MCCL_DEBUG("Rank %d: barrier '%s' passed", rank_, tag.c_str());
}

} // namespace mccl
