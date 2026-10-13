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

std::string Rendezvous::barrier_key(const std::string& tag, uint64_t epoch, int rank) {
    return "mccl/barrier/" + tag + "/" + std::to_string(epoch) + "/" +
           std::to_string(rank);
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
    // Epoch makes the barrier reusable: without it, a second barrier with the
    // same tag returns immediately for every rank because the keys from the
    // first barrier still exist in the store.
    uint64_t epoch;
    {
        std::lock_guard<std::mutex> lock(barrier_mu_);
        epoch = barrier_epochs_[tag]++;
    }

    std::string key = barrier_key(tag, epoch, rank_);
    std::vector<uint8_t> val = {1};
    store_->set(key, val);

    // Wait on all peers in one store call: waiting peer by peer costs a store
    // round trip per peer, so the barrier grows linearly with world size.
    // The timeout restarts whenever a peer arrives, as the per-peer wait did.
    std::vector<int> pending_ranks;
    std::vector<std::string> pending_keys;
    for (int r = 0; r < world_size_; r++) {
        if (r == rank_) continue;
        pending_ranks.push_back(r);
        pending_keys.push_back(barrier_key(tag, epoch, r));
    }

    while (!pending_keys.empty()) {
        try {
            store_->wait(pending_keys, timeout_);
            break;
        } catch (const std::exception& e) {
            std::vector<int> still_ranks;
            std::vector<std::string> still_keys;
            for (size_t i = 0; i < pending_keys.size(); i++) {
                bool present = false;
                try {
                    present = store_->check({pending_keys[i]});
                } catch (const std::exception&) {
                }
                if (!present) {
                    still_ranks.push_back(pending_ranks[i]);
                    still_keys.push_back(pending_keys[i]);
                }
            }
            if (still_keys.size() == pending_keys.size()) {
                std::string missing;
                for (int r : still_ranks) {
                    if (!missing.empty()) missing += ",";
                    missing += std::to_string(r);
                }
                throw TimeoutError(
                    "Barrier '" + tag + "': rank " + std::to_string(rank_) +
                    " timed out waiting for rank " + missing +
                    ": " + e.what()
                );
            }
            pending_ranks.swap(still_ranks);
            pending_keys.swap(still_keys);
        }
    }

    MCCL_DEBUG("Rank %d: barrier '%s' passed", rank_, tag.c_str());
}

} // namespace mccl
