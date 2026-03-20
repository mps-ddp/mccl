#pragma once

#include <torch/torch.h>
#include <c10d/ProcessGroup.hpp>
#include <chrono>
#include <string>

#include "compression/Compression.hpp"

namespace mccl {

struct MCCLOptions : public c10d::Backend::Options {
    MCCLOptions()
        : c10d::Backend::Options("mccl", std::chrono::milliseconds(30000)) {}

    explicit MCCLOptions(std::chrono::milliseconds timeout)
        : c10d::Backend::Options("mccl", timeout) {}

    // Transport
    std::string transport = "auto";  // "auto", "tcp", "rdma"
    std::string listen_addr = "0.0.0.0";
    uint16_t port_base = 29500;
    std::string ifname;
    size_t chunk_bytes = 4 * 1024 * 1024;
    size_t small_msg_threshold = 65536;
    bool transport_crc = false;

    // Compute
    bool fast_math = true;
    uint32_t gpu_threshold = 4096;
    bool overlap_comm = true;

    // Engine
    size_t max_queue_depth = 1024;

    // Compression
    CompressionMode compression = CompressionMode::NONE;
    double topk_ratio = 0.01;

    // Watchdog
    std::chrono::milliseconds watchdog_timeout{300000};

    // Health monitor
    std::chrono::milliseconds heartbeat_interval{5000};
};

} // namespace mccl
