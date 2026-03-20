#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace mccl {

class MCCLError : public std::runtime_error {
public:
    explicit MCCLError(const std::string& msg)
        : std::runtime_error("[MCCL] " + msg) {}
};

class TransportError : public MCCLError {
public:
    explicit TransportError(const std::string& msg)
        : MCCLError("Transport: " + msg) {}
};

class MetalError : public MCCLError {
public:
    explicit MetalError(const std::string& msg)
        : MCCLError("Metal: " + msg) {}
};

class TensorError : public MCCLError {
public:
    explicit TensorError(const std::string& msg)
        : MCCLError("Tensor: " + msg) {}
};

class TimeoutError : public MCCLError {
public:
    explicit TimeoutError(const std::string& msg)
        : MCCLError("Timeout: " + msg) {}
};

class ProtocolError : public MCCLError {
public:
    explicit ProtocolError(const std::string& msg)
        : MCCLError("Protocol: " + msg) {}
};

#define MCCL_CHECK(cond, ...)                                          \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::ostringstream _mccl_ss;                                \
            _mccl_ss << "Check failed: " #cond " at "                  \
                     << __FILE__ << ":" << __LINE__;                    \
            std::string _extra = std::string(__VA_ARGS__);              \
            if (!_extra.empty()) _mccl_ss << " — " << _extra;          \
            throw ::mccl::MCCLError(_mccl_ss.str());                   \
        }                                                               \
    } while (false)

#define MCCL_CHECK_TENSOR(cond, ...)                                   \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::ostringstream _mccl_ss;                                \
            _mccl_ss << "Tensor check failed: " #cond;                 \
            std::string _extra = std::string(__VA_ARGS__);              \
            if (!_extra.empty()) _mccl_ss << " — " << _extra;          \
            throw ::mccl::TensorError(_mccl_ss.str());                 \
        }                                                               \
    } while (false)

} // namespace mccl
