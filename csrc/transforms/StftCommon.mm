#include "transforms/WindowUtils.hpp"
#include "transforms/Stft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace mccl {

int64_t stft_num_frames(int64_t signal_length, const StftParams& params) {
    if (signal_length < 0) {
        throw std::invalid_argument("stft_num_frames: negative signal_length");
    }
    if (!params.center) {
        if (signal_length < params.n_fft) {
            return 0;
        }
        return 1 + (signal_length - params.n_fft) / params.hop_length;
    }
    return 1 + signal_length / params.hop_length;
}

namespace {

int64_t reflect_index(int64_t idx, int64_t len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * (len - 1) - idx;
        }
    }
    return idx;
}

} // namespace

void reflect_pad_1d(
    const float* input,
    int64_t length,
    int64_t pad,
    std::vector<float>& output) {
    const int64_t out_len = length + 2 * pad;
    output.resize(static_cast<size_t>(out_len));
    for (int64_t i = 0; i < out_len; ++i) {
        const int64_t src = i - pad;
        if (src >= 0 && src < length) {
            output[static_cast<size_t>(i)] = input[static_cast<size_t>(src)];
        } else {
            const int64_t mapped = reflect_index(src, length);
            output[static_cast<size_t>(i)] = input[static_cast<size_t>(mapped)];
        }
    }
}

void prepare_window(
    const float* src,
    int64_t src_len,
    int64_t win_length,
    std::vector<float>& out) {
    out.assign(static_cast<size_t>(win_length), 0.f);
    const int64_t n = std::min(src_len, win_length);
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = src[static_cast<size_t>(i)];
    }
}

StftBackend parse_stft_backend(const std::string& name) {
    if (name == "vdsp" || name == "VDSP") {
        return StftBackend::Vdsp;
    }
    if (name == "metal" || name == "METAL") {
        return StftBackend::Metal;
    }
    if (name == "auto" || name == "AUTO") {
        return StftBackend::Auto;
    }
    throw std::invalid_argument("unknown STFT backend: " + name);
}

StftBackend resolve_stft_backend(StftBackend requested) {
    if (requested != StftBackend::Auto) {
        return requested;
    }
    const char* env = std::getenv("MCCL_STFT_BACKEND");
    if (env != nullptr && env[0] != '\0') {
        return parse_stft_backend(env);
    }
    return StftBackend::Vdsp;
}

} // namespace mccl
