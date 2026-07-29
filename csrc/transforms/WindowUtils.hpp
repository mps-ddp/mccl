#pragma once

#include <cstdint>
#include <vector>

namespace mccl {

/// Number of STFT frames (PyTorch ``center=True``).
int64_t stft_num_frames(int64_t signal_length, const struct StftParams& params);

/// Reflect-pad 1D waveform (PyTorch ``pad_mode=reflect``).
void reflect_pad_1d(
    const float* input,
    int64_t length,
    int64_t pad,
    std::vector<float>& output);

/// Prepare window of length ``win_length`` (zero-pad/truncate source).
void prepare_window(
    const float* src,
    int64_t src_len,
    int64_t win_length,
    std::vector<float>& out);

} // namespace mccl
