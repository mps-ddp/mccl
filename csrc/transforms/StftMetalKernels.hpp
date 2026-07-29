#pragma once

#include "transforms/Stft.hpp"

#include <vector>

namespace mccl {

void metal_stft_init();

/// Batched STFT forward on MCCL Metal queue (radix-2 FFT in shader).
void metal_stft_forward(
    const float* waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& spec_real,
    std::vector<float>& spec_imag);

void metal_stft_backward(
    const float* grad_spec_real,
    const float* grad_spec_imag,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& grad_waveform);

void metal_istft_forward(
    const float* spec_real,
    const float* spec_imag,
    int64_t batch,
    int64_t n_frames,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t length,
    std::vector<float>& waveform);

void metal_istft_backward(
    const float* grad_waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t n_frames,
    std::vector<float>& grad_spec_real,
    std::vector<float>& grad_spec_imag);

} // namespace mccl
