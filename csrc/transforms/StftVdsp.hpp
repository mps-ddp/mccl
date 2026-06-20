#pragma once

#include "transforms/Stft.hpp"

#include <complex>
#include <cstdint>
#include <vector>

namespace mccl {

/// CPU float STFT (vDSP). ``spec`` linear layout matches torch ``[batch, freq, frames]``.
void stft_forward_vdsp(
    const float* waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& spec_real,
    std::vector<float>& spec_imag);

void stft_backward_vdsp(
    const float* grad_spec_real,
    const float* grad_spec_imag,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& grad_waveform);

void istft_forward_vdsp(
    const float* spec_real,
    const float* spec_imag,
    int64_t batch,
    int64_t n_frames,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t length,
    std::vector<float>& waveform);

void istft_backward_vdsp(
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
