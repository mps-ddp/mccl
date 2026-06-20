#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <string>

namespace mccl {

enum class StftBackend {
    Vdsp = 0,
    Metal = 1,
    Auto = 2,
};

StftBackend parse_stft_backend(const std::string& name);
StftBackend resolve_stft_backend(StftBackend requested);

struct StftParams {
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t win_length = 0;
    bool center = true;
    bool normalized = false;
};

/// Complex spectrogram ``[batch, n_fft/2+1, n_frames]`` on the input device.
at::Tensor stft_forward(
    const at::Tensor& waveform,
    const at::Tensor& window,
    const StftParams& params,
    StftBackend backend);

/// Gradient w.r.t. waveform ``[batch, time]`` matching ``stft_forward``.
at::Tensor stft_backward(
    const at::Tensor& grad_spec,
    const at::Tensor& window,
    const StftParams& params,
    int64_t signal_length,
    StftBackend backend);

/// Inverse STFT: complex spec -> waveform ``[batch, length]``.
at::Tensor istft_forward(
    const at::Tensor& spec,
    const at::Tensor& window,
    const StftParams& params,
    int64_t length,
    StftBackend backend);

/// Gradient w.r.t. spec from ``istft_forward``.
at::Tensor istft_backward(
    const at::Tensor& grad_waveform,
    const at::Tensor& window,
    const StftParams& params,
    int64_t length,
    StftBackend backend);

} // namespace mccl
