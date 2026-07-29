#include "transforms/StftMetal.hpp"
#include "transforms/StftMetalKernels.hpp"
#include "metal/MPSInterop.hpp"

namespace mccl {

void stft_forward_metal(
    const float* waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& spec_real,
    std::vector<float>& spec_imag) {
    mps_stream_sync();
    metal_stft_forward(
        waveform, batch, signal_length, window, win_length, params, spec_real, spec_imag);
    mps_stream_sync_after_cpu_mps_buffer_write();
}

void stft_backward_metal(
    const float* grad_spec_real,
    const float* grad_spec_imag,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& grad_waveform) {
    mps_stream_sync();
    metal_stft_backward(
        grad_spec_real,
        grad_spec_imag,
        batch,
        signal_length,
        window,
        win_length,
        params,
        grad_waveform);
    mps_stream_sync_after_cpu_mps_buffer_write();
}

void istft_forward_metal(
    const float* spec_real,
    const float* spec_imag,
    int64_t batch,
    int64_t n_frames,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t length,
    std::vector<float>& waveform) {
    mps_stream_sync();
    metal_istft_forward(
        spec_real, spec_imag, batch, n_frames, window, win_length, params, length, waveform);
    mps_stream_sync_after_cpu_mps_buffer_write();
}

void istft_backward_metal(
    const float* grad_waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t n_frames,
    std::vector<float>& grad_spec_real,
    std::vector<float>& grad_spec_imag) {
    mps_stream_sync();
    metal_istft_backward(
        grad_waveform,
        batch,
        signal_length,
        window,
        win_length,
        params,
        n_frames,
        grad_spec_real,
        grad_spec_imag);
    mps_stream_sync_after_cpu_mps_buffer_write();
}

} // namespace mccl
