#include "transforms/Stft.hpp"
#include "transforms/StftVdsp.hpp"
#include "transforms/StftMetal.hpp"
#include "transforms/WindowUtils.hpp"
#include "metal/MPSInterop.hpp"
#include "common/Errors.hpp"

#include <ATen/ATen.h>
#include <cstring>
#include <stdexcept>

namespace mccl {

namespace {

void check_waveform(const at::Tensor& waveform) {
    MCCL_CHECK(waveform.is_mps() || waveform.is_cpu(),
               "stft: waveform must be on MPS or CPU");
    MCCL_CHECK(waveform.scalar_type() == at::kFloat,
               "stft: waveform must be float32");
    MCCL_CHECK(waveform.is_contiguous(), "stft: waveform must be contiguous");
}

void check_window(const at::Tensor& window) {
    MCCL_CHECK(window.scalar_type() == at::kFloat, "stft: window must be float32");
    MCCL_CHECK(window.dim() == 1, "stft: window must be 1-D");
}

std::tuple<at::Tensor, int64_t, int64_t> flatten_waveform(const at::Tensor& waveform) {
    at::Tensor w = waveform;
    if (w.dim() == 3 && w.size(1) == 1) {
        w = w.squeeze(1);
    }
    MCCL_CHECK(w.dim() == 2, "stft: waveform must be [batch, time] or [batch, 1, time]");
    return {w, w.size(0), w.size(1)};
}

at::Tensor read_float_mps(const at::Tensor& tensor, std::vector<float>& buf) {
    at::Tensor cpu = tensor;
    if (tensor.is_mps()) {
        mps_stream_sync();
        if (tensor_cpu_accessible(tensor)) {
            auto view = extract_mps_buffer(tensor);
            buf.resize(view.nbytes / sizeof(float));
            std::memcpy(buf.data(), view.cpu_ptr, view.nbytes);
            return tensor;
        }
        cpu = tensor.detach().cpu().contiguous();
    } else {
        cpu = tensor.contiguous();
    }
    buf.resize(static_cast<size_t>(cpu.numel()));
    std::memcpy(buf.data(), cpu.data_ptr<float>(), buf.size() * sizeof(float));
    return cpu;
}

void write_float_mps(const std::vector<float>& buf, at::Tensor& out) {
    if (out.is_mps()) {
        if (tensor_cpu_accessible(out)) {
            auto view = extract_mps_buffer(out);
            MCCL_CHECK(view.nbytes == buf.size() * sizeof(float), "stft: nbytes mismatch");
            std::memcpy(view.cpu_ptr, buf.data(), view.nbytes);
            mps_stream_sync_after_cpu_mps_buffer_write();
            return;
        }
        auto cpu = torch::from_blob(
            const_cast<float*>(buf.data()),
            out.sizes(),
            torch::TensorOptions().dtype(at::kFloat));
        out.copy_(cpu.to(out.device()));
        return;
    }
    std::memcpy(out.data_ptr<float>(), buf.data(), buf.size() * sizeof(float));
}

const float* waveform_ptr(const at::Tensor& tensor, std::vector<float>& owned) {
    if (tensor.is_cpu() && tensor.is_contiguous()) {
        return tensor.data_ptr<float>();
    }
    read_float_mps(tensor, owned);
    return owned.data();
}

at::Tensor make_complex_spec(
    const std::vector<float>& re,
    const std::vector<float>& im,
    int64_t batch,
    int64_t n_freq,
    int64_t n_frames,
    c10::Device device) {
    const int64_t n = batch * n_freq * n_frames;
    MCCL_CHECK(
        static_cast<int64_t>(re.size()) == n && static_cast<int64_t>(im.size()) == n,
        "stft: spec buffer size mismatch");

    auto options = at::TensorOptions().dtype(at::kComplexFloat).device(device);
    at::Tensor spec = at::empty({batch, n_freq, n_frames}, options);

    if (device.is_cpu()) {
        auto* out = reinterpret_cast<c10::complex<float>*>(spec.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = c10::complex<float>(re[static_cast<size_t>(i)], im[static_cast<size_t>(i)]);
        }
        return spec;
    }

    at::Tensor cpu_spec = at::empty({batch, n_freq, n_frames}, at::TensorOptions().dtype(at::kComplexFloat));
    auto* out = reinterpret_cast<c10::complex<float>*>(cpu_spec.data_ptr());
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = c10::complex<float>(re[static_cast<size_t>(i)], im[static_cast<size_t>(i)]);
    }
    spec.copy_(cpu_spec.to(device));
    return spec;
}

void read_complex_spec(
    const at::Tensor& spec,
    std::vector<float>& re,
    std::vector<float>& im) {
    at::Tensor cpu = spec.is_mps() ? spec.detach().cpu() : spec.contiguous();
    MCCL_CHECK(cpu.scalar_type() == at::kComplexFloat, "stft: spec must be complex64");
    const int64_t n = cpu.numel();
    re.resize(static_cast<size_t>(n));
    im.resize(static_cast<size_t>(n));
    auto* ptr = reinterpret_cast<c10::complex<float>*>(cpu.data_ptr());
    for (int64_t i = 0; i < n; ++i) {
        re[static_cast<size_t>(i)] = ptr[static_cast<size_t>(i)].real();
        im[static_cast<size_t>(i)] = ptr[static_cast<size_t>(i)].imag();
    }
}

} // namespace

at::Tensor stft_forward(
    const at::Tensor& waveform,
    const at::Tensor& window,
    const StftParams& params,
    StftBackend backend) {
    check_waveform(waveform);
    check_window(window);
    const StftBackend resolved = resolve_stft_backend(backend);

    auto [w2d, batch, signal_length] = flatten_waveform(waveform);
    const int64_t n_freq = params.n_fft / 2 + 1;
    const int64_t n_frames = stft_num_frames(signal_length, params);

    std::vector<float> wav_buf;
    const float* wave_ptr = waveform_ptr(w2d, wav_buf);

    at::Tensor win_cpu = window.is_mps() ? window.detach().cpu() : window.contiguous();

    std::vector<float> spec_re, spec_im;
    if (resolved == StftBackend::Metal) {
        stft_forward_metal(
            wave_ptr,
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            spec_re,
            spec_im);
    } else {
        stft_forward_vdsp(
            wave_ptr,
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            spec_re,
            spec_im);
    }

    return make_complex_spec(spec_re, spec_im, batch, n_freq, n_frames, w2d.device());
}

at::Tensor stft_backward(
    const at::Tensor& grad_spec,
    const at::Tensor& window,
    const StftParams& params,
    int64_t signal_length,
    StftBackend backend) {
    check_window(window);
    const StftBackend resolved = resolve_stft_backend(backend);

    const int64_t batch = grad_spec.size(0);
    std::vector<float> gre, gim;
    read_complex_spec(grad_spec, gre, gim);

    at::Tensor win_cpu = window.is_mps() ? window.detach().cpu() : window.contiguous();

    std::vector<float> grad_wav;
    if (resolved == StftBackend::Metal) {
        stft_backward_metal(
            gre.data(),
            gim.data(),
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            grad_wav);
    } else {
        stft_backward_vdsp(
            gre.data(),
            gim.data(),
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            grad_wav);
    }

    // vDSP/Metal irfft implements the mathematical inverse FFT (÷ n_fft). PyTorch's
    // rfft backward adjoint scales by n_fft/2 for the one-sided real spectrum.
    if (!params.normalized && params.n_fft > 0) {
        const float scale = static_cast<float>(params.n_fft) * 0.5f;
        for (float& v : grad_wav) {
            v *= scale;
        }
    }

    at::Tensor out = at::empty(
        {batch, signal_length},
        at::TensorOptions().dtype(at::kFloat).device(grad_spec.device()));
    write_float_mps(grad_wav, out);
    return out;
}

at::Tensor istft_forward(
    const at::Tensor& spec,
    const at::Tensor& window,
    const StftParams& params,
    int64_t length,
    StftBackend backend) {
    check_window(window);
    const StftBackend resolved = resolve_stft_backend(backend);

    const int64_t batch = spec.size(0);
    const int64_t n_frames = spec.size(2);

    std::vector<float> sre, sim;
    read_complex_spec(spec, sre, sim);

    at::Tensor win_cpu = window.is_mps() ? window.detach().cpu() : window.contiguous();

    std::vector<float> wav;
    if (resolved == StftBackend::Metal) {
        istft_forward_metal(
            sre.data(),
            sim.data(),
            batch,
            n_frames,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            length,
            wav);
    } else {
        istft_forward_vdsp(
            sre.data(),
            sim.data(),
            batch,
            n_frames,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            length,
            wav);
    }

    at::Tensor out = at::empty(
        {batch, length},
        at::TensorOptions().dtype(at::kFloat).device(spec.device()));
    write_float_mps(wav, out);
    return out;
}

at::Tensor istft_backward(
    const at::Tensor& grad_waveform,
    const at::Tensor& window,
    const StftParams& params,
    int64_t length,
    StftBackend backend) {
    check_window(window);
    const StftBackend resolved = resolve_stft_backend(backend);

    auto [w2d, batch, signal_length] = flatten_waveform(grad_waveform);
    MCCL_CHECK(signal_length == length, "istft_backward: length mismatch");

    const int64_t n_frames = stft_num_frames(length, params);

    std::vector<float> gw;
    const float* gw_ptr = waveform_ptr(w2d, gw);

    at::Tensor win_cpu = window.is_mps() ? window.detach().cpu() : window.contiguous();

    std::vector<float> gspec_re, gspec_im;
    if (resolved == StftBackend::Metal) {
        istft_backward_metal(
            gw_ptr,
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            n_frames,
            gspec_re,
            gspec_im);
    } else {
        istft_backward_vdsp(
            gw_ptr,
            batch,
            signal_length,
            win_cpu.data_ptr<float>(),
            win_cpu.size(0),
            params,
            n_frames,
            gspec_re,
            gspec_im);
    }

    // istft_backward reuses stft_forward; adjoint of OLA+iFFT needs ÷ n_fft vs raw STFT.
    if (!params.normalized && params.n_fft > 0) {
        const float scale = 1.f / static_cast<float>(params.n_fft);
        for (float& v : gspec_re) {
            v *= scale;
        }
        for (float& v : gspec_im) {
            v *= scale;
        }
    }

    const int64_t n_freq = params.n_fft / 2 + 1;
    return make_complex_spec(gspec_re, gspec_im, batch, n_freq, n_frames, grad_waveform.device());
}

} // namespace mccl
