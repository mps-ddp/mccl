#include "transforms/StftVdsp.hpp"
#include "transforms/WindowUtils.hpp"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mccl {

namespace {

constexpr int64_t kFreqBins(int64_t n_fft) { return n_fft / 2 + 1; }

/// Torch layout ``[batch, freq, frames]`` row-major linear index.
inline size_t spec_index(int64_t b, int64_t k, int64_t f, int64_t n_freq, int64_t n_frames) {
    return static_cast<size_t>((b * n_freq + k) * n_frames + f);
}

int log2_int(int64_t n_fft) {
    int l = 0;
    int64_t v = n_fft;
    while (v > 1) {
        v >>= 1;
        ++l;
    }
    return l;
}

struct FftSetupCache {
    FFTSetup setup = nullptr;
    int64_t n_fft = 0;
    int log2n = 0;

    void ensure(int64_t n) {
        if (n == n_fft && setup != nullptr) {
            return;
        }
        destroy();
        n_fft = n;
        log2n = log2_int(n);
        if ((1LL << log2n) != n) {
            throw std::invalid_argument("stft vDSP: n_fft must be a power of two");
        }
        setup = vDSP_create_fftsetup(static_cast<vDSP_Length>(log2n), FFT_RADIX2);
        if (setup == nullptr) {
            throw std::runtime_error("vDSP_create_fftsetup failed");
        }
    }

    void destroy() {
        if (setup != nullptr) {
            vDSP_destroy_fftsetup(setup);
            setup = nullptr;
        }
        n_fft = 0;
        log2n = 0;
    }

    ~FftSetupCache() { destroy(); }
};

thread_local FftSetupCache g_fft;

struct VdspWorkspace {
    std::vector<float> padded;
    std::vector<float> frames;
    std::vector<float> fft_re;
    std::vector<float> fft_im;
    std::vector<float> acc;
    std::vector<float> wsum;

    void ensure_frames(int64_t n_frames, int64_t n_fft) {
        frames.resize(static_cast<size_t>(n_frames * n_fft));
        if (fft_re.size() < static_cast<size_t>(n_fft)) {
            fft_re.resize(static_cast<size_t>(n_fft));
            fft_im.resize(static_cast<size_t>(n_fft));
        }
    }
};

thread_local VdspWorkspace g_ws;

/// Unfold ``[padded_len]`` into contiguous ``[n_frames, n_fft]`` (``memcpy`` rows).
void extract_frames(
    const float* padded,
    int64_t padded_len,
    int64_t n_frames,
    int64_t n_fft,
    int64_t hop,
    float* frames) {
    for (int64_t f = 0; f < n_frames; ++f) {
        const int64_t start = f * hop;
        float* row = frames + f * n_fft;
        if (start >= 0 && start + n_fft <= padded_len) {
            std::memcpy(row, padded + start, static_cast<size_t>(n_fft) * sizeof(float));
        } else {
            for (int64_t i = 0; i < n_fft; ++i) {
                const int64_t idx = start + i;
                row[static_cast<size_t>(i)] =
                    (idx >= 0 && idx < padded_len) ? padded[static_cast<size_t>(idx)] : 0.f;
            }
        }
    }
}

void window_frames(float* frames, int64_t n_frames, int64_t n_fft, const float* win) {
    const vDSP_Length len = static_cast<vDSP_Length>(n_fft);
    for (int64_t f = 0; f < n_frames; ++f) {
        vDSP_vmul(frames + f * n_fft, 1, win, 1, frames + f * n_fft, 1, len);
    }
}

void rfft_row(const float* row, int64_t n_fft, float* out_re, float* out_im) {
    g_fft.ensure(n_fft);
    const int64_t n_freq = kFreqBins(n_fft);
    float* re = g_ws.fft_re.data();
    float* im = g_ws.fft_im.data();
    std::memcpy(re, row, static_cast<size_t>(n_fft) * sizeof(float));
    std::memset(im, 0, static_cast<size_t>(n_fft) * sizeof(float));
    DSPSplitComplex split{re, im};
    vDSP_fft_zip(
        g_fft.setup, &split, 1, static_cast<vDSP_Length>(g_fft.log2n), FFT_FORWARD);
    for (int64_t k = 0; k < n_freq; ++k) {
        out_re[k] = re[static_cast<size_t>(k)];
        out_im[k] = im[static_cast<size_t>(k)];
    }
}

void irfft_row(const float* in_re, const float* in_im, int64_t n_fft, float* row) {
    g_fft.ensure(n_fft);
    const int64_t n_freq = kFreqBins(n_fft);
    float* re = g_ws.fft_re.data();
    float* im = g_ws.fft_im.data();
    for (int64_t k = 0; k < n_freq; ++k) {
        re[static_cast<size_t>(k)] = in_re[k];
        im[static_cast<size_t>(k)] = in_im[k];
    }
    for (int64_t k = n_freq; k < n_fft; ++k) {
        const int64_t m = n_fft - k;
        re[static_cast<size_t>(k)] = re[static_cast<size_t>(m)];
        im[static_cast<size_t>(k)] = -im[static_cast<size_t>(m)];
    }
    vDSP_vneg(im, 1, im, 1, static_cast<vDSP_Length>(n_fft));
    DSPSplitComplex split{re, im};
    vDSP_fft_zip(
        g_fft.setup, &split, 1, static_cast<vDSP_Length>(g_fft.log2n), FFT_FORWARD);
    const float inv_n = 1.f / static_cast<float>(n_fft);
    vDSP_vsmul(re, 1, &inv_n, re, 1, static_cast<vDSP_Length>(n_fft));
    vDSP_vsmul(im, 1, &inv_n, im, 1, static_cast<vDSP_Length>(n_fft));
    vDSP_vneg(im, 1, im, 1, static_cast<vDSP_Length>(n_fft));
    std::memcpy(row, re, static_cast<size_t>(n_fft) * sizeof(float));
}

struct FftParallelCtx {
    FFTSetup setup = nullptr;
    vDSP_Length log2n = 0;
    int64_t n_fft = 0;
    int64_t n_freq = 0;
    int64_t n_frames = 0;
    int64_t batch = 0;
    int64_t n_freq_total = 0;
    const float* frames = nullptr;
    float* spec_re = nullptr;
    float* spec_im = nullptr;
};

void fft_frames_forward_parallel(FftParallelCtx ctx) {
    dispatch_apply(static_cast<size_t>(ctx.n_frames), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t f) {
        thread_local std::vector<float> re;
        thread_local std::vector<float> im;
        if (re.size() < static_cast<size_t>(ctx.n_fft)) {
            re.resize(static_cast<size_t>(ctx.n_fft));
            im.resize(static_cast<size_t>(ctx.n_fft));
        }
        const float* row = ctx.frames + static_cast<int64_t>(f) * ctx.n_fft;
        std::memcpy(re.data(), row, static_cast<size_t>(ctx.n_fft) * sizeof(float));
        std::memset(im.data(), 0, static_cast<size_t>(ctx.n_fft) * sizeof(float));
        DSPSplitComplex split{re.data(), im.data()};
        vDSP_fft_zip(ctx.setup, &split, 1, ctx.log2n, FFT_FORWARD);
        for (int64_t k = 0; k < ctx.n_freq; ++k) {
            const size_t idx = spec_index(ctx.batch, k, static_cast<int64_t>(f), ctx.n_freq, ctx.n_frames);
            ctx.spec_re[idx] = re[static_cast<size_t>(k)];
            ctx.spec_im[idx] = im[static_cast<size_t>(k)];
        }
    });
}

struct IrfftParallelCtx {
    FFTSetup setup = nullptr;
    vDSP_Length log2n = 0;
    int64_t n_fft = 0;
    int64_t n_freq = 0;
    int64_t n_frames = 0;
    int64_t batch = 0;
    const float* spec_re = nullptr;
    const float* spec_im = nullptr;
    float* frames = nullptr;
};

void irfft_frames_parallel(IrfftParallelCtx ctx) {
    const float inv_n = 1.f / static_cast<float>(ctx.n_fft);
    dispatch_apply(static_cast<size_t>(ctx.n_frames), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t f) {
        thread_local std::vector<float> re;
        thread_local std::vector<float> im;
        if (re.size() < static_cast<size_t>(ctx.n_fft)) {
            re.resize(static_cast<size_t>(ctx.n_fft));
            im.resize(static_cast<size_t>(ctx.n_fft));
        }
        for (int64_t k = 0; k < ctx.n_freq; ++k) {
            const size_t idx = spec_index(ctx.batch, k, static_cast<int64_t>(f), ctx.n_freq, ctx.n_frames);
            re[static_cast<size_t>(k)] = ctx.spec_re[idx];
            im[static_cast<size_t>(k)] = ctx.spec_im[idx];
        }
        for (int64_t k = ctx.n_freq; k < ctx.n_fft; ++k) {
            const int64_t m = ctx.n_fft - k;
            re[static_cast<size_t>(k)] = re[static_cast<size_t>(m)];
            im[static_cast<size_t>(k)] = -im[static_cast<size_t>(m)];
        }
        vDSP_vneg(im.data(), 1, im.data(), 1, static_cast<vDSP_Length>(ctx.n_fft));
        DSPSplitComplex split{re.data(), im.data()};
        vDSP_fft_zip(ctx.setup, &split, 1, ctx.log2n, FFT_FORWARD);
        vDSP_vsmul(re.data(), 1, &inv_n, re.data(), 1, static_cast<vDSP_Length>(ctx.n_fft));
        vDSP_vsmul(im.data(), 1, &inv_n, im.data(), 1, static_cast<vDSP_Length>(ctx.n_fft));
        vDSP_vneg(im.data(), 1, im.data(), 1, static_cast<vDSP_Length>(ctx.n_fft));
        std::memcpy(
            ctx.frames + static_cast<int64_t>(f) * ctx.n_fft,
            re.data(),
            static_cast<size_t>(ctx.n_fft) * sizeof(float));
    });
}

void overlap_add_frame(
    float* acc,
    int64_t acc_len,
    const float* frame,
    int64_t frame_start,
    int64_t n_fft) {
    const int64_t i0 = std::max<int64_t>(0, -frame_start);
    const int64_t i1 = std::min<int64_t>(n_fft, acc_len - frame_start);
    if (i0 < i1) {
        vDSP_vadd(
            acc + frame_start + i0,
            1,
            frame + i0,
            1,
            acc + frame_start + i0,
            1,
            static_cast<vDSP_Length>(i1 - i0));
    }
}

void overlap_add_weight(
    float* wsum,
    int64_t acc_len,
    const float* window,
    int64_t win_length,
    int64_t frame_start) {
    const int64_t i0 = std::max<int64_t>(0, -frame_start);
    const int64_t i1 = std::min<int64_t>(win_length, acc_len - frame_start);
    for (int64_t i = i0; i < i1; ++i) {
        const float w = window[static_cast<size_t>(i)];
        wsum[static_cast<size_t>(frame_start + i)] += w * w;
    }
}

} // namespace

void stft_forward_vdsp(
    const float* waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& spec_real,
    std::vector<float>& spec_imag) {
    const int64_t n_fft = params.n_fft;
    const int64_t hop = params.hop_length;
    const int64_t n_freq = kFreqBins(n_fft);
    const int64_t n_frames = stft_num_frames(signal_length, params);
    const int64_t pad = params.center ? n_fft / 2 : 0;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    const size_t spec_elems = static_cast<size_t>(batch * n_freq * n_frames);
    spec_real.assign(spec_elems, 0.f);
    spec_imag.assign(spec_elems, 0.f);

    g_fft.ensure(n_fft);
    g_ws.ensure_frames(n_frames, n_fft);

    for (int64_t b = 0; b < batch; ++b) {
        const float* x = waveform + b * signal_length;
        if (params.center) {
            reflect_pad_1d(x, signal_length, pad, g_ws.padded);
        } else {
            g_ws.padded.assign(x, x + signal_length);
        }
        const int64_t padded_len = static_cast<int64_t>(g_ws.padded.size());

        extract_frames(
            g_ws.padded.data(),
            padded_len,
            n_frames,
            n_fft,
            hop,
            g_ws.frames.data());
        window_frames(g_ws.frames.data(), n_frames, n_fft, win.data());

        FftParallelCtx ctx{
            g_fft.setup,
            static_cast<vDSP_Length>(g_fft.log2n),
            n_fft,
            n_freq,
            n_frames,
            b,
            n_freq,
            g_ws.frames.data(),
            spec_real.data(),
            spec_imag.data(),
        };
        fft_frames_forward_parallel(ctx);
    }
}

void stft_backward_vdsp(
    const float* grad_spec_real,
    const float* grad_spec_imag,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& grad_waveform) {
    const int64_t n_fft = params.n_fft;
    const int64_t hop = params.hop_length;
    const int64_t n_freq = kFreqBins(n_fft);
    const int64_t n_frames = stft_num_frames(signal_length, params);
    const int64_t pad = params.center ? n_fft / 2 : 0;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    grad_waveform.assign(static_cast<size_t>(batch * signal_length), 0.f);
    const int64_t padded_len = params.center ? signal_length + n_fft : signal_length;

    g_fft.ensure(n_fft);
    g_ws.ensure_frames(n_frames, n_fft);
    g_ws.acc.assign(static_cast<size_t>(padded_len), 0.f);

    const vDSP_Length win_len = static_cast<vDSP_Length>(n_fft);

    for (int64_t b = 0; b < batch; ++b) {
        std::fill(g_ws.acc.begin(), g_ws.acc.end(), 0.f);

        IrfftParallelCtx ctx{
            g_fft.setup,
            static_cast<vDSP_Length>(g_fft.log2n),
            n_fft,
            n_freq,
            n_frames,
            b,
            grad_spec_real,
            grad_spec_imag,
            g_ws.frames.data(),
        };
        irfft_frames_parallel(ctx);

        for (int64_t f = 0; f < n_frames; ++f) {
            float* row = g_ws.frames.data() + f * n_fft;
            vDSP_vmul(row, 1, win.data(), 1, row, 1, win_len);
            overlap_add_frame(g_ws.acc.data(), padded_len, row, f * hop, n_fft);
        }

        float* out = grad_waveform.data() + b * signal_length;
        if (params.center) {
            std::memcpy(
                out,
                g_ws.acc.data() + pad,
                static_cast<size_t>(signal_length) * sizeof(float));
        } else {
            std::memcpy(out, g_ws.acc.data(), static_cast<size_t>(signal_length) * sizeof(float));
        }
    }
}

void istft_forward_vdsp(
    const float* spec_real,
    const float* spec_imag,
    int64_t batch,
    int64_t n_frames,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t length,
    std::vector<float>& waveform) {
    const int64_t n_fft = params.n_fft;
    const int64_t hop = params.hop_length;
    const int64_t n_freq = kFreqBins(n_fft);
    const int64_t pad = params.center ? n_fft / 2 : 0;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    const int64_t out_len = params.center ? length + n_fft : hop * (n_frames - 1) + n_fft;
    waveform.assign(static_cast<size_t>(batch * length), 0.f);

    g_fft.ensure(n_fft);
    g_ws.ensure_frames(n_frames, n_fft);
    g_ws.acc.assign(static_cast<size_t>(out_len), 0.f);
    g_ws.wsum.assign(static_cast<size_t>(out_len), 0.f);

    const vDSP_Length win_len = static_cast<vDSP_Length>(n_fft);

    for (int64_t b = 0; b < batch; ++b) {
        std::fill(g_ws.acc.begin(), g_ws.acc.end(), 0.f);
        std::fill(g_ws.wsum.begin(), g_ws.wsum.end(), 0.f);

        IrfftParallelCtx ctx{
            g_fft.setup,
            static_cast<vDSP_Length>(g_fft.log2n),
            n_fft,
            n_freq,
            n_frames,
            b,
            spec_real,
            spec_imag,
            g_ws.frames.data(),
        };
        irfft_frames_parallel(ctx);

        for (int64_t f = 0; f < n_frames; ++f) {
            float* row = g_ws.frames.data() + f * n_fft;
            vDSP_vmul(row, 1, win.data(), 1, row, 1, win_len);
            overlap_add_frame(g_ws.acc.data(), out_len, row, f * hop, n_fft);
            overlap_add_weight(g_ws.wsum.data(), out_len, win.data(), n_fft, f * hop);
        }

        for (int64_t i = 0; i < out_len; ++i) {
            const float denom = std::max(g_ws.wsum[static_cast<size_t>(i)], 1e-8f);
            g_ws.acc[static_cast<size_t>(i)] /= denom;
        }

        float* out = waveform.data() + b * length;
        if (params.center) {
            std::memcpy(
                out,
                g_ws.acc.data() + pad,
                static_cast<size_t>(length) * sizeof(float));
        } else {
            const int64_t n = std::min(length, out_len);
            std::memcpy(out, g_ws.acc.data(), static_cast<size_t>(n) * sizeof(float));
        }
    }
}

void istft_backward_vdsp(
    const float* grad_waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t n_frames,
    std::vector<float>& grad_spec_real,
    std::vector<float>& grad_spec_imag) {
    stft_forward_vdsp(
        grad_waveform, batch, signal_length, window, win_length, params, grad_spec_real, grad_spec_imag);
}

} // namespace mccl
