#include "transforms/StftVdsp.hpp"
#include "transforms/WindowUtils.hpp"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
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
    std::vector<float> acc;
    std::vector<float> wsum;
    std::vector<float> win_sq;
};

thread_local VdspWorkspace g_ws;

size_t irfft_ola_worker_count(int64_t n_frames) {
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<size_t>(std::max<int64_t>(1, std::min<int64_t>(n_frames, static_cast<int64_t>(hw))));
}

/// Hermitian mirror + forward FFT + scale → real time frame in ``row``.
void irfft_frame_to_row(
    FFTSetup setup,
    vDSP_Length log2n,
    int64_t n_fft,
    int64_t n_freq,
    const float* in_re,
    const float* in_im,
    float* row) {
    thread_local std::vector<float> re;
    thread_local std::vector<float> im;
    if (re.size() < static_cast<size_t>(n_fft)) {
        re.resize(static_cast<size_t>(n_fft));
        im.resize(static_cast<size_t>(n_fft));
    }
    for (int64_t k = 0; k < n_freq; ++k) {
        re[static_cast<size_t>(k)] = in_re[k];
        im[static_cast<size_t>(k)] = in_im[k];
    }
    for (int64_t k = n_freq; k < n_fft; ++k) {
        const int64_t m = n_fft - k;
        re[static_cast<size_t>(k)] = re[static_cast<size_t>(m)];
        im[static_cast<size_t>(k)] = -im[static_cast<size_t>(m)];
    }
    vDSP_vneg(im.data(), 1, im.data(), 1, static_cast<vDSP_Length>(n_fft));
    DSPSplitComplex split{re.data(), im.data()};
    vDSP_fft_zip(setup, &split, 1, log2n, FFT_FORWARD);
    const float inv_n = 1.f / static_cast<float>(n_fft);
    vDSP_vsmul(re.data(), 1, &inv_n, re.data(), 1, static_cast<vDSP_Length>(n_fft));
    vDSP_vsmul(im.data(), 1, &inv_n, im.data(), 1, static_cast<vDSP_Length>(n_fft));
    vDSP_vneg(im.data(), 1, im.data(), 1, static_cast<vDSP_Length>(n_fft));
    std::memcpy(row, re.data(), static_cast<size_t>(n_fft) * sizeof(float));
}

void overlap_add_frame(
    float* acc,
    int64_t acc_len,
    const float* frame,
    int64_t frame_start,
    int64_t n_fft);

void overlap_add_weight(
    float* wsum,
    int64_t acc_len,
    const float* win_sq,
    int64_t win_length,
    int64_t frame_start);

struct IrfftOlaChunkCtx {
    FFTSetup setup = nullptr;
    vDSP_Length log2n = 0;
    int64_t n_fft = 0;
    int64_t n_freq = 0;
    int64_t n_frames = 0;
    int64_t hop = 0;
    int64_t acc_len = 0;
    int64_t batch = 0;
    int64_t frame_begin = 0;
    int64_t frame_end = 0;
    const float* spec_re = nullptr;
    const float* spec_im = nullptr;
    const std::complex<float>* spec_cplx = nullptr;
    const float* win = nullptr;
    const float* win_sq = nullptr;
    float* partial_acc = nullptr;
    float* partial_wsum = nullptr;
};

void irfft_ola_chunk(IrfftOlaChunkCtx ctx) {
    thread_local std::vector<float> frame_re;
    thread_local std::vector<float> frame_im;
    thread_local std::vector<float> row;
    if (frame_re.size() < static_cast<size_t>(ctx.n_freq)) {
        frame_re.resize(static_cast<size_t>(ctx.n_freq));
        frame_im.resize(static_cast<size_t>(ctx.n_freq));
    }
    if (row.size() < static_cast<size_t>(ctx.n_fft)) {
        row.resize(static_cast<size_t>(ctx.n_fft));
    }
    const vDSP_Length n_fft_len = static_cast<vDSP_Length>(ctx.n_fft);
    for (int64_t f = ctx.frame_begin; f < ctx.frame_end; ++f) {
        if (ctx.spec_cplx != nullptr) {
            for (int64_t k = 0; k < ctx.n_freq; ++k) {
                const size_t idx = spec_index(
                    ctx.batch, k, f, ctx.n_freq, ctx.n_frames);
                frame_re[static_cast<size_t>(k)] = ctx.spec_cplx[idx].real();
                frame_im[static_cast<size_t>(k)] = ctx.spec_cplx[idx].imag();
            }
        } else {
            for (int64_t k = 0; k < ctx.n_freq; ++k) {
                const size_t idx = spec_index(
                    ctx.batch, k, f, ctx.n_freq, ctx.n_frames);
                frame_re[static_cast<size_t>(k)] = ctx.spec_re[idx];
                frame_im[static_cast<size_t>(k)] = ctx.spec_im[idx];
            }
        }
        irfft_frame_to_row(
            ctx.setup,
            ctx.log2n,
            ctx.n_fft,
            ctx.n_freq,
            frame_re.data(),
            frame_im.data(),
            row.data());
        vDSP_vmul(row.data(), 1, ctx.win, 1, row.data(), 1, n_fft_len);
        overlap_add_frame(ctx.partial_acc, ctx.acc_len, row.data(), f * ctx.hop, ctx.n_fft);
        if (ctx.partial_wsum != nullptr && ctx.win_sq != nullptr) {
            overlap_add_weight(
                ctx.partial_wsum, ctx.acc_len, ctx.win_sq, ctx.n_fft, f * ctx.hop);
        }
    }
}

void merge_partial_buffers(
    float* acc,
    int64_t acc_len,
    const std::vector<std::vector<float>>& parts) {
    if (parts.empty()) {
        return;
    }
    std::memcpy(acc, parts[0].data(), static_cast<size_t>(acc_len) * sizeof(float));
    const vDSP_Length len = static_cast<vDSP_Length>(acc_len);
    for (size_t w = 1; w < parts.size(); ++w) {
        vDSP_vadd(acc, 1, parts[w].data(), 1, acc, 1, len);
    }
}

void irfft_ola_parallel(IrfftOlaChunkCtx base) {
    const size_t n_workers = irfft_ola_worker_count(base.n_frames);
    std::vector<std::vector<float>> ola_parts(
        n_workers, std::vector<float>(static_cast<size_t>(base.acc_len), 0.f));
    std::vector<float*> ola_ptrs(n_workers);
    for (size_t w = 0; w < n_workers; ++w) {
        ola_ptrs[w] = ola_parts[w].data();
    }

    std::vector<std::vector<float>> wsum_parts;
    std::vector<float*> wsum_ptrs;
    if (base.partial_wsum != nullptr) {
        wsum_parts.assign(
            n_workers, std::vector<float>(static_cast<size_t>(base.acc_len), 0.f));
        wsum_ptrs.resize(n_workers);
        for (size_t w = 0; w < n_workers; ++w) {
            wsum_ptrs[w] = wsum_parts[w].data();
        }
    }

    float** ola_ptrs_raw = ola_ptrs.data();
    float** wsum_ptrs_raw = wsum_ptrs.empty() ? nullptr : wsum_ptrs.data();

    dispatch_apply(n_workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
        const int64_t f0 = static_cast<int64_t>(worker) * base.n_frames / static_cast<int64_t>(n_workers);
        const int64_t f1 = static_cast<int64_t>(worker + 1) * base.n_frames / static_cast<int64_t>(n_workers);
        IrfftOlaChunkCtx ctx = base;
        ctx.frame_begin = f0;
        ctx.frame_end = f1;
        ctx.partial_acc = ola_ptrs_raw[worker];
        ctx.partial_wsum = wsum_ptrs_raw != nullptr ? wsum_ptrs_raw[worker] : nullptr;
        irfft_ola_chunk(ctx);
    });

    merge_partial_buffers(base.partial_acc, base.acc_len, ola_parts);
    if (base.partial_wsum != nullptr) {
        merge_partial_buffers(base.partial_wsum, base.acc_len, wsum_parts);
    }
}

/// Copy one STFT frame from padded audio, zero-fill at edges when needed.
void load_frame(
    const float* padded,
    int64_t padded_len,
    int64_t start,
    int64_t n_fft,
    float* row) {
    if (start >= 0 && start + n_fft <= padded_len) {
        std::memcpy(row, padded + start, static_cast<size_t>(n_fft) * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n_fft; ++i) {
        const int64_t idx = start + i;
        row[static_cast<size_t>(i)] =
            (idx >= 0 && idx < padded_len) ? padded[static_cast<size_t>(idx)] : 0.f;
    }
}

struct StftForwardFusedCtx {
    FFTSetup setup = nullptr;
    vDSP_Length log2n = 0;
    int64_t n_fft = 0;
    int64_t n_freq = 0;
    int64_t n_frames = 0;
    int64_t hop = 0;
    int64_t padded_len = 0;
    int64_t batch = 0;
    const float* padded = nullptr;
    const float* win = nullptr;
    float* spec_re = nullptr;
    float* spec_im = nullptr;
};

/// Fused extract + Hann window + rFFT per frame (avoids full ``[n_frames, n_fft]`` staging).
void stft_forward_fft_fused(StftForwardFusedCtx ctx) {
    const vDSP_Length n_fft_len = static_cast<vDSP_Length>(ctx.n_fft);
    dispatch_apply(
        static_cast<size_t>(ctx.n_frames),
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t f) {
            thread_local std::vector<float> row;
            thread_local std::vector<float> re;
            thread_local std::vector<float> im;
            if (row.size() < static_cast<size_t>(ctx.n_fft)) {
                row.resize(static_cast<size_t>(ctx.n_fft));
                re.resize(static_cast<size_t>(ctx.n_fft));
                im.resize(static_cast<size_t>(ctx.n_fft));
            }

            load_frame(
                ctx.padded,
                ctx.padded_len,
                static_cast<int64_t>(f) * ctx.hop,
                ctx.n_fft,
                row.data());
            vDSP_vmul(row.data(), 1, ctx.win, 1, row.data(), 1, n_fft_len);
            std::memcpy(re.data(), row.data(), static_cast<size_t>(ctx.n_fft) * sizeof(float));
            std::memset(im.data(), 0, static_cast<size_t>(ctx.n_fft) * sizeof(float));
            DSPSplitComplex split{re.data(), im.data()};
            vDSP_fft_zip(ctx.setup, &split, 1, ctx.log2n, FFT_FORWARD);
            for (int64_t k = 0; k < ctx.n_freq; ++k) {
                const size_t idx = spec_index(
                    ctx.batch, k, static_cast<int64_t>(f), ctx.n_freq, ctx.n_frames);
                ctx.spec_re[idx] = re[static_cast<size_t>(k)];
                ctx.spec_im[idx] = im[static_cast<size_t>(k)];
            }
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
    const float* win_sq,
    int64_t win_length,
    int64_t frame_start) {
    const int64_t i0 = std::max<int64_t>(0, -frame_start);
    const int64_t i1 = std::min<int64_t>(win_length, acc_len - frame_start);
    if (i0 < i1) {
        vDSP_vadd(
            wsum + frame_start + i0,
            1,
            win_sq + i0,
            1,
            wsum + frame_start + i0,
            1,
            static_cast<vDSP_Length>(i1 - i0));
    }
}

void prepare_win_sq(const std::vector<float>& win, int64_t n_fft) {
    g_ws.win_sq.resize(static_cast<size_t>(n_fft));
    vDSP_vsq(win.data(), 1, g_ws.win_sq.data(), 1, static_cast<vDSP_Length>(n_fft));
}

void vdsp_waveform_from_spec_planes(
    const float* spec_re,
    const float* spec_im,
    const std::complex<float>* spec_cplx,
    int64_t batch,
    int64_t n_frames,
    int64_t signal_length,
    int64_t length,
    const StftParams& params,
    const std::vector<float>& win,
    bool normalize_ola,
    std::vector<float>& waveform) {
    const int64_t n_fft = params.n_fft;
    const int64_t hop = params.hop_length;
    const int64_t n_freq = kFreqBins(n_fft);
    const int64_t pad = params.center ? n_fft / 2 : 0;
    const int64_t acc_len = normalize_ola
        ? (params.center ? length + n_fft : hop * (n_frames - 1) + n_fft)
        : (params.center ? signal_length + n_fft : signal_length);
    const int64_t out_len = normalize_ola ? length : signal_length;

    g_fft.ensure(n_fft);
    g_ws.acc.assign(static_cast<size_t>(acc_len), 0.f);
    if (normalize_ola) {
        prepare_win_sq(win, n_fft);
        g_ws.wsum.assign(static_cast<size_t>(acc_len), 0.f);
    }

    for (int64_t b = 0; b < batch; ++b) {
        std::fill(g_ws.acc.begin(), g_ws.acc.end(), 0.f);
        if (normalize_ola) {
            std::fill(g_ws.wsum.begin(), g_ws.wsum.end(), 0.f);
        }

        IrfftOlaChunkCtx base{
            g_fft.setup,
            static_cast<vDSP_Length>(g_fft.log2n),
            n_fft,
            n_freq,
            n_frames,
            hop,
            acc_len,
            b,
            0,
            0,
            spec_re,
            spec_im,
            spec_cplx,
            win.data(),
            normalize_ola ? g_ws.win_sq.data() : nullptr,
            g_ws.acc.data(),
            normalize_ola ? g_ws.wsum.data() : nullptr,
        };
        irfft_ola_parallel(base);

        if (normalize_ola) {
            for (int64_t i = 0; i < acc_len; ++i) {
                const float denom = std::max(g_ws.wsum[static_cast<size_t>(i)], 1e-8f);
                g_ws.acc[static_cast<size_t>(i)] /= denom;
            }
        }

        float* out = waveform.data() + b * out_len;
        if (params.center) {
            std::memcpy(
                out,
                g_ws.acc.data() + pad,
                static_cast<size_t>(out_len) * sizeof(float));
        } else if (normalize_ola) {
            const int64_t n = std::min(out_len, acc_len);
            std::memcpy(out, g_ws.acc.data(), static_cast<size_t>(n) * sizeof(float));
        } else {
            std::memcpy(out, g_ws.acc.data(), static_cast<size_t>(out_len) * sizeof(float));
        }
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

    for (int64_t b = 0; b < batch; ++b) {
        const float* x = waveform + b * signal_length;
        if (params.center) {
            reflect_pad_1d(x, signal_length, pad, g_ws.padded);
        } else {
            g_ws.padded.assign(x, x + signal_length);
        }
        const int64_t padded_len = static_cast<int64_t>(g_ws.padded.size());

        StftForwardFusedCtx ctx{
            g_fft.setup,
            static_cast<vDSP_Length>(g_fft.log2n),
            n_fft,
            n_freq,
            n_frames,
            hop,
            padded_len,
            b,
            g_ws.padded.data(),
            win.data(),
            spec_real.data(),
            spec_imag.data(),
        };
        stft_forward_fft_fused(ctx);
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
    const int64_t n_frames = stft_num_frames(signal_length, params);

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);
    grad_waveform.assign(static_cast<size_t>(batch * signal_length), 0.f);

    vdsp_waveform_from_spec_planes(
        grad_spec_real,
        grad_spec_imag,
        nullptr,
        batch,
        n_frames,
        signal_length,
        signal_length,
        params,
        win,
        false,
        grad_waveform);
}

void stft_backward_vdsp_cplx(
    const std::complex<float>* grad_spec,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& grad_waveform) {
    const int64_t n_fft = params.n_fft;
    const int64_t n_frames = stft_num_frames(signal_length, params);

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);
    grad_waveform.assign(static_cast<size_t>(batch * signal_length), 0.f);

    vdsp_waveform_from_spec_planes(
        nullptr,
        nullptr,
        grad_spec,
        batch,
        n_frames,
        signal_length,
        signal_length,
        params,
        win,
        false,
        grad_waveform);
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

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);
    waveform.assign(static_cast<size_t>(batch * length), 0.f);

    vdsp_waveform_from_spec_planes(
        spec_real,
        spec_imag,
        nullptr,
        batch,
        n_frames,
        length,
        length,
        params,
        win,
        true,
        waveform);
}

void istft_forward_vdsp_cplx(
    const std::complex<float>* spec,
    int64_t batch,
    int64_t n_frames,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t length,
    std::vector<float>& waveform) {
    const int64_t n_fft = params.n_fft;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);
    waveform.assign(static_cast<size_t>(batch * length), 0.f);

    vdsp_waveform_from_spec_planes(
        nullptr,
        nullptr,
        spec,
        batch,
        n_frames,
        length,
        length,
        params,
        win,
        true,
        waveform);
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
