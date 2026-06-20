#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "transforms/StftMetalKernels.hpp"
#include "transforms/WindowUtils.hpp"
#include "metal/MetalKernels.hpp"
#include "metal/MPSInterop.hpp"
#include "common/Errors.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

namespace mccl {

namespace {

struct StftMetalCache {
    id<MTLComputePipelineState> rfft = nil;
    id<MTLComputePipelineState> irfft = nil;
    std::atomic<bool> initialized{false};
};

StftMetalCache& stft_cache() {
    static StftMetalCache c;
    return c;
}

id<MTLComputePipelineState> load_pipeline(NSString* name) {
    metal_kernels_init();
    id<MTLDevice> dev = (__bridge id<MTLDevice>)get_mtl_device();
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)get_mccl_mtl_library();
    MCCL_CHECK(lib != nil, "StftMetal: MCCL shader library not initialized");

    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    MCCL_CHECK(fn != nil, std::string("StftMetal: missing kernel ") + [name UTF8String]);
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    MCCL_CHECK(pso != nil, std::string("StftMetal pipeline failed: ") +
                                (err ? [[err localizedDescription] UTF8String] : "unknown"));
    return pso;
}

void ensure_stft_pipelines() {
    auto& c = stft_cache();
    if (c.initialized.load()) {
        return;
    }
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (c.initialized.load()) {
        return;
    }
    c.rfft = load_pipeline(@"stft_rfft_frame");
    c.irfft = load_pipeline(@"stft_irfft_frame");
    c.initialized.store(true);
}

uint32_t log2_u32(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) {
        ++l;
    }
    return l;
}

id<MTLBuffer> make_buffer(id<MTLDevice> dev, const void* data, size_t nbytes) {
    id<MTLBuffer> buf = [dev newBufferWithBytes:data length:nbytes options:MTLResourceStorageModeShared];
    MCCL_CHECK(buf != nil, "StftMetal: newBufferWithBytes failed");
    return buf;
}

void dispatch_rfft(
    id<MTLBuffer> padded,
    id<MTLBuffer> window,
    id<MTLBuffer> spec_re,
    id<MTLBuffer> spec_im,
    uint32_t batch,
    uint32_t padded_length,
    uint32_t n_fft,
    uint32_t hop,
    uint32_t n_frames,
    uint32_t n_freq) {
    ensure_stft_pipelines();
    auto& c = stft_cache();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)get_mccl_command_queue();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    cmd.label = @"mccl_stft_rfft";

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.rfft];
    [enc setBuffer:padded offset:0 atIndex:0];
    [enc setBuffer:window offset:0 atIndex:1];
    [enc setBuffer:spec_re offset:0 atIndex:2];
    [enc setBuffer:spec_im offset:0 atIndex:3];

    uint32_t signal_length = padded_length;
    uint32_t log2n = log2_u32(n_fft);
    [enc setBytes:&batch length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&signal_length length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&padded_length length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&n_fft length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&hop length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&n_frames length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&n_freq length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&log2n length:sizeof(uint32_t) atIndex:11];

    MTLSize grid = MTLSizeMake(batch, n_frames, 1);
    NSUInteger tg_w = c.rfft.threadExecutionWidth;
    NSUInteger tg_h = std::max<NSUInteger>(1, c.rfft.maxTotalThreadsPerThreadgroup / tg_w);
    MTLSize tg = MTLSizeMake(tg_w, tg_h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void dispatch_irfft(
    id<MTLBuffer> spec_re,
    id<MTLBuffer> spec_im,
    id<MTLBuffer> window,
    id<MTLBuffer> frames,
    uint32_t batch,
    uint32_t n_fft,
    uint32_t n_frames,
    uint32_t n_freq) {
    ensure_stft_pipelines();
    auto& c = stft_cache();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)get_mccl_command_queue();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    cmd.label = @"mccl_stft_irfft";

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.irfft];
    [enc setBuffer:spec_re offset:0 atIndex:0];
    [enc setBuffer:spec_im offset:0 atIndex:1];
    [enc setBuffer:window offset:0 atIndex:2];
    [enc setBuffer:frames offset:0 atIndex:3];

    uint32_t log2n = log2_u32(n_fft);
    [enc setBytes:&batch length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&n_fft length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&n_frames length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&n_freq length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&log2n length:sizeof(uint32_t) atIndex:8];

    MTLSize grid = MTLSizeMake(batch, n_frames, 1);
    NSUInteger tg_w = c.irfft.threadExecutionWidth;
    NSUInteger tg_h = std::max<NSUInteger>(1, c.irfft.maxTotalThreadsPerThreadgroup / tg_w);
    MTLSize tg = MTLSizeMake(tg_w, tg_h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void overlap_add_frames_cpu(
    float* acc,
    int64_t acc_len,
    const float* frames,
    int64_t n_frames,
    int64_t n_fft,
    int64_t hop) {
    for (int64_t f = 0; f < n_frames; ++f) {
        const float* frame = frames + f * n_fft;
        const int64_t start = f * hop;
        for (int64_t i = 0; i < n_fft; ++i) {
            const int64_t pos = start + i;
            if (pos >= 0 && pos < acc_len) {
                acc[static_cast<size_t>(pos)] += frame[static_cast<size_t>(i)];
            }
        }
    }
}

void overlap_add_window_sq_cpu(
    float* wsum,
    int64_t acc_len,
    const float* window,
    int64_t win_length,
    int64_t n_frames,
    int64_t hop) {
    for (int64_t f = 0; f < n_frames; ++f) {
        const int64_t start = f * hop;
        for (int64_t i = 0; i < win_length; ++i) {
            const int64_t pos = start + i;
            if (pos >= 0 && pos < acc_len) {
                const float w = window[static_cast<size_t>(i)];
                wsum[static_cast<size_t>(pos)] += w * w;
            }
        }
    }
}

} // namespace

void metal_stft_init() { ensure_stft_pipelines(); }

void metal_stft_forward(
    const float* waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    std::vector<float>& spec_real,
    std::vector<float>& spec_imag) {
    MCCL_CHECK(params.n_fft <= 4096, "StftMetal: n_fft > 4096 not supported in shader");
    const int64_t n_fft = params.n_fft;
    const int64_t hop = params.hop_length;
    const int64_t n_freq = n_fft / 2 + 1;
    const int64_t n_frames = stft_num_frames(signal_length, params);
    const int64_t pad = params.center ? n_fft / 2 : 0;
    const int64_t padded_len = params.center ? signal_length + n_fft : signal_length;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    std::vector<float> padded(static_cast<size_t>(batch * padded_len), 0.f);
    for (int64_t b = 0; b < batch; ++b) {
        if (params.center) {
            std::vector<float> one;
            reflect_pad_1d(waveform + b * signal_length, signal_length, pad, one);
            std::memcpy(
                padded.data() + b * padded_len,
                one.data(),
                static_cast<size_t>(padded_len) * sizeof(float));
        } else {
            std::memcpy(
                padded.data() + b * padded_len,
                waveform + b * signal_length,
                static_cast<size_t>(signal_length) * sizeof(float));
        }
    }

    spec_real.assign(static_cast<size_t>(batch * n_freq * n_frames), 0.f);
    spec_imag.assign(static_cast<size_t>(batch * n_freq * n_frames), 0.f);

    id<MTLDevice> dev = (__bridge id<MTLDevice>)get_mtl_device();
    id<MTLBuffer> buf_pad = make_buffer(dev, padded.data(), padded.size() * sizeof(float));
    id<MTLBuffer> buf_win = make_buffer(dev, win.data(), win.size() * sizeof(float));
    id<MTLBuffer> buf_re = make_buffer(dev, spec_real.data(), spec_real.size() * sizeof(float));
    id<MTLBuffer> buf_im = make_buffer(dev, spec_imag.data(), spec_imag.size() * sizeof(float));

    dispatch_rfft(
        buf_pad,
        buf_win,
        buf_re,
        buf_im,
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(padded_len),
        static_cast<uint32_t>(n_fft),
        static_cast<uint32_t>(hop),
        static_cast<uint32_t>(n_frames),
        static_cast<uint32_t>(n_freq));

    std::memcpy(spec_real.data(), buf_re.contents, spec_real.size() * sizeof(float));
    std::memcpy(spec_imag.data(), buf_im.contents, spec_imag.size() * sizeof(float));
}

void metal_stft_backward(
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
    const int64_t n_freq = n_fft / 2 + 1;
    const int64_t n_frames = stft_num_frames(signal_length, params);
    const int64_t pad = params.center ? n_fft / 2 : 0;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    std::vector<float> frames(static_cast<size_t>(batch * n_frames * n_fft), 0.f);
    id<MTLDevice> dev = (__bridge id<MTLDevice>)get_mtl_device();
    id<MTLBuffer> buf_re = make_buffer(
        dev, grad_spec_real, static_cast<size_t>(batch * n_freq * n_frames) * sizeof(float));
    id<MTLBuffer> buf_im = make_buffer(
        dev, grad_spec_imag, static_cast<size_t>(batch * n_freq * n_frames) * sizeof(float));
    id<MTLBuffer> buf_win = make_buffer(dev, win.data(), win.size() * sizeof(float));
    id<MTLBuffer> buf_frames = make_buffer(dev, frames.data(), frames.size() * sizeof(float));

    dispatch_irfft(
        buf_re,
        buf_im,
        buf_win,
        buf_frames,
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(n_fft),
        static_cast<uint32_t>(n_frames),
        static_cast<uint32_t>(n_freq));
    std::memcpy(frames.data(), buf_frames.contents, frames.size() * sizeof(float));

    grad_waveform.assign(static_cast<size_t>(batch * signal_length), 0.f);
    const int64_t padded_len = params.center ? signal_length + n_fft : signal_length;
    std::vector<float> acc(static_cast<size_t>(padded_len), 0.f);

    for (int64_t b = 0; b < batch; ++b) {
        std::fill(acc.begin(), acc.end(), 0.f);
        overlap_add_frames_cpu(
            acc.data(),
            padded_len,
            frames.data() + b * n_frames * n_fft,
            n_frames,
            n_fft,
            hop);
        float* out = grad_waveform.data() + b * signal_length;
        if (params.center) {
            for (int64_t t = 0; t < signal_length; ++t) {
                out[static_cast<size_t>(t)] = acc[static_cast<size_t>(t + pad)];
            }
        } else {
            std::memcpy(out, acc.data(), static_cast<size_t>(signal_length) * sizeof(float));
        }
    }
}

void metal_istft_forward(
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
    const int64_t n_freq = n_fft / 2 + 1;
    const int64_t pad = params.center ? n_fft / 2 : 0;

    std::vector<float> win(static_cast<size_t>(n_fft), 0.f);
    prepare_window(window, win_length, n_fft, win);

    const int64_t out_len = params.center ? length + n_fft : hop * (n_frames - 1) + n_fft;
    std::vector<float> frames(static_cast<size_t>(batch * n_frames * n_fft), 0.f);

    id<MTLDevice> dev = (__bridge id<MTLDevice>)get_mtl_device();
    id<MTLBuffer> buf_re = make_buffer(
        dev, spec_real, static_cast<size_t>(batch * n_freq * n_frames) * sizeof(float));
    id<MTLBuffer> buf_im = make_buffer(
        dev, spec_imag, static_cast<size_t>(batch * n_freq * n_frames) * sizeof(float));
    id<MTLBuffer> buf_win = make_buffer(dev, win.data(), win.size() * sizeof(float));
    id<MTLBuffer> buf_frames = make_buffer(dev, frames.data(), frames.size() * sizeof(float));

    dispatch_irfft(
        buf_re,
        buf_im,
        buf_win,
        buf_frames,
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(n_fft),
        static_cast<uint32_t>(n_frames),
        static_cast<uint32_t>(n_freq));
    std::memcpy(frames.data(), buf_frames.contents, frames.size() * sizeof(float));

    waveform.assign(static_cast<size_t>(batch * length), 0.f);
    std::vector<float> acc(static_cast<size_t>(out_len), 0.f);
    std::vector<float> wsum(static_cast<size_t>(out_len), 0.f);

    for (int64_t b = 0; b < batch; ++b) {
        std::fill(acc.begin(), acc.end(), 0.f);
        std::fill(wsum.begin(), wsum.end(), 0.f);
        overlap_add_frames_cpu(
            acc.data(),
            out_len,
            frames.data() + b * n_frames * n_fft,
            n_frames,
            n_fft,
            hop);
        overlap_add_window_sq_cpu(wsum.data(), out_len, win.data(), n_fft, n_frames, hop);
        for (int64_t i = 0; i < out_len; ++i) {
            const float denom = std::max(wsum[static_cast<size_t>(i)], 1e-8f);
            acc[static_cast<size_t>(i)] /= denom;
        }
        float* out = waveform.data() + b * length;
        if (params.center) {
            for (int64_t t = 0; t < length; ++t) {
                out[static_cast<size_t>(t)] = acc[static_cast<size_t>(t + pad)];
            }
        } else {
            const int64_t n = std::min(length, out_len);
            std::memcpy(out, acc.data(), static_cast<size_t>(n) * sizeof(float));
        }
    }
}

void metal_istft_backward(
    const float* grad_waveform,
    int64_t batch,
    int64_t signal_length,
    const float* window,
    int64_t win_length,
    const StftParams& params,
    int64_t n_frames,
    std::vector<float>& grad_spec_real,
    std::vector<float>& grad_spec_imag) {
    metal_stft_forward(
        grad_waveform, batch, signal_length, window, win_length, params, grad_spec_real, grad_spec_imag);
}

} // namespace mccl
