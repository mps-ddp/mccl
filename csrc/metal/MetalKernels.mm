#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

#include "metal/MetalKernels.hpp"
#include "metal/MPSInterop.hpp"
#include "runtime/MemoryPool.hpp"
#include "metal/AccelerateOps.hpp"
#include "metal/EventSync.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"
#include "common/TensorChecks.hpp"
#include <c10d/Types.hpp>

namespace mccl {

namespace {

struct KernelCache {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> accumulate_f32 = nil;
    id<MTLComputePipelineState> accumulate_f16 = nil;
    id<MTLComputePipelineState> accumulate_bf16 = nil;
    id<MTLComputePipelineState> scale_f32 = nil;
    id<MTLComputePipelineState> scale_f16 = nil;
    id<MTLComputePipelineState> scale_bf16 = nil;
    id<MTLComputePipelineState> accum_scale_f32 = nil;
    id<MTLComputePipelineState> accum_scale_f16 = nil;
    id<MTLComputePipelineState> accum_scale_bf16 = nil;
    id<MTLComputePipelineState> min_f32 = nil;
    id<MTLComputePipelineState> min_f16 = nil;
    id<MTLComputePipelineState> min_bf16 = nil;
    id<MTLComputePipelineState> max_f32 = nil;
    id<MTLComputePipelineState> max_f16 = nil;
    id<MTLComputePipelineState> max_bf16 = nil;
    id<MTLComputePipelineState> product_f32 = nil;
    id<MTLComputePipelineState> product_f16 = nil;
    id<MTLComputePipelineState> product_bf16 = nil;
    id<MTLComputePipelineState> accumulate_i32 = nil;
    id<MTLComputePipelineState> accumulate_i64 = nil;
    id<MTLComputePipelineState> accumulate_u8 = nil;
    id<MTLComputePipelineState> min_i32 = nil;
    id<MTLComputePipelineState> max_i32 = nil;
    id<MTLComputePipelineState> product_i32 = nil;
    id<MTLComputePipelineState> min_i64 = nil;
    id<MTLComputePipelineState> max_i64 = nil;
    id<MTLComputePipelineState> product_i64 = nil;
    id<MTLComputePipelineState> min_u8 = nil;
    id<MTLComputePipelineState> max_u8 = nil;
    id<MTLComputePipelineState> product_u8 = nil;
    id<MTLComputePipelineState> scale_i32 = nil;
    id<MTLComputePipelineState> scale_i64 = nil;
    id<MTLComputePipelineState> scale_u8 = nil;

    std::atomic<bool> initialized{false};
};

KernelCache& cache() {
    static KernelCache c;
    return c;
}

/// Serializes MCCL Metal command buffer / encoder use across threads (reduce_engine,
/// Python test hooks, etc.). Recursive: metal_reduce_op calls metal_accumulate_chunk.
static std::recursive_mutex g_metal_kernel_mutex;

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev,
                                           id<MTLLibrary> lib,
                                           NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    MCCL_CHECK(fn != nil,
               "Metal function not found: " +
               std::string([name UTF8String]));

    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    MCCL_CHECK(pso != nil,
               "Pipeline creation failed for " +
               std::string([name UTF8String]) + ": " +
               std::string(err ? [[err localizedDescription] UTF8String] : "unknown error"));

    MCCL_DEBUG("Pipeline %s: threadExecutionWidth=%lu maxThreads=%lu",
               [name UTF8String],
               (unsigned long)pso.threadExecutionWidth,
               (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    return pso;
}

id<MTLComputePipelineState> make_pipeline_if_present(id<MTLDevice> dev,
                                                     id<MTLLibrary> lib,
                                                     NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        MCCL_DEBUG("Optional Metal function unavailable: %s", [name UTF8String]);
        return nil;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    MCCL_CHECK(pso != nil,
               "Pipeline creation failed for " +
               std::string([name UTF8String]) + ": " +
               std::string(err ? [[err localizedDescription] UTF8String] : "unknown error"));
    return pso;
}

/// Compute dispatch grid for vectorized kernels.
/// Kernels process 8 elements per thread (two float4/half4/bfloat4 vectors).
/// Returns {total_threads, threadgroup_size} sized to SIMD width.
struct DispatchParams {
    NSUInteger grid_width;
    NSUInteger threadgroup_width;
};

DispatchParams compute_dispatch(id<MTLComputePipelineState> pso, uint64_t element_count) {
    NSUInteger total_threads = (element_count + 7) / 8;

    NSUInteger simd = pso.threadExecutionWidth;
    NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger tg = (max_tg / simd) * simd;
    if (tg == 0) tg = simd;

    return {total_threads, tg};
}

bool metal_fast_math_enabled() {
    const char* v = std::getenv("MCCL_FAST_MATH");
    if (!v) return true;
    std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE");
}

uint32_t small_gpu_threshold() {
    static const uint32_t cached = []() -> uint32_t {
        const char* v = std::getenv("MCCL_GPU_THRESHOLD");
        if (!v) return 4096;
        long long parsed = std::atoll(v);
        return static_cast<uint32_t>(std::max<long long>(0, parsed));
    }();
    return cached;
}

bool is_vector_aligned(const MPSBufferView& view, at::ScalarType dtype) {
    size_t alignment = 1;
    switch (dtype) {
        case at::kFloat:
            alignment = 16;
            break;
        case at::kHalf:
        case at::kBFloat16:
            alignment = 8;
            break;
        case at::kInt:
        case at::kLong:
            alignment = 16;
            break;
        case at::kBool:
            alignment = 4;
            break;
        default:
            break;
    }
    return (view.byte_offset % alignment) == 0;
}

bool binary_vector_aligned(const MPSBufferView& dst_view, const MPSBufferView& src_view,
                           at::ScalarType dtype) {
    return is_vector_aligned(dst_view, dtype) && is_vector_aligned(src_view, dtype);
}

bool should_use_small_cpu_path(const at::Tensor& tensor) {
    static bool cpu_reduce = [] {
        auto* v = std::getenv("MCCL_FP32_CPU_REDUCE");
        if (!v) return false;
        std::string s(v);
        return s == "1" || s == "true" || s == "on" || s == "yes";
    }();
    if (!cpu_reduce) return false;
    auto dtype = tensor.scalar_type();
    return (dtype == at::kHalf || dtype == at::kBFloat16) &&
           tensor.numel() <= small_gpu_threshold() &&
           tensor_cpu_accessible(tensor);
}

bool should_use_small_cpu_binary_path(const at::Tensor& dst, const at::Tensor& src) {
    return should_use_small_cpu_path(dst) && tensor_cpu_accessible(src);
}

id<MTLCommandBuffer> acquire_command_buffer(KernelCache& c, const char* label) {
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    cmd.label = @(label);
    return cmd;
}

void finish_command_buffer(id<MTLCommandBuffer> cmd) {
    [cmd commit];
}

/// Unified fp32: seed dst from GPU, reduce with staged src, blit back.
/// Metal GPU binary ops read stale bytes on Apple Silicon UMA after wire recv.
bool unified_fp32_binary_reduce(const at::Tensor& dst, const at::Tensor& src,
                                c10d::ReduceOp::RedOpType op) {
    if (dst.scalar_type() != at::kFloat) return false;
    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);
    if (!dst_view.cpu_accessible || !dst_view.cpu_ptr ||
        !src_view.cpu_accessible) {
        return false;
    }
    StagingBuffer dst_staged = stage_for_send(dst);
    memcpy(dst_view.cpu_ptr, dst_staged.data, dst_view.nbytes);
    StagingBuffer src_staged = stage_for_send(src);
    cpu_reduce_op(static_cast<float*>(dst_view.cpu_ptr),
                  static_cast<const float*>(src_staged.data),
                  dst.numel(), op);
    blit_buffer_to_tensor(dst_view.cpu_ptr, dst);
    mps_stream_sync_after_cpu_mps_buffer_write();
    return true;
}

void cpu_small_reduce(const at::Tensor& dst, const at::Tensor& src,
                      c10d::ReduceOp::RedOpType op) {
    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);
    MCCL_CHECK(dst_view.cpu_accessible && src_view.cpu_accessible,
               "Small CPU reduce path requires shared MPS storage");

    int64_t count = dst.numel();
    auto dtype = dst.scalar_type();
    if (dtype == at::kFloat) {
        cpu_reduce_op(static_cast<float*>(dst_view.cpu_ptr),
                      static_cast<const float*>(src_view.cpu_ptr), count, op);
    } else if (dtype == at::kHalf) {
        cpu_reduce_op_half(static_cast<c10::Half*>(dst_view.cpu_ptr),
                           static_cast<const c10::Half*>(src_view.cpu_ptr),
                           count, op);
    } else if (dtype == at::kBFloat16) {
        cpu_reduce_op_bf16(static_cast<c10::BFloat16*>(dst_view.cpu_ptr),
                           static_cast<const c10::BFloat16*>(src_view.cpu_ptr),
                           count, op);
    } else if (dtype == at::kBool || dtype == at::kInt || dtype == at::kLong) {
        cpu_reduce_op_integral(dst_view.cpu_ptr, src_view.cpu_ptr, count, dtype, op);
    } else {
        cpu_reduce_op_bf16(static_cast<c10::BFloat16*>(dst_view.cpu_ptr),
                           static_cast<const c10::BFloat16*>(src_view.cpu_ptr),
                           count, op);
    }
}

void cpu_small_scale_inplace(const at::Tensor& buf, float scale) {
    MPSBufferView view = extract_mps_buffer(buf);
    MCCL_CHECK(view.cpu_accessible, "Small CPU scale path requires shared MPS storage");

    int64_t count = buf.numel();
    if (buf.scalar_type() == at::kHalf) {
        cpu_scale_inplace_half(static_cast<c10::Half*>(view.cpu_ptr), count, scale);
    } else {
        cpu_scale_inplace_bf16(static_cast<c10::BFloat16*>(view.cpu_ptr), count, scale);
    }
}

void cpu_small_accumulate_and_scale(const at::Tensor& dst, const at::Tensor& src,
                                    float scale) {
    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);
    MCCL_CHECK(dst_view.cpu_accessible && src_view.cpu_accessible,
               "Small CPU fused path requires shared MPS storage");

    int64_t count = dst.numel();
    if (dst.scalar_type() == at::kHalf) {
        cpu_accumulate_and_scale_half(static_cast<c10::Half*>(dst_view.cpu_ptr),
                                      static_cast<const c10::Half*>(src_view.cpu_ptr),
                                      count, scale);
    } else {
        cpu_accumulate_and_scale_bf16(static_cast<c10::BFloat16*>(dst_view.cpu_ptr),
                                      static_cast<const c10::BFloat16*>(src_view.cpu_ptr),
                                      count, scale);
    }
}

uint32_t safe_numel(const at::Tensor& t) {
    int64_t n = t.numel();
    MCCL_CHECK(n <= static_cast<int64_t>(UINT32_MAX),
               "Tensor too large for Metal kernels (" + std::to_string(n) +
               " elements, max " + std::to_string(UINT32_MAX) +
               "). Use CPU reduction path for this tensor.");
    return static_cast<uint32_t>(n);
}

id<MTLComputePipelineState> select_accumulate_pipeline(KernelCache& c, at::ScalarType dtype) {
    if (dtype == at::kFloat) return c.accumulate_f32;
    if (dtype == at::kHalf) return c.accumulate_f16;
    if (dtype == at::kBFloat16) return c.accumulate_bf16;
    if (dtype == at::kBool) return c.accumulate_u8;
    if (dtype == at::kInt) return c.accumulate_i32;
    if (dtype == at::kLong) return c.accumulate_i64;
    return nil;
}

id<MTLComputePipelineState> select_integral_binary_pipeline(
    KernelCache& c, at::ScalarType dtype, c10d::ReduceOp::RedOpType op) {
    const bool is_u8 = (dtype == at::kBool);
    const bool is_i32 = (dtype == at::kInt);
    const bool is_i64 = (dtype == at::kLong);
    if (!is_u8 && !is_i32 && !is_i64) return nil;
    switch (op) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
            if (is_u8) return c.accumulate_u8;
            if (is_i64) return c.accumulate_i64;
            return c.accumulate_i32;
        case c10d::ReduceOp::MIN:
            if (is_u8) return c.min_u8;
            if (is_i64) return c.min_i64;
            return c.min_i32;
        case c10d::ReduceOp::MAX:
            if (is_u8) return c.max_u8;
            if (is_i64) return c.max_i64;
            return c.max_i32;
        case c10d::ReduceOp::PRODUCT:
            if (is_u8) return c.product_u8;
            if (is_i64) return c.product_i64;
            return c.product_i32;
        default:
            return nil;
    }
}

id<MTLComputePipelineState> select_scale_pipeline(KernelCache& c, at::ScalarType dtype) {
    if (dtype == at::kFloat) return c.scale_f32;
    if (dtype == at::kHalf) return c.scale_f16;
    if (dtype == at::kBFloat16) return c.scale_bf16;
    if (dtype == at::kBool) return c.scale_u8;
    if (dtype == at::kInt) return c.scale_i32;
    if (dtype == at::kLong) return c.scale_i64;
    return nil;
}

id<MTLComputePipelineState> select_accum_scale_pipeline(KernelCache& c, at::ScalarType dtype) {
    if (dtype == at::kFloat) return c.accum_scale_f32;
    if (dtype == at::kHalf) return c.accum_scale_f16;
    if (dtype == at::kBFloat16) return c.accum_scale_bf16;
    return nil;
}

} // anonymous namespace


void metal_kernels_init() {
    KernelCache& c = cache();
    if (c.initialized.load(std::memory_order_acquire)) return;

    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    if (c.initialized.load(std::memory_order_acquire)) return;

    @autoreleasepool {
        c.device = (__bridge id<MTLDevice>)get_mtl_device();
        c.queue  = (__bridge id<MTLCommandQueue>)get_mccl_command_queue();

        NSError* err = nil;
        NSFileManager* fm = [NSFileManager defaultManager];

        // Search for precompiled metallib in priority order:
        // 1. MCCL_SHADER_PATH env var (if it points to a .metallib)
        // 2. Next to this shared library (where setup.py installs it)
        // 3. Main bundle (for app embedding)
        // 4. Fall back to runtime compilation from .metal source

        NSMutableArray* metallib_search = [NSMutableArray array];

        const char* env_path = std::getenv("MCCL_SHADER_PATH");
        if (env_path) {
            NSString* envStr = @(env_path);
            if ([envStr hasSuffix:@".metallib"]) {
                [metallib_search addObject:envStr];
            }
        }

        // Directory containing this .so (wheel / site-packages layout)
        NSString* soDir = nil;
        Dl_info dl_info;
        if (dladdr((void*)metal_kernels_init, &dl_info) && dl_info.dli_fname) {
            NSString* soPath = @(dl_info.dli_fname);
            soDir = [soPath stringByDeletingLastPathComponent];
            [metallib_search addObject:[soDir stringByAppendingPathComponent:@"mccl_shaders.metallib"]];
        }

        NSString* bundlePath = [[NSBundle mainBundle] pathForResource:@"mccl_shaders"
                                                               ofType:@"metallib"];
        if (bundlePath) [metallib_search addObject:bundlePath];

        for (NSString* libPath in metallib_search) {
            if ([fm fileExistsAtPath:libPath]) {
                NSURL* url = [NSURL fileURLWithPath:libPath];
                c.library = [c.device newLibraryWithURL:url error:&err];
                if (c.library) {
                    MCCL_INFO("Loaded precompiled metallib: %s",
                              [libPath UTF8String]);
                    break;
                }
            }
        }

        if (!c.library) {
            MCCL_INFO("Compiling Metal shaders at runtime (no precompiled metallib found)");

            NSString* srcPath = nil;

            if (env_path) {
                NSString* envStr = @(env_path);
                if ([envStr hasSuffix:@".metal"] && [fm fileExistsAtPath:envStr]) {
                    srcPath = envStr;
                }
            }

            if (!srcPath) {
                NSArray* searchPaths = @[
                    @"csrc/metal/shaders.metal",
                    @"../csrc/metal/shaders.metal",
                    @"../../csrc/metal/shaders.metal",
                ];
                for (NSString* p in searchPaths) {
                    if ([fm fileExistsAtPath:p]) {
                        srcPath = p;
                        break;
                    }
                }
            }

            // Pip wheel: setup.py installs shaders.metal next to _C.so
            if (!srcPath && soDir) {
                NSString* besideSo = [soDir stringByAppendingPathComponent:@"shaders.metal"];
                if ([fm fileExistsAtPath:besideSo]) {
                    srcPath = besideSo;
                }
            }

            MCCL_CHECK(srcPath != nil,
                "Cannot find shaders.metal or mccl_shaders.metallib. "
                "Set MCCL_SHADER_PATH env var.");

            NSString* src = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&err];
            MCCL_CHECK(src != nil, "Failed to read shaders.metal: " +
                       std::string([[err localizedDescription] UTF8String]));

            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = metal_fast_math_enabled();

            if (@available(macOS 15.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_1;
            } else if (@available(macOS 14.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_0;
            } else {
                opts.languageVersion = MTLLanguageVersion2_4;
            }

            c.library = [c.device newLibraryWithSource:src options:opts error:&err];
            MCCL_CHECK(c.library != nil,
                "Metal shader compilation failed: " +
                std::string([[err localizedDescription] UTF8String]));
        }

        c.accumulate_f32  = make_pipeline(c.device, c.library, @"accumulate_chunk_f32");
        c.accumulate_f16  = make_pipeline(c.device, c.library, @"accumulate_chunk_f16");
        c.accumulate_bf16 = make_pipeline_if_present(c.device, c.library, @"accumulate_chunk_bf16");
        c.scale_f32       = make_pipeline(c.device, c.library, @"scale_inplace_f32");
        c.scale_f16       = make_pipeline(c.device, c.library, @"scale_inplace_f16");
        c.scale_bf16      = make_pipeline_if_present(c.device, c.library, @"scale_inplace_bf16");
        c.accum_scale_f32 = make_pipeline(c.device, c.library, @"accumulate_scale_f32");
        c.accum_scale_f16 = make_pipeline(c.device, c.library, @"accumulate_scale_f16");
        c.accum_scale_bf16 = make_pipeline_if_present(c.device, c.library, @"accumulate_scale_bf16");
        c.min_f32         = make_pipeline(c.device, c.library, @"elementwise_min_f32");
        c.min_f16         = make_pipeline(c.device, c.library, @"elementwise_min_f16");
        c.min_bf16        = make_pipeline_if_present(c.device, c.library, @"elementwise_min_bf16");
        c.max_f32         = make_pipeline(c.device, c.library, @"elementwise_max_f32");
        c.max_f16         = make_pipeline(c.device, c.library, @"elementwise_max_f16");
        c.max_bf16        = make_pipeline_if_present(c.device, c.library, @"elementwise_max_bf16");
        c.product_f32     = make_pipeline(c.device, c.library, @"elementwise_product_f32");
        c.product_f16     = make_pipeline(c.device, c.library, @"elementwise_product_f16");
        c.product_bf16    = make_pipeline_if_present(c.device, c.library, @"elementwise_product_bf16");
        c.accumulate_i32  = make_pipeline(c.device, c.library, @"accumulate_chunk_i32");
        c.accumulate_i64  = make_pipeline(c.device, c.library, @"accumulate_chunk_i64");
        c.accumulate_u8   = make_pipeline(c.device, c.library, @"accumulate_chunk_u8");
        c.min_i32         = make_pipeline(c.device, c.library, @"elementwise_min_i32");
        c.max_i32         = make_pipeline(c.device, c.library, @"elementwise_max_i32");
        c.product_i32     = make_pipeline(c.device, c.library, @"elementwise_product_i32");
        c.min_i64         = make_pipeline(c.device, c.library, @"elementwise_min_i64");
        c.max_i64         = make_pipeline(c.device, c.library, @"elementwise_max_i64");
        c.product_i64     = make_pipeline(c.device, c.library, @"elementwise_product_i64");
        c.min_u8          = make_pipeline(c.device, c.library, @"elementwise_min_u8");
        c.max_u8          = make_pipeline(c.device, c.library, @"elementwise_max_u8");
        c.product_u8      = make_pipeline(c.device, c.library, @"elementwise_product_u8");
        c.scale_i32       = make_pipeline(c.device, c.library, @"scale_inplace_i32");
        c.scale_i64       = make_pipeline(c.device, c.library, @"scale_inplace_i64");
        c.scale_u8        = make_pipeline(c.device, c.library, @"scale_inplace_u8");

        c.initialized.store(true, std::memory_order_release);
        MCCL_INFO("Metal kernel cache initialized (fastMath=%s)",
                  metal_fast_math_enabled() ? "on" : "off");
    }
}

void* get_mccl_mtl_library() {
    metal_kernels_init();
    return (__bridge void*)cache().library;
}

void metal_accumulate_chunk(const at::Tensor& dst, const at::Tensor& src) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");
    check_same_shape_dtype(dst, src);

    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);

    // Unified MPS memory: GPU binary kernels read stale bytes after recv blit.
    if (unified_fp32_binary_reduce(dst, src, c10d::ReduceOp::SUM)) {
        return;
    }

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)dst_view.mtl_buffer;
    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)src_view.mtl_buffer;

    uint32_t count = safe_numel(dst);
    auto pso = select_accumulate_pipeline(c, dst.scalar_type());
    MCCL_CHECK(pso != nil || should_use_small_cpu_binary_path(dst, src),
               "No Metal accumulate pipeline for dtype " +
               std::string(at::toString(dst.scalar_type())));
    if (should_use_small_cpu_binary_path(dst, src)) {
        cpu_small_reduce(dst, src, c10d::ReduceOp::SUM);
        return;
    }
    auto dp = compute_dispatch(pso, count);
    bool aligned = binary_vector_aligned(dst_view, src_view, dst.scalar_type());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = acquire_command_buffer(c, "mccl_accumulate");

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:dst_buf offset:dst_view.byte_offset atIndex:0];
        [enc setBuffer:src_buf offset:src_view.byte_offset atIndex:1];
        [enc setBytes:&count length:sizeof(count) atIndex:2];
        [enc setBytes:&aligned length:sizeof(aligned) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(dp.grid_width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(dp.threadgroup_width, 1, 1)];
        [enc endEncoding];

        finish_command_buffer(cmd);
    }
}


void metal_scale_inplace(const at::Tensor& buf, double scale) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");

    MPSBufferView view = extract_mps_buffer(buf);
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;

    uint32_t count = safe_numel(buf);
    auto pso = select_scale_pipeline(c, buf.scalar_type());
    if (should_use_small_cpu_path(buf)) {
        cpu_small_scale_inplace(buf, static_cast<float>(scale));
        return;
    }
    MCCL_CHECK(pso != nil,
               "No Metal scale pipeline for dtype " +
               std::string(at::toString(buf.scalar_type())));
    auto dp = compute_dispatch(pso, count);
    bool aligned = is_vector_aligned(view, buf.scalar_type());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = acquire_command_buffer(c, "mccl_scale");

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:mtl_buf offset:view.byte_offset atIndex:0];

        if (buf.scalar_type() == at::kHalf) {
            __fp16 s = static_cast<__fp16>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:1];
        } else if (buf.scalar_type() == at::kBFloat16) {
            c10::BFloat16 s = static_cast<float>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:1];
        } else {
            float s = static_cast<float>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:1];
        }
        [enc setBytes:&count length:sizeof(count) atIndex:2];
        [enc setBytes:&aligned length:sizeof(aligned) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(dp.grid_width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(dp.threadgroup_width, 1, 1)];
        [enc endEncoding];

        finish_command_buffer(cmd);
    }
}


void metal_accumulate_and_scale(const at::Tensor& dst, const at::Tensor& src,
                                double scale) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");
    check_same_shape_dtype(dst, src);

    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)dst_view.mtl_buffer;
    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)src_view.mtl_buffer;

    uint32_t count = safe_numel(dst);
    auto dtype = dst.scalar_type();
    if (dtype == at::kBool || dtype == at::kInt || dtype == at::kLong) {
        // Integral AVG is uncommon and does not justify three fused kernels.
        // The recursive kernel lock allows this exact two-command fallback.
        metal_accumulate_chunk(dst, src);
        metal_scale_inplace(dst, scale);
        return;
    }
    if (should_use_small_cpu_binary_path(dst, src)) {
        cpu_small_accumulate_and_scale(dst, src, static_cast<float>(scale));
        return;
    }
    auto pso = select_accum_scale_pipeline(c, dst.scalar_type());
    MCCL_CHECK(pso != nil,
               "No Metal fused pipeline for dtype " +
               std::string(at::toString(dst.scalar_type())));
    auto dp = compute_dispatch(pso, count);
    bool aligned = binary_vector_aligned(dst_view, src_view, dst.scalar_type());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = acquire_command_buffer(c, "mccl_accumulate_scale");

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:dst_buf offset:dst_view.byte_offset atIndex:0];
        [enc setBuffer:src_buf offset:src_view.byte_offset atIndex:1];

        if (dst.scalar_type() == at::kHalf) {
            __fp16 s = static_cast<__fp16>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:2];
        } else if (dst.scalar_type() == at::kBFloat16) {
            c10::BFloat16 s = static_cast<float>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:2];
        } else {
            float s = static_cast<float>(scale);
            [enc setBytes:&s length:sizeof(s) atIndex:2];
        }
        [enc setBytes:&count length:sizeof(count) atIndex:3];
        [enc setBytes:&aligned length:sizeof(aligned) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(dp.grid_width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(dp.threadgroup_width, 1, 1)];
        [enc endEncoding];

        finish_command_buffer(cmd);
    }
}


namespace {

void dispatch_binary_op(id<MTLComputePipelineState> pso,
                        const at::Tensor& dst, const at::Tensor& src,
                        const char* label) {
    KernelCache& c = cache();
    MPSBufferView dst_view = extract_mps_buffer(dst);
    MPSBufferView src_view = extract_mps_buffer(src);

    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)dst_view.mtl_buffer;
    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)src_view.mtl_buffer;

    uint32_t count = safe_numel(dst);
    auto dp = compute_dispatch(pso, count);
    bool aligned = binary_vector_aligned(dst_view, src_view, dst.scalar_type());

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = acquire_command_buffer(c, label);

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:dst_buf offset:dst_view.byte_offset atIndex:0];
        [enc setBuffer:src_buf offset:src_view.byte_offset atIndex:1];
        [enc setBytes:&count length:sizeof(count) atIndex:2];
        [enc setBytes:&aligned length:sizeof(aligned) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(dp.grid_width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(dp.threadgroup_width, 1, 1)];
        [enc endEncoding];
        finish_command_buffer(cmd);
    }
}

} // anonymous namespace

void metal_elementwise_min(const at::Tensor& dst, const at::Tensor& src) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");
    check_same_shape_dtype(dst, src);
    if (unified_fp32_binary_reduce(dst, src, c10d::ReduceOp::MIN)) return;
    if (should_use_small_cpu_binary_path(dst, src)) {
        cpu_small_reduce(dst, src, c10d::ReduceOp::MIN);
        return;
    }
    id<MTLComputePipelineState> pso = select_integral_binary_pipeline(
        c, dst.scalar_type(), c10d::ReduceOp::MIN);
    if (pso == nil) {
        pso = dst.scalar_type() == at::kFloat ? c.min_f32 :
              dst.scalar_type() == at::kHalf ? c.min_f16 :
              c.min_bf16;
    }
    MCCL_CHECK(pso != nil,
               "No Metal min pipeline for dtype " +
               std::string(at::toString(dst.scalar_type())));
    dispatch_binary_op(pso, dst, src, "mccl_min");
}

void metal_elementwise_max(const at::Tensor& dst, const at::Tensor& src) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");
    check_same_shape_dtype(dst, src);
    if (unified_fp32_binary_reduce(dst, src, c10d::ReduceOp::MAX)) return;
    if (should_use_small_cpu_binary_path(dst, src)) {
        cpu_small_reduce(dst, src, c10d::ReduceOp::MAX);
        return;
    }
    id<MTLComputePipelineState> pso = select_integral_binary_pipeline(
        c, dst.scalar_type(), c10d::ReduceOp::MAX);
    if (pso == nil) {
        pso = dst.scalar_type() == at::kFloat ? c.max_f32 :
              dst.scalar_type() == at::kHalf ? c.max_f16 :
              c.max_bf16;
    }
    MCCL_CHECK(pso != nil,
               "No Metal max pipeline for dtype " +
               std::string(at::toString(dst.scalar_type())));
    dispatch_binary_op(pso, dst, src, "mccl_max");
}

void metal_elementwise_product(const at::Tensor& dst, const at::Tensor& src) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    KernelCache& c = cache();
    MCCL_CHECK(c.initialized, "metal_kernels_init() not called");
    check_same_shape_dtype(dst, src);
    if (unified_fp32_binary_reduce(dst, src, c10d::ReduceOp::PRODUCT)) return;
    if (should_use_small_cpu_binary_path(dst, src)) {
        cpu_small_reduce(dst, src, c10d::ReduceOp::PRODUCT);
        return;
    }
    id<MTLComputePipelineState> pso = select_integral_binary_pipeline(
        c, dst.scalar_type(), c10d::ReduceOp::PRODUCT);
    if (pso == nil) {
        pso = dst.scalar_type() == at::kFloat ? c.product_f32 :
              dst.scalar_type() == at::kHalf ? c.product_f16 :
              c.product_bf16;
    }
    MCCL_CHECK(pso != nil,
               "No Metal product pipeline for dtype " +
               std::string(at::toString(dst.scalar_type())));
    dispatch_binary_op(pso, dst, src, "mccl_product");
}

void metal_reduce_op(const at::Tensor& dst, const at::Tensor& src,
                     c10d::ReduceOp::RedOpType op) {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    switch (op) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
            metal_accumulate_chunk(dst, src);
            break;
        case c10d::ReduceOp::MIN:
            metal_elementwise_min(dst, src);
            break;
        case c10d::ReduceOp::MAX:
            metal_elementwise_max(dst, src);
            break;
        case c10d::ReduceOp::PRODUCT:
            metal_elementwise_product(dst, src);
            break;
        default:
            throw MCCLError("Unsupported ReduceOp: " +
                            std::to_string(static_cast<int>(op)));
    }
}

uint64_t metal_reduce_op_fenced(const at::Tensor& dst, const at::Tensor& src,
                                c10d::ReduceOp::RedOpType op) {
    metal_reduce_op(dst, src, op);
    if (event_sync_available()) {
        uint64_t v = next_fence_event_value();
        signal_mccl_fence_gpu(v);
        return v;
    }
    metal_sync_queue_only();
    return 0;
}

void metal_sync() {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    mps_sync();
}

void metal_sync_queue_only() {
    std::lock_guard<std::recursive_mutex> lock(g_metal_kernel_mutex);
    mccl_queue_drain();
}

} // namespace mccl
