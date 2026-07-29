#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>

#include "backend/ProcessGroupMCCL.hpp"
#include "backend/Options.hpp"
#include "compression/Compression.hpp"
#include "runtime/Metrics.hpp"
#include "runtime/MemoryPool.hpp"
#include "metal/MetalKernels.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/EventSync.hpp"
#include "metal/AccelerateOps.hpp"
#include "transforms/Stft.hpp"
#include "common/Logging.hpp"
#include "common/Version.hpp"
#include "common/TensorChecks.hpp"
#include "common/Errors.hpp"

#include <mutex>

namespace py = pybind11;

namespace mccl {

static std::mutex g_pg_mu;
static ProcessGroupMCCL* g_active_pg = nullptr;

void set_active_pg(ProcessGroupMCCL* pg) {
    std::lock_guard<std::mutex> lock(g_pg_mu);
    g_active_pg = pg;
}

void clear_active_pg_if(ProcessGroupMCCL* pg) {
    std::lock_guard<std::mutex> lock(g_pg_mu);
    if (g_active_pg == pg) g_active_pg = nullptr;
}

ProcessGroupMCCL* get_active_pg() {
    std::lock_guard<std::mutex> lock(g_pg_mu);
    return g_active_pg;
}

void register_backend_py() {
    MCCL_INFO("Registering MCCL backend v%s (protocol v%d)",
              MCCL_VERSION_STRING, MCCL_PROTOCOL_VERSION);

    py::module_ dist = py::module_::import("torch.distributed");
    py::object Backend = dist.attr("Backend");

    py::object create_fn = py::cpp_function(
        [](const c10::intrusive_ptr<c10d::Store>& store,
           int rank, int world_size,
           std::chrono::milliseconds timeout) -> c10::intrusive_ptr<c10d::Backend> {
            auto pg = createProcessGroupMCCL(store, rank, world_size, timeout);
            set_active_pg(static_cast<ProcessGroupMCCL*>(pg.get()));
            return pg;
        },
        py::arg("store"),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("timeout"),
        py::call_guard<py::gil_scoped_release>()
    );

    py::list devices;
    devices.append("mps");

    Backend.attr("register_backend")(
        "mccl",
        create_fn,
        py::arg("devices") = devices
    );

    MCCL_INFO("MCCL backend registered successfully");
}

} // namespace mccl


PYBIND11_MODULE(_C, m) {
    m.doc() = "MCCL — MPS-native ProcessGroup backend for PyTorch Distributed";

    m.def("_register_backend", &mccl::register_backend_py,
          "Register the MCCL backend with torch.distributed");

    m.def("_get_metrics_summary", []() -> std::optional<mccl::Metrics::Summary> {
        std::lock_guard<std::mutex> lock(mccl::g_pg_mu);
        auto* pg = mccl::g_active_pg;
        if (!pg) return std::nullopt;
        return pg->get_metrics_summary();
    }, "Return a snapshot of collective metrics, or None if no ProcessGroup exists",
       py::call_guard<py::gil_scoped_release>());

    m.def("_log_metrics", []() {
        std::lock_guard<std::mutex> lock(mccl::g_pg_mu);
        auto* pg = mccl::g_active_pg;
        if (pg) pg->log_metrics();
    }, "Dump metrics to the MCCL logger at INFO level",
       py::call_guard<py::gil_scoped_release>());

    m.def("_reset_metrics", []() {
        std::lock_guard<std::mutex> lock(mccl::g_pg_mu);
        auto* pg = mccl::g_active_pg;
        if (pg) pg->reset_metrics();
    }, "Reset all metric counters",
       py::call_guard<py::gil_scoped_release>());

    m.def("_trim_staging_pool", []() {
        mccl::staging_memory_pool().trim();
    }, "Release pooled host staging buffers (safe after epoch barrier)",
       py::call_guard<py::gil_scoped_release>());

    m.def("_staging_pool_bytes", []() {
        return mccl::staging_memory_pool().total_allocated();
    }, "Bytes retained by the MCCL host staging pool",
       py::call_guard<py::gil_scoped_release>());

    m.def("_tensor_cpu_accessible",
          [](const at::Tensor& tensor) {
              return mccl::tensor_cpu_accessible(tensor);
          },
          py::arg("tensor"),
          "True when MTLBuffer storage is Shared (CPU can read/write without blit)");

    m.def("_mps_storage_mode",
          [](const at::Tensor& tensor) {
              return mccl::mps_storage_mode_string(tensor);
          },
          py::arg("tensor"),
          "MTLStorageMode name: cpu|private|shared|managed|none");

    m.def("_collective_send_uses_blit",
          [](const at::Tensor& tensor) {
              return mccl::collective_send_uses_blit(tensor);
          },
          py::arg("tensor"),
          "True when stage_for_send_collective blits instead of reading cpu_ptr");

    m.def("_stage_for_send_uses_blit",
          [](const at::Tensor& tensor) {
              return mccl::stage_for_send_uses_blit(tensor);
          },
          py::arg("tensor"),
          "True when stage_for_send blits private MPS storage to staging pool");

    m.def("_unified_collective_enabled",
          []() { return mccl::unified_collective_enabled(); },
          "True when unified collective is enabled (default on; set MCCL_UNIFIED_COLLECTIVE=0 to disable)");

    m.def("_unstage_from_recv",
          [](const at::Tensor& tensor, const at::Tensor& src, bool cpu_unified_stage) {
              MCCL_CHECK(src.is_cpu() && src.is_contiguous(),
                         "_unstage_from_recv: src must be contiguous CPU tensor");
              MCCL_CHECK(static_cast<size_t>(src.numel()) * src.element_size() ==
                             mccl::tensor_nbytes(tensor),
                         "_unstage_from_recv: src nbytes mismatch");
              mccl::unstage_from_recv(
                  tensor, src.data_ptr(),
                  static_cast<size_t>(src.numel()) * src.element_size(),
                  cpu_unified_stage);
          },
          py::arg("tensor"),
          py::arg("src"),
          py::arg("cpu_unified_stage") = false,
          "Test helper: copy wire bytes into tensor (blit or unified cpu_ptr path)");

    m.def("_ensure_shared_storage",
          [](const at::Tensor& tensor) {
              mccl::mps_stream_sync();
              return mccl::ensure_shared_storage(tensor);
          },
          py::arg("tensor"),
          "Blit private MPS bytes into a new CPU/shared buffer (opt-in copy, not zero-copy)");

    m.def("_stage_for_send_collective_bench",
          [](const at::Tensor& tensor, int iters) {
              mccl::mps_stream_sync();
              auto t0 = std::chrono::steady_clock::now();
              const int n = std::max(1, iters);
              for (int i = 0; i < n; ++i) {
                  mccl::StagingBuffer staged = mccl::stage_for_send_collective(tensor);
                  (void)staged;
              }
              auto t1 = std::chrono::steady_clock::now();
              return std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     static_cast<double>(n);
          },
          py::arg("tensor"),
          py::arg("iters") = 20,
          "Average ms per stage_for_send_collective (after mps sync)");

    m.def("_test_event_sync_fence_churn",
          [](int ws) {
              mccl::event_sync_init();
              if (!mccl::event_sync_available()) {
                  return false;
              }
              uint64_t v1 = mccl::next_mps_event_value();
              mccl::commit_mps_and_signal(v1);
              mccl::wait_for_mps(v1);
              const int steps = std::max(0, 2 * (ws - 1));
              for (int i = 0; i < steps; ++i) {
                  uint64_t fv = mccl::next_fence_event_value();
                  mccl::signal_mccl_fence_gpu(fv);
                  mccl::wait_for_mccl_fence(fv);
              }
              uint64_t v2 = mccl::next_mps_event_value();
              mccl::commit_mps_and_signal(v2);
              mccl::wait_for_mps(v2);
              return true;
          },
          py::arg("ws"),
          "Simulate ring_chunked fence churn; must not hang wait_for_mps",
          py::call_guard<py::gil_scoped_release>());

    m.attr("__version__") = MCCL_VERSION_STRING;
    m.attr("__protocol_version__") = MCCL_PROTOCOL_VERSION;

    // Expose CompressionMode enum
    py::enum_<mccl::CompressionMode>(m, "CompressionMode")
        .value("NONE", mccl::CompressionMode::NONE)
        .value("FP16", mccl::CompressionMode::FP16)
        .value("TOPK", mccl::CompressionMode::TOPK);

    // Expose MCCLOptions for extended API usage
    py::class_<mccl::MCCLOptions, c10d::Backend::Options,
               c10::intrusive_ptr<mccl::MCCLOptions>>(m, "MCCLOptions")
        .def(py::init<>())
        .def(py::init<std::chrono::milliseconds>(), py::arg("timeout"))
        .def_readwrite("transport", &mccl::MCCLOptions::transport)
        .def_readwrite("listen_addr", &mccl::MCCLOptions::listen_addr)
        .def_readwrite("port_base", &mccl::MCCLOptions::port_base)
        .def_readwrite("ifname", &mccl::MCCLOptions::ifname)
        .def_readwrite("chunk_bytes", &mccl::MCCLOptions::chunk_bytes)
        .def_readwrite("small_msg_threshold", &mccl::MCCLOptions::small_msg_threshold)
        .def_readwrite("transport_crc", &mccl::MCCLOptions::transport_crc)
        .def_readwrite("fast_math", &mccl::MCCLOptions::fast_math)
        .def_readwrite("gpu_threshold", &mccl::MCCLOptions::gpu_threshold)
        .def_readwrite("overlap_comm", &mccl::MCCLOptions::overlap_comm)
        .def_readwrite("max_queue_depth", &mccl::MCCLOptions::max_queue_depth)
        .def_readwrite("compression", &mccl::MCCLOptions::compression)
        .def_readwrite("topk_ratio", &mccl::MCCLOptions::topk_ratio)
        .def_readwrite("watchdog_timeout", &mccl::MCCLOptions::watchdog_timeout)
        .def_readwrite("heartbeat_interval", &mccl::MCCLOptions::heartbeat_interval);

    // ── Metal kernel test helpers ─────────────────────────────────────
    // These thin wrappers are used by test_local_kernels.py to exercise the
    // actual MCCL Metal compute shaders, as opposed to PyTorch's own MPS ops.

    m.def("_metal_accumulate_chunk",
          [](at::Tensor dst, const at::Tensor& src) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_accumulate_chunk(dst, src);
              mccl::metal_sync();
              return dst;
          },
          "dst += src via MCCL Metal kernel; returns dst");

    m.def("_metal_elementwise_min",
          [](at::Tensor dst, const at::Tensor& src) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_elementwise_min(dst, src);
              mccl::metal_sync();
              return dst;
          },
          "dst = min(dst, src) via MCCL Metal kernel; returns dst");

    m.def("_metal_elementwise_max",
          [](at::Tensor dst, const at::Tensor& src) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_elementwise_max(dst, src);
              mccl::metal_sync();
              return dst;
          },
          "dst = max(dst, src) via MCCL Metal kernel; returns dst");

    m.def("_metal_elementwise_product",
          [](at::Tensor dst, const at::Tensor& src) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_elementwise_product(dst, src);
              mccl::metal_sync();
              return dst;
          },
          "dst *= src via MCCL Metal kernel; returns dst");

    m.def("_metal_scale_inplace",
          [](at::Tensor buf, double scale) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_scale_inplace(buf, scale);
              mccl::metal_sync();
              return buf;
          },
          "buf *= scale via MCCL Metal kernel; returns buf");

    m.def("_metal_accumulate_and_scale",
          [](at::Tensor dst, const at::Tensor& src, double scale) {
              mccl::metal_kernels_init();
              mccl::mps_stream_sync();
              mccl::metal_accumulate_and_scale(dst, src, scale);
              mccl::metal_sync();
              return dst;
          },
          "dst = (dst + src) * scale via MCCL Metal kernel; returns dst");

    // ── MCCL transforms (STFT / iSTFT) ────────────────────────────────
    m.def("_stft_forward",
          [](const at::Tensor& waveform,
             const at::Tensor& window,
             int64_t n_fft,
             int64_t hop_length,
             int64_t win_length,
             bool center,
             bool normalized,
             const std::string& backend) {
              mccl::StftParams params{n_fft, hop_length, win_length, center, normalized};
              return mccl::stft_forward(
                  waveform,
                  window,
                  params,
                  mccl::parse_stft_backend(backend));
          },
          py::arg("waveform"),
          py::arg("window"),
          py::arg("n_fft"),
          py::arg("hop_length"),
          py::arg("win_length"),
          py::arg("center") = true,
          py::arg("normalized") = false,
          py::arg("backend") = "vdsp",
          py::call_guard<py::gil_scoped_release>());

    m.def("_stft_backward",
          [](const at::Tensor& grad_spec,
             const at::Tensor& window,
             int64_t n_fft,
             int64_t hop_length,
             int64_t win_length,
             bool center,
             bool normalized,
             int64_t signal_length,
             const std::string& backend) {
              mccl::StftParams params{n_fft, hop_length, win_length, center, normalized};
              return mccl::stft_backward(
                  grad_spec, window, params, signal_length, mccl::parse_stft_backend(backend));
          },
          py::arg("grad_spec"),
          py::arg("window"),
          py::arg("n_fft"),
          py::arg("hop_length"),
          py::arg("win_length"),
          py::arg("center") = true,
          py::arg("normalized") = false,
          py::arg("signal_length"),
          py::arg("backend") = "vdsp",
          py::call_guard<py::gil_scoped_release>());

    m.def("_istft_forward",
          [](const at::Tensor& spec,
             const at::Tensor& window,
             int64_t n_fft,
             int64_t hop_length,
             int64_t win_length,
             bool center,
             bool normalized,
             int64_t length,
             const std::string& backend) {
              mccl::StftParams params{n_fft, hop_length, win_length, center, normalized};
              return mccl::istft_forward(
                  spec, window, params, length, mccl::parse_stft_backend(backend));
          },
          py::arg("spec"),
          py::arg("window"),
          py::arg("n_fft"),
          py::arg("hop_length"),
          py::arg("win_length"),
          py::arg("center") = true,
          py::arg("normalized") = false,
          py::arg("length"),
          py::arg("backend") = "vdsp",
          py::call_guard<py::gil_scoped_release>());

    m.def("_istft_backward",
          [](const at::Tensor& grad_waveform,
             const at::Tensor& window,
             int64_t n_fft,
             int64_t hop_length,
             int64_t win_length,
             bool center,
             bool normalized,
             int64_t length,
             const std::string& backend) {
              mccl::StftParams params{n_fft, hop_length, win_length, center, normalized};
              return mccl::istft_backward(
                  grad_waveform, window, params, length, mccl::parse_stft_backend(backend));
          },
          py::arg("grad_waveform"),
          py::arg("window"),
          py::arg("n_fft"),
          py::arg("hop_length"),
          py::arg("win_length"),
          py::arg("center") = true,
          py::arg("normalized") = false,
          py::arg("length"),
          py::arg("backend") = "vdsp",
          py::call_guard<py::gil_scoped_release>());

    // ── Expose Metrics summary ────────────────────────────────────────
    py::class_<mccl::Metrics::Summary>(m, "MetricsSummary")
        .def_readonly("total_ops", &mccl::Metrics::Summary::total_ops)
        .def_readonly("total_bytes_sent", &mccl::Metrics::Summary::total_bytes_sent)
        .def_readonly("total_bytes_recv", &mccl::Metrics::Summary::total_bytes_recv)
        .def_readonly("total_errors", &mccl::Metrics::Summary::total_errors)
        .def_readonly("avg_latency_ms", &mccl::Metrics::Summary::avg_latency_ms)
        .def_readonly("p50_latency_ms", &mccl::Metrics::Summary::p50_latency_ms)
        .def_readonly("p99_latency_ms", &mccl::Metrics::Summary::p99_latency_ms)
        .def_readonly("peak_throughput_gbps", &mccl::Metrics::Summary::peak_throughput_gbps)
        .def_readonly("avg_sync_ms", &mccl::Metrics::Summary::avg_sync_ms)
        .def_readonly("avg_network_ms", &mccl::Metrics::Summary::avg_network_ms)
        .def_readonly("avg_reduce_ms", &mccl::Metrics::Summary::avg_reduce_ms)
        .def_readonly("avg_queue_wait_ms", &mccl::Metrics::Summary::avg_queue_wait_ms)
        .def_readonly("avg_execution_ms", &mccl::Metrics::Summary::avg_execution_ms);
}
