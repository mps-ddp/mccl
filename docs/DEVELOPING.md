# Developing MCCL

Guide for contributors and anyone reading the codebase for the first time.

## What MCCL is

MCCL is a **PyTorch `ProcessGroup` backend** named `mccl` for **`torch.distributed` on MPS** (Apple Silicon). It lets you run `dist.init_process_group(backend="mccl", device_id=...)` and use **DDP**, **collectives**, and **point-to-point** ops on Metal-backed tensors across processes—on one Mac or many.

PyTorch ships NCCL for CUDA and Gloo for CPU; **MPS did not have a first-party multi-node collective stack** in the same way. MCCL fills that gap with a native extension: TCP (and optionally RDMA) transport plus reductions that respect **unified memory** (shared `MTLBuffer` / Accelerate / Metal).

## Repository layout

| Path | Role |
|------|------|
| `mccl/` | Python package: `init()`, `MCCLConfig`, metrics, backend registration |
| `csrc/backend/` | `ProcessGroupMCCL`, `WorkMCCL`, MPS dispatch registration |
| `csrc/metal/` | MPS interop, `EventSync`, Metal kernels, Accelerate ops |
| `csrc/transport/` | TCP transport, protocol, optional RDMA (`transport/rdma/`) |
| `csrc/runtime/` | Progress engine, watchdog, metrics, memory pool, rendezvous |
| `csrc/compression/` | Optional fp16 / top-k compression |
| `examples/` | `ddp_dummy_train.py`, compare scripts |
| `docs/MULTINODE.md` | **2+ Macs**: `MASTER_ADDR` (head), firewall, ports, tuning |
| `tests/` | Pytest suite (see [TESTING.md](../TESTING.md)) |

## Build (from source)

**Requirements**

- macOS on **Apple Silicon** (arm64)
- Xcode command-line tools (`xcode-select --install`)
- Python **3.11+**
- PyTorch **2.5+** with MPS (install `torch` first so headers/libs resolve)

**Install editable**

```bash
pip install torch
pip install -e ".[dev]"
```

This compiles the C++/Objective-C++ extension (`mccl._C`). Builds **only** on Darwin/arm64 (see `setup.py`).

**Rebuild after C++ changes**

```bash
pip install -e . --no-build-isolation
# or
python setup.py build_ext --inplace
```

## Architecture (mental model)

1. **Python** imports `mccl` → pybind registers the `mccl` backend with `c10d`.
2. **`init_process_group(backend="mccl", device_id=mps_device)`** constructs `ProcessGroupMCCL`.
3. **Collectives** record GPU completion via **event sync** (`commit_mps_and_signal` on PyTorch’s MPS command buffer + `wait_for_mps` on the **ProgressEngine** thread before reading tensor memory), then run **network / CPU or Metal** reduction. Legacy comment: full-stream `torch::mps::synchronize()` on the autograd thread is unsafe mid-backward; the event path serializes via `dispatch_sync` on PyTorch’s MPS dispatch queue instead.

   **DDP**: `examples/ddp_dummy_train.py` uses **bucketed** allreduce during `backward()` (no `no_sync()` workaround). **`MCCL_SYNC_MODE=coalesced` must not** be used with hook-driven multi-bucket DDP (skips per-bucket GPU waits → corrupt traffic).

   **Multi-node**: see [MULTINODE.md](MULTINODE.md) for `MASTER_ADDR`, ports / firewall, and tuning.
4. **Float32 + shared memory**: often **CPU-side reduction** via Accelerate (vDSP) reading the same pointers as MPS.
5. **Float16 / bf16**: **Metal** kernels on a dedicated MCCL queue; may sync more heavily.
6. **Transport**: overlapped **TCP** by default; **RDMA** optional when OS/hardware support it (`MCCL_TRANSPORT`).

For configuration, see the main [README.md](../README.md) env table and `mccl.config.MCCLConfig`.

## Debugging tips

- Set **`MCCL_LOG_LEVEL=INFO`** (or `DEBUG`) to print resolved config and runtime logs from C++.
- **`TORCH_DISTRIBUTED_DEBUG=DETAIL`** helps PyTorch-side distributed tracing.
- Multi-node “hangs” are often **ports** (`MASTER_PORT` vs `MCCL_PORT_BASE`) or **firewall** — see README “Multi-node appears hung”.
- Example smoke tests: `examples/ddp_dummy_train.py --baseline` (single GPU) and `torchrun --nproc_per_node=2 ...` (local DDP).

## Code style

- Match surrounding C++ and Python style; keep changes scoped to the task.
- Objective-C++ (`.mm`) for Metal / Foundation interop.

## Version and protocol

Extension version lives in `mccl/version.py` and C++ `common/Version.hpp` (keep in sync). Wire protocol version is asserted in tests (`test_build.py`).
