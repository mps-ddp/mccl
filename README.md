# MCCL

**Collective communication for PyTorch Distributed on Apple Silicon (MPS).**

MCCL implements a custom `torch.distributed` backend (`backend="mccl"`) so you can run **DDP** and **collectives** on **MPS tensors** across processes—one Mac or several. It uses Apple Silicon **unified memory** where possible: float32 reductions often go through **Accelerate / vDSP** on CPU-visible shared buffers while Metal stays available for compute; fp16/bf16 paths use **Metal** shaders. The default transport is **overlapped TCP**; **RDMA** is optional when the OS and hardware support it.

### New here?

| Goal | Where to start |
|------|----------------|
| **Clone, build, run a smoke test** | [Quick start](#quick-start) → `examples/ddp_dummy_train.py` |
| **Understand the codebase** | [docs/DEVELOPING.md](docs/DEVELOPING.md) |
| **Run tests** | [TESTING.md](TESTING.md) |
| **Version / platform matrix** | [COMPATIBILITY.md](COMPATIBILITY.md) |

### Is this “novel” or never done before?

**Not in the abstract.** Distributed training, ring allreduce, and TCP collectives are well-studied; NVIDIA’s **NCCL** and PyTorch’s **Gloo** are the familiar stacks on Linux/CUDA/CPU.

What **is** specific here: PyTorch’s **MPS** path historically did not ship a **production multi-process collective library** comparable to NCCL. MCCL is an **engineering project** that plugs that gap: a **native ProcessGroup for `DeviceType::MPS`**, with transports and reduction paths tuned for **Apple Silicon unified memory**. So the *idea* of distributed training is old; the *integration*—this backend + Metal/Accelerate + optional Apple RDMA—is what this repository provides. Don’t treat it as a research claim of world-first algorithms; treat it as **infrastructure** for Mac clusters.

## Architecture

- **AMX/Accelerate reductions (f32)**: Element-wise operations (SUM, MIN, MAX, PRODUCT, AVG) use vDSP functions that leverage the AMX coprocessor. Because MPS tensors use `MTLStorageModeShared`, the CPU reads gradient data at the same physical addresses as the GPU without a blit on the send path. Received data is written directly into the tensor's shared-memory buffer (one `memcpy` from the network buffer). The GPU remains fully available for the next forward pass during reduction.

- **Metal compute fallback (f16)**: Half-precision tensors use vectorized Metal shaders (float4/half4 per thread) dispatched through a dedicated command queue. 14 precompiled pipeline states cover all reduce operations.

- **Overlapped transport**: Each ring step sends to the right neighbor and receives from the left neighbor simultaneously via a `poll()`-based state machine on the progress engine thread. TCP is full-duplex, so even 2-rank clusters benefit. No extra threads, no context switches.

- **Ring allreduce**: Bandwidth-optimal reduce-scatter + allgather with algorithm dispatch: tree-gather for small messages, direct exchange for 2 ranks, ring for 3+ ranks.

- **Production infrastructure**: Single-thread progress engine, per-collective watchdog, passive TCP liveness monitor, lock-free metrics with p50/p99 latency tracking, 64-byte aligned memory pool, ABORT protocol for coordinated failure.

## Supported collectives

allreduce, broadcast, barrier, allgather, reduce_scatter, send, recv

## Quick start

**Prerequisites:** macOS on **Apple Silicon**, Python **3.11+**, Xcode command-line tools, **PyTorch 2.5+** with MPS.

```bash
git clone …   # your fork or upstream URL
cd mccl
pip install torch
pip install -e ".[dev]"
```

**Smoke test (single GPU, no distributed launcher):**

```bash
python examples/ddp_dummy_train.py --baseline
```

**Local two-process DDP on one Mac:**

```bash
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py
```

See [examples/ddp_dummy_train.py](examples/ddp_dummy_train.py) for env vars (model size, steps, ports).

**Application code (minimal):**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import mccl

# Option 1: env vars (backward-compatible)
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

# Option 2: Python config (recommended)
mccl.init(compression="fp16", fast_math=True, listen_addr="192.168.1.10")
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

# Option 3: full config object
cfg = mccl.MCCLConfig(
    compression="fp16",
    gpu_threshold=8192,
    small_msg_threshold=131072,
    log_level="INFO",
)
mccl.init(cfg)
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

model = DDP(MyModel().to("mps"))
```

## Configuration

All settings can be controlled via the Python `MCCLConfig` API **or** environment variables.
Python config takes priority when `mccl.init()` is called before `init_process_group`.

| Env variable | Python field | Default | Description |
|---|---|---|---|
| `MCCL_TRANSPORT` | `transport` | `auto` | Transport mode: `auto`, `tcp`, `rdma` |
| `MCCL_LOG_LEVEL` | `log_level` | `WARN` | TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF |
| `MCCL_LISTEN_ADDR` | `listen_addr` | auto-detect | Bind address (auto-detects Thunderbolt bridge) |
| `MCCL_LINK_PROFILE` | — | (unset) | Set to `thunderbolt` for production TCP defaults on TB links: larger default socket buffers and chunk size (see `Connection.cpp`, `TcpTransport.cpp`) |
| `MCCL_PORT_BASE` | `port_base` | `29600` | Base port (rank N listens on port_base + N). **Must differ from `MASTER_PORT`** (PyTorch’s TCP store uses `MASTER_PORT` on the master; default `29600` avoids colliding with typical `MASTER_PORT=29500`). |
| `MCCL_IFNAME` | `ifname` | (auto) | Advisory network interface hint |
| `MCCL_CHUNK_BYTES` | `chunk_bytes` | `4194304` | Chunk size for CRC-enabled transport |
| `MCCL_SMALL_MSG_THRESHOLD` | `small_msg_threshold` | `65536` | Bytes below which allreduce uses gather-reduce |
| `MCCL_TRANSPORT_CRC` | `transport_crc` | `false` | Per-chunk CRC32 integrity checks |
| `MCCL_FAST_MATH` | `fast_math` | `true` | Metal shader FMA / relaxed NaN (set `0` for IEEE) |
| `MCCL_GPU_THRESHOLD` | `gpu_threshold` | `4096` | Elements below which f16/bf16 falls back to CPU |
| `MCCL_COMPRESSION` | `compression` | `none` | `none`, `fp16`, or `topk` |
| `MCCL_TOPK_RATIO` | `topk_ratio` | `0.01` | Sparsification ratio for topk mode |
| `MCCL_SHADER_PATH` | `shader_path` | auto-detect | Path to `shaders.metal` or `mccl_shaders.metallib` |

## Diagnostics

Runtime metrics are accessible from Python after `init_process_group`:

```python
summary = mccl.get_metrics()
if summary:
    print(f"ops={summary.total_ops}  p50={summary.p50_latency_ms:.2f}ms  "
          f"throughput={summary.peak_throughput_gbps:.2f} Gbps")

mccl.log_metrics()    # dump to stderr via C++ logger
mccl.reset_metrics()  # zero all counters
```

The C++ backend also logs the full resolved config at INFO level on every rank at startup, so setting `MCCL_LOG_LEVEL=INFO` shows every tunable value per node.

### Multi-node appears hung

Typical causes:

1. **Not all ranks running** — `torchrun` / elastic block until every node joins; killing the master yields `DistNetworkError` / recv 0 bytes (expected).
2. **`MCCL_PORT_BASE` equals `MASTER_PORT`** — PyTorch’s TCP rendezvous uses `MASTER_PORT` on the master host; MCCL rank 0 listens on `MCCL_PORT_BASE + 0`. They must be different ports (defaults use `29600` vs `29500`, or set `MCCL_PORT_BASE=$((MASTER_PORT+100))` on all nodes).
3. **Firewall / wrong IP** — Peers must reach `MASTER_ADDR:MASTER_PORT` and each published MCCL endpoint; open `MCCL_PORT_BASE` through `MCCL_PORT_BASE + world_size - 1` if needed.

**Full multi-node checklist, fair baseline vs DDP, and perf tuning** (`DDP_BUCKET_MB`, sockets, compression): see [docs/MULTINODE.md](docs/MULTINODE.md).

## RDMA over Thunderbolt 5

MCCL supports RDMA transport for ultra-low-latency collective communication over Thunderbolt 5 connections between Apple Silicon nodes. When available, RDMA bypasses the TCP/IP stack entirely for sub-10µs latency and ~80 Gbps bandwidth.

### Requirements

- **Hardware**: Apple Silicon M4 Pro, M4 Max, or M4 Ultra with Thunderbolt 5 ports
- **macOS**: 26.2 or later (ships `librdma.dylib`)
- **Connectivity**: Direct Thunderbolt 5 cable between nodes (not a hub/switch)

### Setup

1. **Enable RDMA** (one-time, requires Recovery OS):
   ```bash
   # Boot into Recovery OS, then in Terminal:
   rdma_ctl enable
   ```

2. **Verify RDMA** is available:
   ```bash
   # After reboot, check that the library loads:
   python -c "import ctypes; ctypes.cdll.LoadLibrary('librdma.dylib'); print('RDMA OK')"
   ```

3. **Configure MCCL** to use RDMA:
   ```python
   import mccl
   mccl.init(transport="rdma")
   # Or via environment variable:
   # MCCL_TRANSPORT=rdma
   ```

### Transport modes

| `MCCL_TRANSPORT` | Behavior |
|---|---|
| `auto` (default) | Try RDMA first; fall back to TCP if unavailable |
| `rdma` | Use RDMA; fall back to TCP if init fails |
| `tcp` | TCP only (skip RDMA probe entirely) |

### Expected performance

| Metric | TCP | RDMA (TB5) |
|---|---|---|
| Latency (small msg) | ~50-100µs | ~5-10µs |
| Bandwidth | ~10 Gbps | ~80 Gbps |
| CPU overhead | Moderate (poll + memcpy) | Minimal (zero-copy) |

### Troubleshooting

- **"no RDMA devices found"**: Ensure Thunderbolt 5 cable is connected and `rdma_ctl enable` was run from Recovery OS
- **Falls back to TCP**: Check `MCCL_LOG_LEVEL=INFO` output for detailed RDMA init diagnostics
- **MR registration failures**: Apple limits ~100 memory regions per device; reduce world size or buffer count

## Performance characteristics

For f32 allreduce on Apple Silicon:

- **Reduction**: AMX via `vDSP_vadd` at memory bandwidth (~100+ GB/s on M4). Zero Metal kernel launch overhead. Send path reads directly from the tensor's CPU-accessible shared-memory pointer (zero-copy). Receive path writes into the same pointer (one memcpy from the network staging buffer).
- **Transport**: Overlapped send+recv halves per-step network time vs serial. macOS TCP auto-tuning enabled (no fixed buffer sizes).
- **Sync**: Single `torch::mps::synchronize()` per collective. No GPU sync at the end for f32 path.

## Compatibility

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested matrix of macOS, PyTorch, and hardware versions.

## Contributing / development

See [docs/DEVELOPING.md](docs/DEVELOPING.md) for repository layout, how the backend fits together, rebuild commands, and debugging.

## Testing

See [TESTING.md](TESTING.md) for pytest commands and test layout.

## License

MIT -- see [LICENSE](LICENSE).
