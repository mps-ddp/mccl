# MCCL

**Distributed PyTorch across Apple Silicon Macs using MPS.**

## Performance Reality

**Current performance:** Expect **~10x slower** than single-GPU training in most cases. This is not expected to be a performance boost with the current implementation.

**Future potential:** Performance could be significantly improved with additional work on RDMA over Thunderbolt 5, better collective routing algorithms from PyTorch, or other transport optimizations.

## Quick start

**Prerequisites:** PyTorch must be installed before installing MCCL. This has been tested with PyTorch nightlies and the latest stable version 2.10.0.

```bash
pip install -e .
```

```python
import torch.distributed as dist
import mccl

dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)
# Use standard DDP on MPS tensors
```

**Tested:** M1 Max + M4 Max MacBook Pro, Thunderbolt 3, macOS 14–15, PyTorch 2.5+.

## When this makes sense

- **Experimenting** with distributed training on Apple Silicon
- **Playing** with multi-Mac setups and Apple Silicon clusters
- **Research** into PyTorch backends and MPS collective operations

**Don't expect performance gains** — use for learning and experimentation.

## What this is

MCCL is a `torch.distributed` backend for **DDP** and **collectives** on **MPS tensors**. Uses TCP over Thunderbolt/Ethernet by default; optional RDMA when available.

**How it's built:** Unified memory where it helps; f32 reductions through **Accelerate / vDSP** on CPU-visible buffers; fp16/bf16 can use **Metal**; overlapped TCP transport on progress thread.

**Not novel:** Allreduce over TCP, rings, etc. are old (NCCL, Gloo). This fills the gap for PyTorch **MPS** multi-process collectives.

**Why build this?** To explore if multi-Mac training was feasible and understand how PyTorch backends work under the hood. Turns out it's a lot of plumbing but pretty satisfying when two MacBooks actually sync gradients over a Thunderbolt cable.

## Setup guides

| Goal | Where |
|------|--------|
| Two Macs + Thunderbolt | [docs/THUNDERBOLT_SETUP.md](docs/THUNDERBOLT_SETUP.md) |
| Tuning, firewall, buckets | [docs/MULTINODE.md](docs/MULTINODE.md) |
| Code layout | [docs/DEVELOPING.md](docs/DEVELOPING.md) |
| Tests | [TESTING.md](TESTING.md) |
| Versions | [COMPATIBILITY.md](COMPATIBILITY.md) |
| Community timings | [RESULTS.md](RESULTS.md) |
| Launch / demo notes | [docs/LAUNCH.md](docs/LAUNCH.md) |

## Benchmarking

```bash
bash scripts/benchmark_matrix.sh
```

See [examples/ddp_dummy_train.py](examples/ddp_dummy_train.py) for env vars and [RESULTS.md](RESULTS.md) to add your numbers.

## What we haven't tested

- More than **2** nodes in one run  
- **Thunderbolt 5 + RDMA** end-to-end (development hardware was TB3 + TCP)  
- Every Mac SKU (Studio, mini, mixed generations)  
- Real production jobs beyond the example script  
- Huge models (10B+)  

If you run it on something else, **PR a row to [RESULTS.md](RESULTS.md)** or open an issue with `mccl.get_metrics()` + setup.

## Examples

**Smoke test (single GPU, no launcher):**

```bash
python examples/ddp_dummy_train.py --baseline
```

**Two processes, one Mac:**

```bash
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py
```

**Full app code:**

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

## Supported collectives

allreduce, broadcast, barrier, allgather, reduce_scatter, send, recv

## Configuration

All settings can be controlled via the Python `MCCLConfig` API **or** environment variables.
Python config takes priority when `mccl.init()` is called before `init_process_group`.

| Env variable | Python field | Default | Description |
|---|---|---|---|
| `MCCL_TRANSPORT` | `transport` | `auto` | Transport mode: `auto`, `tcp`, `rdma` |
| `MCCL_LOG_LEVEL` | `log_level` | `WARN` | TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF |
| `MCCL_LISTEN_ADDR` | `listen_addr` | auto-detect | Bind address (auto-detects Thunderbolt bridge) |
| `MCCL_LINK_PROFILE` | — | (unset) | `thunderbolt` → larger default socket buffers + chunk size (see `Connection.cpp`, `TcpTransport.cpp`) |
| `MCCL_PORT_BASE` | `port_base` | `29600` | Base port (rank N listens on port_base + N). **Must differ from `MASTER_PORT`**. |
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
2. **`MCCL_PORT_BASE` equals `MASTER_PORT`** — PyTorch's TCP rendezvous uses `MASTER_PORT` on the master host; MCCL rank 0 listens on `MCCL_PORT_BASE + 0`. They must be different ports (defaults use `29600` vs `29500`, or set `MCCL_PORT_BASE=$((MASTER_PORT+100))` on all nodes).
3. **Firewall / wrong IP** — Peers must reach `MASTER_ADDR:MASTER_PORT` and each published MCCL endpoint; open `MCCL_PORT_BASE` through `MCCL_PORT_BASE + world_size - 1` if needed.

**Checklist + tuning:** [docs/MULTINODE.md](docs/MULTINODE.md). **Thunderbolt 2-Mac wiring:** [docs/THUNDERBOLT_SETUP.md](docs/THUNDERBOLT_SETUP.md).

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

## How it works

- **Accelerate / vDSP (f32):** Element-wise ops (SUM, MIN, MAX, PRODUCT, AVG) on shared MPS buffers; CPU sees the same memory as the GPU for staging. Receive path typically one `memcpy` from the socket buffer into the tensor.

- **Metal (fp16/bf16):** Vectorized shaders on a dedicated queue where that path is used.

- **Overlapped TCP:** `poll()`-driven send/recv overlap on the progress engine; full-duplex helps even with 2 ranks.

- **Algorithms:** Small messages, 2-rank direct path, ring for 3+; reduce-scatter + allgather style allreduce.

- **Ops stuff:** Progress engine, watchdog, TCP keepalive-ish behavior, metrics, pooled buffers, ABORT path for failures.

## Compatibility

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested matrix of macOS, PyTorch, and hardware versions.

## Contributing / development

See [docs/DEVELOPING.md](docs/DEVELOPING.md) for repository layout, how the backend fits together, rebuild commands, and debugging.

## Testing

See [TESTING.md](TESTING.md) for pytest commands and test layout.

## License

MIT -- see [LICENSE](LICENSE).