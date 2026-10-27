# Configuration

MCCL reads these environment variables when a process group is created. Set them on every
rank before `torch.distributed.init_process_group(backend="mccl")`, or set the same values
from Python:

```python
import mccl

mccl.init(compression="fp16", log_level="INFO")  # same as MCCL_COMPRESSION=fp16 MCCL_LOG_LEVEL=INFO
```

The defaults are the settings used for the paper's 24-Mac results. For multi-node setup
and troubleshooting, see [MULTINODE.md](MULTINODE.md).

## Networking

| Variable | Default | Description |
| --- | --- | --- |
| `MCCL_IFNAME` | unset | Network interface to use, such as `en0`. Defaults to the interface on `MASTER_ADDR`'s subnet. |
| `MCCL_LISTEN_ADDR` | auto | Address published to peers, overriding interface selection. |
| `MCCL_PORT_BASE` | `20100` | Rank *r* listens on `base + r`. Keep clear of `MASTER_PORT`. |
| `MCCL_TRANSPORT` | `auto` | `tcp` or `rdma`. RDMA over Thunderbolt 5 is experimental. |
| `MCCL_SOCK_BUFSIZE` | 32 MiB | Socket buffer size. `0` lets the kernel auto-tune. |
| `MCCL_CHUNK_BYTES` | 16 MB | Maximum message size on the wire. |
| `MCCL_LINK_PROFILE` | unset | `thunderbolt` raises the minimum transport chunk to 16 MB. |
| `MCCL_TRANSPORT_CRC` | off | CRC32 on every payload. |

## Collective algorithms

| Variable | Default | Description |
| --- | --- | --- |
| `MCCL_SMALL_MSG_THRESHOLD` | 256 KiB | Allreduces at or below this size use the small-message algorithm. |
| `MCCL_SMALL_ALGO` | `tree` | Small-message algorithm: `tree` (recursive doubling) or `star` (via rank 0). |
| `MCCL_RING_ALGO` | `auto` | `chunked` (double-buffered, 2N chunks) or `basic`. |
| `MCCL_RING_PIPELINE` | on | Streaming ring. `0` selects a lock-step fallback for debugging. |
| `MCCL_PIPELINE_DEPTH` | `4` | Receives posted ahead per ring pipeline (1–8). |
| `MCCL_PIPELINE_INFLIGHT_BYTES` | 8 MB | Per-connection in-flight budget; caps the effective pipeline depth. |
| `MCCL_CREDIT_MIN_CHUNK` | 1 MB | Minimum chunk size for credit flow control. `0` disables it. |
| `MCCL_COLLECTIVE_CONCURRENCY` | `1` | Collectives in flight (1–8). |
| `MCCL_CONCURRENT_RINGS` | off | Allow ring allreduces to overlap on the wire. Requires concurrency of 2 or more. |
| `MCCL_DEMUX_PARK_BYTES` | auto | Per-peer limit on messages buffered before their receive is posted. |
| `MCCL_DEMUX_INFLIGHT_BUDGET_BYTES` | 1 GiB | Caps effective concurrency at budget ÷ DDP bucket size. |

## Reduction and compression

| Variable | Default | Description |
| --- | --- | --- |
| `MCCL_COMPRESSION` | `none` | `fp16` or `topk` gradient compression. |
| `MCCL_TOPK_RATIO` | `0.01` | Fraction of values kept by `topk` compression. |
| `MCCL_UNIFIED_COLLECTIVE` | on | In-place path for shared MPS storage. `0` stages through Metal. |
| `MCCL_UNIFIED_CPU_REDUCE` | on | Accelerate reduction in unified memory. `0` uses Metal kernels. |
| `MCCL_FP32_CPU_REDUCE` | off | Force fp32 CPU reduction on every path. |
| `MCCL_FAST_MATH` | on | Metal fast math. `0` selects strict math. |

## Synchronization, timeouts, and logging

| Variable | Default | Description |
| --- | --- | --- |
| `MCCL_EVENT_SYNC` | on | Shared-event ordering. `0` uses a blocking stream sync and removes overlap (debugging only). |
| `MCCL_OVERLAP_COMM` | on | Overlap communication with computation. |
| `MCCL_CPU_WRITE_SYNC` | none | `full` synchronizes MPS after CPU-path collectives (debugging only). |
| `MCCL_COLLECTIVE_STORE_BARRIER` | off | Store barrier before every collective (debugging only). |
| `MCCL_WATCHDOG_TIMEOUT_MS` | `300000` | Timeout for a running collective. The clock starts when the operation begins executing. |
| `MCCL_LOG_LEVEL` | `WARN` | `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL`, or `OFF`. `INFO` logs the selected interface, address, MTU, and link speed. |

## Transforms

| Variable | Default | Description |
| --- | --- | --- |
| `MCCL_STFT_BACKEND` | `auto` | Backend for `mccl.transforms.stft`: `vdsp`, `metal`, or `auto`. |
