# MCCL

`torch.distributed` backend for **DDP + collectives on MPS** (Apple Silicon). **TCP** by default; **RDMA** if the OS exposes it (`MCCL_TRANSPORT`).

## Install

- Apple Silicon Mac, **Python 3.11+**, **PyTorch first** (nightlies / 2.10.0 tested), **Xcode CLT** (`xcode-select --install`). Optional: full Xcode for Metal precompile.

```bash
pip install torch
pip install -e .
```

```python
import torch.distributed as dist
import mccl
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)
```

Demo: https://github.com/user-attachments/assets/21865149-b077-4b65-93cc-f9e319ff0328  
Ethernet/Wi‑Fi OK; slower than a wired link.

## Docs

| | |
|--|--|
| Two Macs / TB | [docs/THUNDERBOLT_SETUP.md](docs/THUNDERBOLT_SETUP.md) |
| Firewall, ports, buckets | [docs/MULTINODE.md](docs/MULTINODE.md) |
| Hacking | [docs/DEVELOPING.md](docs/DEVELOPING.md) |
| Tests | [TESTING.md](TESTING.md) |
| Versions | [COMPATIBILITY.md](COMPATIBILITY.md) |
| Timings (please add yours) | [RESULTS.md](RESULTS.md) |

## Examples

```bash
python examples/ddp_dummy_train.py --baseline
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py
```

```python
import mccl
from torch.nn.parallel import DistributedDataParallel as DDP
# mccl.init(...) optional — see Configuration
dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)
model = DDP(MyModel().to("mps"))
```

## Throughput chart (one run)

**Numbers from that run** (`--save-stats` + `examples/benchmark_throughput.py`): **~59.6** vs **~22.9** samples/s (baseline vs 2-rank DDP), **~2.6×** throughput ratio — **only** for that JSON, not a spec.

**Setup was wrong for a fair fight:**

| | Baseline | DDP |
|--|--|--|
| Hardware | **1× M1 Max** | **2 Macs, mixed SoCs** (e.g. M1 Max + M4 Max) |
| Network | — | **TCP**, TB3-class link |
| Model / batch | ~96M-param example, **global batch 8** | same |

So: **slowest rank wins the step**, baseline never hits the network, chips don’t match — **don’t read this as “MCCL is 2.6× slower” in general.** No same-SKU cluster to test on; **PR [RESULTS.md](RESULTS.md)** if you have better data.

```bash
python examples/ddp_dummy_train.py --baseline --save-stats baseline_stats.json
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py --save-stats ddp_stats.json
python examples/benchmark_throughput.py --baseline baseline_stats.json --ddp ddp_stats.json -o bench
```

`bash scripts/benchmark_matrix.sh` — other checks. Env knobs: [examples/ddp_dummy_train.py](examples/ddp_dummy_train.py).

![bench](bench.png)  
![bars](bench_bars.png)

## Collectives

`allreduce`, `broadcast`, `barrier`, `allgather`, `reduce_scatter`, `send`, `recv`

## Configuration

Python `MCCL.init` / `MCCLConfig` or env; Python wins if set before `init_process_group`.

| Env | Default | Notes |
|-----|---------|--------|
| `MCCL_TRANSPORT` | `auto` | `tcp`, `rdma`, `auto` |
| `MCCL_LOG_LEVEL` | `WARN` | |
| `MCCL_LISTEN_ADDR` | auto | |
| `MCCL_LINK_PROFILE` | — | `thunderbolt` → bigger buffers |
| `MCCL_PORT_BASE` | `29600` | **≠** `MASTER_PORT` |
| `MCCL_IFNAME` | auto | |
| `MCCL_CHUNK_BYTES` | `4194304` | |
| `MCCL_SMALL_MSG_THRESHOLD` | `65536` | |
| `MCCL_TRANSPORT_CRC` | `false` | |
| `MCCL_FAST_MATH` | `true` | |
| `MCCL_GPU_THRESHOLD` | `4096` | |
| `MCCL_COMPRESSION` | `none` | `fp16`, `topk` |
| `MCCL_TOPK_RATIO` | `0.01` | |
| `MCCL_SHADER_PATH` | auto | |

## Diagnostics

```python
mccl.get_metrics(); mccl.log_metrics(); mccl.reset_metrics()
```

`MCCL_LOG_LEVEL=INFO` — full config on startup. **Hung multi-node:** [docs/MULTINODE.md](docs/MULTINODE.md).

## Thunderbolt, TCP, and what was tested

- **Throughput charts in this repo:** two Macs talking over a **Thunderbolt bridge IP** using the **TCP** transport path (TB3-class link in our case). **Not RDMA.**
- **Ethernet / Wi‑Fi:** same code paths; expect worse RTT and throughput than a direct TB cable between hosts.
- **`MCCL_LINK_PROFILE=thunderbolt`:** optional buffer/chunk tuning when peers are on a TB link — see [docs/MULTINODE.md](docs/MULTINODE.md).
- **Cabling / `169.254.x.x` / firewall:** [docs/THUNDERBOLT_SETUP.md](docs/THUNDERBOLT_SETUP.md).

## RDMA (Thunderbolt 5)

RDMA is **optional** and **not** what produced the benchmark plots above. It needs **TB5 hardware** (e.g. **M4 Pro / Max / Ultra** with TB5 ports), a **direct TB5 cable** (no hub), and a **macOS build that ships `librdma.dylib`** (Apple’s docs call out newer macOS versions). We have **not** done serious end-to-end training benchmarks on RDMA here — treat latency/BW claims you see elsewhere as **hypotheses until you measure**.

**Enable once (Recovery OS):** `rdma_ctl enable` → reboot.

**Sanity check:**

```bash
python -c "import ctypes; ctypes.cdll.LoadLibrary('librdma.dylib'); print('RDMA lib OK')"
```

**Use in MCCL:** `MCCL_TRANSPORT=rdma` or `mccl.init(transport="rdma")`.

| Mode | Behavior |
|------|----------|
| `auto` (default) | Try RDMA, fall back to TCP |
| `rdma` | RDMA preferred; TCP if init fails |
| `tcp` | Skip RDMA; TCP only |

**Gotchas:** if RDMA init fails, logs at `MCCL_LOG_LEVEL=INFO` say why. Apple limits **~100 memory regions** per device — very large jobs can hit MR registration errors (reduce ranks or buffer churn). **PR [RESULTS.md](RESULTS.md)** if you have real TB5+RDMA training numbers.

## Internals

**f32 reductions on CPU-visible MPS memory:** element-wise adds, scales, min/max, product go through **Apple Accelerate / vDSP** (`vDSP_vadd`, `vDSP_vsmul`, …) in `csrc/metal/AccelerateOps.mm`, with a **parallel chunk loop** so big tensors use all performance cores. On Apple Silicon, those **vector library paths are AMX-backed where Accelerate routes them** — we’re not hand-writing AMX kernels; we lean on **vDSP + UMA** (CPU pointer is valid for shared MPS buffers, so reductions can run without extra copies in the common f32 case).

**fp16 / bf16:** Metal shaders when above `MCCL_GPU_THRESHOLD`; small tensors can widen → **f32 vDSP** → narrow.

**Network:** TCP progress thread, overlap, 2-rank vs ring allreduce. **CRC:** ARM `crc32` hw when enabled.

More layout / build: [docs/DEVELOPING.md](docs/DEVELOPING.md).

## License

MIT — [LICENSE](LICENSE)
