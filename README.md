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

`ddp_dummy_train.py` defaults: **DDP** `BATCH_SIZE=128` per rank (global **256** with 2 ranks); **`--baseline`** global **256** unless you set `BASELINE_BATCH_SIZE` / `BATCH_SIZE`. Lower if you OOM.

## Throughput (example — **your mileage will vary**)

Latest author run (`examples/ddp_dummy_train.py` + `--save-stats` + `examples/benchmark_throughput.py`):

```
single M1 Max (MPS):  78.3 samples/s   (global_batch=256, world=1)
DDP (MCCL):          134.2 samples/s   (global_batch=256, world=2)
baseline / DDP:      0.58×  → DDP ~172% of baseline samples/s
params:              96,510,024 (same both sides)
```

**Global batch 256:** baseline = `BASELINE_BATCH_SIZE=256` (or `BATCH_SIZE=256` on one process). DDP = **`BATCH_SIZE=128` per rank × 2 ranks** (or 256×1 on two nodes — same global).

**Takeaway:** with this **~96M-param** model, **big batch + higher GPU/memory util**, **2-rank MCCL DDP beat one M1 Max** on **samples/s** in our JSON. With **small global batch** (we’ve seen **global 8**), **comm/sync dominated** and DDP looked **much slower**. **Hardware mix still matters** (slowest rank sets the step); **PR [RESULTS.md](RESULTS.md)** with your setup.

```bash
# Match global batch: e.g. DDP BATCH_SIZE=128 × world 2 → baseline BASELINE_BATCH_SIZE=256
python examples/ddp_dummy_train.py --baseline --save-stats baseline_stats.json
torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/ddp_dummy_train.py --save-stats ddp_stats.json
python examples/benchmark_throughput.py --baseline baseline_stats.json --ddp ddp_stats.json -o bench
```

`bash scripts/benchmark_matrix.sh` — other checks. Env: [examples/ddp_dummy_train.py](examples/ddp_dummy_train.py).

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

- **Throughput charts in this repo:** **TCP** over a **Thunderbolt bridge IP** (TB3-class link here). **Not RDMA.** Plots match the **high global-batch** run above unless you regenerate.
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
