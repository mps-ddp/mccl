# Multi-node DDP (2+ Macs, MCCL)

Checklist and tuning for **one MPS GPU per machine** (`torchrun --nnodes=N --nproc_per_node=1`).

**Two Macs + Thunderbolt cable (step-by-step):** [THUNDERBOLT_SETUP.md](THUNDERBOLT_SETUP.md)

## Phase 0 — Wiring (do this before tuning performance)

1. **`MASTER_ADDR` / `MASTER_PORT` (head / rank 0)** — On **every** node, set `MASTER_ADDR` to the **rank-0 machine’s reachable IP** (Thunderbolt bridge `169.254.x.x`, LAN, etc.) and the same `MASTER_PORT` everywhere. Standard PyTorch multi-node; MCCL follows from that.

2. **Firewall (both sides)**  
   - PyTorch store: `MASTER_PORT` (e.g. 29500).  
   - MCCL: `MCCL_PORT_BASE` through `MCCL_PORT_BASE + world_size - 1` (default base is often `MASTER_PORT + 100` — see `ddp_dummy_train._setup_mccl_env`). Keep **`MCCL_PORT_BASE` ≠ `MASTER_PORT`**.

3. **Link**: Prefer **Thunderbolt bridge** or **wired Ethernet** over WiFi for bandwidth and stable latency.

4. **Same software**: Matching **PyTorch** and **MCCL** build on every node (nightly vs stable mismatch causes subtle failures).

5. **Elastic / launcher**: With `--nnodes=2`, workers start only after all nodes join — starting one machine alone can look like a hang until the second runs the same command.

## Production Thunderbolt profile (TCP)

Direct Mac-to-Mac Thunderbolt often uses **link-local** IPv4 (`169.254.x.x`). For a **production-ready, high-throughput TCP** setup:

1. Set **`MASTER_ADDR`** on every node to the **rank-0 Mac’s Thunderbolt IPv4** (the address on the bridge interface peers can reach).
2. **Multi-homed Macs**: when `MASTER_ADDR` is in `169.254.0.0/16`, MCCL **prefers the Thunderbolt bridge** for peer-visible addresses (`resolve_best_local_addr` in [`csrc/transport/TcpTransport.cpp`](../csrc/transport/TcpTransport.cpp)) so ranks do not pick Wi‑Fi/LAN by mistake.
3. **Opt-in tuning**: on **all** nodes, before `init_process_group`:
   - `export MCCL_LINK_PROFILE=thunderbolt`, or
   - `source scripts/thunderbolt_prod.sh`
   This enables **32 MB** default TCP send/recv buffers (if `MCCL_SOCK_BUFSIZE` is unset) in [`Connection.cpp`](../csrc/transport/Connection.cpp) and **at least 16 MB** transport chunks (if `MCCL_CHUNK_BYTES` is unset) in [`TransportConfig::from_env`](../csrc/transport/TcpTransport.cpp).
4. **Python**: `mccl.apply_thunderbolt_production_defaults(training_defaults=True)` before `dist.init_process_group` sets the profile plus optional `DDP_BUCKET_MB` / `MCCL_OVERLAP_COMM` defaults.
5. **RDMA** (Thunderbolt 5, when the OS exposes it): optional TCP bypass — [README.md](../README.md#rdma-over-thunderbolt-5).
6. **Validate**: run **`iperf3`** between the same two IPs you use for training; then compare `mccl.get_metrics()` with and without `MCCL_LINK_PROFILE=thunderbolt`.

Firewall: still allow **`MASTER_PORT`** and **`MCCL_PORT_BASE` … + world_size − 1** on **both** Macs.

See also the docstring in [`examples/ddp_dummy_train.py`](../examples/ddp_dummy_train.py).

## Fair measurement vs single-GPU baseline

DDP uses **`BATCH_SIZE` per rank**; **global batch** = `BATCH_SIZE * WORLD_SIZE`.

For apples-to-apples compute load vs **one Mac**:

```bash
# Example: DDP with BATCH_SIZE=4, world_size=2 → global batch 8
BASELINE_BATCH_SIZE=8 python examples/ddp_dummy_train.py --baseline
```

Or set env **`BASELINE_BATCH_SIZE`** to that global batch before `--baseline`.

Record **`mccl.get_metrics()`** on rank 0 after a DDP run: `avg_sync_ms`, `avg_network_ms`, `avg_reduce_ms`, `total_ops`, bytes.

## Why 2-node often feels slower than “single training” (and not sub-second)

This is **usually expected** with the default dummy model and TCP between Macs—not a sign that MCCL is “wrong.”

### 1. Apples-to-apples: global batch

- DDP: **`BATCH_SIZE` per rank** → global batch = `BATCH_SIZE × world_size`.
- If you compare **2-node DDP with `BATCH_SIZE=4`** to **one Mac with `BATCH_SIZE=4`**, the single-GPU run does **half the examples per step**. It *should* be faster per step.
- Fair baseline: **`BASELINE_BATCH_SIZE=(BATCH_SIZE × world_size)`** with `--baseline` (see above).

### 2. Even when the comparison is fair, wall-clock ≠ “half the work”

Each rank runs **full forward + full backward** on **its** microbatch. You buy throughput in **samples/sec** (global batch / step time), not a free halving of step time. Step time is roughly:

**max(compute on rank 0, compute on rank 1) + time spent in gradient collectives on the critical path.**

With **hook-driven bucketing**, many **allreduce** calls run during backward; if the **inter-host link has high latency**, those waits **add** (fully or partly—overlap depends on PyTorch/MCCL and bucket timing).

### 3. Order-of-magnitude: default small model

`examples/ddp_dummy_train.py` **defaults to a small MLP** (few M params) for quick runs. At that size, communication overhead often dominates compute time, making DDP slower than single GPU.

Gradients scale with parameter count (fp32 → **~4 GB** per full model per rank before compression at ~1B scale). With **many buckets/step** and **~100–200 ms average network time per op** on WiFi or even GbE, **communication alone** can be **multiple seconds**, on top of compute. **Sub-second** at huge width usually needs **smaller models**, **fp16** (compute + wire), **fewer/larger buckets**, and a **fast, stable link** (e.g. Thunderbolt bridge), not “better DDP” alone.

### 4. What to send when debugging (paste into an issue or notes)

For **single baseline** and **2-node DDP** runs (same `MODEL_*` / global batch):

| Item | Example |
|------|--------|
| Link | TB bridge / GbE / WiFi |
| `BATCH_SIZE`, `world_size`, global batch | 4, 2, 8 |
| `DDP_BUCKET_MB`, `MCCL_COMPRESSION`, `TRAIN_AUTOCAST_FP16` | |
| Avg step time (s), steps timed | |
| `mccl.get_metrics()`: `total_ops`, `avg_sync_ms`, `avg_network_ms`, `avg_reduce_ms`, MB sent/recv | |
| PyTorch + MCCL build (commit) | |

From that we can see whether you’re **compute-bound** (baseline ≈ DDP), **network-bound** (`avg_network_ms × ops/step` ≈ extra time), or **reduce/CPU-bound** (`avg_reduce_ms` large).

## Multi-node–first performance tuning

| Knob | Notes |
|------|--------|
| **`DDP_BUCKET_MB`** | Larger (e.g. 50–200) → **fewer** allreduces per step → fewer TCP round-trips over the inter-host link. Trade-off: memory / peak message size. |
| **`MCCL_COMPRESSION=fp16`** | Halves wire volume when compression is enabled in the build (`ProcessGroupMCCL` compressor path). |
| **FP16 training** | `TRAIN_AUTOCAST_FP16=1` in `ddp_dummy_train.py` (or your script) uses `torch.autocast("mps", dtype=torch.float16)` where supported. |
| **`MCCL_SYNC_MODE=full`** | Required for DDP gradient buckets. **Do not** use `coalesced` with hook-driven DDP (stale grads / broken pipe). |
| **`MCCL_SOCK_BUFSIZE`** | Override kernel socket buffer (bytes); default is large in `Connection.cpp`. Set `0` to let the kernel auto-tune. |
| **`MCCL_CHUNK_BYTES`** | Transport chunk size (see `TransportConfig::from_env()` in `TcpTransport.cpp`); affects CRC/chunked paths. |
| **`MCCL_TRANSPORT`** | `tcp` default; RDMA when available and configured (see `transport/rdma/`). |
| **`MCCL_LINK_PROFILE=thunderbolt`** | Production TCP defaults for Thunderbolt IP links: larger default **socket buffers** and **chunk size** (see [Production Thunderbolt profile](#production-thunderbolt-profile-tcp)). Use `scripts/thunderbolt_prod.sh` or `mccl.apply_thunderbolt_production_defaults()`. |
| **Model size** | Use larger models (custom `MODEL_HIDDEN`, `MODEL_DEPTH` env vars) to see where DDP becomes worthwhile vs single GPU. |

### Example: sweep `DDP_BUCKET_MB` (local or multi-node)

```bash
for mb in 25 50 100 200; do
  echo "=== DDP_BUCKET_MB=$mb ==="
  DDP_BUCKET_MB=$mb torchrun --nproc_per_node=2 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/ddp_dummy_train.py
done
```

On two Macs, use the same `torchrun` line as your real job (with `MASTER_ADDR`, `nnodes`, etc.) and only vary `DDP_BUCKET_MB`. Optionally add `MCCL_COMPRESSION=fp16` and/or `TRAIN_AUTOCAST_FP16=1` per row in the benchmark table below.

The dummy train example defaults to a **small** MLP where DDP is typically slower than single GPU due to communication overhead.

## TCP / transport (reference)

- **Socket buffers**: [`csrc/transport/Connection.cpp`](../csrc/transport/Connection.cpp) — `MCCL_SOCK_BUFSIZE`, `TCP_NODELAY`, macOS `TCP_NOTSENT_LOWAT` via `MCCL_TCP_LOWAT`.
- **Large message overlap**: [`csrc/transport/TcpTransport.cpp`](../csrc/transport/TcpTransport.cpp) — `send_recv_overlap` threshold and env-driven chunk sizes.
- **RDMA**: Optional; same `Transport` API — use when OS/hardware supports it for your topology.

## Optional: Metal gradient reduce (future work)

If **`avg_reduce_ms`** dominates at **large** buckets (vs `avg_network_ms`), a **Metal** reduction path could shrink CPU vDSP time. Scope: `allreduce_two_rank` / `AccelerateOps` and Metal kernels — non-trivial; profile first.

## Optional: `asyncOp` on allreduce

When PyTorch calls `allreduce` with **`async_op=True`**, MCCL can **defer** MPS `commit_mps_and_signal` to the **ProgressEngine** thread so the caller returns sooner (`ProcessGroupMCCL::allreduce`). Standard **DDP** still tends to wait on each `Work` in hooks; benefit is workload-dependent. Requires **`MCCL_OVERLAP_COMM`** (event sync) on.

## Quick benchmark table (fill in when comparing)

| Link (TB / GbE / WiFi) | `DDP_BUCKET_MB` | Step time (s) | `avg_network_ms` | `total_ops` / step |
|------------------------|-----------------|---------------|------------------|---------------------|
|                        |                 |               |                  |                     |
