# Benchmark results

Community-contributed timings. **Add a row via PR** after you run something reproducible.

## Template

| Hardware | Link | Model | Batch | DDP_BUCKET_MB | Step time | MCCL ops/step | avg_network_ms | Notes |
|----------|------|-------|-------|---------------|-----------|---------------|----------------|-------|
| M1 Max + M4 Max | TB3 | ~4M params | 8 | 100 | ~14ms | ~1.3 | ~10ms | Default model, 2 ranks local `torchrun` |

## How to contribute

1. Run `bash scripts/benchmark_matrix.sh` from the repo root (or your own `torchrun` setup).
2. Note:
   - Mac models (e.g. “M2 Ultra + M4 Pro”)
   - Link (TB3 / TB4 / Ethernet / Wi‑Fi)
   - Model config (default params or custom env vars)
   - **`mccl.get_metrics()`** summary if available
   - Any flags: `MCCL_COMPRESSION`, `MCCL_LINK_PROFILE`, `MCCL_TRANSPORT`, etc.
3. Open a PR that appends a row to the table above (or paste into an issue if you prefer).
