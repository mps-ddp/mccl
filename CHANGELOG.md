# Changelog

## v5.4 — Small allgather star (multi-node DDP init_sync)

### Fixed
- **`allgather` for `nbytes <= small_msg_threshold`**: use rank-ordered **star** exchange (direct mesh send src→dst) instead of lock-step **ring**. Ring neighbor hops could leave distant slots zeroed on multi-node TCP (`RuntimeError: Rank N has 111 params, rank 0 has inconsistent 0 params`) even with `MCCL_RING_PIPELINE=0` and mccl 0.5.0. Star matches the reliability model of `broadcast_tree_small` for small payloads.

### Added
- Test: `test_allgather_int64_param_vec_ws8_prod_env` (111×int64, `MCCL_OVERLAP_COMM=1`).

## v5.3 — Multi-node broadcast wire buffers (broadcast_object_list / large init_sync)

### Fixed
- **`broadcast_ring_pipelined`**: large broadcasts (`nbytes > small_msg_threshold`, e.g. `broadcast_object_list` payloads) no longer recv/send into tensor unified-memory / `data_ptr` over TCP. Uses one pooled wire buffer per rank + `unstage_from_recv` on non-root — same rule as `broadcast_tree_small` (fixes multi-node `EOFError` on unpickle).
- **Small fan-out broadcast recv** (`world_size < 4` non-root path): always recv into pooled buffer + `unstage_from_recv` (removed direct `cpu_ptr` recv).
- **`stage_for_send_collective`**: CPU tensors use direct `data_ptr` (no StagingPool blit hop) for DDP verify / metadata collectives.

## v5.2 — Small allgather lock-step (DDP init_sync)

### Fixed
- **Standalone `allgather` / `reduce_scatter` with `MCCL_RING_PIPELINE=1`**: small messages (below `small_msg_threshold`, e.g. DDP param-count int64 metadata) now use lock-step ring + pooled recv + `unstage_from_recv` instead of the streaming ring pipeline COPY path, which could report rank 0 as 0 params at ws=8 (`RuntimeError: Rank 1 has 111 params, rank 0 has 0`).

### Added
- Tests: `test_allgather_int64_ws8_ring_pipeline`, `test_ddp_init_sync_ws8_ring_pipeline`.

## v4.8 — Ring pipeline correctness at ws=8+

### Fixed
- **Pipeline TX gate/credit protocol**: always wait on gate and credit for step `g` even when `send_bytes==0` (no wire send); fixes credit misalignment deadlocks on multi-node.
- **Demux park limit**: scale with `world_size × MCCL_COLLECTIVE_CONCURRENCY` (2GB cap) as documented in v4.4.

### Added
- Tests: ws=8 async 25MB buckets, odd chunk boundaries, skewed-start lockstep, 25MB f64 gradient reference.

## v4.4 — Mac cluster scaling (ws > 4)

### Fixed
- **Ring allreduce above 4 workers**: default to basic ring (not chunked) when `WORLD_SIZE > 4` and `MCCL_RING_ALGO` unset; collective concurrency defaults to 1.
- **Demux park limit**: auto-scale `park_limit_bytes` with `world_size × MCCL_COLLECTIVE_CONCURRENCY` when `MCCL_DEMUX_PARK_BYTES` unset (cap 2GB).
- **Chunked ring fallback**: on chunked-ring failure, retry basic ring when `MCCL_RING_FALLBACK_BASIC=1` (default for ws > 4).
- **Integral reduce ops**: bool/int32/int64 Metal + CPU paths for tree/small collectives.

### Added
- `MCCLConfig.for_world_size()` and `apply_to_env()` in Python (`mccl.config`).
- Tests: `test_config_profiles.py`, `test_ring_ws8.py`.

### Integration (ml-style-fx-transfer)
- `submit_job.sh`: lab profile defaults for ws > 4 (basic ring, concurrency 1, 25–50MB DDP buckets).
- `trainers/ray_mccl_backend.py`: scale-aware MCCL profile; Thunderbolt detection via 169.254.x.x only.
- `train_mask_mccl_fx_transfer.py`: auto `--mccl_ring basic` when `num_workers > 4`; conservative sync no longer sets dead `MCCL_SYNC_MODE=full`.

### Deferred
- float64 DDP reduce paths
- Removing app-level int allreduce patches (until cluster matrix green)
