# Changelog

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
