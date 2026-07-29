# Changelog

## v6.5 — Production multi-node defaults

### Changed
- **Unset-env defaults** aligned with the lab multi-node profile: `MCCL_PIPELINE_DEPTH=1`, `MCCL_COLLECTIVE_CONCURRENCY=1`, `MCCL_CHUNK_BYTES=16MB`, `MCCL_PORT_BASE=20100`, `MCCL_UNIFIED_COLLECTIVE` on, `MCCL_DEMUX_INFLIGHT_BUDGET_BYTES=1GB`. (`MCCL_FAST_MATH` stays on by default; SAO opts into strict math via env.)
- **Demux park auto-scale** retained when `MCCL_DEMUX_PARK_BYTES` is unset; hard cap raised from 2 GiB to **4 GiB**.

### Fixed
- **`stage_for_send_collective`**: drop redundant pre/post ``mccl_queue_drain()`` when EventSync is on — ``chunked_blit_to_staging`` already ``waitUntilCompleted`` on the MCCL queue (orders behind prior reduces). Removes two empty-buffer GPU barriers per DDP bucket.

## v6.1 — Concurrency ceiling 8 + bucket budget

### Changed
- **`collective_concurrency_max`**: ``MCCL_MAX_COLLECTIVE_CONCURRENCY`` (default 8) replaces hard ``min(requested, 2)``.
- **`demux_inflight_budget_bytes`**: caps effective concurrency as ``budget / DDP_bucket`` (override ``MCCL_DEMUX_INFLIGHT_BUDGET_BYTES``). ws=8 default budget 128 MiB → conc=4 at 8 MiB buckets, conc=2 at 25 MiB, conc=1 at 64 MiB+.

### Added
- **`test_conc4_small_bucket_ddp_multibucket`**: ws=8, 8 MiB buckets, ``MCCL_COLLECTIVE_CONCURRENCY=4``.

## v6.0 — Bucket-aware collective concurrency

### Changed
- **`effective_collective_concurrency`**: replaces blind ws≥5 → 1 cap. `MCCL_COLLECTIVE_CONCURRENCY=2` is allowed when `MCCL_DEMUX_MAX_COLLECTIVE_BYTES` ≤ 25 MiB (ws≥5) or ≤ 16 MiB (ws≥8). Larger DDP buckets force concurrency 1 to avoid TCP demux / ENOBUFS under overlapped 64 MiB ring collectives.
- Wire **`MCCL_DEMUX_MAX_COLLECTIVE_BYTES`** to **`DDP_BUCKET_MB`** on workers so bucket growth auto-serializes collectives.

### Added
- **`test_collective_concurrency_v6.py`**: dual async allreduce + DDP multibucket at ws=5/8; small-bucket concurrency=2 vs large-bucket cap-to-1.

## v5.8 — Overlap producer pipeline fence (step-0 backward Metal fix)

### Fixed
- **Producer pipeline fence**: with `MCCL_OVERLAP_COMM=1`, each new collective on the autograd thread calls `wait_prior_overlap_release_on_producer()` before `commit_mps_and_signal`. Bucket *N+1* no longer commits PyTorch MPS while bucket *N* MCCL GPU work is still in flight (step-0 backward `command buffer exited with error status`).
- **Release cursor**: `arm_work_release` publishes monotonic tokens; `Work::wait` updates the consumed cursor so producer and consumer fences stay aligned.

## v5.7 — DDP consumer release (NCCL stream-return semantics)

### Fixed
- **`WorkMCCL::wait` consumer fence**: `publish_collective_release` arms an MCCL shared-event token per collective; the autograd/DDP thread must `wait_for_mccl` in `Work::wait` before PyTorch MPS resumes encoding. Previously `markComplete` consumed the release on the ProgressEngine thread, so overlapped buckets could resume backward on stale reduced gradients (train/val boundary Metal command-buffer death).
- **`barrier`**: arms the same release token so `dist.barrier()` at epoch boundaries fences the MCCL queue when `MCCL_OVERLAP_COMM=1`.
- **Token=0 fallback**: `Work::wait` drains the MCCL queue when EventSync is on but no per-op token was published (e.g. overlap off).

### Added
- **`test_ddp_consumer_release.py`**: weight-trajectory parity without grad `.item()` sync between backward and `optimizer.step`; train→barrier→rank-skewed val→train; barrier + post-barrier allreduce + MPS burst. Catches the engine-thread `release_waited_` bug that grad-only parity masked.

## v5.5 — ws≥5 collective concurrency cap

### Fixed
- **`effective_collective_concurrency`**: ws≥5 caps `MCCL_COLLECTIVE_CONCURRENCY` to 1 in both the collective pool and demux park-limit scaling. Prevents ENOBUFS / peer death when env requests concurrency=2 with large async DDP buckets (cluster ws=5 regression).
- **Test harness**: `mccl_test_utils.next_port()` uses pid+jitter to avoid EADDRINUSE when pytest runs files in parallel.

## v5.4 — Small allgather star (multi-node DDP init_sync)

### Fixed
- **`allgather_star_small` completeness**: send to all `dst != src` (not only `dst > src`).
- **`recv_chunks` on BROADCAST/ALLGATHER**: mesh hops use symmetric `collective_send_only` / `collective_recv_only` on the ALLREDUCE wire opcode (`send_recv_overlap` with nbytes on both legs). One-sided `send_chunks`/`recv_chunks` and BROADCAST/ALLGATHER demux tags dropped payload on multi-hop paths.
- **2-rank broadcast**: mirror `allreduce_two_rank` transport — default fp32 uses `compressed_send`/`compressed_recv` with separate `recv_tensor` + root ack recv; `MCCL_FP32_CPU_REDUCE=1` uses symmetric `send_recv_overlap`.
- **ws≥3 small broadcast**: root-star (`broadcast_star_small`) replaces tree/fanout for payloads below `small_msg_threshold`.
- **small allgather**: star uses compressed send/recv + ack (same wire as broadcast).
- **Collective entry sync**: store rendezvous (`rendezvous_collective_enter`) before broadcast/allgather wire I/O so fast ranks cannot send before slow ranks post recv.
- **`wait_recv`**: fail loudly on `received < nbytes` instead of returning success.
- **MPS recv → Metal reduce**: `unstage_from_recv` / `blit_buffer_to_tensor` no longer memcpy into unified `cpu_ptr` for MPS tensors (Metal kernels read device memory). Fixes ring allreduce, ring pipeline, and DDP gradient multibucket parity.
- **Metal default for allreduce**: CPU unified-buffer reduce (tree, ring `cpu_ptr`, inplace pipeline recv) is opt-in only (`MCCL_FP32_CPU_REDUCE=1`). Default uses Metal kernels + blit staging for all dtypes.
- **`metal_reduce_op_fenced`**: wait on fence / queue drain before return so async engine paths (two-rank split, ring) expose completed gradients to DDP.

### Added
- Test: `test_allgather_int64_param_vec_ws8_prod_env` (111×int64, `MCCL_OVERLAP_COMM=1`).
- Test: `test_allgather_distinct_values_ws8` (per-rank unique scalars; catches incomplete star).

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
