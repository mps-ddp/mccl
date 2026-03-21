# Compatibility

Values below mirror `mccl.version.COMPATIBILITY_MATRIX` (see `mccl/version.py`).

| Area | Supported / notes |
|------|-------------------|
| **OS** | macOS (tested matrix includes 15.x; see `version.py` for exact patch list) |
| **CPU** | Apple Silicon **arm64** only (no Intel) |
| **Python** | 3.11, 3.12 |
| **PyTorch** | 2.5.x, 2.6.x (project depends on `torch>=2.5`) |
| **Device** | **MPS** tensors for collectives |
| **Backend name** | `mccl` |

**RDMA** (optional): requires supported hardware, recent macOS, and system setup — see [README.md](README.md#rdma-over-thunderbolt-5).

When upgrading PyTorch or macOS, run the test suite and a small `examples/ddp_dummy_train.py` run.
