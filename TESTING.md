# Testing

## Prerequisites

- Build the package: `pip install -e ".[dev]"` on **macOS Apple Silicon**.

## Run all tests

```bash
pytest tests/
```

## Useful subsets

```bash
# Import / build / backend registration
pytest tests/test_build.py -

# Process group and collectives (local)
pytest tests/test_process_group_local.py -v

# Metal / kernel paths
pytest tests/test_local_kernels.py -v

# Longer soak (may take time)
pytest tests/test_soak.py -v
```

## Multi-host / DDP integration

Tests like `tests/test_two_host_ddp.py` may assume specific environments (multiple machines, env vars). Read each file’s docstring and skip conditions.

**Manual 2-Mac checklist and perf knobs** (firewall, `MCCL_LISTEN_ADDR`, `DDP_BUCKET_MB`, `MCCL_SOCK_BUFSIZE`, …): [docs/MULTINODE.md](docs/MULTINODE.md).

**Fair baseline vs DDP**: use the same **global batch** when comparing single-GPU `--baseline` to multi-node DDP (see `BASELINE_BATCH_SIZE` in `examples/ddp_dummy_train.py`).

## Logs

`pyproject.toml` configures pytest to write `test_failures.log` on failures.

## CI

If you add CI, mirror: Python 3.11+, macOS arm64 agent, Xcode CLT, `pip install -e ".[dev]"`, then `pytest`.
