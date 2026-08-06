"""MCCLConfig world-size invariant defaults."""

import os

import pytest

from mccl.config import MCCLConfig


def test_for_world_size_invariant():
    for ws in (4, 8, 16):
        cfg = MCCLConfig.for_world_size(ws)
        assert cfg.ring_algo == "auto"
        assert cfg.collective_concurrency == 1
        assert cfg.pipeline_depth == 1
        assert cfg.port_base == 20100
        assert cfg.fast_math is True
        assert cfg.unified_collective is True
        assert cfg.demux_inflight_budget_bytes == 1 * 1024 * 1024 * 1024
        assert cfg.ddp_bucket_mb == 25


def test_apply_to_env_only_if_unset(monkeypatch):
    monkeypatch.delenv("MCCL_RING_ALGO", raising=False)
    monkeypatch.setenv("MCCL_COLLECTIVE_CONCURRENCY", "3")
    cfg = MCCLConfig.for_world_size(8)
    applied = cfg.apply_to_env(only_if_unset=True)
    assert os.environ.get("MCCL_RING_ALGO") is None  # "auto" skipped
    assert os.environ.get("MCCL_COLLECTIVE_CONCURRENCY") == "3"
    assert "MCCL_COLLECTIVE_CONCURRENCY" not in applied


def test_perf_mode():
    cfg = MCCLConfig.for_world_size(8, mode="perf")
    assert cfg.ring_algo == "chunked"
    assert cfg.collective_concurrency >= 2


def test_ddp_bucket_and_demux_reservation_stay_coupled(monkeypatch):
    monkeypatch.delenv("DDP_BUCKET_MB", raising=False)
    monkeypatch.delenv("MCCL_DEMUX_MAX_COLLECTIVE_BYTES", raising=False)
    cfg = MCCLConfig(ddp_bucket_mb=96)
    applied = cfg.apply_ddp_bucket_env()
    assert applied == {
        "DDP_BUCKET_MB": "96",
        "MCCL_DEMUX_MAX_COLLECTIVE_BYTES": str(96 * 1024 * 1024),
    }


def test_ddp_bucket_coupling_preserves_explicit_env(monkeypatch):
    monkeypatch.setenv("DDP_BUCKET_MB", "64")
    monkeypatch.setenv("MCCL_DEMUX_MAX_COLLECTIVE_BYTES", "123")
    cfg = MCCLConfig(ddp_bucket_mb=96)
    assert cfg.apply_ddp_bucket_env(only_if_unset=True) == {}
    assert os.environ["DDP_BUCKET_MB"] == "64"
    assert os.environ["MCCL_DEMUX_MAX_COLLECTIVE_BYTES"] == "123"
