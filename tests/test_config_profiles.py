"""MCCLConfig world-size invariant defaults."""

import os

import pytest

from mccl.config import MCCLConfig


def test_for_world_size_invariant():
    for ws in (4, 8, 16):
        cfg = MCCLConfig.for_world_size(ws)
        assert cfg.ring_algo == "auto"
        assert cfg.collective_concurrency == 2
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
