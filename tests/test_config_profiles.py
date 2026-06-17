"""MCCLConfig.for_world_size lab/perf profiles."""

import os

import pytest

from mccl.config import MCCLConfig


def test_for_world_size_lab_ws4():
    cfg = MCCLConfig.for_world_size(4)
    assert cfg.ring_algo == "chunked"
    assert cfg.collective_concurrency == 2
    assert cfg.ddp_bucket_mb == 25


def test_for_world_size_lab_ws8():
    cfg = MCCLConfig.for_world_size(8)
    assert cfg.ring_algo == "basic"
    assert cfg.collective_concurrency == 1
    assert cfg.ddp_bucket_mb == 25
    assert cfg.demux_park_bytes == 1024 * 1024 * 1024


def test_for_world_size_lab_ws16():
    cfg = MCCLConfig.for_world_size(16)
    assert cfg.ring_algo == "basic"
    assert cfg.collective_concurrency == 1
    assert cfg.ddp_bucket_mb == 50


def test_apply_to_env_only_if_unset(monkeypatch):
    monkeypatch.delenv("MCCL_RING_ALGO", raising=False)
    monkeypatch.setenv("MCCL_COLLECTIVE_CONCURRENCY", "3")
    cfg = MCCLConfig.for_world_size(8)
    applied = cfg.apply_to_env(only_if_unset=True)
    assert os.environ.get("MCCL_RING_ALGO") == "basic"
    assert os.environ.get("MCCL_COLLECTIVE_CONCURRENCY") == "3"
    assert "MCCL_COLLECTIVE_CONCURRENCY" not in applied


def test_perf_mode():
    cfg = MCCLConfig.for_world_size(8, mode="perf")
    assert cfg.ring_algo == "chunked"
    assert cfg.collective_concurrency >= 2
