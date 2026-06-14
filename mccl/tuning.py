"""
Optional performance presets for MCCL (set ``os.environ`` before ``init_process_group``).
"""
from __future__ import annotations

import os


def apply_thunderbolt_production_defaults(
    *,
    training_defaults: bool = False,
) -> None:
    """Opt in to Thunderbolt-oriented TCP defaults (large buffers, 16MB chunks).

    Sets ``MCCL_LINK_PROFILE=thunderbolt`` if unset. The native layer then applies:

    - Larger default socket buffers (see ``Connection::configure_socket``).
    - At least **16 MiB** ``MCCL_CHUNK_BYTES`` if that env var is unset.

    If ``training_defaults`` is True, also sets (when unset) ``MCCL_OVERLAP_COMM=1``
    and ``DDP_BUCKET_MB=512`` for PyTorch DDP workloads.

    Call **before** ``dist.init_process_group(backend="mccl", ...)`` on every node.
    """
    os.environ.setdefault("MCCL_LINK_PROFILE", "thunderbolt")
    if training_defaults:
        os.environ.setdefault("MCCL_OVERLAP_COMM", "1")
        os.environ.setdefault("DDP_BUCKET_MB", "512")


def apply_ethernet_cluster_defaults(
    *,
    training_defaults: bool = False,
) -> None:
    """Conservative defaults for multi-rank TCP/Ethernet clusters (e.g. 24 Macs).

    Sets when unset:

    - ``MCCL_LINK_PROFILE=ethernet`` — 4 MiB transport chunks, pipeline depth 3.
    - ``MCCL_PIPELINE_DEPTH=3`` — extra posted-ahead receives for higher latency links.
    - ``MCCL_COLLECTIVE_CONCURRENCY=2`` — safe bucket overlap (raise to 3 after profiling).

    Demux park bytes auto-scale from concurrency × pipeline depth × chunk size in C++.

    If ``training_defaults`` is True, also sets ``DDP_BUCKET_MB=100`` (fewer, larger
    buckets over GbE) and ``MCCL_OVERLAP_COMM=1``.

    Call **before** ``dist.init_process_group(backend="mccl", ...)`` on every node.
    """
    os.environ.setdefault("MCCL_LINK_PROFILE", "ethernet")
    os.environ.setdefault("MCCL_PIPELINE_DEPTH", "3")
    os.environ.setdefault("MCCL_COLLECTIVE_CONCURRENCY", "2")
    if training_defaults:
        os.environ.setdefault("MCCL_OVERLAP_COMM", "1")
        os.environ.setdefault("DDP_BUCKET_MB", "100")


__all__ = [
    "apply_thunderbolt_production_defaults",
    "apply_ethernet_cluster_defaults",
]
