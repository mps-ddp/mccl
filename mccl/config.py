"""
Unified configuration for MCCL.

MCCLConfig is the single source of truth for every tunable.  It can be
constructed from defaults, env vars, keyword arguments, or a plain dict.
The ``to_env`` method writes the values back as ``MCCL_*`` environment
variables so the C++ layer (which reads env vars at init time) picks them
up without any C++ refactoring.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional


@dataclasses.dataclass
class MCCLConfig:
    """All MCCL tunables in one place."""

    # ── Transport ────────────────────────────────────────────────────
    transport: str = "auto"  # "auto", "tcp", "rdma"
    listen_addr: str = "auto"
    port_base: int = 29600
    ifname: str = ""
    chunk_bytes: int = 4 * 1024 * 1024
    small_msg_threshold: int = 65536
    connect_timeout_ms: int = 30000
    transport_crc: bool = False

    # ── Compute ──────────────────────────────────────────────────────
    fast_math: bool = True
    gpu_threshold: int = 4096
    shader_path: str = ""
    overlap_comm: bool = True

    # ── Compression ──────────────────────────────────────────────────
    compression: str = "none"
    topk_ratio: float = 0.01

    # ── Runtime ──────────────────────────────────────────────────────
    watchdog_timeout_ms: int = 300000
    heartbeat_interval_ms: int = 5000
    max_queue_depth: int = 1024
    log_level: str = "WARN"

    # ── env var name ↔ field mapping ─────────────────────────────────

    _ENV_MAP: dict[str, str] = dataclasses.field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._ENV_MAP = {
            "MCCL_TRANSPORT": "transport",
            "MCCL_LISTEN_ADDR": "listen_addr",
            "MCCL_PORT_BASE": "port_base",
            "MCCL_IFNAME": "ifname",
            "MCCL_CHUNK_BYTES": "chunk_bytes",
            "MCCL_SMALL_MSG_THRESHOLD": "small_msg_threshold",
            "MCCL_CONNECT_TIMEOUT_MS": "connect_timeout_ms",
            "MCCL_TRANSPORT_CRC": "transport_crc",
            "MCCL_FAST_MATH": "fast_math",
            "MCCL_GPU_THRESHOLD": "gpu_threshold",
            "MCCL_SHADER_PATH": "shader_path",
            "MCCL_OVERLAP_COMM": "overlap_comm",
            "MCCL_COMPRESSION": "compression",
            "MCCL_TOPK_RATIO": "topk_ratio",
            "MCCL_WATCHDOG_TIMEOUT_MS": "watchdog_timeout_ms",
            "MCCL_HEARTBEAT_INTERVAL_MS": "heartbeat_interval_ms",
            "MCCL_MAX_QUEUE_DEPTH": "max_queue_depth",
            "MCCL_LOG_LEVEL": "log_level",
        }
        self._validate()

    def _validate(self) -> None:
        valid_transports = {"auto", "tcp", "rdma"}
        if self.transport not in valid_transports:
            raise ValueError(
                f"transport must be one of {sorted(valid_transports)}, "
                f"got {self.transport!r}"
            )
        valid_compressions = {"none", "fp16", "topk"}
        if self.compression not in valid_compressions:
            raise ValueError(
                f"compression must be one of {sorted(valid_compressions)}, "
                f"got {self.compression!r}"
            )
        valid_log_levels = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"log_level must be one of {sorted(valid_log_levels)}, "
                f"got {self.log_level!r}"
            )
        if self.port_base < 1 or self.port_base > 65535:
            raise ValueError(
                f"port_base must be in [1, 65535], got {self.port_base}"
            )
        if self.gpu_threshold < 0:
            raise ValueError(
                f"gpu_threshold must be >= 0, got {self.gpu_threshold}"
            )
        if not (0.0 < self.topk_ratio <= 1.0):
            raise ValueError(
                f"topk_ratio must be in (0.0, 1.0], got {self.topk_ratio}"
            )
        if self.max_queue_depth < 1:
            raise ValueError(
                f"max_queue_depth must be >= 1, got {self.max_queue_depth}"
            )
        if self.chunk_bytes < 1:
            raise ValueError(
                f"chunk_bytes must be >= 1, got {self.chunk_bytes}"
            )
        if self.watchdog_timeout_ms < 1:
            raise ValueError(
                f"watchdog_timeout_ms must be >= 1, got {self.watchdog_timeout_ms}"
            )

    # ── Construction helpers ─────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "MCCLConfig":
        """Build a config by reading every ``MCCL_*`` env var."""
        cfg = cls()
        env_map = cfg._ENV_MAP
        for env_key, field_name in env_map.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            fld = _field_type(cfg, field_name)
            setattr(cfg, field_name, _coerce(raw, fld))
        return cfg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCCLConfig":
        """Build from a plain dict (e.g. parsed YAML/JSON)."""
        known = {f.name for f in dataclasses.fields(cls) if f.name != "_ENV_MAP"}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    # ── Export ────────────────────────────────────────────────────────

    def to_env(self) -> Dict[str, str]:
        """Write all values into ``os.environ`` and return the mapping."""
        out: Dict[str, str] = {}
        for env_key, field_name in self._ENV_MAP.items():
            val = getattr(self, field_name)
            if isinstance(val, bool):
                s = "1" if val else "0"
            else:
                s = str(val)
            if s and s != "auto" and s != "":
                os.environ[env_key] = s
                out[env_key] = s
            elif env_key in os.environ:
                del os.environ[env_key]
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for logging / JSON export)."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "_ENV_MAP"
        }

    def __str__(self) -> str:
        lines = ["MCCLConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ── Internal helpers ─────────────────────────────────────────────────

def _field_type(obj: Any, name: str) -> type:
    for f in dataclasses.fields(obj):
        if f.name == name:
            return f.type  # type: ignore[return-value]
    return str


def _coerce(raw: str, target_type: Any) -> Any:
    if target_type is bool or target_type == "bool":
        return raw not in ("0", "false", "FALSE", "no", "NO", "")
    if target_type is int or target_type == "int":
        return int(raw)
    if target_type is float or target_type == "float":
        return float(raw)
    return raw
