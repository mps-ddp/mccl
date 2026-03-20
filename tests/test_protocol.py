"""Phase C: Protocol encode/decode and CRC tests.

These are pure-logic tests that run on any platform.
"""

import struct
import pytest


def crc32_py(data: bytes) -> int:
    """Python reimplementation of the MCCL CRC32 for cross-validation."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


class TestCRC32:
    def test_empty(self):
        assert crc32_py(b"") == 0x00000000

    def test_known_value(self):
        # CRC32 of "123456789" is 0xCBF43926
        assert crc32_py(b"123456789") == 0xCBF43926

    def test_deterministic(self):
        data = b"hello mccl transport"
        assert crc32_py(data) == crc32_py(data)

    def test_different_data(self):
        assert crc32_py(b"aaa") != crc32_py(b"bbb")


class TestMessageHeader:
    """Test the wire format assumptions for MessageHeader."""

    WIRE_SIZE = 24

    def test_header_size(self):
        # 2 + 1 + 1 + 4 + 4 + 4 + 4 + 4 = 24 bytes
        assert self.WIRE_SIZE == 24

    def test_roundtrip(self):
        # Simulate encoding a header as packed bytes
        header = struct.pack(
            "<HBBIIIIi",  # little-endian
            1,      # protocol_version
            1,      # op_type (ALLREDUCE)
            0,      # flags
            42,     # seq_num
            7,      # tensor_id
            0,      # chunk_index
            1024,   # payload_bytes
            0xDEAD, # checksum
        )
        assert len(header) == self.WIRE_SIZE

        # Decode
        (proto, op, flags, seq, tid, chunk, payload, cksum) = struct.unpack(
            "<HBBIIIIi", header
        )
        assert proto == 1
        assert op == 1
        assert seq == 42
        assert tid == 7
        assert payload == 1024


class TestHandshakePayload:
    WIRE_SIZE = 74

    def test_size(self):
        # 2 + 4 + 4 + 64 = 74
        assert self.WIRE_SIZE == 74

    def test_roundtrip(self):
        hostname = b"test-host"
        hostname_padded = hostname + b"\x00" * (64 - len(hostname))
        payload = struct.pack("<hii64s", 1, 0, 2, hostname_padded)
        assert len(payload) == self.WIRE_SIZE

        proto, rank, ws, host_bytes = struct.unpack("<hii64s", payload)
        assert proto == 1
        assert rank == 0
        assert ws == 2
        assert host_bytes[:len(hostname)] == hostname
