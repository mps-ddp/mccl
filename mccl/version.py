__version__ = "0.3.2"

COMPATIBILITY_MATRIX = {
    "macos": ["15.0", "15.1", "15.2", "15.3", "15.4"],
    "python": ["3.11", "3.12"],
    "pytorch": ["2.5.x", "2.6.x"],
    "hardware": "Apple Silicon (M1/M2/M3/M4 family)",
    "backend_name": "mccl",
    "supported_devices": ["mps"],
    "supported_collectives": [
        "allreduce", "broadcast", "barrier",
        "allgather", "reduce_scatter",
        "send", "recv",
    ],
    "compression_modes": ["none", "fp16", "topk"],
}
