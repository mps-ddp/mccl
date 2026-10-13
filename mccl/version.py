__version__ = "6.8.0"

COMPATIBILITY_MATRIX = {
    "macos": ["15.x", "26.x"],
    "python": ["3.11", "3.12", "3.13"],
    "pytorch": ["2.13.x", "2.14.x"],
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
