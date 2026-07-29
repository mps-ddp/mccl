import torch
from mccl import _C

x = torch.randn(1, 4096)
w = torch.hann_window(2048)
print("calling _stft_forward", flush=True)
y = _C._stft_forward(x, w, 2048, 512, 2048, True, False, "vdsp")
print("done", y.shape, flush=True)
