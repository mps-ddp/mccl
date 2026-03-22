#!/usr/bin/env python3
"""
Minimal DDP training on a **learnable synthetic task** (Gaussian inputs, fixed linear
labeler) using MCCL + MPS.

Run with torch.distributed.run (``torchrun``). MCCL must be importable on every node.

**Single-GPU baseline (same model/hparams as DDP, no torchrun)**::

    python examples/ddp_dummy_train.py --baseline

Or: ``SINGLE_GPU=1 python examples/ddp_dummy_train.py`` (same behavior).

**Single Mac (2 processes, one GPU)**::

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/ddp_dummy_train.py

**Compare single vs DDP on one machine**::

    ./examples/compare_single_vs_ddp.sh

**Two Macs (1 process per machine)** — pick an IP on machine A that machine B can reach
(Thunderbolt bridge ``169.254.x.x`` or LAN). Open firewall for ``MASTER_PORT`` and
``MCCL_PORT_BASE .. MCCL_PORT_BASE + world_size - 1`` (defaults: master_port + 100).

Machine A (rank 0 / master)::

    export MASTER_ADDR=192.168.1.10   # or 169.254.x.x on TB bridge
    export MASTER_PORT=29500
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        examples/ddp_dummy_train.py

Machine B::

    export MASTER_ADDR=192.168.1.10   # same as machine A
    export MASTER_PORT=29500
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        examples/ddp_dummy_train.py

**Model size** — **defaults are a few‑M‑param** MLP (``INPUT_DIM=512``, ``NUM_CLASSES=64``,
``MODEL_HIDDEN=1024``, ``MODEL_DEPTH=4``) so runs finish quickly and multi-node **TCP overhead**
is easier to see. Override any dim with env:

- ``INPUT_DIM``, ``NUM_CLASSES``, ``MODEL_HIDDEN``, ``MODEL_DEPTH``

**Large stress model (~1B params)** — ``MCCL_STRESS_MODEL=1`` restores the old wide defaults
(``2048 / 128 / 8192 / 16``) unless you override the vars above.

If MPS runs out of memory, lower ``BATCH_SIZE`` (e.g. 2) or reduce ``MODEL_HIDDEN`` / ``MODEL_DEPTH``.

Optional env (see MCCL docs): ``MCCL_LISTEN_ADDR``, ``MCCL_PORT_BASE``, ``MCCL_TRANSPORT``,
``MCCL_LINK_PROFILE=thunderbolt`` (production TCP tuning on Thunderbolt IP — see ``scripts/thunderbolt_prod.sh``).
Training env: ``TRAIN_STEPS`` (default 500), ``BATCH_SIZE`` (default **128** per DDP rank → global **256** with 2 ranks),
``DDP_BUCKET_MB`` (default 25; **512** if ``MCCL_LINK_PROFILE=thunderbolt`` and unset; else try **50–200+** for 2-node),
``TRAIN_AUTOCAST_FP16=1`` for ``torch.autocast`` fp16 forward+loss (smaller/faster on MPS),
``MCCL_COMPRESSION=fp16`` when supported (halves cross-host bytes).

**Multi-node guide**: ``docs/MULTINODE.md`` (firewall, listen addr, baseline matching).

**Fair baseline vs DDP**: default baseline batch is **256** (global on one GPU).
DDP default is **128 per rank** × **world 2** = **256** global. Override with
``BASELINE_BATCH_SIZE`` / ``BATCH_SIZE`` if you OOM.

**Why it looks like a "hang"**

1. **Torch elastic** — With ``--nnodes=2``, the launcher often **waits until both nodes join**
   before worker processes start. Starting only machine A can show **no script output** for a long
   time; that is normal until machine B runs the same command.

2. **PyTorch store** — Every rank must reach ``init_process_group``. If machine B cannot open
   TCP to ``MASTER_ADDR:MASTER_PORT`` (default 29500), it retries forever. **Open inbound TCP
   on the master** for that port (macOS Firewall / Security).

3. **MCCL** — After the store works, ranks exchange ``IP:port`` for MCCL's own sockets. If the
   published address is wrong (e.g. loopback or another interface), peers **hang in
   ``connect_all``**. On each Mac set ``MCCL_LISTEN_ADDR`` to that machine's **reachable**
   unicast IP on the link between the two (not ``.255`` broadcast, not ``127.0.0.1``).

4. **Ports to allow** (world_size=2, default): ``MASTER_PORT`` (e.g. 29500) and
   ``MCCL_PORT_BASE`` … ``MCCL_PORT_BASE+1`` (e.g. 29600–29601 if ``MCCL_PORT_BASE=29600``).

Debug: ``export TORCH_DISTRIBUTED_DEBUG=DETAIL`` and ``export PYTHONUNBUFFERED=1``.

**Stats for plots:** ``--save-stats PATH.json`` writes step times and throughput; pair with
``examples/benchmark_throughput.py`` for ``.npz`` / charts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Register MCCL with torch.distributed before touching the process group.
import mccl  # noqa: F401

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _model_dims_from_env() -> tuple[int, int, int, int]:
    """Return INPUT_DIM, NUM_CLASSES, MODEL_HIDDEN, MODEL_DEPTH from env.

    Default is a few-M-param MLP. ``MCCL_STRESS_MODEL=1`` selects ~1B-param stress defaults.
    Override any dim with INPUT_DIM / NUM_CLASSES / MODEL_HIDDEN / MODEL_DEPTH.
    """
    stress = os.environ.get("MCCL_STRESS_MODEL", "").lower() in ("1", "true", "yes")
    if stress:
        input_dim = int(os.environ.get("INPUT_DIM", "2048"))
        num_classes = int(os.environ.get("NUM_CLASSES", "128"))
        hidden = int(os.environ.get("MODEL_HIDDEN", "8192"))
        depth = int(os.environ.get("MODEL_DEPTH", "16"))
    else:
        input_dim = int(os.environ.get("INPUT_DIM", "512"))
        num_classes = int(os.environ.get("NUM_CLASSES", "64"))
        hidden = int(os.environ.get("MODEL_HIDDEN", "1024"))
        depth = int(os.environ.get("MODEL_DEPTH", "4"))
    return input_dim, num_classes, hidden, depth


class SyntheticDataset:
    """Well-separated Gaussian clusters — trivially learnable for any model.

    Fixed class centroids are spread far apart (scale ``separation``).  Each batch
    picks a random class per sample, adds small Gaussian noise, and returns (x, y).
    Identical centroids on every rank; per-step RNG gives different samples per rank.
    """

    def __init__(
        self, input_dim: int, num_classes: int, seed: int = 424242, separation: float = 5.0,
    ) -> None:
        g = torch.Generator()
        g.manual_seed(seed)
        self.centroids = torch.randn(num_classes, input_dim, generator=g) * separation
        self.input_dim = input_dim
        self.num_classes = num_classes

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        step: int,
        rank: int,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(1000 + step * world_size + rank)
        y = torch.randint(0, self.num_classes, (batch_size,))
        centroids = self.centroids.to(device=device)
        noise = torch.randn(batch_size, self.input_dim, device=device) * 0.3
        x = centroids[y] + noise
        y = y.to(device=device)
        return x, y


def _write_training_stats(
    path: str,
    mode: str,
    step_times: list[float],
    losses: list[float],
    batch_size: int,
    world_size: int,
    total_params: int,
) -> None:
    avg_time = sum(step_times) / len(step_times) if step_times else 0.0
    throughput = (batch_size * world_size) / avg_time if avg_time > 0 else 0.0
    payload = {
        "mode": mode,
        "step_times": step_times,
        "losses": losses,
        "avg_step_time_s": avg_time,
        "throughput_samples_per_sec": throughput,
        "batch_size_per_rank": batch_size,
        "world_size": world_size,
        "global_batch_size": batch_size * world_size,
        "total_params": total_params,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class MultiHeadAttention(nn.Module):
    """Simple multi-head attention for testing MPS performance."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation (computationally expensive)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(attn_output)


class ConvBlock(nn.Module):
    """Convolutional block with large kernels and multiple operations."""
    def __init__(self, in_channels: int, out_channels: int, use_large_kernels: bool = True):
        super().__init__()
        
        if use_large_kernels:
            # Large kernel convolutions - much more compute intensive
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=4)  # Very large kernel
            
            # Depthwise separable convolution with large kernel
            self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=11, padding=5, groups=out_channels)
            self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        else:
            # Standard small kernels
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Batch normalization for each conv layer
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        # Activations and regularization
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # More compute-intensive activation
        self.dropout = nn.Dropout2d(0.1)
        
        # Squeeze-and-excitation block for more computation
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(out_channels, out_channels // 4)
        self.se_fc2 = nn.Linear(out_channels // 4, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        
        # Large kernel convolution sequence
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.gelu(self.bn3(self.conv3(x)))
        
        # Depthwise separable convolution
        x = self.relu(self.bn4(self.depthwise(x)))
        x = self.gelu(self.bn5(self.pointwise(x)))
        
        # Squeeze-and-excitation attention
        b, c, h, w = x.shape
        se_weight = self.se_pool(x).view(b, c)
        se_weight = self.relu(self.se_fc1(se_weight))
        se_weight = self.sigmoid(self.se_fc2(se_weight)).view(b, c, 1, 1)
        x = x * se_weight
        
        # Residual connection if dimensions match
        if identity.shape == x.shape:
            x = x + identity
            
        return x


class HeavyDummyModel(nn.Module):
    """More computationally intensive model for better MPS testing."""
    def __init__(self, input_dim: int, num_classes: int, hidden: int, depth: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden)
        
        # CNN path with large kernels - FIXED architecture regardless of depth
        conv_spatial = 64
        self.conv_proj = nn.Linear(hidden, conv_spatial * conv_spatial * 16)
        
        # Always build the same conv layers (ignore depth for conv)
        self.conv1 = ConvBlock(16, 32, use_large_kernels=True)
        self.conv2 = ConvBlock(32, 64, use_large_kernels=True)
        self.conv3 = ConvBlock(64, 64, use_large_kernels=True)
        
        # Dilated convolutions
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        
        # Attention path - FIXED number of layers
        self.attn_proj = nn.Linear(64 * 16 * 16, hidden)
        self.seq_len = 16
        self.pos_encoding = nn.Parameter(torch.randn(self.seq_len, hidden))
        
        # Always 2 attention layers (ignore depth)
        self.attn1 = MultiHeadAttention(hidden, n_heads=8)
        self.attn2 = MultiHeadAttention(hidden, n_heads=8)
        
        # Final MLP - FIXED architecture
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initial projection
        x = self.input_proj(x)  # [batch, hidden]
        
        # Convolutional path with large kernels
        conv_x = self.conv_proj(x)  # [batch, 64*64*16]
        conv_x = conv_x.view(batch_size, 16, 64, 64)  # [batch, 16, 64, 64]
        
        # Apply fixed conv layers
        conv_x = self.conv1(conv_x)  # [batch, 32, 64, 64]
        conv_x = self.conv2(conv_x)  # [batch, 64, 64, 64]
        conv_x = self.conv3(conv_x)  # [batch, 64, 64, 64]
        
        # Apply dilated convolutions
        conv_x = self.dilated_conv(conv_x)  # [batch, 64, 64, 64]
            
        # Pool and flatten
        conv_x = self.pool(conv_x)  # [batch, 64, 16, 16]
        conv_x = conv_x.flatten(1)  # [batch, 64*16*16]
        
        # Project back to hidden dim for attention
        attn_x = self.attn_proj(conv_x)  # [batch, hidden]
        
        # Reshape for attention with fixed sequence length
        attn_x = attn_x.unsqueeze(1).expand(batch_size, self.seq_len, self.hidden)
        
        # Add positional encoding
        attn_x = attn_x + self.pos_encoding.unsqueeze(0)
        
        # Apply fixed attention layers
        attn_x = self.attn1(attn_x) + attn_x  # Residual
        attn_x = self.attn2(attn_x) + attn_x  # Residual
            
        # Global average pooling over sequence dimension
        x = attn_x.mean(dim=1)  # [batch, hidden]
        
        # Final MLP
        return self.mlp(x)


def build_dummy_classifier() -> nn.Module:
    """Heavy model for MCCL/DDP demos with conv, attention, and MLP layers."""
    input_dim, num_classes, hidden, depth = _model_dims_from_env()
    if depth < 1:
        raise ValueError("MODEL_DEPTH must be >= 1")
    
    # Always use the heavy model - removed MCCL_SIMPLE_MODEL check for consistency
    return HeavyDummyModel(input_dim, num_classes, hidden, depth)


def _setup_mccl_env() -> None:
    """Avoid MCCL listen port colliding with PyTorch's TCP store on MASTER_PORT."""
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        # Reserve MASTER_PORT for c10d store; MCCL uses port_base + rank.
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)

    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


def _apply_thunderbolt_profile_training_defaults() -> None:
    """When MCCL_LINK_PROFILE=thunderbolt, apply DDP-friendly defaults if unset."""
    if os.environ.get("MCCL_LINK_PROFILE", "").lower() != "thunderbolt":
        return
    os.environ.setdefault("DDP_BUCKET_MB", "512")
    os.environ.setdefault("MCCL_OVERLAP_COMM", "1")


def single_gpu_baseline(save_stats: str | None = None) -> None:
    """Run single GPU training for comparison (same model + hparams as DDP path)."""
    import time

    device = torch.device("mps")
    torch.manual_seed(42)

    model = build_dummy_classifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    steps = int(os.environ.get("TRAIN_STEPS", "500"))
    if os.environ.get("BASELINE_BATCH_SIZE"):
        batch_size = int(os.environ["BASELINE_BATCH_SIZE"])
    else:
        # Default 256 global on one GPU — matches 2-rank DDP with BATCH_SIZE=128 each.
        batch_size = int(os.environ.get("BATCH_SIZE", "256"))
    input_dim, num_classes, _, _ = _model_dims_from_env()
    dataset = SyntheticDataset(input_dim, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    st = os.environ.get("MCCL_STRESS_MODEL", "").lower() in ("1", "true", "yes")
    print(
        f"Single GPU baseline | device={device}\n"
        f"  Model: {total_params:,} params{'  MCCL_STRESS_MODEL' if st else ''}\n"
        f"  Batch: {batch_size} global (override BASELINE_BATCH_SIZE / BATCH_SIZE if OOM)\n"
        f"  Steps: {steps}\n"
        f"  INPUT_DIM={input_dim} NUM_CLASSES={num_classes} "
        f"(MODEL_HIDDEN/MODEL_DEPTH from env)",
        flush=True,
    )

    warmup_steps = 5
    step_times = []
    losses = []

    for step in range(warmup_steps + steps):
        x, y = dataset.get_batch(batch_size, device, step, rank=0, world_size=1)

        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # Synchronize so timing reflects actual GPU work, not just dispatch
        torch.mps.synchronize()

        step_time = time.perf_counter() - start_time

        if step >= warmup_steps:
            step_times.append(step_time)
            losses.append(loss.item())

        if step % 5 == 0 or step == warmup_steps + steps - 1:
            status = "warmup" if step < warmup_steps else "train"
            print(
                f"  {status} step {step:4d}  loss={loss.item():.6f}  time={step_time:.3f}s",
                flush=True,
            )

    if step_times:
        avg_time = sum(step_times) / len(step_times)
        print(
            f"\n=== Single GPU Stats ===\n"
            f"  Steps completed: {len(step_times)}\n"
            f"  Avg step time: {avg_time:.3f}s ({1/avg_time:.1f} steps/sec)\n"
            f"  Final loss: {losses[-1]:.6f}\n"
            f"  Throughput: {batch_size / avg_time:.1f} samples/sec",
            flush=True,
        )

    if save_stats and step_times:
        _write_training_stats(
            save_stats,
            "baseline",
            step_times,
            losses,
            batch_size,
            world_size=1,
            total_params=total_params,
        )
        print(f"[ddp_dummy_train] wrote stats to {save_stats}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDP dummy train (MCCL+MPS) or single-GPU baseline for comparison.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Single-process MPS run (no torchrun); same model/env as DDP for fair comparison.",
    )
    parser.add_argument(
        "--save-stats",
        metavar="PATH",
        default=None,
        help="Write timing/loss JSON for examples/benchmark_throughput.py (rank 0 only for DDP).",
    )
    args = parser.parse_args()

    if args.baseline or os.environ.get("SINGLE_GPU"):
        if not torch.backends.mps.is_available():
            print("MPS is not available; this example expects Apple Silicon + MPS.", file=sys.stderr)
            sys.exit(1)
        single_gpu_baseline(save_stats=args.save_stats)
        return

    if "RANK" not in os.environ:
        print(
            "Distributed mode requires torchrun (RANK/WORLD_SIZE in env).\n"
            "  Single-GPU baseline:  python examples/ddp_dummy_train.py --baseline\n"
            "  DDP (2 ranks local): torchrun --nproc_per_node=2 --nnodes=1 "
            "--master_addr=127.0.0.1 --master_port=29500 examples/ddp_dummy_train.py",
            file=sys.stderr,
        )
        sys.exit(2)

    # Print before any distributed init so you see progress if something blocks later.
    print(
        "[ddp_dummy_train] starting "
        f"RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
        flush=True,
    )

    _setup_mccl_env()
    _apply_thunderbolt_profile_training_defaults()
    print(
        f"[ddp_dummy_train] MCCL_PORT_BASE={os.environ.get('MCCL_PORT_BASE')} "
        f"MCCL_LISTEN_ADDR={os.environ.get('MCCL_LISTEN_ADDR', '(unset)')} "
        f"MCCL_LINK_PROFILE={os.environ.get('MCCL_LINK_PROFILE', '(unset)')}",
        flush=True,
    )
    master_addr = os.environ.get("MASTER_ADDR", "")
    if master_addr and master_addr not in ("127.0.0.1", "localhost", "::1"):
        if "MCCL_LISTEN_ADDR" not in os.environ:
            print(
                "[ddp_dummy_train] hint: multi-node? set MCCL_LISTEN_ADDR on **each** machine to "
                "that machine's own IP on the peer link (not MASTER_ADDR). "
                "Open firewall for MCCL_PORT_BASE..+world_size-1.",
                flush=True,
            )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # MPS memory caching: let PyTorch's allocator cache GPU memory.
    # The old HIGH_WATERMARK_RATIO=0.0 disabled caching and crippled performance.
    # Event-based sync (Phase 1) makes allreduce safe during backward, so the
    # workaround is no longer needed.

    if world_size < 2:
        print(
            "Need world_size >= 2 (MCCL transport). Examples:\n"
            "  torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 ...\n"
            "  torchrun --nproc_per_node=1 --nnodes=2 ...\n"
            "Or use: python examples/ddp_dummy_train.py --baseline",
            file=sys.stderr,
        )
        sys.exit(1)

    if not torch.backends.mps.is_available():
        print("MPS is not available; this example expects Apple Silicon + MPS.", file=sys.stderr)
        sys.exit(1)

    # Each Mac in this setup has a single MPS device (index 0). LOCAL_RANK is still
    # set by torchrun and is usually 0 when --nproc_per_node=1.
    device = torch.device("mps:0")

    print(f"[ddp_dummy_train] rank {rank}: calling init_process_group(mccl)...", flush=True)
    dist.init_process_group(
        backend="mccl",
        device_id=device,
    )
    print(f"[ddp_dummy_train] rank {rank}: init_process_group done", flush=True)

    # CRITICAL: Use same seed for model initialization across all ranks
    torch.manual_seed(42)

    # DDP() runs cross-rank collectives; if one rank is slow/OOM/stuck, others block here.
    print(f"[ddp_dummy_train] rank {rank}: allocating model on {device}...", flush=True)
    model = build_dummy_classifier().to(device)
    
    # Now set different seed per rank for data generation
    torch.manual_seed(42 + rank)
    print(f"[ddp_dummy_train] rank {rank}: wrapping DDP (syncs with other ranks)...", flush=True)
    # Multi-node: larger buckets (e.g. 75–200) reduce allreduce count and RTT cost.
    bucket_mb = int(os.environ.get("DDP_BUCKET_MB", "25"))
    ddp = DDP(model, find_unused_parameters=False, bucket_cap_mb=bucket_mb)
    use_autocast = os.environ.get("TRAIN_AUTOCAST_FP16", "").lower() in (
        "1",
        "true",
        "yes",
    )
    print(f"[ddp_dummy_train] rank {rank}: DDP wrapper ready", flush=True)

    optimizer = torch.optim.AdamW(ddp.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    steps = int(os.environ.get("TRAIN_STEPS", "500"))
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    input_dim, num_classes, _, _ = _model_dims_from_env()
    dataset = SyntheticDataset(input_dim, num_classes)

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stress_flag = os.environ.get("MCCL_STRESS_MODEL", "").lower() in ("1", "true", "yes")
    if rank == 0:
        print(
            f"DDP training | world_size={world_size} device={device}\n"
            f"  Model: {total_params:,} params ({trainable_params:,} trainable)"
            f"{'  MCCL_STRESS_MODEL' if stress_flag else ''}\n"
            f"  Batch: {batch_size} per rank ({batch_size * world_size} global)\n"
            f"  Steps: {steps}  bucket_cap_mb={bucket_mb}"
            f"{'  autocast_fp16' if use_autocast else ''}\n"
            f"  INPUT_DIM={input_dim} NUM_CLASSES={num_classes}",
            flush=True,
        )

    # Warmup to get stable timing
    warmup_steps = 5
    import time

    step_times = []
    losses = []

    verbose = os.environ.get("DDP_DUMMY_VERBOSE", "")

    for step in range(warmup_steps + steps):
        # Different x per rank; same linear labeler W,b on all ranks (learnable task)
        x, y = dataset.get_batch(batch_size, device, step, rank, world_size)

        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: forward pass...", flush=True)

        # DDP hooks fire bucketed allreduce DURING backward -- each ~25MB
        # bucket is communicated while the next bucket's gradients are still
        # being computed.  Event-based sync (Phase 1) makes this safe on MPS:
        # we encode a signal + commit instead of draining the full MPS stream.
        if use_autocast:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = ddp(x)
                loss = loss_fn(logits, y)
        else:
            logits = ddp(x)
            loss = loss_fn(logits, y)

        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: backward pass (with bucketed allreduce)...", flush=True)

        loss.backward()

        if verbose and rank == 0 and step < 3:
            print(f"    step {step}: optimizer step...", flush=True)
        optimizer.step()

        step_time = time.perf_counter() - start_time

        # Skip warmup for timing stats
        if step >= warmup_steps:
            step_times.append(step_time)
            losses.append(loss.item())

        if rank == 0 and (step % 5 == 0 or step == warmup_steps + steps - 1):
            status = "warmup" if step < warmup_steps else "train"
            print(f"  {status} step {step:4d}  loss={loss.item():.6f}  time={step_time:.3f}s", flush=True)

    # Performance stats
    if step_times:
        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)

        # Collect MCCL metrics if available
        try:
            metrics = mccl.get_metrics()
            if metrics:
                mccl_info = (
                    f"MCCL: {metrics.total_ops} ops, "
                    f"{metrics.total_bytes_sent/1e6:.1f}MB sent, "
                    f"{metrics.total_bytes_recv/1e6:.1f}MB recv, "
                    f"avg_lat={metrics.avg_latency_ms:.2f}ms"
                )
                phase_info = ""
                for attr in ("avg_sync_ms", "avg_network_ms", "avg_reduce_ms"):
                    val = getattr(metrics, attr, None)
                    if val is not None:
                        phase_info += f"  {attr}={val:.2f}ms"
                if phase_info:
                    mccl_info += f"\n  Phase breakdown:{phase_info}"
            else:
                mccl_info = "MCCL metrics unavailable"
        except Exception:
            mccl_info = "MCCL metrics error"

        if rank == 0:
            print(
                f"\n=== Training Stats (rank {rank}) ===\n"
                f"  Steps completed: {len(step_times)}\n"
                f"  Avg step time: {avg_time:.3f}s ({1/avg_time:.1f} steps/sec)\n"
                f"  Min/Max time: {min_time:.3f}s / {max_time:.3f}s\n"
                f"  Final loss: {losses[-1]:.6f} (started: {losses[0]:.6f})\n"
                f"  {mccl_info}\n"
                f"  Global batch size: {batch_size * world_size}\n"
                f"  Throughput: {batch_size * world_size / avg_time:.1f} samples/sec",
                flush=True,
            )

        if args.save_stats and rank == 0 and step_times:
            _write_training_stats(
                args.save_stats,
                "ddp",
                step_times,
                losses,
                batch_size,
                world_size,
                total_params,
            )
            print(f"[ddp_dummy_train] wrote stats to {args.save_stats}", flush=True)

    # Sanity: parameters stay synced
    head = next(model.parameters()).detach().flatten()[:8].to(device)
    ref = head.clone()
    dist.broadcast(ref, src=0)
    if not torch.allclose(head, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Parameter mismatch across ranks after DDP — check MCCL / network.")

    if rank == 0:
        print("✓ Parameters are in sync across ranks.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
