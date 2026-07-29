"""Multi-bucket DDP gradient parity tests.

A model large enough to produce many DDP gradient buckets (small
bucket_cap_mb forces several allreduces per backward).  Gradients after
DDP backward must match a single-process model fed the concatenation of all
ranks' batches (the mathematical definition of data-parallel averaging).
"""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL tests require macOS on Apple Silicon",
)


def _ddp_multibucket_parity_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(42)  # identical init on all ranks

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    model = make_model().to("mps")
    # 1 MB buckets -> several gradient buckets per backward.
    ddp = DDP(model, bucket_cap_mb=1)

    # Per-rank batch; the reference model sees all batches together.
    batches = []
    for r in range(world_size):
        g = torch.Generator().manual_seed(1000 + r)
        x = torch.randn(64, 256, generator=g)
        y = torch.randint(0, 10, (64,), generator=g)
        batches.append((x, y))

    x_mine, y_mine = batches[rank]
    out = ddp(x_mine.to("mps"))
    loss = torch.nn.functional.cross_entropy(out, y_mine.to("mps"))
    loss.backward()

    # Reference: single process, mean over the concatenated batch equals the
    # average of per-rank mean losses only if batch sizes match (they do).
    torch.manual_seed(42)
    ref = make_model()  # CPU reference, fresh but identical init
    ref_losses = []
    for r in range(world_size):
        xr, yr = batches[r]
        out_r = ref(xr)
        ref_losses.append(torch.nn.functional.cross_entropy(out_r, yr))
    ref_loss = sum(ref_losses) / world_size
    ref_loss.backward()

    for (name, p_ddp), (_, p_ref) in zip(
        ddp.module.named_parameters(), ref.named_parameters()
    ):
        g_ddp = p_ddp.grad.detach().cpu()
        g_ref = p_ref.grad.detach()
        max_err = (g_ddp - g_ref).abs().max().item()
        scale = g_ref.abs().max().item() + 1e-8
        assert max_err / scale < 5e-4, (
            f"gradient mismatch on {name}: rel max err {max_err / scale:.2e}"
        )


def _ddp_multistep_stability_fn(rank, world_size):
    """Several optimizer steps with many buckets; weights must remain
    identical across ranks at every step (a single corrupted allreduce
    desynchronizes the replicas)."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
    ).to("mps")
    ddp = DDP(model, bucket_cap_mb=1)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    for step in range(10):
        g = torch.Generator().manual_seed(step * 100 + rank)
        x = torch.randn(32, 128, generator=g).to("mps")
        opt.zero_grad()
        loss = (ddp(x) ** 2).mean()
        loss.backward()
        opt.step()

        # Cross-rank weight digest comparison via allgather.
        digest = torch.cat([p.detach().flatten()[:16] for p in ddp.parameters()])
        outs = [torch.zeros_like(digest) for _ in range(world_size)]
        dist.all_gather(outs, digest)
        for r in range(world_size):
            assert torch.equal(outs[r].cpu(), outs[rank].cpu()), (
                f"replicas diverged at step {step} (rank {r} vs {rank})"
            )


class TestDDPMultiBucket:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_gradient_parity_vs_single_process(self, world_size):
        run_workers(_ddp_multibucket_parity_fn, world_size=world_size, timeout=420)

    @pytest.mark.parametrize("world_size", [2])
    def test_replica_weights_stay_identical(self, world_size):
        run_workers(_ddp_multistep_stability_fn, world_size=world_size, timeout=420)
