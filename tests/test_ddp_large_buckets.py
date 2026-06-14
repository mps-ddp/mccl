"""Large-bucket DDP gradient parity: verifies MCCL allreduce over big buckets."""

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL DDP tests require macOS on Apple Silicon",
)


class TestDDPLargeBuckets:
    @pytest.mark.parametrize("bucket_mb", [25, 100])
    @pytest.mark.parametrize("world_size", [3, 4])
    def test_large_bucket_finite_grads(self, world_size, bucket_mb):
        def fn(rank, world_size):
            import os
            import torch
            import torch.nn as nn
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            bucket_mb = int(os.environ["MCCL_TEST_BUCKET_MB"])
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
            ).to("mps")
            torch.manual_seed(42 + rank)
            ddp = DDP(model, bucket_cap_mb=bucket_mb, find_unused_parameters=False)
            loss_fn = nn.MSELoss()
            x = torch.randn(32, 512, device="mps")
            y = torch.randn(32, 64, device="mps")
            for step in range(2):
                ddp.zero_grad(set_to_none=True)
                loss_fn(ddp(x), y).backward()
                for p in ddp.parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"step {step} non-finite"
                        gathered = [
                            torch.empty_like(p.grad) for _ in range(world_size)
                        ]
                        dist.all_gather(gathered, p.grad)
                        ref = gathered[0].cpu()
                        for r, g in enumerate(gathered[1:], start=1):
                            assert torch.allclose(g.cpu(), ref, atol=1e-5, rtol=1e-5), (
                                f"step {step} rank {r} gradient mismatch"
                            )

        run_workers(
            fn,
            world_size=world_size,
            env={"MCCL_TEST_BUCKET_MB": str(bucket_mb)},
            timeout=420,
        )
