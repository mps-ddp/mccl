"""Multi-host DDP correctness test for 3+ ranks.

Launch on each host with matching MASTER_ADDR / MASTER_PORT / WORLD_SIZE and a
unique RANK. Intended for one process per host.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class MultiHostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        return self.net(x)


def _setup_mccl_env() -> None:
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)
    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


def _grad_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    grad_sum = torch.zeros(1, dtype=torch.float32, device=device)
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += p.grad.detach().float().sum()
    return grad_sum


def _param_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros(1, dtype=torch.float32, device=device)
    for p in model.parameters():
        checksum += p.detach().float().sum()
    return checksum


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 3:
        raise RuntimeError("test_multi_host_ddp.py expects WORLD_SIZE >= 3")

    _setup_mccl_env()

    import mccl  # noqa: F401

    dist.init_process_group(
        backend="mccl",
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("mps")
    torch.manual_seed(42)
    model = MultiHostModel().to(device)
    ddp_model = DDP(model, bucket_cap_mb=int(os.environ.get("DDP_BUCKET_MB", "8")))
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_steps = int(os.environ.get("MCCL_PROD_STEPS", "80"))
    validate_grads = os.environ.get("MCCL_VALIDATE_GRAD_SYNC", "1").lower() in ("1", "true", "yes")
    losses = []

    for step in range(num_steps):
        torch.manual_seed(7000 + step * world_size + rank)
        x = torch.randn(24, 256, device=device)
        y = torch.randn(24, 64, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(ddp_model(x), y)
        loss.backward()

        if validate_grads:
            grad_sum = _grad_checksum(model, device)
            grad_global = grad_sum.clone()
            dist.all_reduce(grad_global, op=dist.ReduceOp.SUM)
            grad_expected = grad_global / world_size
            grad_div = (grad_sum - grad_expected).abs().item()
            if grad_div > 1e-3:
                raise RuntimeError(
                    f"[rank {rank}] gradient checksum mismatch at step {step}: "
                    f"local={grad_sum.item():.6f} expected={grad_expected.item():.6f} "
                    f"divergence={grad_div:.2e}"
                )

        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"Rank {rank} | Step {step} | Loss: {loss.item():.6f}")

    torch.mps.synchronize()
    dist.barrier()

    checksum = _param_checksum(model, device)
    checksum_global = checksum.clone()
    dist.all_reduce(checksum_global, op=dist.ReduceOp.SUM)
    checksum_expected = checksum_global / world_size
    diff = (checksum - checksum_expected).abs().item()
    print(
        f"Rank {rank} | param checksum={checksum.item():.6f} "
        f"expected={checksum_expected.item():.6f} diff={diff:.8f}"
    )

    ok_tensor = torch.tensor([1 if diff < 1e-3 else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    all_ok = ok_tensor.item() == 1

    if rank == 0:
        print(f"Rank 0 | Final loss: {losses[-1]:.6f}")
        if all_ok:
            print("\n*** MULTI-HOST 3+ DDP CORRECTNESS TEST PASSED ***")
        else:
            print("\n*** WARNING: Multi-host param divergence detected ***", file=sys.stderr)
        try:
            import mccl

            metrics = mccl.get_metrics()
            if metrics is not None:
                print(
                    "Rank 0 | metrics: "
                    f"avg_network_ms={metrics.avg_network_ms:.3f} "
                    f"avg_reduce_ms={metrics.avg_reduce_ms:.3f} "
                    f"avg_queue_wait_ms={metrics.avg_queue_wait_ms:.3f} "
                    f"avg_send_ms={metrics.avg_send_ms:.3f} "
                    f"avg_recv_ms={metrics.avg_recv_ms:.3f} "
                    f"avg_stage_ms={metrics.avg_stage_ms:.3f} "
                    f"avg_writeback_ms={metrics.avg_writeback_ms:.3f} "
                    f"avg_backpressure_ms={metrics.avg_backpressure_ms:.3f} "
                    f"avg_pipeline_depth={metrics.avg_pipeline_depth:.3f} "
                    f"max_pipeline_depth={metrics.max_pipeline_depth}"
                )
        except Exception:
            pass

    dist.barrier()
    dist.destroy_process_group()

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
