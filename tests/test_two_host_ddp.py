"""Phase E: Two-host DDP correctness test.

This test is designed to run across two Apple Silicon machines.
Launch with the scripts/launch_two_host.sh helper.

NOT meant to be run via pytest directly — use the launch script.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyModel(nn.Module):
    """Small model for DDP correctness testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _setup_mccl_env() -> None:
    """Keep MCCL listen ports off MASTER_PORT (PyTorch TCP store). Same as examples/ddp_dummy_train.py."""
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)
    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    _setup_mccl_env()

    import mccl  # noqa: F401

    dist.init_process_group(
        backend="mccl",
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("mps")

    # Seed identically so initial params match
    torch.manual_seed(42)
    model = TinyModel().to(device)
    ddp_model = DDP(model)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    num_steps = int(os.environ.get("MCCL_PROD_STEPS", "50"))
    validate_grads = os.environ.get("MCCL_VALIDATE_GRAD_SYNC", "0").lower() in ("1", "true", "yes")
    losses = []

    for step in range(num_steps):
        # Deterministic data per step (same across ranks)
        torch.manual_seed(1000 + step)
        inputs = torch.randn(32, 128, device=device)
        labels = torch.randint(0, 10, (32,), device=device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        if validate_grads:
            grad_sum = torch.zeros(1, dtype=torch.float32, device=device)
            for p in model.parameters():
                if p.grad is not None:
                    grad_sum += p.grad.detach().float().sum()
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

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 10 == 0:
            print(f"Rank {rank} | Step {step} | Loss: {loss_val:.6f}")

    # Flush GPU work and synchronize all ranks before reading parameters —
    # on slower transports (TCP) the last DDP allreduce may still be in-flight.
    # The dist.barrier() here goes through MCCL's barrier() which now drains
    # all per-peer net engines first, closing the race where the 2-rank
    # large-message split allreduce could leave its local reduction pending
    # on reduce_engine_ while barrier overtook it.
    torch.mps.synchronize()
    dist.barrier()

    # Final parameter checksum — symmetric: every rank participates via
    # all_reduce(SUM) so no rank can report false success.
    param_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Rank {rank} | Final param checksum: {param_sum:.6f}")
    print(f"Rank {rank} | Final loss: {losses[-1]:.6f}")

    # Regression: this check previously failed on slow TCP because the
    # broadcast-from-rank-0 pattern let rank 0 print "PASSED" while rank 1
    # still had stale parameters and crashed.  The all_reduce(SUM) approach
    # ensures every rank contributes before any rank evaluates the result.
    param_tensor = torch.tensor([param_sum], device=device)
    dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)

    avg_param = param_tensor.item() / world_size
    diff = abs(avg_param - param_sum)
    print(f"Rank {rank} | param checksum: {param_sum:.6f}  avg: {avg_param:.6f}  "
          f"divergence: {diff:.8f}")

    check_passed = diff < 1e-3

    # Share pass/fail across all ranks so neither rank hangs in the final
    # barrier while the other has already exited with an error.
    ok_tensor = torch.tensor([1 if check_passed else 0],
                             dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    all_ok = ok_tensor.item() == 1

    if rank == 0:
        if all_ok:
            print("\n*** DDP CORRECTNESS TEST PASSED ***")
        else:
            print("\n*** WARNING: Param divergence detected ***", file=sys.stderr)
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
