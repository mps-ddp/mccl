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

    num_steps = 50
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
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 10 == 0:
            print(f"Rank {rank} | Step {step} | Loss: {loss_val:.6f}")

    # Flush GPU work and synchronize all ranks before reading parameters —
    # on slower transports (TCP) the last DDP allreduce may still be in-flight.
    torch.mps.synchronize()
    dist.barrier()

    # Final parameter checksum
    param_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Rank {rank} | Final param checksum: {param_sum:.6f}")
    print(f"Rank {rank} | Final loss: {losses[-1]:.6f}")

    # Verify params match across ranks: all_reduce the checksum with SUM,
    # then check that every rank's individual checksum equals the average
    # (which is only true when all parameter tensors are identical).
    param_tensor = torch.tensor([param_sum], device=device)
    dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)

    avg_param = param_tensor.item() / world_size
    diff = abs(avg_param - param_sum)
    print(f"Rank {rank} | param checksum: {param_sum:.6f}  avg: {avg_param:.6f}  "
          f"divergence: {diff:.8f}")

    if rank == 0:
        if diff < 1e-3:
            print("\n*** DDP CORRECTNESS TEST PASSED ***")
        else:
            print("\n*** WARNING: Param divergence detected ***")
            sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
