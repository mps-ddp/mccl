"""DDP backward smoke: gradient magnitudes must stay finite (catches ~inf corruption)."""

from __future__ import annotations

import platform

import pytest

from mccl_test_utils import run_workers

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MCCL DDP smoke requires macOS Apple Silicon",
)


def _ddp_backward_finite_fn(rank, world_size):
    import os
    import pickle

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    dtype_name = os.environ.get("MCCL_TEST_DTYPE", "float32")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    bucket_mb = int(os.environ.get("MCCL_TEST_BUCKET_MB", "25"))

    # Identical initialization isolates gradient synchronization.  Using
    # rank-dependent weights here made the parity assertion also depend on
    # DDP's constructor-time parameter broadcast.
    torch.manual_seed(77)
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    ).to(dtype=dtype, device="mps")
    ddp = DDP(model, bucket_cap_mb=bucket_mb)

    g = torch.Generator().manual_seed(500 + rank)
    x = torch.randn(32, 512, generator=g, dtype=dtype).to("mps")
    y = torch.randint(0, 10, (32,), generator=g).to("mps")
    loss = F.cross_entropy(ddp(x), y)
    if not torch.isfinite(loss):
        raise AssertionError(f"rank={rank} dtype={dtype_name}: non-finite forward loss")
    loss.backward()
    # Materialize DDP's asynchronous MCCL reductions before using the gradients
    # as inputs to a second collective for full-tensor cross-rank comparison.
    torch.mps.synchronize()

    max_grad = 0.0
    flat_grads = []
    for p in ddp.module.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise AssertionError(
                    f"rank={rank} dtype={dtype_name}: non-finite gradient"
                )
            gmax = p.grad.detach().abs().max().item()
            max_grad = max(max_grad, gmax)
            flat_grads.append(p.grad.detach().flatten())

    if not (max_grad > 0 and max_grad < 1e6):
        raise AssertionError(
            f"rank={rank} dtype={dtype_name}: grad.abs().max()={max_grad} (expected finite, <1e6)"
        )

    # Compare through the rendezvous store rather than another MCCL collective:
    # otherwise an all-gather failure could be misreported as DDP gradient
    # corruption.  This preserves every element and isolates backward sync.
    grad_vector = torch.cat(flat_grads).cpu()
    store = dist.distributed_c10d._get_default_store()
    store.set(f"bf16_diag_grad_{rank}", pickle.dumps(grad_vector))
    for peer in range(world_size):
        peer_grads = pickle.loads(store.get(f"bf16_diag_grad_{peer}"))
        if not torch.equal(peer_grads, grad_vector):
            max_diff = (peer_grads.float() - grad_vector.float()).abs().max().item()
            raise AssertionError(
                f"rank={rank} dtype={dtype_name}: full gradient mismatch "
                f"against rank={peer}, max_diff={max_diff}"
            )
    # Rank 0 owns the TCPStore server in this harness.  Keep it alive until
    # every peer has finished reading the full gradient payloads.
    if rank == 0:
        for peer in range(1, world_size):
            store.get(f"bf16_diag_done_{peer}")
        for peer in range(1, world_size):
            store.set(f"bf16_diag_release_{peer}", b"1")
    else:
        store.set(f"bf16_diag_done_{rank}", b"1")
        store.get(f"bf16_diag_release_{rank}")


def _bf16_ddp_analytical_gradient_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(11)
    model = torch.nn.Linear(8, 4, bias=False).to(
        device="mps", dtype=torch.bfloat16
    )
    ddp = DDP(model, bucket_cap_mb=1)

    x_cpu = (
        (torch.arange(16, dtype=torch.float32).reshape(2, 8) % 5)
        - 2.0
        + float(rank)
    ).to(torch.bfloat16)
    x = x_cpu.to("mps")
    output = ddp(x)
    assert torch.isfinite(output).all(), f"rank={rank}: non-finite forward output"
    loss = output.sum() / x.shape[0]
    assert torch.isfinite(loss), f"rank={rank}: non-finite loss"
    loss.backward()

    grad = ddp.module.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all(), f"rank={rank}: non-finite weight gradient"

    # d(sum(linear(x)) / batch)/dW is mean(x), repeated for each output.
    means = []
    for r in range(world_size):
        xr = (
            (torch.arange(16, dtype=torch.float32).reshape(2, 8) % 5)
            - 2.0
            + float(r)
        ).to(torch.bfloat16).float()
        means.append(xr.mean(dim=0))
    expected_row = torch.stack(means).mean(dim=0)
    expected = expected_row.repeat(4, 1)
    torch.testing.assert_close(
        grad.float().cpu(), expected, rtol=2e-2, atol=2e-2
    )

    gathered = [torch.empty_like(grad) for _ in range(world_size)]
    dist.all_gather(gathered, grad)
    for peer, peer_grad in enumerate(gathered):
        assert torch.equal(peer_grad, grad), (
            f"rank={rank}: analytical gradient differs from rank={peer}"
        )


def _bf16_ddp_multistep_finite_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(29)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
    ).to(device="mps", dtype=torch.bfloat16)
    ddp = DDP(model, bucket_cap_mb=1)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-3)

    for step in range(5):
        generator = torch.Generator().manual_seed(900 + 10 * step + rank)
        x = torch.randn(16, 32, generator=generator, dtype=torch.bfloat16).to("mps")
        optimizer.zero_grad(set_to_none=True)
        output = ddp(x)
        assert torch.isfinite(output).all(), (
            f"rank={rank}: non-finite activation at step={step}"
        )
        loss = output.float().square().mean()
        assert torch.isfinite(loss), f"rank={rank}: non-finite loss at step={step}"
        loss.backward()

        for name, parameter in ddp.module.named_parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all(), (
                f"rank={rank}: non-finite gradient {name} at step={step}"
            )

        optimizer.step()
        for name, parameter in ddp.module.named_parameters():
            assert torch.isfinite(parameter).all(), (
                f"rank={rank}: non-finite parameter {name} at step={step}"
            )

        digest = torch.cat(
            [parameter.detach().flatten()[:16] for parameter in ddp.parameters()]
        )
        gathered = [torch.empty_like(digest) for _ in range(world_size)]
        dist.all_gather(gathered, digest)
        for peer, peer_digest in enumerate(gathered):
            assert torch.equal(peer_digest, digest), (
                f"rank={rank}: parameters diverged from rank={peer} at step={step}"
            )


def _bf16_autocast_ddp_boundary_fn(rank, world_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(41)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).to("mps")
    ddp = DDP(model, bucket_cap_mb=1)

    generator = torch.Generator().manual_seed(1200 + rank)
    x = torch.randn(8, 16, generator=generator).to("mps")
    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        output = ddp(x)
        if output.dtype != torch.bfloat16:
            raise AssertionError(
                f"rank={rank}: BF16 autocast was expected after parent preflight, "
                f"but output dtype was {output.dtype}"
            )
        assert torch.isfinite(output).all(), (
            f"rank={rank}: non-finite BF16 autocast output"
        )
        loss = output.float().square().mean()
    assert torch.isfinite(loss), f"rank={rank}: non-finite autocast loss"
    loss.backward()

    flat_grads = []
    for name, parameter in ddp.module.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all(), (
            f"rank={rank}: non-finite autocast gradient {name}"
        )
        flat_grads.append(parameter.grad.detach().flatten())

    # FP32 master parameters produce FP32 DDP gradient buckets even though the
    # forward activations were BF16. Record and verify that boundary explicitly.
    grad_vector = torch.cat(flat_grads)
    assert grad_vector.dtype == torch.float32, (
        f"rank={rank}: unexpected autocast gradient dtype {grad_vector.dtype}"
    )
    gathered = [torch.empty_like(grad_vector) for _ in range(world_size)]
    dist.all_gather(gathered, grad_vector)
    for peer, peer_grads in enumerate(gathered):
        assert torch.equal(peer_grads, grad_vector), (
            f"rank={rank}: autocast gradients differ from rank={peer}"
        )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("world_size", [4, 7])
def test_ddp_backward_grad_finite(dtype, world_size):
    env = {
        "MCCL_RING_ALGO": "chunked",
        "MCCL_COLLECTIVE_CONCURRENCY": "2",
        "MCCL_TEST_DTYPE": dtype,
        "MCCL_TEST_BUCKET_MB": "25",
    }
    run_workers(_ddp_backward_finite_fn, world_size=world_size, env=env, timeout=600)


def test_bf16_ddp_gradient_matches_analytical_reference():
    run_workers(_bf16_ddp_analytical_gradient_fn, world_size=3, timeout=300)


def test_bf16_ddp_multistep_stays_finite_and_synchronized():
    run_workers(_bf16_ddp_multistep_finite_fn, world_size=2, timeout=300)


def test_bf16_autocast_ddp_boundary_when_supported():
    import warnings

    import torch

    probe = torch.nn.Linear(4, 4).to("mps")
    x = torch.ones(2, 4, device="mps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                output = probe(x)
        except (RuntimeError, TypeError) as exc:
            pytest.skip(f"installed PyTorch MPS does not support BF16 autocast: {exc}")
    if output.dtype != torch.bfloat16:
        pytest.skip(
            "installed PyTorch MPS disables BF16 autocast "
            f"(probe output dtype={output.dtype})"
        )
    assert torch.isfinite(output).all()
    run_workers(_bf16_autocast_ddp_boundary_fn, world_size=2, timeout=300)
