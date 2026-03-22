# Thunderbolt direct connection (2 Macs)

Short walkthrough for **two Apple Silicon Macs** on a **Thunderbolt 3/4** cable, **TCP** (MCCL default). For tuning and firewall detail, see [MULTINODE.md](MULTINODE.md).

## Physical setup

1. **Cable:** Thunderbolt 3 or 4 cable between the two Macs (USB-C ports that support TB).
2. **Network:** macOS often brings up a **bridge** interface with **link-local** IPv4 (`169.254.x.x`).
3. **Check:** `ifconfig | grep -A 3 bridge` — look for `inet 169.254.x.x` on the Thunderbolt-related interface.

## Software setup

**On both Macs** (same repo / same `pip install -e .` build helps):

```bash
pip install -e .

# Find your Thunderbolt-side IP (example path; interface name varies)
ifconfig | grep -A 3 bridge
# Example: inet 169.254.238.250
```

Pick **one machine as rank 0** (the **head**). **Every** node uses the same **`MASTER_ADDR`** (that machine’s Thunderbolt IP) and **`MASTER_PORT`**.

**Machine A (rank 0):**

```bash
export MASTER_ADDR=169.254.238.250   # Machine A’s TB IP (example)
export MASTER_PORT=29500

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ddp_dummy_train.py
```

**Machine B (rank 1):**

```bash
export MASTER_ADDR=169.254.238.250   # Rank-0’s IP (same on both)
export MASTER_PORT=29500

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/ddp_dummy_train.py
```

Use the real `169.254.*` from `ifconfig` for **rank 0** in `MASTER_ADDR`.

## Firewall

macOS may block incoming connections. Allow at least:

- **`MASTER_PORT`** (e.g. 29500) — PyTorch TCP store / rendezvous  
- **`MCCL_PORT_BASE` … + world_size − 1`** — MCCL peer sockets (default base often `29600`; must **not** equal `MASTER_PORT`)

**System Settings → Network → Firewall → Options** — allow incoming for **Python** and/or **Terminal** (or add explicit rules if you use a custom tool).

## How it fits together

1. **`torchrun`** uses the TCP store on **`MASTER_ADDR:MASTER_PORT`** so all ranks discover each other.
2. **MCCL** uses **separate TCP sockets** (per peer) for collective traffic, coordinated after the store comes up ([MULTINODE.md](MULTINODE.md)).
3. During training, **DDP** triggers **allreduce** (etc.) on gradient buckets; MCCL moves bytes over the Thunderbolt path and applies the reduction, then writes back into **MPS** tensors.

## Troubleshooting

- **Hangs at init:** Firewall; wrong `MASTER_ADDR`; not all nodes started; `MCCL_PORT_BASE` colliding with `MASTER_PORT`.
- **Slow training:** Expected for tiny models; for large models tune buckets — [MULTINODE.md](MULTINODE.md) (`DDP_BUCKET_MB`, `MCCL_LINK_PROFILE=thunderbolt`, etc.).
- **Connection refused:** Blocked `MASTER_PORT` or MCCL port range; fix firewall.
