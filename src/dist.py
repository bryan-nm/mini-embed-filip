"""Distributed bootstrap for Aurora (Intel XPU + oneCCL).

Single place that turns an `mpiexec`-launched process into a torch.distributed
rank pinned to one GPU *tile*. Aurora exposes 6 Max-1550 GPUs x 2 tiles =
12 tiles/node; we run one rank per tile.

Rank/size/local-rank are read from the launcher's environment. We try PALS
(the Aurora/HPE PBS launcher) first, then MPICH `PMI_*`, then generic torch
`RANK/WORLD_SIZE/LOCAL_RANK`. The shell script is expected to export
MASTER_ADDR / MASTER_PORT (head node of PBS_NODEFILE) for the env:// rendezvous.

Everything degrades gracefully: if no launcher env is present (e.g. a laptop
smoke test), `init_distributed` returns a single-rank world on the best local
device and never touches the process group.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


# Intel extension registers the `xpu` backend and the oneCCL process-group
# backend. Both imports are optional so the module still loads on CPU/CUDA/Mac.
try:  # noqa: SIM105
    import intel_extension_for_pytorch as ipex  # noqa: F401
except Exception:
    ipex = None
try:  # registers the "ccl" backend for torch.distributed
    import oneccl_bindings_for_pytorch  # noqa: F401
except Exception:
    pass


def _first_env(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return int(v)
    return default


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str
    group_size: int = 1          # contrastive subgroup size (for grouped all-gather)
    group = None                 # torch.distributed.ProcessGroup for this rank's subgroup
    group_rank: int = 0          # this rank's index within its subgroup

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _detect_topology() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank).

    Ask MPI directly first — this is ground truth under `mpiexec` and immune to
    launcher env-var naming differences (the env-var path silently returned
    world=1 on Aurora MPICH, making every rank encode the whole dataset). Local
    rank comes from a shared-memory communicator split, so it's correct even
    when no *_LOCAL_RANK* env var is exported.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world = comm.Get_size()
        if world > 1:
            rank = comm.Get_rank()
            local = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
            return rank, world, local
    except Exception:
        pass
    # Fallback: launcher environment variables (covers non-MPI / other stacks).
    world = _first_env("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", default=1)
    rank = _first_env("PALS_RANKID", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK", default=0)
    local = _first_env(
        "PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
        "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", default=0,
    )
    return rank, world, local


def _pick_local_device(local_rank: int, device_name: str) -> torch.device:
    if device_name not in ("auto", "xpu") and not device_name.startswith("xpu"):
        # Honour an explicit cpu/cuda/mps request (smoke tests).
        return torch.device(device_name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        n = torch.xpu.device_count()
        idx = local_rank % max(n, 1)
        torch.xpu.set_device(idx)
        return torch.device(f"xpu:{idx}")
    if torch.cuda.is_available():
        idx = local_rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def init_distributed(device_name: str = "auto", group_size: int = 1, init_pg: bool = True) -> DistEnv:
    """Initialise the process group (if launched under MPI) and pin the tile.

    `group_size` > 1 additionally builds contiguous rank subgroups of that size
    and records this rank's subgroup, used by `grouped_all_gather` for the
    bounded FILIP negative pool.

    `init_pg=False` skips the torch (oneCCL) process group entirely: useful for
    embarrassingly-parallel jobs (precompute) that only need rank/world for
    sharding plus an MPI barrier — no collective backend required.
    """
    rank, world, local = _detect_topology()
    device = _pick_local_device(local, device_name)

    if world <= 1:
        return DistEnv(rank=0, world_size=1, local_rank=0, device=device,
                       backend="none", group_size=1, group_rank=0)
    if not init_pg:
        if rank == 0:
            print(f"[dist] world={world} ranks, no process group (MPI barrier only); "
                  f"rank0 device={device}", flush=True)
        return DistEnv(rank=rank, world_size=world, local_rank=local, device=device,
                       backend="mpi", group_size=1, group_rank=0)

    # env:// rendezvous; MASTER_ADDR/PORT come from the launch script.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local)

    backend = "ccl" if device.type == "xpu" else ("nccl" if device.type == "cuda" else "gloo")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", world_size=world, rank=rank
        )
    if rank == 0:
        print(f"[dist] process group up: backend={backend} world={world} "
              f"group_size={group_size} rank0 device={device}", flush=True)
    env = DistEnv(rank=rank, world_size=world, local_rank=local, device=device,
                  backend=backend, group_size=max(group_size, 1))

    if group_size > 1:
        if world % group_size != 0 and rank == 0:
            print(f"[dist] WARNING: world_size={world} not divisible by "
                  f"group_size={group_size}; last group will be ragged.")
        # Build every subgroup on every rank (collective requirement), keep ours.
        for g0 in range(0, world, group_size):
            members = list(range(g0, min(g0 + group_size, world)))
            grp = torch.distributed.new_group(ranks=members, backend=backend)
            if rank in members:
                env.group = grp
                env.group_rank = members.index(rank)
                env.group_size = len(members)
    return env


def barrier() -> None:
    """Cross-rank barrier. Uses the torch process group if up, else falls back
    to an MPI barrier (so precompute, which runs without a process group, still
    gets a real barrier between the encode and merge phases)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        return
    try:
        from mpi4py import MPI
        if MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except Exception:
        pass


def cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Grouped all-gather of variable-length per-token tensors (with gradient on
# the local slice, open_clip style).
# ---------------------------------------------------------------------------
def grouped_all_gather(local: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """All-gather `local` ([B, L, D] or [B, L]) within this rank's subgroup.

    Pads the L axis to the subgroup-max so shapes match, then concatenates the
    gathered tensors along dim 0 → [group_size*B, L_max, ...]. The local slice
    keeps its autograd graph (the gathered copies are detached), so gradients
    flow to the local rank's embeddings only — the correct contrastive form.
    """
    if env.group is None or env.group_size <= 1:
        return local

    # Agree on a common L across the subgroup.
    L = torch.tensor([local.size(1)], device=local.device)
    torch.distributed.all_reduce(L, op=torch.distributed.ReduceOp.MAX, group=env.group)
    L_max = int(L.item())
    if local.size(1) < L_max:
        pad = list(local.shape)
        pad[1] = L_max - local.size(1)
        local = torch.cat([local, local.new_zeros(*pad)], dim=1)

    gathered = [torch.empty_like(local) for _ in range(env.group_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=env.group)
    # Splice the local (grad-carrying) tensor back into its slot.
    gathered[env.group_rank] = local
    return torch.cat(gathered, dim=0)
