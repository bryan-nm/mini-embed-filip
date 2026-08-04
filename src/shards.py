"""Shard-directory hygiene for the embarrassingly-parallel phases.

`precompute`, `precompute_decoys`, `export_embeddings --live` and
`roundtrip_eval` all follow the same pattern: every rank writes
`<something>.<rank>.<ext>` into a shard directory, then one process globs that
directory and concatenates. The glob is the problem — it merges whatever is on
disk, not what THIS run produced.

The failure that motivated this module: a 192-rank precompute crashed partway,
leaving shards 00000-00191. The retry ran on 24 ranks, wrote 00000-00023 over
the top, and the merge then walked into the stale 00024 (which began at the
192-rank partition's offset 71952, not the 24-rank run's 575503) and aborted
with a confusing "gap/overlap" error. Same corpus, same code, different world
size — and nothing had cleaned up.

Two defences, because they fail differently:

  reset_shard_dir()   PREVENTION. Rank 0 empties the directory before any rank
                      writes, so a run can only ever see its own shards. Cheap,
                      and it fixes the case above outright.

  check_shard_world() DETECTION. Each shard records the world size that wrote
                      it; the merge rejects a directory holding more than one
                      generation. This is the backstop for the paths where
                      prevention is deliberately skipped — `--merge-only`,
                      `--score-only` — and for a directory assembled by hand.

Detection matters most where the merge has no other integrity check:
`precompute` and `precompute_decoys` verify their shards tile [0,N)
contiguously and so at least fail loudly, but `roundtrip_eval` and
`export_embeddings` just concatenate. There, stale shards mean extra rows in
the scored pool — a silently wrong metric rather than a crash.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional


def reset_shard_dir(shards_dir, env, *, patterns=("*",)) -> int:
    """Rank 0 empties `shards_dir`, then all ranks wait. Returns files removed.

    Call BEFORE any rank writes a shard. Files only — subdirectories are left
    alone, so pointing this at a directory that also holds something else does
    not destroy it. The barrier is what makes it safe: without it rank 0 could
    still be deleting while rank 5 is writing.
    """
    from src.dist import barrier

    shards_dir = Path(shards_dir)
    removed = 0
    if env.is_main:
        shards_dir.mkdir(parents=True, exist_ok=True)
        for pat in patterns:
            for p in glob.glob(str(shards_dir / pat)):
                if os.path.isfile(p):
                    os.remove(p)
                    removed += 1
        if removed:
            print(f"[shards] cleared {removed} stale file(s) from {shards_dir} "
                  f"(previous run's shards)", flush=True)
    barrier()
    return removed


def check_shard_world(metas: list, shards_dir, what: str,
                      expected: Optional[int] = None) -> None:
    """Reject a shard set that spans more than one run.

    `metas` is the list of per-shard dicts, each of which MAY carry a "world"
    key (shards written before that key existed simply skip the check, so an
    older shard directory still merges). `expected`, when given, is the world
    size the caller believes produced them.

    Raises RuntimeError naming the fix, because the fix is always the same:
    delete the directory and re-run the encode.
    """
    worlds = {m["world"] for m in metas if "world" in m}
    if not worlds:
        return                                   # pre-stamp shards; nothing to check
    if len(worlds) > 1:
        raise RuntimeError(
            f"{what}: {shards_dir} holds shards from {len(worlds)} different runs "
            f"(world sizes {sorted(worlds)}). A re-run at a smaller world size "
            f"overwrites the low-numbered shards and leaves the rest behind.\n"
            f"    rm -rf {shards_dir}\n"
            f"and re-run the encode.")
    (world,) = worlds
    stamped = [m for m in metas if "world" in m]
    if len(stamped) != world:
        raise RuntimeError(
            f"{what}: {shards_dir} holds {len(stamped)} shards but they were "
            f"written by a {world}-rank run. The set is incomplete (a rank died) "
            f"or polluted.\n"
            f"    rm -rf {shards_dir}\n"
            f"and re-run the encode.")
    if expected is not None and world != expected:
        raise RuntimeError(
            f"{what}: {shards_dir} holds shards from a {world}-rank run, but this "
            f"is a {expected}-rank run.\n"
            f"    rm -rf {shards_dir}\n"
            f"and re-run the encode.")
