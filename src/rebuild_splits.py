"""Regenerate a cache's `splits.json` from the cache alone (no CSV, no training).

`splits.json` is a *derived* file: `make_splits` seeds `np.random.default_rng(seed)`
and permutes the connected components of (same protein) ∪ (same caption), so the
split is a pure function of

    (pair_ids.json, text_group_ids.json, cfg.data.splits, seed)

`train_retrieval` writes it as a side effect of `build_or_load_splits`, rebuilding
it whenever the file's `n` doesn't match the cache. That means a stale file — most
often a small `--subset-size` smoke run left behind in a directory that was later
re-precomputed at full scale — is recoverable rather than fatal: rebuilding here
with the same seed and ratios reproduces byte-identical splits.

It is needed standalone because `src/precompute_decoys` reads `splits.json` to keep
decoy donors inside the owner's split (so held-out caption text never leaks into a
training decoy), and it runs before any trainer would have recreated the file.

    python -m src.rebuild_splits --cache-dir cache
    python -m src.rebuild_splits --cache-dir cache --check    # report only

IMPORTANT: pass the same `--seed` the retrieval run used (`train_retrieval --seed`,
default 0). A different seed produces a *valid* split that is not the one the
checkpoint was trained on, which would quietly put trained-on rows in val.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.data import (
    group_ids_from_accessions,
    load_splits,
    make_splits,
    merged_split_group_ids,
    save_splits,
    splits_are_valid,
)


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--seed", type=int, default=cfg.data.seed,
                    help="MUST match the retrieval run's --seed (default 0)")
    ap.add_argument("--csv", default=None,
                    help="only used when the cache has no text_group_ids.json; "
                         "the CSV the cache was built from (default: cfg.data.csv_path)")
    ap.add_argument("--check", action="store_true",
                    help="report whether the existing file is current; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rewrite even when the existing file is already valid")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    pair_ids_path = cache / "pair_ids.json"
    if not pair_ids_path.exists():
        raise FileNotFoundError(
            f"{pair_ids_path} missing — {cache} is not a built per-token cache.")
    with open(pair_ids_path) as f:
        accessions = json.load(f)
    n = len(accessions)

    # Caption-identity ids. Preferred source is the file precompute writes; a cache
    # predating that (pre-2026-07-22) has none, and `train_retrieval` falls back to
    # hashing the CSV. We mirror that fallback exactly — same function, same row
    # order, guarded by the same accession check — so the split this reproduces is
    # the one such a run built, rather than dead-ending on a multi-hour re-precompute.
    tgi_path = cache / "text_group_ids.json"
    if tgi_path.exists():
        with open(tgi_path) as f:
            row_text_np = np.asarray(json.load(f), dtype=np.int64)
        if len(row_text_np) != n:
            raise RuntimeError(
                f"{tgi_path} has {len(row_text_np)} entries but pair_ids has {n}; "
                f"the cache is internally inconsistent. Re-run "
                f"`python -m src.precompute`.")
        print(f"[splits] caption ids from {tgi_path.name}")
    else:
        csv_path = args.csv or cfg.data.csv_path
        print(f"[splits] {tgi_path.name} missing (cache predates it); deriving "
              f"caption ids from {csv_path}")
        from src.data import group_ids_from_texts, load_pairs
        fb = load_pairs(csv_path, id_col=cfg.data.csv_id_col,
                        protein_col=cfg.data.csv_protein_col,
                        text_col=cfg.data.csv_text_col,
                        pfam_col=cfg.data.csv_pfam_col,
                        subset_size=len(accessions))
        if len(fb) != n or [p.uid for p in fb] != list(accessions):
            raise RuntimeError(
                f"CSV rows do not match the cache's pair_ids ({len(fb)} vs {n} "
                f"rows); cannot derive caption ids. Point --csv at the CSV this "
                f"cache was built from.")
        row_text_np = group_ids_from_texts([p.text for p in fb])

    row_group_np = group_ids_from_accessions(accessions)
    split_group_np = merged_split_group_ids(row_group_np, row_text_np)
    n_groups = int(np.unique(split_group_np).shape[0])
    n_prot = int(row_group_np.max()) + 1 if n else 0
    n_caps = int(row_text_np.max()) + 1 if n else 0
    print(f"[splits] cache {cache}: {n} rows, {n_prot} proteins, "
          f"{n_caps} unique captions -> {n_groups} split groups")

    path = cache / "splits.json"
    if path.exists():
        try:
            existing = load_splits(str(path))
            ok = splits_are_valid(existing, n, args.seed, cfg.data.splits,
                                  n_groups=n_groups)
            print(f"[splits] existing {path.name}: n={existing.get('n')} "
                  f"n_groups={existing.get('n_groups')} seed={existing.get('seed')} "
                  f"ratios={existing.get('ratios')} -> "
                  f"{'CURRENT' if ok else 'STALE'}")
        except Exception as e:
            ok = False
            print(f"[splits] existing {path.name} unreadable ({e}); treating as stale")
    else:
        ok = False
        print(f"[splits] no {path} yet")

    if args.check:
        raise SystemExit(0 if ok else 1)
    if ok and not args.force:
        print("[splits] already current; nothing to do (use --force to rewrite)")
        return

    splits = make_splits(n, cfg.data.splits, args.seed, group_ids=split_group_np)
    save_splits(splits, str(path))
    print(f"[splits] wrote {path}: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])} "
          f"over {splits['n_groups']} groups (seed={args.seed}, "
          f"ratios={list(cfg.data.splits)})")
    print("[splits] deterministic: the same cache + seed + ratios always reproduce "
          "this file, so it matches what train_retrieval built for the same inputs.")


if __name__ == "__main__":
    main()
