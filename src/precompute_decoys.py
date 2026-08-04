"""Per-token cache builder for hard decoy captions (distributed, sharded).

Companion to `src/precompute.py`. That script encodes the corpus's real
(protein, caption) pairs; this one encodes the *decoys* — captions built from a
real caption by swapping exactly one templated field in from a different caption
(see `src/decoys.py`). Decoys are text-only: the protein side is unchanged, so
this reuses the base cache's proteins and only adds a third packed modality.

Two phases:

  1. plan   One rank parses every caption, indexes the swappable field spans, and
            assigns each row up to K (field, donor) swaps. Saved as
            `decoy_plan.pt`; deterministic given (CSV, seed), so encoding can be
            re-run or resumed without re-deriving anything. Cheap enough (a few
            minutes of one core) that doing it on one rank beats making all ranks
            repeat the same full-corpus parse.
  2. encode  Every rank materializes and encodes a contiguous slice of the plan,
            writing packed bf16 per-token hidden states exactly like
            `precompute.py`. Rank 0 then merges the shards in rank order.

Truncation guard: the text encoder caps at `max_text_tokens`. If a decoy's
swapped field sits past that cap, the decoy tokenizes *identically* to the true
caption — an impossible negative that would ask the loss to separate two equal
inputs. The plan builder avoids this approximately (a character budget on the
swap position) and the encoder checks it exactly, tokenizing the true caption
alongside and flagging any decoy whose truncated ids match. Flagged decoys stay
in the file (so offsets remain a pure function of the plan) but are excluded via
`decoy_keep.pt` at read time.

Final output layout (under --decoy-cache-dir):
  decoy_h.bin             bf16, total_decoy_tokens x 768
  decoy_offsets.pt        int64 [M+1]
  decoy_mask.bin          uint8, total_decoy_tokens
  decoy_row_ptr.pt        int64 [N_rows+1]; CSR, row -> its decoys' cache rows
  decoy_keep.pt           bool  [M]; False = swap fell outside the token window
  decoy_plan.pt           the plan (owner/donor rows, field ids, char spans)
  decoy_stats.json        plan + encode statistics (coverage, per-field counts)
  decoy_fingerprint.json  base cache fingerprint + plan params

Usage:
  # plan + distributed encode + merge in one job:
  mpiexec ... python -m src.precompute_decoys --device xpu --batch-size 64
  # split phases explicitly:
  python -m src.precompute_decoys --plan-only
  mpiexec ... python -m src.precompute_decoys --device xpu --encode-only
  python -m src.precompute_decoys --merge-only
  # laptop smoke test:
  python -m src.precompute_decoys --device cpu --batch-size 8 --subset-size 500
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.data import cache_fingerprint, read_cache_fingerprint, fingerprint_matches
from src.decoys import (
    build_decoy_plan,
    decoy_fingerprint,
    load_caption_rows,
    plan_decoy_texts,
    write_decoy_fingerprint,
)
from src.dist import barrier, cleanup, init_distributed
from src.shards import check_shard_world, reset_shard_dir
from src.encoders import (
    encode_text_batch,
    load_text_encoder,
    text_encoder_max_len,
)
from src.precompute import _bf16_to_uint16_bytes, _copy_into, _shard_range


PLAN_KEYS = ("owner_row", "label_id", "owner_start", "owner_end",
             "donor_row", "donor_start", "donor_end", "row_ptr")


# ---------------------------------------------------------------------------
# Corpus loading (shared by every phase; validated against the base cache)
# ---------------------------------------------------------------------------
def load_corpus(cfg, args, *, verbose: bool = True):
    """Read (accessions, captions) and check they line up with the base cache.

    The decoy cache is addressed by base-cache row index, so a silent row-order or
    row-count divergence between the CSV and `pair_ids.json` would mis-associate
    every decoy. Checked here, loudly, once per process.
    """
    uids, texts = load_caption_rows(
        cfg.data.csv_path, id_col=cfg.data.csv_id_col,
        text_col=cfg.data.csv_text_col, subset_size=args.subset_size)

    pair_ids_path = Path(args.cache_dir) / "pair_ids.json"
    if pair_ids_path.exists():
        with open(pair_ids_path) as f:
            pair_ids = json.load(f)
        if len(pair_ids) != len(uids) or pair_ids != uids:
            n_diff = sum(1 for a, b in zip(pair_ids, uids) if a != b)
            raise RuntimeError(
                f"CSV rows do not match the base cache at {args.cache_dir}: "
                f"{len(uids)} CSV rows vs {len(pair_ids)} cached rows, "
                f"{n_diff} accession mismatches in the overlap. The decoy cache is "
                f"indexed by base-cache row, so these must agree exactly — check "
                f"--subset-size and FILIP_DATA_CSV.")
    elif verbose:
        print(f"[decoy] WARNING: {pair_ids_path} not found; cannot verify that the "
              f"CSV row order matches the base cache.", flush=True)
    return uids, texts


def resolve_swap_labels(cfg, args):
    """(all labels, ids of the ones eligible for swapping)."""
    labels = list(cfg.data.caption_field_labels)
    if not args.swap_fields:
        return labels, None
    unknown = [f for f in args.swap_fields if f not in labels]
    if unknown:
        raise ValueError(
            f"--swap-fields {unknown} are not caption field labels. "
            f"Known labels: {labels}")
    return labels, [labels.index(f) for f in args.swap_fields]


def auto_max_value_start(max_text_tokens: int, chars_per_token: float) -> int:
    """Character budget for where a swappable field may start.

    A deliberately conservative estimate of the text encoder's truncation window:
    biomedical WordPiece averages ~4 characters per token, so the default 3.0
    keeps chosen swaps comfortably inside the window rather than right at its
    edge. The exact token-level check in `encode_shard` is what actually
    guarantees correctness; this only keeps the plan from wasting decoy slots.
    """
    return max(int(max_text_tokens * chars_per_token), 1)


# ---------------------------------------------------------------------------
# Phase 1: plan
# ---------------------------------------------------------------------------
def build_plan(args, cfg) -> None:
    out_dir = Path(args.decoy_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uids, texts = load_corpus(cfg, args)
    labels, swap_ids = resolve_swap_labels(cfg, args)

    # Donors are drawn from the owner's own split so no held-out caption text
    # reaches a training decoy. splits.json is written by train_retrieval into the
    # base cache dir; we are resuming one of its checkpoints, so it exists in
    # practice. Without it, donors are drawn corpus-wide and we say so.
    split_of_row = None
    splits_path = Path(args.splits) if args.splits else Path(args.cache_dir) / "splits.json"
    if args.allow_cross_split_donors:
        print("[decoy-plan] --allow-cross-split-donors: donor pool is the whole "
              "corpus (val/test caption fragments may appear in train decoys)")
    elif splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        if int(sp.get("n", -1)) != len(texts):
            # `load_corpus` already matched the CSV against the base cache's
            # pair_ids.json, so reaching here means the cache and CSV agree and
            # only the split file is out of date. That is worth spelling out,
            # because it is *also* evidence that train_retrieval never wrote
            # splits for this cache dir: build_or_load_splits rebuilds a stale
            # file and saves it back, so a run against this cache would have
            # fixed it. Either the checkpoint was trained against a different
            # --cache-dir (in which case the decoys belong there, since they are
            # indexed by base-cache row), or this cache was re-precomputed at
            # full scale after that run and the smoke-test split was left behind.
            raise RuntimeError(
                f"{splits_path} covers {sp.get('n')} rows but the cache/CSV has "
                f"{len(texts)} (they agree with each other — only the split file "
                f"is stale).\n"
                f"Splits are a deterministic function of the cache + seed + "
                f"ratios, so regenerating reproduces exactly what train_retrieval "
                f"used:\n"
                f"    python -m src.rebuild_splits --cache-dir {args.cache_dir} "
                f"--seed <the retrieval run's --seed, default 0>\n"
                f"First confirm this IS the cache the checkpoint was trained "
                f"against: decoys are indexed by base-cache row, so building them "
                f"against the wrong cache attaches every decoy to the wrong "
                f"protein. Alternatively pass --splits <path> to point at the "
                f"right file, or --allow-cross-split-donors to drop the "
                f"same-split requirement (leaks held-out caption text into "
                f"training decoys).")
        split_of_row = np.full(len(texts), 2, dtype=np.int8)     # default: test
        split_of_row[np.asarray(sp["train"], dtype=np.int64)] = 0
        split_of_row[np.asarray(sp["val"], dtype=np.int64)] = 1
        print(f"[decoy-plan] donors restricted to the owner's split "
              f"({splits_path}): train={int((split_of_row == 0).sum())} "
              f"val={int((split_of_row == 1).sum())} "
              f"test={int((split_of_row == 2).sum())}")
    else:
        raise FileNotFoundError(
            f"No split file at {splits_path}. The decoy plan keeps donors inside "
            f"the owner's split so held-out caption text never leaks into training "
            f"decoys. Point --splits at the retrieval run's splits.json, or pass "
            f"--allow-cross-split-donors to accept corpus-wide donors.")

    max_value_start = (args.max_swap_char if args.max_swap_char > 0
                       else auto_max_value_start(args.max_text_tokens, args.chars_per_token))
    t0 = time.time()
    plan = build_decoy_plan(
        texts, uids, labels,
        decoys_per_row=args.decoys_per_row,
        seed=args.decoy_seed,
        swap_label_ids=swap_ids,
        max_value_start=max_value_start,
        split_of_row=split_of_row,
        allow_same_protein_donor=args.allow_same_protein_donor,
        donor_tries=args.donor_tries,
    )
    plan["stats"]["plan_seconds"] = round(time.time() - t0, 1)
    torch.save({k: torch.from_numpy(plan[k]) for k in PLAN_KEYS}
               | {"labels": plan["labels"], "stats": plan["stats"]},
               out_dir / "decoy_plan.pt")
    with open(out_dir / "decoy_stats.json", "w") as f:
        json.dump(plan["stats"], f, indent=2)
    print(f"[decoy-plan] saved {out_dir / 'decoy_plan.pt'} in "
          f"{plan['stats']['plan_seconds']}s: {json.dumps(plan['stats'])}", flush=True)


def load_plan(decoy_cache_dir: str) -> dict:
    path = Path(decoy_cache_dir) / "decoy_plan.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; build it first with "
            f"`python -m src.precompute_decoys --plan-only`.")
    raw = torch.load(path, map_location="cpu")
    plan = {k: raw[k].numpy() for k in PLAN_KEYS}
    plan["labels"] = raw["labels"]
    plan["stats"] = raw["stats"]
    return plan


# ---------------------------------------------------------------------------
# Phase 2: encode this rank's slice of the plan
# ---------------------------------------------------------------------------
def encode_shard(args, cfg, env) -> None:
    shards_dir = Path(args.shards_dir)
    # See src/shards.py: a retry at a different world size leaves orphans.
    reset_shard_dir(shards_dir, env)
    device = env.device

    _, texts = load_corpus(cfg, args, verbose=env.is_main)
    plan = load_plan(args.decoy_cache_dir)
    m = int(plan["owner_row"].shape[0])
    if env.is_main:
        print(f"[decoy] plan holds {m} decoys over {len(texts)} rows; "
              f"world={env.world_size}", flush=True)

    if args.load_stagger > 0 and env.world_size > 1:
        time.sleep((env.local_rank % 12) * args.load_stagger)
    model, tok = load_text_encoder(
        cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
    barrier()

    mask_specials = not args.no_mask_text_specials
    mask_field_labels = not args.no_mask_text_field_labels
    effective_max = min(args.max_text_tokens, text_encoder_max_len(model))

    start, end = _shard_range(m, env.rank, env.world_size)
    tag = f"{env.rank:05d}"
    print(f"[decoy][rank {env.rank}] decoys [{start},{end}) -> {end - start} "
          f"on {device}", flush=True)

    h_fh = open(shards_dir / f"decoy_h.{tag}.bin", "wb")
    mask_fh = open(shards_dir / f"decoy_mask.{tag}.bin", "wb")
    lens: list[int] = []
    keep: list[int] = []
    n_dropped = 0
    bs = args.batch_size
    t0 = time.time()
    last_log = t0
    for s in range(start, end, bs):
        e = min(s + bs, end)
        decoy_texts = plan_decoy_texts(plan, texts, s, e)
        # Exact truncation check: if the swapped field falls outside the encoder's
        # window the decoy tokenizes to the same ids as the truth, and asking the
        # loss to separate two identical inputs is pure gradient noise.
        owner_texts = [texts[int(plan["owner_row"][i])] for i in range(s, e)]
        ids_d = tok(decoy_texts, truncation=True, max_length=effective_max)["input_ids"]
        ids_o = tok(owner_texts, truncation=True, max_length=effective_max)["input_ids"]
        batch_keep = [int(a != b) for a, b in zip(ids_d, ids_o)]
        n_dropped += len(batch_keep) - sum(batch_keep)
        keep.extend(batch_keep)

        h, mask = encode_text_batch(
            model, tok, decoy_texts, device, args.max_text_tokens,
            mask_specials=mask_specials, mask_field_labels=mask_field_labels)
        for row in range(h.size(0)):
            k = mask[row]
            h_row = h[row][k].to(torch.bfloat16)
            h_fh.write(_bf16_to_uint16_bytes(h_row).tobytes())
            mask_fh.write(mask[row][k].cpu().numpy().astype(np.uint8).tobytes())
            lens.append(int(h_row.size(0)))
        if env.is_main and (e == end or (time.time() - last_log) > 10.0):
            done = e - start
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[decoy][rank 0] {done}/{end - start}  {rate:.1f} decoys/s "
                  f"eta={(end - e) / max(rate, 1e-6) / 60:.1f} min", flush=True)
            last_log = time.time()
    h_fh.close()
    mask_fh.close()

    with open(shards_dir / f"decoymeta.{tag}.json", "w") as f:
        json.dump({"rank": env.rank, "world": env.world_size,
                   "start": start, "end": end,
                   "lens": lens, "keep": keep}, f)
    print(f"[decoy][rank {env.rank}] shard done: {len(lens)} decoys, "
          f"{sum(lens)} tokens, {n_dropped} dropped (swap outside the token window)",
          flush=True)


# ---------------------------------------------------------------------------
# Phase 3: merge shards -> single-file decoy cache
# ---------------------------------------------------------------------------
def merge_shards(args, cfg) -> None:
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.decoy_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = load_plan(args.decoy_cache_dir)
    m = int(plan["owner_row"].shape[0])
    n_rows = int(plan["row_ptr"].shape[0] - 1)

    meta_paths = sorted(glob.glob(str(shards_dir / "decoymeta.*.json")))
    if not meta_paths:
        raise RuntimeError(f"No decoy shard metadata in {shards_dir}; encode first.")
    metas = []
    for mp in meta_paths:
        with open(mp) as f:
            metas.append(json.load(f))
    check_shard_world(metas, shards_dir, "decoy shards")

    h_out = open(out_dir / "decoy_h.bin", "wb")
    mask_out = open(out_dir / "decoy_mask.bin", "wb")
    offsets = [0]
    keep: list[int] = []
    expect_start = 0
    for mp, meta in zip(meta_paths, metas):
        if meta["start"] != expect_start:
            raise RuntimeError(
                f"decoy shard gap/overlap: {mp} starts at {meta['start']}, expected "
                f"{expect_start}. Shards must tile [0,M) contiguously.")
        expect_start = meta["end"]
        tag = f"{meta['rank']:05d}"
        _copy_into(shards_dir / f"decoy_h.{tag}.bin", h_out)
        _copy_into(shards_dir / f"decoy_mask.{tag}.bin", mask_out)
        for l in meta["lens"]:
            offsets.append(offsets[-1] + l)
        keep.extend(meta["keep"])
    h_out.close()
    mask_out.close()

    if expect_start != m:
        raise RuntimeError(
            f"decoy shards tile [0,{expect_start}) but the plan holds {m} decoys. "
            f"Missing or extra shards.")
    if len(keep) != m:
        raise RuntimeError(f"keep flags ({len(keep)}) != plan decoys ({m}).")

    torch.save(torch.tensor(offsets, dtype=torch.long), out_dir / "decoy_offsets.pt")
    keep_t = torch.tensor(keep, dtype=torch.bool)
    torch.save(keep_t, out_dir / "decoy_keep.pt")
    torch.save(torch.from_numpy(plan["row_ptr"].astype(np.int64)),
               out_dir / "decoy_row_ptr.pt")

    # Usable coverage is what actually matters to training: rows whose decoys all
    # got dropped contribute nothing to the decoy loss.
    owner = torch.from_numpy(plan["owner_row"].astype(np.int64))
    per_row_kept = torch.zeros(n_rows, dtype=torch.long)
    per_row_kept.index_add_(0, owner, keep_t.long())
    n_kept = int(keep_t.sum())
    stats = dict(plan["stats"])
    stats.update({
        "n_decoys_encoded": m,
        "n_decoys_kept": n_kept,
        "n_decoys_dropped_truncated": m - n_kept,
        "rows_with_usable_decoy": int((per_row_kept > 0).sum()),
        "rows_total": n_rows,
        "mean_usable_decoys_per_row": round(n_kept / max(n_rows, 1), 3),
    })
    with open(out_dir / "decoy_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    base_fp = read_cache_fingerprint(args.cache_dir)
    if not base_fp:
        base_fp = cache_fingerprint(
            cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
            args.max_text_tokens, cfg.data.max_protein_tokens,
            not args.no_mask_text_specials, cfg.retrieval.mask_protein_special_tokens,
            cfg.data.caption_field_labels, not args.no_mask_text_field_labels)
        print("[decoy-merge] WARNING: no base fingerprint found; synthesizing one "
              "from config. Verify --cache-dir points at the retrieval cache.")
    write_decoy_fingerprint(str(out_dir), decoy_fingerprint(
        base_fp, decoys_per_row=args.decoys_per_row, seed=args.decoy_seed,
        swap_fields=args.swap_fields or cfg.data.caption_field_labels,
        max_value_start=stats.get("max_value_start", 0),
        allow_same_protein_donor=args.allow_same_protein_donor, n_rows=n_rows))

    size_gb = (out_dir / "decoy_h.bin").stat().st_size / 1e9
    print(f"[decoy-merge] done. {m} decoys ({n_kept} usable) for {n_rows} rows; "
          f"{stats['rows_with_usable_decoy']} rows have at least one. "
          f"{size_gb:.2f} GB", flush=True)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir,
                    help="base per-token cache (read for row order + fingerprint)")
    ap.add_argument("--decoy-cache-dir", default=cfg.hard_neg.decoy_cache_dir)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--splits", default=None,
                    help="splits.json used to keep donors inside the owner's split "
                         "(default: <cache-dir>/splits.json)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--max-text-tokens", type=int, default=cfg.data.max_text_tokens)
    ap.add_argument("--no-mask-text-specials", action="store_true")
    ap.add_argument("--no-mask-text-field-labels", action="store_true")

    ap.add_argument("--decoys-per-row", type=int, default=cfg.hard_neg.decoys_per_row)
    ap.add_argument("--decoy-seed", type=int, default=cfg.hard_neg.decoy_seed)
    ap.add_argument("--swap-fields", nargs="*", default=list(cfg.hard_neg.swap_fields),
                    help="restrict swapping to these caption field labels "
                         "(default: all of DataCfg.caption_field_labels)")
    ap.add_argument("--donor-tries", type=int, default=8,
                    help="donor samples attempted per decoy before giving up")
    ap.add_argument("--allow-same-protein-donor", action="store_true",
                    default=cfg.hard_neg.donor_same_protein,
                    help="permit donors from another caption of the SAME protein "
                         "(off by default: another caption of the same protein "
                         "usually has a field that is also "
                         "true of this protein, making the decoy a false negative)")
    ap.add_argument("--allow-cross-split-donors", action="store_true",
                    default=cfg.hard_neg.donor_cross_split,
                    help="permit donors from other splits (leaks held-out caption "
                         "fragments into training decoys)")
    ap.add_argument("--max-swap-char", type=int, default=0,
                    help="only swap fields whose value starts before this character "
                         "offset (0 = derive from --max-text-tokens)")
    ap.add_argument("--chars-per-token", type=float, default=3.0,
                    help="conservative chars/token used to derive --max-swap-char")

    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--encode-only", action="store_true", help="write shards, skip merge")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--load-stagger", type=float, default=0.5)
    args = ap.parse_args()
    if args.shards_dir is None:
        args.shards_dir = str(Path(args.decoy_cache_dir) / "shards")

    base_fp = read_cache_fingerprint(args.cache_dir)
    expected = cache_fingerprint(
        cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
        args.max_text_tokens, cfg.data.max_protein_tokens,
        not args.no_mask_text_specials, cfg.retrieval.mask_protein_special_tokens,
        cfg.data.caption_field_labels, not args.no_mask_text_field_labels)
    if base_fp and not fingerprint_matches(base_fp, expected):
        raise RuntimeError(
            f"Base cache fingerprint mismatch at {args.cache_dir}.\n"
            f"  expected: {expected}\n  found:    {base_fp}\n"
            f"Decoys must be encoded exactly like the base text cache — same text "
            f"encoder, length cap and masking flags — or their scores are not "
            f"comparable with the true captions'.")

    if args.plan_only:
        # Guarded so `--plan-only` is safe under mpiexec too: the plan is a single
        # file written once, and N ranks racing on it would interleave bytes.
        env = init_distributed(args.device, group_size=1, init_pg=False)
        if env.is_main:
            build_plan(args, cfg)
        barrier()
        cleanup()
        return
    if args.merge_only:
        merge_shards(args, cfg)
        return

    # Embarrassingly parallel like precompute: rank/world for sharding + an MPI
    # barrier, no oneCCL process group.
    env = init_distributed(args.device, group_size=1, init_pg=False)
    if env.is_main and not (Path(args.decoy_cache_dir) / "decoy_plan.pt").exists():
        build_plan(args, cfg)
    barrier()
    encode_shard(args, cfg, env)
    barrier()
    if not args.encode_only and env.is_main:
        merge_shards(args, cfg)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
