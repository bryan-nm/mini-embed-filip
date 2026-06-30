"""One-shot per-token cache builder (distributed, sharded).

For every (protein, text) pair, runs the frozen encoders and writes packed
bf16 per-token hidden states + valid-token masks to disk, plus an offsets
table that lets us read each row's variable-length slice.

Distributed model (Aurora): launched under `mpiexec` with one rank per GPU
tile. Each rank encodes a *contiguous* slice of the dataset (so global row
order is preserved) and writes per-rank shard files into `<cache>/shards/`.
A single merge pass then concatenates the shards, in rank order, into the
final single-file cache the trainers/readers expect.

Protein dedup: the augmented corpus repeats each protein across ~8.87 caption
rows. The protein modality is encoded + stored once per *unique* protein
(keyed by accession), the text modality once per CSV row, and a row_protein_idx
map joins them at read time. This avoids ~9x of the protein encoder pass (the
precompute bottleneck) and ~1.5 TB of duplicated protein cache.

Final output layout (under cache_dir):
  protein_h.bin       bf16, total_protein_tokens × 960   (UNIQUE proteins)
  protein_offsets.pt  int64 [N_unique+1]
  protein_mask.bin    uint8, total_protein_tokens
  protein_ids.json    list of accessions, unique-protein order [N_unique]
  text_h.bin          bf16, total_text_tokens × 768      (per CSV row)
  text_offsets.pt     int64 [N_rows+1]
  text_mask.bin       uint8, total_text_tokens
  pair_ids.json       list of accessions, one per CSV row [N_rows]
  row_protein_idx.pt  int64 [N_rows]; CSV row -> unique-protein index
  fingerprint.json    (format tag, encoder paths, length caps, special-mask flags)

Usage:
  # distributed encode + merge in one job (all ranks encode; rank 0 merges):
  mpiexec ... python -m src.precompute --device xpu --batch-size 64
  # split phases explicitly:
  mpiexec ... python -m src.precompute --device xpu --encode-only
  python -m src.precompute --merge-only            # one process
  # laptop smoke test (single process, no MPI):
  python -m src.precompute --device cpu --batch-size 8 --subset-size 1000
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
from src.data import (
    cache_fingerprint,
    dedup_proteins,
    load_pairs,
    write_cache_fingerprint,
)
from src.dist import barrier, cleanup, init_distributed
from src.encoders import (
    encode_protein_batch,
    encode_text_batch,
    load_protein_encoder,
    load_text_encoder,
)


def _bf16_to_uint16_bytes(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> uint16 numpy view (binary-equivalent storage)."""
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _shard_range(n: int, rank: int, world: int) -> tuple[int, int]:
    """Contiguous, balanced [start, end) for this rank. Order-preserving."""
    per, rem = divmod(n, world)
    start = rank * per + min(rank, rem)
    end = start + per + (1 if rank < rem else 0)
    return start, end


# ---------------------------------------------------------------------------
# Encode one modality's slice -> packed shard files (generic over modality)
# ---------------------------------------------------------------------------
def _encode_modality(items, ids, start, end, prefix, encode_fn,
                     shards_dir: Path, tag: str, env, bs: int) -> None:
    """Encode `items` (list of strings) in batches, writing a packed shard.

    Each row's invalid (masked-out) positions are dropped before packing, so
    storage matches the FILIP/uniformity valid-token set. Writes
    <prefix>_h.<tag>.bin, <prefix>_mask.<tag>.bin and <prefix>meta.<tag>.json.
    """
    h_fh = open(shards_dir / f"{prefix}_h.{tag}.bin", "wb")
    mask_fh = open(shards_dir / f"{prefix}_mask.{tag}.bin", "wb")
    lens: list[int] = []
    m = len(items)
    t0 = time.time()
    last_log = t0
    for s in range(0, m, bs):
        e = min(s + bs, m)
        h, mask = encode_fn(items[s:e])
        for row in range(h.size(0)):
            keep = mask[row]
            h_row = h[row][keep].to(torch.bfloat16)
            h_fh.write(_bf16_to_uint16_bytes(h_row).tobytes())
            mask_fh.write(mask[row][keep].cpu().numpy().astype(np.uint8).tobytes())
            lens.append(int(h_row.size(0)))
        if env.is_main and (e == m or (time.time() - last_log) > 10.0):
            rate = e / max(time.time() - t0, 1e-6)
            print(f"[precompute][rank 0][{prefix}] {e}/{m}  {rate:.1f} items/s "
                  f"eta={(m-e)/max(rate,1e-6)/60:.1f} min", flush=True)
            last_log = time.time()
    h_fh.close()
    mask_fh.close()
    with open(shards_dir / f"{prefix}meta.{tag}.json", "w") as f:
        json.dump({"rank": env.rank, "start": start, "end": end,
                   "lens": lens, "ids": ids}, f)
    print(f"[precompute][rank {env.rank}][{prefix}] shard done: {len(ids)} rows, "
          f"{sum(lens)} tokens", flush=True)


def encode_shard(args, cfg, env) -> None:
    device = env.device
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    mask_text_specials = not args.no_mask_text_specials
    mask_protein_specials = not args.no_mask_protein_specials

    pairs = load_pairs(
        cfg.data.csv_path,
        id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col,
        text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.subset_size,
    )
    n_rows = len(pairs)
    # Deterministic, world-wide dedup (first-appearance by accession). Every rank
    # derives the identical unique-protein ordering from the same pairs list.
    _, unique_seqs, unique_ids = dedup_proteins(pairs)
    n_unique = len(unique_seqs)
    if env.is_main:
        print(f"[precompute] n_rows={n_rows} unique_proteins={n_unique} "
              f"world={env.world_size} max_text={args.max_text_tokens} "
              f"max_protein={args.max_protein_tokens}", flush=True)

    # Stagger encoder load to avoid thousands of ranks hammering the files at once.
    if args.load_stagger > 0 and env.world_size > 1:
        time.sleep((env.local_rank % 12) * args.load_stagger)
    text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
    barrier()  # everyone past model load before timing the encode

    tag = f"{env.rank:05d}"
    bs = args.batch_size

    # --- Protein pass: shard over UNIQUE proteins ---
    p_start, p_end = _shard_range(n_unique, env.rank, env.world_size)
    print(f"[precompute][rank {env.rank}] proteins [{p_start},{p_end}) "
          f"-> {p_end - p_start} on {device}", flush=True)
    _encode_modality(
        unique_seqs[p_start:p_end], unique_ids[p_start:p_end], p_start, p_end,
        "protein",
        lambda batch: encode_protein_batch(
            prot_model, prot_tok, batch, device, args.max_protein_tokens,
            mask_specials=mask_protein_specials),
        shards_dir, tag, env, bs,
    )

    # --- Text pass: shard over CSV rows ---
    t_start, t_end = _shard_range(n_rows, env.rank, env.world_size)
    print(f"[precompute][rank {env.rank}] text rows [{t_start},{t_end}) "
          f"-> {t_end - t_start} on {device}", flush=True)
    _encode_modality(
        [p.text for p in pairs[t_start:t_end]],
        [p.uid for p in pairs[t_start:t_end]], t_start, t_end, "text",
        lambda batch: encode_text_batch(
            text_model, text_tok, batch, device, args.max_text_tokens,
            mask_specials=mask_text_specials),
        shards_dir, tag, env, bs,
    )


# ---------------------------------------------------------------------------
# Merge all shards (one process) -> final single-file cache
# ---------------------------------------------------------------------------
_MERGE_BUFSIZE = 64 * 1024 * 1024


def _copy_into(src: Path, dst) -> None:
    with open(src, "rb") as fh:
        while True:
            b = fh.read(_MERGE_BUFSIZE)
            if not b:
                break
            dst.write(b)


def _merge_modality(shards_dir: Path, cache_dir: Path, prefix: str,
                    expected_total: int):
    """Concatenate one modality's shards (rank order) -> final cache files.

    Validates that the shards tile [0, expected_total) contiguously. Returns
    (offsets_tensor [rows+1], ids list).
    """
    metas = sorted(glob.glob(str(shards_dir / f"{prefix}meta.*.json")))
    if not metas:
        raise RuntimeError(
            f"No {prefix} shard metadata in {shards_dir}; run encode first.")
    h_out = open(cache_dir / f"{prefix}_h.bin", "wb")
    mask_out = open(cache_dir / f"{prefix}_mask.bin", "wb")
    offsets = [0]
    ids: list[str] = []
    expect_start = 0
    for mp in metas:
        with open(mp) as f:
            meta = json.load(f)
        if meta["start"] != expect_start:
            raise RuntimeError(
                f"{prefix} shard gap/overlap: {mp} starts at {meta['start']}, "
                f"expected {expect_start}. Shards must tile [0,N) contiguously.")
        expect_start = meta["end"]
        tag = f"{meta['rank']:05d}"
        _copy_into(shards_dir / f"{prefix}_h.{tag}.bin", h_out)
        _copy_into(shards_dir / f"{prefix}_mask.{tag}.bin", mask_out)
        for l in meta["lens"]:
            offsets.append(offsets[-1] + l)
        ids.extend(meta["ids"])
    h_out.close()
    mask_out.close()
    if expect_start != expected_total:
        raise RuntimeError(
            f"{prefix} shards tile [0,{expect_start}) but expected "
            f"[0,{expected_total}). Missing or extra shards.")
    off = torch.tensor(offsets, dtype=torch.long)
    torch.save(off, cache_dir / f"{prefix}_offsets.pt")
    return off, ids


def merge_shards(args, cfg) -> None:
    shards_dir = Path(args.shards_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Re-derive the row->protein map deterministically (same function the encode
    # ranks used), so it is consistent with the protein shard ordering.
    pairs = load_pairs(
        cfg.data.csv_path,
        id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col,
        text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.subset_size,
    )
    row_protein_idx, unique_seqs, _ = dedup_proteins(pairs)
    n_rows = len(pairs)
    n_unique = len(unique_seqs)
    print(f"[merge] n_rows={n_rows} unique_proteins={n_unique} from {shards_dir}",
          flush=True)

    _, protein_ids = _merge_modality(shards_dir, cache_dir, "protein", n_unique)
    _, pair_ids = _merge_modality(shards_dir, cache_dir, "text", n_rows)

    torch.save(torch.as_tensor(row_protein_idx, dtype=torch.long),
               cache_dir / "row_protein_idx.pt")
    with open(cache_dir / "protein_ids.json", "w") as f:
        json.dump(protein_ids, f)
    with open(cache_dir / "pair_ids.json", "w") as f:
        json.dump(pair_ids, f)
    fp = cache_fingerprint(
        cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
        args.max_text_tokens, args.max_protein_tokens,
        not args.no_mask_text_specials, not args.no_mask_protein_specials,
    )
    write_cache_fingerprint(str(cache_dir), fp)
    bytes_p = (cache_dir / "protein_h.bin").stat().st_size
    bytes_t = (cache_dir / "text_h.bin").stat().st_size
    print(f"[merge] done. {n_rows} rows, {n_unique} unique proteins. "
          f"protein {bytes_p/1e9:.2f} GB, text {bytes_t/1e9:.2f} GB", flush=True)


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--max-text-tokens", type=int, default=cfg.data.max_text_tokens)
    ap.add_argument("--max-protein-tokens", type=int, default=cfg.data.max_protein_tokens)
    ap.add_argument("--no-mask-text-specials", action="store_true")
    ap.add_argument("--no-mask-protein-specials", action="store_true")
    ap.add_argument("--encode-only", action="store_true", help="write shards, skip merge")
    ap.add_argument("--merge-only", action="store_true",
                    help="single-process merge of existing shards, skip encode")
    ap.add_argument("--load-stagger", type=float, default=0.5,
                    help="seconds * local_rank delay before encoder load (I/O herd)")
    args = ap.parse_args()
    if args.shards_dir is None:
        args.shards_dir = str(Path(args.cache_dir) / "shards")

    if args.merge_only:
        merge_shards(args, cfg)
        return

    # Precompute is embarrassingly parallel: no collectives beyond a barrier,
    # so skip the oneCCL process group and rely on the MPI barrier.
    env = init_distributed(args.device, group_size=1, init_pg=False)
    encode_shard(args, cfg, env)
    barrier()
    if not args.encode_only and env.is_main:
        merge_shards(args, cfg)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
