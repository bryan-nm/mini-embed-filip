"""One-shot per-token cache builder (distributed, sharded).

For every (protein, text) pair, runs the frozen encoders and writes packed
bf16 per-token hidden states + valid-token masks to disk, plus an offsets
table that lets us read each row's variable-length slice.

Distributed model (Aurora): launched under `mpiexec` with one rank per GPU
tile. Each rank encodes a *contiguous* slice of the dataset (so global row
order is preserved) and writes per-rank shard files into `<cache>/shards/`.
A single merge pass then concatenates the shards, in rank order, into the
final single-file cache the trainers/readers expect.

Final output layout (under cache_dir), identical to the original format:
  protein_h.bin       bf16, total_protein_tokens × 640
  protein_offsets.pt  int64 [N+1]
  protein_mask.bin    uint8, total_protein_tokens
  text_h.bin          bf16, total_text_tokens × 768
  text_offsets.pt     int64 [N+1]
  text_mask.bin       uint8, total_text_tokens
  pair_ids.json       list of UniProt IDs
  fingerprint.json    (encoder paths, length caps, special-mask flags)

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
# Encode one rank's slice -> shard files
# ---------------------------------------------------------------------------
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
    n = len(pairs)
    start, end = _shard_range(n, env.rank, env.world_size)
    my_pairs = pairs[start:end]
    if env.is_main:
        print(f"[precompute] n={n} world={env.world_size} "
              f"max_text={args.max_text_tokens} max_protein={args.max_protein_tokens}",
              flush=True)
    print(f"[precompute][rank {env.rank}] rows [{start},{end}) -> {len(my_pairs)} pairs "
          f"on {device}", flush=True)

    # Stagger encoder load to avoid 3072 ranks hammering the model files at once.
    if args.load_stagger > 0 and env.world_size > 1:
        time.sleep((env.local_rank % 12) * args.load_stagger)
    text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
    barrier()  # everyone past model load before timing the encode

    tag = f"{env.rank:05d}"
    p_h_fh = open(shards_dir / f"protein_h.{tag}.bin", "wb")
    p_mask_fh = open(shards_dir / f"protein_mask.{tag}.bin", "wb")
    t_h_fh = open(shards_dir / f"text_h.{tag}.bin", "wb")
    t_mask_fh = open(shards_dir / f"text_mask.{tag}.bin", "wb")

    p_lens: list[int] = []
    t_lens: list[int] = []
    ids: list[str] = []

    bs = args.batch_size
    t0 = time.time()
    last_log = t0
    m = len(my_pairs)
    for s in range(0, m, bs):
        e = min(s + bs, m)
        chunk = my_pairs[s:e]
        ids.extend(c.uid for c in chunk)

        h_t, mask_t = encode_text_batch(
            text_model, text_tok, [c.text for c in chunk],
            device, args.max_text_tokens, mask_specials=mask_text_specials,
        )
        h_p, mask_p = encode_protein_batch(
            prot_model, prot_tok, [c.protein for c in chunk],
            device, args.max_protein_tokens, mask_specials=mask_protein_specials,
        )
        for row in range(h_t.size(0)):
            keep_t = mask_t[row]
            keep_p = mask_p[row]
            ht_row = h_t[row][keep_t].to(torch.bfloat16)
            hp_row = h_p[row][keep_p].to(torch.bfloat16)
            p_h_fh.write(_bf16_to_uint16_bytes(hp_row).tobytes())
            t_h_fh.write(_bf16_to_uint16_bytes(ht_row).tobytes())
            p_mask_fh.write(mask_p[row][keep_p].cpu().numpy().astype(np.uint8).tobytes())
            t_mask_fh.write(mask_t[row][keep_t].cpu().numpy().astype(np.uint8).tobytes())
            p_lens.append(int(hp_row.size(0)))
            t_lens.append(int(ht_row.size(0)))

        if env.is_main and (e == m or (time.time() - last_log) > 10.0):
            rate = e / max(time.time() - t0, 1e-6)
            print(f"[precompute][rank 0] {e}/{m}  {rate:.1f} pairs/s "
                  f"eta={ (m-e)/max(rate,1e-6)/60:.1f} min", flush=True)
            last_log = time.time()

    for fh in (p_h_fh, p_mask_fh, t_h_fh, t_mask_fh):
        fh.close()
    with open(shards_dir / f"meta.{tag}.json", "w") as f:
        json.dump({"rank": env.rank, "start": start, "end": end,
                   "p_lens": p_lens, "t_lens": t_lens, "ids": ids}, f)
    print(f"[precompute][rank {env.rank}] shard done: {len(ids)} rows, "
          f"{sum(p_lens)} p-tokens, {sum(t_lens)} t-tokens", flush=True)


# ---------------------------------------------------------------------------
# Merge all shards (one process) -> final single-file cache
# ---------------------------------------------------------------------------
def merge_shards(args, cfg) -> None:
    shards_dir = Path(args.shards_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(glob.glob(str(shards_dir / "meta.*.json")))
    if not metas:
        raise RuntimeError(f"No shard metadata found in {shards_dir}; run encode first.")
    print(f"[merge] {len(metas)} shards from {shards_dir}", flush=True)

    p_h_out = open(cache_dir / "protein_h.bin", "wb")
    p_mask_out = open(cache_dir / "protein_mask.bin", "wb")
    t_h_out = open(cache_dir / "text_h.bin", "wb")
    t_mask_out = open(cache_dir / "text_mask.bin", "wb")

    p_offsets = [0]
    t_offsets = [0]
    ids: list[str] = []
    expect_start = 0
    bufsize = 64 * 1024 * 1024

    def _copy(src: Path, dst):
        with open(src, "rb") as fh:
            while True:
                b = fh.read(bufsize)
                if not b:
                    break
                dst.write(b)

    for mp in metas:
        with open(mp) as f:
            meta = json.load(f)
        if meta["start"] != expect_start:
            raise RuntimeError(
                f"Shard gap/overlap: {mp} starts at {meta['start']}, expected "
                f"{expect_start}. Shards must tile [0,N) contiguously.")
        expect_start = meta["end"]
        tag = f"{meta['rank']:05d}"
        _copy(shards_dir / f"protein_h.{tag}.bin", p_h_out)
        _copy(shards_dir / f"protein_mask.{tag}.bin", p_mask_out)
        _copy(shards_dir / f"text_h.{tag}.bin", t_h_out)
        _copy(shards_dir / f"text_mask.{tag}.bin", t_mask_out)
        for pl in meta["p_lens"]:
            p_offsets.append(p_offsets[-1] + pl)
        for tl in meta["t_lens"]:
            t_offsets.append(t_offsets[-1] + tl)
        ids.extend(meta["ids"])

    for fh in (p_h_out, p_mask_out, t_h_out, t_mask_out):
        fh.close()

    torch.save(torch.tensor(p_offsets, dtype=torch.long), cache_dir / "protein_offsets.pt")
    torch.save(torch.tensor(t_offsets, dtype=torch.long), cache_dir / "text_offsets.pt")
    with open(cache_dir / "pair_ids.json", "w") as f:
        json.dump(ids, f)
    fp = cache_fingerprint(
        cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
        args.max_text_tokens, args.max_protein_tokens,
        not args.no_mask_text_specials, not args.no_mask_protein_specials,
    )
    write_cache_fingerprint(str(cache_dir), fp)
    bytes_p = (cache_dir / "protein_h.bin").stat().st_size
    bytes_t = (cache_dir / "text_h.bin").stat().st_size
    print(f"[merge] done. {len(ids)} rows. protein {bytes_p/1e9:.2f} GB, "
          f"text {bytes_t/1e9:.2f} GB", flush=True)


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

    env = init_distributed(args.device, group_size=1)
    encode_shard(args, cfg, env)
    barrier()
    if not args.encode_only and env.is_main:
        merge_shards(args, cfg)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
