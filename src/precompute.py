"""One-shot per-token cache builder.

For every (protein, text) pair, runs the frozen encoders and writes packed
bf16 per-token hidden states + valid-token masks to disk, plus an offsets
table that lets us read each row's variable-length slice.

Output layout (under cache_dir):
  protein_h.bin       bf16, total_protein_tokens × 640
  protein_offsets.pt  int64 [N+1]
  protein_mask.bin    uint8, total_protein_tokens
  text_h.bin          bf16, total_text_tokens × 768
  text_offsets.pt     int64 [N+1]
  text_mask.bin       uint8, total_text_tokens
  pair_ids.json       list of UniProt IDs
  fingerprint.json    (encoder paths, length caps, special-mask flags)

Usage:
  python -m src.precompute --device cuda --batch-size 64
  python -m src.precompute --device cpu  --batch-size 8 --subset-size 1000
"""
from __future__ import annotations

import argparse
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
from src.encoders import (
    encode_protein_batch,
    encode_text_batch,
    load_protein_encoder,
    load_text_encoder,
)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _bf16_to_uint16_bytes(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> uint16 numpy view (binary-equivalent storage)."""
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.uint16).cpu().numpy()


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=cfg.retrieval.device)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--max-text-tokens", type=int, default=cfg.data.max_text_tokens)
    ap.add_argument("--max-protein-tokens", type=int, default=cfg.data.max_protein_tokens)
    ap.add_argument("--no-mask-text-specials", action="store_true")
    ap.add_argument("--no-mask-protein-specials", action="store_true")
    args = ap.parse_args()

    mask_text_specials = not args.no_mask_text_specials
    mask_protein_specials = not args.no_mask_protein_specials

    device = pick_device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] device={device}  cache_dir={cache_dir}")

    pairs = load_pairs(
        cfg.data.csv_path,
        id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col,
        text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.subset_size,
    )
    n = len(pairs)
    print(f"[precompute] loaded {n} pairs; max_text_tokens={args.max_text_tokens} "
          f"max_protein_tokens={args.max_protein_tokens}")

    print("[precompute] loading text encoder (BioLinkBERT-base)...")
    text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
    print("[precompute] loading protein encoder (SaAMPLIFY-120M)...")
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)

    # Open output files; we stream-append rather than buffer the whole cache.
    p_h_path = cache_dir / "protein_h.bin"
    p_mask_path = cache_dir / "protein_mask.bin"
    t_h_path = cache_dir / "text_h.bin"
    t_mask_path = cache_dir / "text_mask.bin"
    for p in (p_h_path, p_mask_path, t_h_path, t_mask_path):
        if p.exists():
            p.unlink()
    p_h_fh = open(p_h_path, "wb")
    p_mask_fh = open(p_mask_path, "wb")
    t_h_fh = open(t_h_path, "wb")
    t_mask_fh = open(t_mask_path, "wb")

    p_offsets = [0]
    t_offsets = [0]
    ids = []

    bs = args.batch_size
    t0 = time.time()
    last_log = t0
    for start in range(0, n, bs):
        end = min(start + bs, n)
        chunk = pairs[start:end]
        ids.extend(c.uid for c in chunk)

        h_t, mask_t = encode_text_batch(
            text_model, text_tok, [c.text for c in chunk],
            device, args.max_text_tokens, mask_specials=mask_text_specials,
        )
        h_p, mask_p = encode_protein_batch(
            prot_model, prot_tok, [c.protein for c in chunk],
            device, args.max_protein_tokens, mask_specials=mask_protein_specials,
        )

        # For each row in chunk: trim to actual length, write bytes.
        # "Actual length" = where attention_mask was 1 (pre-special-masking).
        # Since `mask_t`/`mask_p` are valid-token masks (already exclude specials
        # when requested), we'd lose padding boundaries if we used them. So we
        # take the row's "non-zero hidden state" tail as the boundary instead:
        # everything from index 0 up to the last position that was attended to.
        # Easiest: derive from the encoder's attention mask via row sum length.
        # We do this by computing per-row length from the bool valid_mask
        # bitwise-OR-ed with the encoder pad mask, which we recompute.

        # To keep this simple, store ONLY valid positions (specials removed
        # if mask_*_specials=True). Saves storage and we don't need pad/special
        # positions downstream anyway.
        for row in range(h_t.size(0)):
            keep_t = mask_t[row]
            keep_p = mask_p[row]
            ht_row = h_t[row][keep_t].to(torch.bfloat16)        # [L_t_valid, 768]
            hp_row = h_p[row][keep_p].to(torch.bfloat16)        # [L_p_valid, 640]
            mt_row = mask_t[row][keep_t]                        # all True
            mp_row = mask_p[row][keep_p]                        # all True

            p_h_fh.write(_bf16_to_uint16_bytes(hp_row).tobytes())
            p_mask_fh.write(mp_row.cpu().numpy().astype(np.uint8).tobytes())
            t_h_fh.write(_bf16_to_uint16_bytes(ht_row).tobytes())
            t_mask_fh.write(mt_row.cpu().numpy().astype(np.uint8).tobytes())

            p_offsets.append(p_offsets[-1] + hp_row.size(0))
            t_offsets.append(t_offsets[-1] + ht_row.size(0))

        now = time.time()
        if now - last_log > 5.0 or end == n:
            rate = end / max(now - t0, 1e-6)
            eta = (n - end) / max(rate, 1e-6)
            print(f"[precompute] {end}/{n}  {rate:.1f} pairs/s  eta={eta/60:.1f} min  "
                  f"p_tokens={p_offsets[-1]} t_tokens={t_offsets[-1]}")
            last_log = now

    p_h_fh.close()
    p_mask_fh.close()
    t_h_fh.close()
    t_mask_fh.close()

    torch.save(torch.tensor(p_offsets, dtype=torch.long), cache_dir / "protein_offsets.pt")
    torch.save(torch.tensor(t_offsets, dtype=torch.long), cache_dir / "text_offsets.pt")
    with open(cache_dir / "pair_ids.json", "w") as f:
        json.dump(ids, f)
    fp = cache_fingerprint(
        cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
        args.max_text_tokens, args.max_protein_tokens,
        mask_text_specials, mask_protein_specials,
    )
    write_cache_fingerprint(str(cache_dir), fp)

    bytes_p = p_h_path.stat().st_size + p_mask_path.stat().st_size
    bytes_t = t_h_path.stat().st_size + t_mask_path.stat().st_size
    print(f"[precompute] done. {n} rows. "
          f"protein cache {bytes_p/1e9:.2f} GB, text cache {bytes_t/1e9:.2f} GB")


if __name__ == "__main__":
    main()
