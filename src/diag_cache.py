"""Diagnostic: does the precomputed token cache contain a max-sim collapse shortcut?

A FILIP max-sim collapse (every cross-token cos -> 1) is trivial to reach if the
cache leaks a token that is (near-)identical across *all* sequences and survives
the valid mask -- e.g. an unmasked BOS/EOS. Then max-sim latches onto that shared
token and every pair scores ~1. This checks the RAW cached hidden states (not the
trainable projection) for that pathology.

    python -m src.diag_cache --cache-dir cache --n 256
"""
from __future__ import annotations

import argparse

import torch

from config import default_cfg
from src.data import PackedPerTokenCache


def modality_report(name: str, cache: PackedPerTokenCache, n_probe: int) -> None:
    within = []          # mean within-sequence pairwise cos of valid tokens
    valid_counts, raw_counts = [], []
    first_valid_vecs = []   # first valid token of each seq (unit-norm) -> cross-seq check
    idxs = torch.randperm(len(cache))[:n_probe].tolist()   # indices valid for THIS cache
    for i in idxs:
        h, m = cache.get(int(i))            # h [L, D] bf16, m [L] bool (no batch dim)
        h, m = h.float(), m.bool()
        raw_counts.append(m.numel())
        valid_counts.append(int(m.sum()))
        v = h[m]                             # [V, D] valid tokens only
        if v.size(0) < 2:
            continue
        vn = torch.nn.functional.normalize(v, dim=-1)
        sim = vn @ vn.t()
        off = ~torch.eye(vn.size(0), dtype=torch.bool)
        within.append(sim[off].mean().item())
        first_valid_vecs.append(vn[0])

    within = torch.tensor(within)
    fv = torch.stack(first_valid_vecs)       # [S, D]
    cross = fv @ fv.t()
    off = ~torch.eye(fv.size(0), dtype=torch.bool)
    cross_first = cross[off].mean().item()   # cross-seq cos of the first valid token

    vc = torch.tensor(valid_counts).float()
    rc = torch.tensor(raw_counts).float()
    print(f"\n=== {name} ===")
    print(f"  seqs probed:                {len(within)}")
    print(f"  raw tokens/seq  (mean):     {rc.mean():.1f}")
    print(f"  valid tokens/seq (mean):    {vc.mean():.1f}   "
          f"(dropped {(rc.mean()-vc.mean()):.1f} as specials/pad)")
    print(f"  within-seq token cos:       {within.mean():.4f}  "
          f"(healthy << 1; ~1 => degenerate tokens)")
    print(f"  cross-seq FIRST-valid cos:  {cross_first:.4f}  "
          f"(~1 => a shared token leaks the mask -> collapse shortcut)")


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--n", type=int, default=256, help="sequences to probe per modality")
    args = ap.parse_args()

    p_cache = PackedPerTokenCache(args.cache_dir, "protein", cfg.model.protein_hidden)
    t_cache = PackedPerTokenCache(args.cache_dir, "text", cfg.model.text_hidden)

    modality_report("PROTEIN (AMPLIFY-350M, 960-d)", p_cache, args.n)
    modality_report("TEXT (BioLinkBERT, 768-d)", t_cache, args.n)
    print("\nInterpretation:")
    print("  - within-seq cos ~1 in a modality  => that encoder's cached tokens are")
    print("    degenerate (precompute/layer/mask bug); no loss tweak will fix it.")
    print("  - cross-seq first-valid cos ~1     => a shared special token survived the")
    print("    mask; fix _protein_valid_mask / _text_valid_mask, re-precompute.")
    print("  - both healthy (<~0.3)             => collapse is training dynamics, not the")
    print("    cache; pursue temperature / uniformity levers instead.")


if __name__ == "__main__":
    main()
