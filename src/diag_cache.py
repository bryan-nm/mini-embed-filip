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


def _pairwise_offdiag_cos(unit: torch.Tensor) -> float:
    sim = unit @ unit.t()
    off = ~torch.eye(unit.size(0), dtype=torch.bool)
    return sim[off].mean().item()


def modality_report(name: str, cache: PackedPerTokenCache, n_probe: int,
                    tokens_per_seq: int = 16) -> None:
    within = []          # mean within-sequence pairwise cos of valid tokens
    pool = []            # random valid tokens across seqs -> clean anisotropy probe
    idxs = torch.randperm(len(cache))[:n_probe].tolist()   # indices valid for THIS cache
    for i in idxs:
        h, m = cache.get(int(i))            # h [L, D] bf16, m [L] bool (no batch dim)
        v = h.float()[m.bool()]             # [V, D] valid tokens only
        if v.size(0) < 2:
            continue
        vn = torch.nn.functional.normalize(v, dim=-1)
        within.append(_pairwise_offdiag_cos(vn))
        sel = torch.randperm(v.size(0))[:tokens_per_seq]
        pool.append(v[sel])                 # keep RAW (un-normalized) for mean-centering

    within = torch.tensor(within)
    pool = torch.cat(pool, dim=0)           # [N, D] raw tokens

    raw_unit = torch.nn.functional.normalize(pool, dim=-1)
    cross_raw = _pairwise_offdiag_cos(raw_unit)          # anisotropy: any two tokens

    mu = pool.mean(dim=0, keepdim=True)                  # global "rogue" mean direction
    mu_unit = torch.nn.functional.normalize(mu, dim=-1)
    to_mean = (raw_unit @ mu_unit.t()).abs().mean().item()   # |cos(token, corpus mean)|

    cent_unit = torch.nn.functional.normalize(pool - mu, dim=-1)
    cross_centered = _pairwise_offdiag_cos(cent_unit)    # anisotropy after mean-centering

    print(f"\n=== {name} ===")
    print(f"  seqs probed:                {len(within)}")
    print(f"  within-seq token cos:       {within.mean():.4f}  (healthy << 1)")
    print(f"  cross-token cos (raw):      {cross_raw:.4f}  (anisotropy; >~0.5 is high)")
    print(f"  |cos(token, corpus mean)|:  {to_mean:.4f}  (~1 => one rogue direction dominates)")
    print(f"  cross-token cos (centered): {cross_centered:.4f}  (mean-centering fix; want << raw)")


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
    print("  - cross-token cos (raw) high (>~0.5) => encoder is anisotropic; the projection")
    print("    input is near-rank-1, so max-sim collapse sits on the data manifold.")
    print("  - if cross-token cos (centered) << raw => mean-centering the encoder features")
    print("    removes the anisotropy: subtract the corpus-mean token vector before project().")
    print("  - if centering barely helps => anisotropy is multi-directional; try whitening")
    print("    or a less anisotropic (earlier) encoder layer.")


if __name__ == "__main__":
    main()
