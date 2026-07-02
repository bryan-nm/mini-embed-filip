"""Corpus-mean token vectors for de-anisotropizing the projection input.

AMPLIFY-350M's last-layer token embeddings are dominated by one 'rogue' mean
direction (|cos(token, mean)| ~= 0.89; see src/diag_cache.py). Subtracting the
corpus mean before the projection head restores isotropy (cross-token cos
0.77 -> 0.00) and removes the FILIP max-sim collapse shortcut.

This streams the packed cache (no re-precompute) and writes
`{prefix}_mean.pt` into the cache dir. train_retrieval calls ensure_means() at
startup and installs the vectors as model buffers, so they save into the
checkpoint and downstream reconstruction/generation inherit the centering.

    python -m src.compute_means --cache-dir cache
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from config import default_cfg


def _mean_path(cache_dir: str, prefix: str) -> Path:
    return Path(cache_dir) / f"{prefix}_mean.pt"


def compute_mean(cache_dir: str, prefix: str, d: int, chunk: int = 1_000_000) -> torch.Tensor:
    """Streaming mean over all packed tokens of one modality. Returns [d] float32."""
    cache = Path(cache_dir)
    offsets = torch.load(cache / f"{prefix}_offsets.pt", map_location="cpu")
    total = int(offsets[-1].item())
    mm = np.memmap(cache / f"{prefix}_h.bin", dtype=np.uint16, mode="r", shape=(total, d))
    acc = torch.zeros(d, dtype=torch.float64)          # float64 accumulator: ~1e8 terms
    for s in range(0, total, chunk):
        e = min(s + chunk, total)
        block = torch.from_numpy(np.asarray(mm[s:e]).copy()).view(torch.bfloat16).float()
        acc += block.sum(dim=0).double()
    return (acc / max(total, 1)).float()


def load_means(cache_dir: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load saved (mean_p, mean_t) if present, else (None, None)."""
    pp, tp = _mean_path(cache_dir, "protein"), _mean_path(cache_dir, "text")
    mean_p = torch.load(pp, map_location="cpu") if pp.exists() else None
    mean_t = torch.load(tp, map_location="cpu") if tp.exists() else None
    return mean_p, mean_t


def ensure_means(cache_dir: str, protein_d: int, text_d: int,
                 verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached means, computing+saving any that are missing. Call on one rank."""
    mean_p, mean_t = load_means(cache_dir)
    if mean_p is None:
        mean_p = compute_mean(cache_dir, "protein", protein_d)
        torch.save(mean_p, _mean_path(cache_dir, "protein"))
        if verbose:
            print(f"[means] computed protein mean; |mean|={mean_p.norm():.3f}")
    if mean_t is None:
        mean_t = compute_mean(cache_dir, "text", text_d)
        torch.save(mean_t, _mean_path(cache_dir, "text"))
        if verbose:
            print(f"[means] computed text mean; |mean|={mean_t.norm():.3f}")
    return mean_p, mean_t


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--force", action="store_true", help="recompute even if files exist")
    args = ap.parse_args()
    if args.force:
        for prefix in ("protein", "text"):
            _mean_path(args.cache_dir, prefix).unlink(missing_ok=True)
    mean_p, mean_t = ensure_means(
        args.cache_dir, cfg.model.protein_hidden, cfg.model.text_hidden, verbose=True)
    print(f"[means] protein: shape={tuple(mean_p.shape)} |mean|={mean_p.norm():.3f}")
    print(f"[means] text:    shape={tuple(mean_t.shape)} |mean|={mean_t.norm():.3f}")


if __name__ == "__main__":
    main()
