"""Best-of-N candidate selection by contrastive round-trip margin (Feature 3).

For one source, score each generated candidate by re-embedding it and comparing
(FILIP) to the source. Plain pos-score is a weak selector — bad generations still
score high (the low-margin failure in EVAL §4) — so the margin criterion subtracts
the candidate's best score against a reference panel of *other* sources, rewarding
candidates that are specifically compatible with their own source:

    margin(cand) = FILIP(cand, src) - max_j FILIP(cand, panel_j)

Pairs naturally with the CVAE: sample N latents -> N structurally distinct
candidates -> select the best margin. Works standalone too (temperature diversity).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.losses import filip_score_matrix_chunked


def pad_stack(seqs: List[torch.Tensor], dim: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of [l_i, dim] per-token tensors -> ([N, Lmax, dim], bool mask [N, Lmax])."""
    n = len(seqs)
    lmax = max(max((t.size(0) for t in seqs), default=1), 1)
    out = torch.zeros(n, lmax, dim, device=device)
    mask = torch.zeros(n, lmax, dtype=torch.bool, device=device)
    for i, t in enumerate(seqs):
        l = t.size(0)
        if l:
            out[i, :l] = t.to(device)
            mask[i, :l] = True
    return out, mask


def select_best_of_n(
    z_cands: torch.Tensor, mask_cands: torch.Tensor,     # [N, Lc, D], [N, Lc]
    z_src: torch.Tensor, mask_src: torch.Tensor,         # [1, Ls, D], [1, Ls]
    z_panel: Optional[torch.Tensor] = None,              # [M, Lp, D]
    mask_panel: Optional[torch.Tensor] = None,           # [M, Lp]
    *,
    mode: str = "margin",
    chunk_rows: int = 8,
) -> Tuple[int, torch.Tensor]:
    """Return (best_idx, scores[N]).

    mode="margin": pos - max over the reference panel (falls back to pos when no
    panel is supplied). mode="pos": just the positive-pair FILIP score.
    """
    pos = filip_score_matrix_chunked(
        z_cands, z_src, mask_cands, mask_src, chunk_rows=chunk_rows
    ).squeeze(1)                                          # [N]
    if mode == "margin" and z_panel is not None and z_panel.size(0) > 0:
        negs = filip_score_matrix_chunked(
            z_cands, z_panel, mask_cands, mask_panel, chunk_rows=chunk_rows
        )                                                # [N, M]
        scores = pos - negs.max(dim=1).values
    else:
        scores = pos
    return int(scores.argmax().item()), scores
