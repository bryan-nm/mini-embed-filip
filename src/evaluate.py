"""Retrieval evaluation: FILIP-based R@K + modality-gap diagnostics.

Operates on per-token z_p / z_t with their valid-token masks. All metrics
use the FILIP score (mean of max-sim, both directions) as the similarity
measure, in place of the pooled cosine used by the previous mini-embed code.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from src.losses import filip_score_matrix_chunked, token_uniformity_loss


@torch.no_grad()
def retrieval_recall(
    z_p: torch.Tensor, z_t: torch.Tensor,
    mask_p: torch.Tensor, mask_t: torch.Tensor,
    ks=(1, 5, 10), chunk_rows: int = 16,
    groups: torch.Tensor = None,
) -> Dict[str, float]:
    """Symmetric retrieval R@K over the FILIP score matrix.

    `groups` (per-row accession id) makes every same-protein candidate a
    positive, so the augmented corpus's sibling captions in a val batch don't
    count as distractors. The reported rank is that of the best-scoring
    positive. Without `groups` this is the diagonal-positive recall.
    """
    sim = filip_score_matrix_chunked(z_p, z_t, mask_p, mask_t, chunk_rows)
    n = sim.size(0)
    target = torch.arange(n, device=sim.device)
    same = None if groups is None else (groups[:, None] == groups[None, :])

    out: Dict[str, float] = {}
    for direction, mat in (("p2t", sim), ("t2p", sim.t())):
        if same is None:
            ranks = (mat >= mat.gather(1, target.unsqueeze(1))).sum(dim=1)
        else:
            neg_inf = torch.finfo(mat.dtype).min
            best_pos = mat.masked_fill(~same, neg_inf).max(dim=1).values
            ranks = ((mat > best_pos[:, None]) & ~same).sum(dim=1) + 1
        for k in ks:
            out[f"R@{k}_{direction}"] = (ranks <= k).float().mean().item()
    for k in ks:
        out[f"R@{k}"] = 0.5 * (out[f"R@{k}_p2t"] + out[f"R@{k}_t2p"])
    return out


@torch.no_grad()
def modality_gap_metrics(
    z_p: torch.Tensor, z_t: torch.Tensor,
    mask_p: torch.Tensor, mask_t: torch.Tensor,
) -> Dict[str, float]:
    """Per-token modality-gap diagnostics.

    Token centroids: mean over all valid tokens in the val split.
    Reports gap on token centroids + within/across-modality token cosines.
    """
    p_flat = z_p[mask_p]                                   # [Np, D]
    t_flat = z_t[mask_t]                                   # [Nt, D]
    if p_flat.size(0) < 2 or t_flat.size(0) < 2:
        return {}

    mu_p = p_flat.mean(dim=0)
    mu_t = t_flat.mean(dim=0)
    gap_l2 = (mu_p - mu_t).norm(p=2).item()

    # Subsample for the O(N^2) metrics to keep eval cheap.
    def _subsample(x, n=2048):
        if x.size(0) <= n:
            return x
        idx = torch.randperm(x.size(0), device=x.device)[:n]
        return x[idx]

    p_sub = _subsample(p_flat)
    t_sub = _subsample(t_flat)
    cross = p_sub @ t_sub.t()                              # [n_p, n_t]
    intra_p = p_sub @ p_sub.t()
    intra_t = t_sub @ t_sub.t()
    np_ = p_sub.size(0); nt = t_sub.size(0)
    eye_p = torch.eye(np_, device=cross.device, dtype=torch.bool)
    eye_t = torch.eye(nt, device=cross.device, dtype=torch.bool)

    return {
        "gap_l2": gap_l2,
        "mean_cross_token_cos": cross.mean().item(),
        "mean_intra_p_token_cos": intra_p.masked_fill(eye_p, 0).sum().item()
            / max(np_ * (np_ - 1), 1),
        "mean_intra_t_token_cos": intra_t.masked_fill(eye_t, 0).sum().item()
            / max(nt * (nt - 1), 1),
        "uniformity_p_tokens": token_uniformity_loss(
            p_sub.unsqueeze(0), torch.ones(1, np_, dtype=torch.bool, device=p_sub.device)
        ).item(),
        "uniformity_t_tokens": token_uniformity_loss(
            t_sub.unsqueeze(0), torch.ones(1, nt, dtype=torch.bool, device=t_sub.device)
        ).item(),
    }


@torch.no_grad()
def evaluate_split(model, loader, device: torch.device, encoders=None,
                   max_protein_tokens: int = 1024, max_text_tokens: int = 1024,
                   filip_chunk_rows: int = 16,
                   row_group_ids: torch.Tensor = None) -> Dict[str, float]:
    """Run a full eval pass. Works for both cached and live modes.

    `row_group_ids` (global row -> accession id) enables accession-grouped
    recall, so multiple captions of one protein in the val set are scored as
    positives rather than distractors.
    """
    model.eval()
    z_ps, z_ts, mps, mts = [], [], [], []
    idxs = []

    for batch in loader:
        if "idx" in batch:
            idxs.append(batch["idx"])
        if "h_p" in batch:
            h_p = batch["h_p"].to(device).float()
            h_t = batch["h_t"].to(device).float()
            mask_p = batch["mask_p"].to(device)
            mask_t = batch["mask_t"].to(device)
        else:
            if encoders is None:
                raise RuntimeError("encoders required for live eval")
            from src.encoders import encode_protein_batch, encode_text_batch
            text_model, text_tok, prot_model, prot_tok = encoders
            h_t, mask_t = encode_text_batch(
                text_model, text_tok, batch["text"], device, max_text_tokens
            )
            h_p, mask_p = encode_protein_batch(
                prot_model, prot_tok, batch["protein"], device, max_protein_tokens
            )
        z_p, z_t = model.project(h_p, h_t)
        # Pad to max in batch shouldn't be needed within one batch.
        z_ps.append(z_p.float().cpu()); z_ts.append(z_t.float().cpu())
        mps.append(mask_p.cpu()); mts.append(mask_t.cpu())

    # Concat with padding to global max length
    def _pad_cat(zs, ms):
        Lmax = max(z.size(1) for z in zs)
        D = zs[0].size(-1)
        chunks_z, chunks_m = [], []
        for z, m in zip(zs, ms):
            B, L, _ = z.shape
            if L < Lmax:
                pad_z = torch.zeros(B, Lmax - L, D, dtype=z.dtype)
                pad_m = torch.zeros(B, Lmax - L, dtype=torch.bool)
                z = torch.cat([z, pad_z], dim=1)
                m = torch.cat([m, pad_m], dim=1)
            chunks_z.append(z); chunks_m.append(m)
        return torch.cat(chunks_z, dim=0), torch.cat(chunks_m, dim=0)

    z_p, mask_p = _pad_cat(z_ps, mps)
    z_t, mask_t = _pad_cat(z_ts, mts)

    groups = None
    if row_group_ids is not None and idxs:
        sel = torch.cat(idxs)                       # global row indices, eval order
        groups = row_group_ids[sel].to(z_p.device)  # aligned with z_p / z_t rows

    metrics: Dict[str, float] = {}
    metrics.update(retrieval_recall(z_p, z_t, mask_p, mask_t,
                                    chunk_rows=filip_chunk_rows, groups=groups))
    metrics.update(modality_gap_metrics(z_p, z_t, mask_p, mask_t))
    return metrics
