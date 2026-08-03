"""Retrieval evaluation: FILIP-based R@K + modality-gap diagnostics.

Operates on per-token z_p / z_t with their valid-token masks. All metrics
use the FILIP score (mean of max-sim, both directions) as the similarity
measure, in place of the pooled cosine used by the previous mini-embed code.
"""
from __future__ import annotations

from typing import Dict

import torch

from src.losses import filip_score_matrix_chunked, token_uniformity_loss


def average_precision(scores: torch.Tensor, relevant: torch.Tensor) -> torch.Tensor:
    """Per-query Average Precision, matching ProTrek's definition.

    scores   [Q, C] similarity of each query row to every candidate column.
    relevant [Q, C] bool; True where a candidate is a correct (multi-positive) hit.

    Ranks candidates by descending score, then for the k-th relevant hit (found at
    1-based rank r) takes precision k/r and averages over all relevant hits — i.e.
    ProTrek's `np.mean([(i+1)/rank for i, rank in enumerate(ranks)])` (see
    protrek_trimodal_model.py `_protein2text`/`_text2protein`), computed here over
    the whole candidate pool exactly as ProTrek searches the full index. Returns
    [Q]; the mean over queries is mAP. Every query has >=1 relevant (its own
    positive), so the normalizer is never zero.
    """
    C = scores.size(1)
    # stable sort -> deterministic tie handling run-to-run (FILIP scores rarely
    # tie exactly except among identical captions, which are all mutually relevant).
    order = scores.argsort(dim=1, descending=True, stable=True)  # [Q, C] cand idx, best first
    rel_sorted = torch.gather(relevant, 1, order).to(scores.dtype)   # [Q, C] in rank order
    ranks = torch.arange(1, C + 1, device=scores.device, dtype=scores.dtype)
    cum_rel = rel_sorted.cumsum(dim=1)                          # relevant-so-far at each rank
    precision_at_hit = cum_rel / ranks[None, :]                 # k / r at every position
    num_rel = rel_sorted.sum(dim=1).clamp_min(1.0)              # R (>=1 via self-positive)
    return (precision_at_hit * rel_sorted).sum(dim=1) / num_rel


@torch.no_grad()
def retrieval_recall(
    z_p: torch.Tensor, z_t: torch.Tensor,
    mask_p: torch.Tensor, mask_t: torch.Tensor,
    ks=(1, 5, 10), chunk_rows: int = 16,
    groups: torch.Tensor = None,
    text_groups: torch.Tensor = None,
) -> Dict[str, float]:
    """Symmetric retrieval R@K + mAP over the FILIP score matrix.

    A candidate is a positive (not a distractor) when it shares the query's
    protein (`groups`, accession id) OR its caption (`text_groups`, caption id).
    Both relations are checked *directly*, mirroring the training-side
    `mask_false_negatives`. This matters because the corpus has byte-identical
    captions across different proteins: without caption grouping, an identical
    caption ties with the positive in p2t and the identical *query* has
    contradictory targets in t2p, both of which deflate R@K (and cap it below 1
    for a caption shared by m proteins). The reported rank is that of the
    best-scoring positive. With neither group supplied this is diagonal recall.

    mAP uses ProTrek's Average-Precision definition (see `average_precision`)
    with the same multi-positive relevance set, so `mAP_p2t` / `mAP_t2p` are
    directly comparable to ProTrek's `protein2text` / `text2protein` `..._map`
    (our p2t = query protein → texts = ProTrek protein2text). The absolute value
    still depends on the candidate-pool composition, so a strict head-to-head
    needs both models scored over the same evaluation set.
    """
    sim = filip_score_matrix_chunked(z_p, z_t, mask_p, mask_t, chunk_rows)
    n = sim.size(0)
    target = torch.arange(n, device=sim.device)
    same = None
    if groups is not None:
        same = groups[:, None] == groups[None, :]
    if text_groups is not None:
        same_text = text_groups[:, None] == text_groups[None, :]
        same = same_text if same is None else (same | same_text)
    # Relevance set for AP: the multi-positive group, or the query's own positive
    # (diagonal) when no groups are supplied — then AP reduces to reciprocal rank.
    relevant = same if same is not None else torch.eye(n, dtype=torch.bool, device=sim.device)

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
        out[f"mAP_{direction}"] = average_precision(mat, relevant).mean().item()
    for k in ks:
        out[f"R@{k}"] = 0.5 * (out[f"R@{k}_p2t"] + out[f"R@{k}_t2p"])
    out["mAP"] = 0.5 * (out["mAP_p2t"] + out["mAP_t2p"])
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

    # Mean within-positive-pair token cosine: the complement to
    # mean_cross_token_cos (which is the across-random-pairs background). Row k of
    # z_p / z_t is a correct pair. mean_{i,j}<z_p[i], z_t[j]> over a pair equals
    # <mean_i z_p[i], mean_j z_t[j]>, so this is just the per-pair dot of the
    # mean-pooled token vectors — no per-pair matmul.
    vp = mask_p.sum(dim=1, keepdim=True).clamp_min(1).to(z_p.dtype)
    vt = mask_t.sum(dim=1, keepdim=True).clamp_min(1).to(z_t.dtype)
    mean_p_pair = (z_p * mask_p.unsqueeze(-1)).sum(dim=1) / vp     # [N, D]
    mean_t_pair = (z_t * mask_t.unsqueeze(-1)).sum(dim=1) / vt     # [N, D]
    mean_pos_token_cos = (mean_p_pair * mean_t_pair).sum(dim=-1).mean().item()

    return {
        "gap_l2": gap_l2,
        "mean_cross_token_cos": cross.mean().item(),
        "mean_pos_token_cos": mean_pos_token_cos,
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
                   filip_chunk_rows: int = 8,
                   row_group_ids: torch.Tensor = None,
                   row_text_ids: torch.Tensor = None) -> Dict[str, float]:
    """Run a full eval pass. Works for both cached and live modes.

    `row_group_ids` (global row -> accession id) and `row_text_ids` (global row
    -> caption id) enable grouped recall: candidates sharing the query's protein
    or its caption are scored as positives rather than distractors, so sibling
    captions and byte-identical captions in the val set don't deflate R@K.

    Scoring stays on `device`: the projected embeddings are tiny (val_subset x
    L x 32), so the FILIP recall + gap metrics run on the GPU instead of being
    copied to CPU — the CPU einsum over the O(N^2 . L_p . L_t) score tensor was
    the eval bottleneck. `filip_chunk_rows` bounds the per-chunk transient so the
    on-device score matrix doesn't blow the tile (no autograd graph here, so only
    the transient matters).
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
        # Keep on device (tensors are small); scoring runs on the GPU.
        z_ps.append(z_p.float()); z_ts.append(z_t.float())
        mps.append(mask_p); mts.append(mask_t)

    # Concat with padding to global max length (pads allocated on z's device).
    def _pad_cat(zs, ms):
        Lmax = max(z.size(1) for z in zs)
        D = zs[0].size(-1)
        chunks_z, chunks_m = [], []
        for z, m in zip(zs, ms):
            B, L, _ = z.shape
            if L < Lmax:
                pad_z = torch.zeros(B, Lmax - L, D, dtype=z.dtype, device=z.device)
                pad_m = torch.zeros(B, Lmax - L, dtype=torch.bool, device=m.device)
                z = torch.cat([z, pad_z], dim=1)
                m = torch.cat([m, pad_m], dim=1)
            chunks_z.append(z); chunks_m.append(m)
        return torch.cat(chunks_z, dim=0), torch.cat(chunks_m, dim=0)

    z_p, mask_p = _pad_cat(z_ps, mps)
    z_t, mask_t = _pad_cat(z_ts, mts)

    groups = text_groups = None
    if idxs and (row_group_ids is not None or row_text_ids is not None):
        sel = torch.cat(idxs)                       # global row indices, eval order
        if row_group_ids is not None:
            groups = row_group_ids[sel].to(z_p.device)      # aligned with z_p/z_t rows
        if row_text_ids is not None:
            text_groups = row_text_ids[sel].to(z_p.device)  # aligned with z_p/z_t rows

    metrics: Dict[str, float] = {}
    metrics.update(retrieval_recall(z_p, z_t, mask_p, mask_t,
                                    chunk_rows=filip_chunk_rows,
                                    groups=groups, text_groups=text_groups))
    metrics.update(modality_gap_metrics(z_p, z_t, mask_p, mask_t))
    return metrics
