"""FILIP late-interaction losses + per-token uniformity + reconstruction.

All inputs are assumed L2-normalized along the last dim where applicable
(projection-head outputs are; expansion-head outputs are not, by design).

The four losses we compose:

  filip_score_matrix    [B_p, B_t] FILIP similarity matrix from per-token z's
                        (mean of max-sim in each direction).
  positive_pair_score   Helper: extract the diagonal of filip_score_matrix as
                        a scalar (mean of positive-pair scores).
  token_uniformity      Wang & Isola (2020) uniformity over all valid tokens
                        within a modality in the batch.
  reconstruction_loss   Per-token MSE between expand(project(h)) and h, masked
                        to valid positions.

phase_r1_loss and phase_r2_loss compose them as described in
PLAN_late_interaction.md sections 3c.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FILIP score (per-pair scalar similarity in [-1, 1])
# ---------------------------------------------------------------------------
def filip_score_matrix(
    z_p: torch.Tensor,        # [B_p, L_p, D] L2-normalized per token
    z_t: torch.Tensor,        # [B_t, L_t, D] L2-normalized per token
    mask_p: torch.Tensor,     # [B_p, L_p] bool; True = valid token
    mask_t: torch.Tensor,     # [B_t, L_t] bool; True = valid token
    neg_inf: float = -1e4,
) -> torch.Tensor:
    """Returns [B_p, B_t] FILIP score: 0.5 * (p2t_max-sim + t2p_max-sim)."""
    # All-pairs token similarities. einsum keeps the math obvious.
    sim = torch.einsum("bld,cmd->bclm", z_p, z_t)  # [B_p, B_t, L_p, L_t]

    mp = mask_p[:, None, :, None]                  # [B_p, 1, L_p, 1]
    mt = mask_t[None, :, None, :]                  # [1, B_t, 1, L_t]
    # Mask invalid positions to a value that never wins a max.
    sim = sim.masked_fill(~mp, neg_inf)
    sim = sim.masked_fill(~mt, neg_inf)

    # p -> t: for each protein position, best matching text token
    max_per_p = sim.max(dim=3).values              # [B_p, B_t, L_p]
    # Zero out invalid p positions so they don't contribute to the mean.
    max_per_p = max_per_p.masked_fill(~mp.squeeze(3), 0.0)
    n_valid_p = mask_p.sum(dim=1).clamp_min(1).to(max_per_p.dtype)  # [B_p]
    score_p2t = max_per_p.sum(dim=2) / n_valid_p[:, None]           # [B_p, B_t]

    # t -> p: for each text token, best matching protein position
    max_per_t = sim.max(dim=2).values              # [B_p, B_t, L_t]
    max_per_t = max_per_t.masked_fill(~mt.squeeze(2), 0.0)
    n_valid_t = mask_t.sum(dim=1).clamp_min(1).to(max_per_t.dtype)  # [B_t]
    score_t2p = max_per_t.sum(dim=2) / n_valid_t[None, :]           # [B_p, B_t]

    return 0.5 * (score_p2t + score_t2p)


def filip_score_matrix_chunked(
    z_p: torch.Tensor,
    z_t: torch.Tensor,
    mask_p: torch.Tensor,
    mask_t: torch.Tensor,
    chunk_rows: int = 16,
) -> torch.Tensor:
    """Chunked version: avoids materializing the full [B, B, L_p, L_t] tensor.

    Iterates over rows of the protein axis. Memory ~ chunk_rows * B_t * L_p * L_t.
    """
    B_p = z_p.size(0)
    rows = []
    for s in range(0, B_p, chunk_rows):
        e = min(s + chunk_rows, B_p)
        rows.append(
            filip_score_matrix(
                z_p[s:e], z_t, mask_p[s:e], mask_t
            )
        )
    return torch.cat(rows, dim=0)


def positive_pair_score(filip_matrix: torch.Tensor) -> torch.Tensor:
    """Mean of the diagonal of a [B, B] FILIP score matrix."""
    return filip_matrix.diagonal().mean()


# Finite "−inf" for masked logits: well below scale*FILIP (|logit| <= ~100), and
# safe across fp32/bf16/autocast (true -inf can poison reductions).
_NEG_LOGIT = -1e4


def mask_false_negatives(
    logits: torch.Tensor,            # [R, C] scaled similarity logits
    target: torch.Tensor,           # [R] index of each row's positive column
    groups_row: Optional[torch.Tensor] = None,   # [R] accession id per row
    groups_col: Optional[torch.Tensor] = None,   # [C] accession id per column
) -> torch.Tensor:
    """Mask same-protein non-target columns out of the InfoNCE denominator.

    The augmented corpus has ~8.87 captions per protein, so a contrastive batch
    can contain several captions of the same protein. Without masking, those
    siblings act as false negatives — the loss would push a protein away from a
    valid caption. We set every column sharing a row's accession (except that
    row's designated positive) to a large negative, removing it from softmax.
    No-op when group ids are not supplied.
    """
    if groups_row is None or groups_col is None:
        return logits
    same = groups_row[:, None] == groups_col[None, :]           # [R, C]
    is_target = torch.zeros_like(same)
    is_target[torch.arange(logits.size(0), device=logits.device), target] = True
    return logits.masked_fill(same & ~is_target, _NEG_LOGIT)


# ---------------------------------------------------------------------------
# Token-level uniformity (Wang & Isola 2020)
# ---------------------------------------------------------------------------
def token_uniformity_loss(
    z: torch.Tensor,          # [B, L, D] L2-normalized per token
    mask: torch.Tensor,       # [B, L] bool
    t: float = 2.0,
    max_tokens: int = 4096,   # cap to control O(N^2) cost
) -> torch.Tensor:
    """Wang & Isola uniformity over all valid tokens in the batch.

    Flattens (B, L) -> N valid tokens, optionally subsamples to max_tokens,
    then computes log E_{i != j} exp(-t * ||z_i - z_j||^2).
    """
    flat = z[mask]                                  # [N, D]
    n = flat.size(0)
    if n < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    if n > max_tokens:
        idx = torch.randperm(n, device=z.device)[:max_tokens]
        flat = flat[idx]
        n = max_tokens
    sim = flat @ flat.t()                           # [n, n]
    sq_dists = 2.0 - 2.0 * sim
    off_diag = ~torch.eye(n, device=z.device, dtype=torch.bool)
    return torch.logsumexp(-t * sq_dists[off_diag], dim=0) - math.log(n * (n - 1))


# ---------------------------------------------------------------------------
# Reconstruction (per-token autoencoder loss)
# ---------------------------------------------------------------------------
def reconstruction_loss(
    h_hat: torch.Tensor,      # [B, L, d_encoder]
    h: torch.Tensor,          # [B, L, d_encoder]
    mask: torch.Tensor,       # [B, L]
) -> torch.Tensor:
    """Masked per-token MSE between reconstructed and true encoder hidden states."""
    diff = (h_hat - h) ** 2                         # [B, L, D]
    diff = diff.mean(dim=-1)                        # [B, L]
    valid = mask.to(diff.dtype)
    return (diff * valid).sum() / valid.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Phase-R1 / Phase-R2 composites
# ---------------------------------------------------------------------------
def phase_r1_loss(
    out: dict,
    h_p: torch.Tensor,
    h_t: torch.Tensor,
    mask_p: torch.Tensor,
    mask_t: torch.Tensor,
    *,
    uniformity_weight: float,
    uniformity_t: float,
    recon_weight: float,
) -> dict:
    """Phase R1 warmup: maximize positive-pair FILIP + token uniformity + recon.

    Note: positive-pair FILIP is computed without negatives; for a B-sized batch
    we still need the [B, B] matrix only if we want the negative direction. To
    keep R1 cheap we compute paired scores only (B FILIP computations, no all-
    pairs matrix). We minimize -filip_pos.
    """
    z_p, z_t = out["z_p"], out["z_t"]
    h_p_hat, h_t_hat = out["h_p_hat"], out["h_t_hat"]

    # Positive-pair FILIP (no negatives): single-pair computation per row.
    # Easiest implementation: use the matrix builder with B_p == B_t and pull
    # the diagonal. This is wasteful (computes B*B pairs to get B), but at
    # phase-1 batch sizes it's cheap, and shares code with phase 2.
    filip_pos_only = filip_score_matrix(z_p, z_t, mask_p, mask_t).diagonal().mean()
    l_align = 1.0 - filip_pos_only

    l_unif = 0.5 * (
        token_uniformity_loss(z_p, mask_p, uniformity_t)
        + token_uniformity_loss(z_t, mask_t, uniformity_t)
    )

    l_recon = 0.5 * (
        reconstruction_loss(h_p_hat, h_p, mask_p)
        + reconstruction_loss(h_t_hat, h_t, mask_t)
    )

    total = l_align + uniformity_weight * l_unif + recon_weight * l_recon
    return {
        "loss": total,
        "align": l_align,
        "unif": l_unif,
        "recon": l_recon,
        "nce": torch.zeros((), device=z_p.device),
        "acc": torch.zeros((), device=z_p.device),
        "filip_pos": filip_pos_only,
    }


def phase_r2_loss_grouped(
    out: dict,
    h_p: torch.Tensor,
    h_t: torch.Tensor,
    mask_p: torch.Tensor,
    mask_t: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    z_p_all: torch.Tensor,        # [G, L_p, D] gathered across the subgroup
    z_t_all: torch.Tensor,        # [G, L_t, D]
    mask_p_all: torch.Tensor,     # [G, L_p]
    mask_t_all: torch.Tensor,     # [G, L_t]
    local_offset: int,            # row index of this rank's first sample within [0, G)
    align_aux_weight: float,
    recon_weight: float,
    chunk_rows: int = 0,
    groups: Optional[torch.Tensor] = None,       # [B] accession id per local anchor
    groups_all: Optional[torch.Tensor] = None,   # [G] accession id per gathered column
) -> dict:
    """Distributed Phase-R2 InfoNCE over a bounded subgroup of negatives.

    Each rank scores its *local* anchors (B rows) against the *gathered* group
    columns (G = group_size * B). Targets are the local anchors' positions
    inside the gathered batch (`local_offset .. local_offset+B`). Only the
    local slice of `z_*_all` carries gradient (see `dist.grouped_all_gather`),
    so this is the standard gather-with-grad contrastive loss — no double count.

    When `groups`/`groups_all` are supplied, columns sharing an anchor's protein
    (other than its own positive) are masked out of the denominator, so the
    corpus's multiple captions per protein don't act as false negatives.
    """
    z_p, z_t = out["z_p"], out["z_t"]
    h_p_hat, h_t_hat = out["h_p_hat"], out["h_t_hat"]
    B = z_p.size(0)
    scale = logit_scale.exp()

    builder = (lambda a, b, ma, mb: filip_score_matrix_chunked(a, b, ma, mb, chunk_rows)) \
        if chunk_rows > 0 else filip_score_matrix

    # p->t: [B_local, G] ; t->p: [B_local, G] (built as [G, B_local] then transposed)
    mat_p2t = builder(z_p, z_t_all, mask_p, mask_t_all)            # [B, G]
    mat_t2p = builder(z_p_all, z_t, mask_p_all, mask_t).t()        # [B, G]

    target = torch.arange(B, device=z_p.device) + local_offset
    # Both directions share the same row/column accession layout.
    logits_pt = mask_false_negatives(scale * mat_p2t, target, groups, groups_all)
    logits_tp = mask_false_negatives(scale * mat_t2p, target, groups, groups_all)
    loss_pt = F.cross_entropy(logits_pt, target)
    loss_tp = F.cross_entropy(logits_tp, target)
    l_nce = 0.5 * (loss_pt + loss_tp)

    with torch.no_grad():
        acc_pt = (logits_pt.argmax(dim=-1) == target).float().mean()
        acc_tp = (logits_tp.argmax(dim=-1) == target).float().mean()
        acc = 0.5 * (acc_pt + acc_tp)
        # positive-pair score = the true-pair FILIP at the local offset (unmasked)
        filip_pos = mat_p2t.gather(1, target[:, None]).mean()

    l_align = 1.0 - filip_pos
    l_recon = 0.5 * (
        reconstruction_loss(h_p_hat, h_p, mask_p)
        + reconstruction_loss(h_t_hat, h_t, mask_t)
    )

    total = l_nce + align_aux_weight * l_align + recon_weight * l_recon
    return {
        "loss": total,
        "align": l_align,
        "unif": torch.zeros((), device=z_p.device),
        "recon": l_recon,
        "nce": l_nce,
        "acc": acc,
        "filip_pos": filip_pos,
    }


def phase_r2_loss(
    out: dict,
    h_p: torch.Tensor,
    h_t: torch.Tensor,
    mask_p: torch.Tensor,
    mask_t: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    align_aux_weight: float,
    recon_weight: float,
    chunk_rows: int = 0,
    groups: Optional[torch.Tensor] = None,       # [B] accession id per pair
) -> dict:
    """Phase R2: FILIP-based symmetric InfoNCE + small align aux + recon.

    `groups` (per-pair accession ids) masks same-protein off-diagonal entries as
    false negatives; no-op when omitted.
    """
    z_p, z_t = out["z_p"], out["z_t"]
    h_p_hat, h_t_hat = out["h_p_hat"], out["h_t_hat"]

    if chunk_rows > 0:
        filip_mat = filip_score_matrix_chunked(z_p, z_t, mask_p, mask_t, chunk_rows)
    else:
        filip_mat = filip_score_matrix(z_p, z_t, mask_p, mask_t)

    scale = logit_scale.exp()
    target = torch.arange(filip_mat.size(0), device=filip_mat.device)
    logits_pt = mask_false_negatives(scale * filip_mat, target, groups, groups)
    logits_tp = mask_false_negatives((scale * filip_mat).t(), target, groups, groups)
    loss_pt = F.cross_entropy(logits_pt, target)
    loss_tp = F.cross_entropy(logits_tp, target)
    l_nce = 0.5 * (loss_pt + loss_tp)

    with torch.no_grad():
        acc_pt = (logits_pt.argmax(dim=-1) == target).float().mean()
        acc_tp = (logits_tp.argmax(dim=-1) == target).float().mean()
        acc = 0.5 * (acc_pt + acc_tp)

    filip_pos = filip_mat.diagonal().mean()
    l_align = 1.0 - filip_pos

    l_recon = 0.5 * (
        reconstruction_loss(h_p_hat, h_p, mask_p)
        + reconstruction_loss(h_t_hat, h_t, mask_t)
    )

    total = l_nce + align_aux_weight * l_align + recon_weight * l_recon
    return {
        "loss": total,
        "align": l_align,
        "unif": torch.zeros((), device=z_p.device),
        "recon": l_recon,
        "nce": l_nce,
        "acc": acc,
        "filip_pos": filip_pos,
    }
