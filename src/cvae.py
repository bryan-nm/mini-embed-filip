"""Generation-side conditional VAE heads (Feature 1).

A small latent `w` captures the one-to-many residual p(target | source) that the
deterministic cross-attention memory expand(project(h)) cannot represent. At
training time the posterior q(w | source, target) sees both pooled 32-d retrieval
embeddings (both produced by frozen heads); at inference only the learned
conditional prior p(w | source) is available, so we sample w from it. w is decoded
to `n_latent_tokens` extra cross-attention memory tokens that are concatenated onto
the per-token expansion memory — the decoder internals are untouched (see
`set_cross_memory` in decoder_adapters.py).

Trained during the generation phase alongside the cross-attention adapters + LoRA.
Saved under `cvae_state` in the generation checkpoint; absent => the decoder runs
without a latent (backward compatible).

Sampling N independent w gives N structurally distinct candidates, which feed the
best-of-N contrastive selection (best_of_n.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CVAECfg:
    d_pool: int = 64          # pooled retrieval-embedding dim (== model embed_dim)
    d_w: int = 32             # latent dim
    n_latent_tokens: int = 4  # memory tokens produced from w
    hidden: int = 256         # prior/posterior MLP width
    mem_dim: int = 768        # cross-attn memory dim (encoder hidden of the source side)
    beta_max: float = 0.1
    free_bits: float = 0.5
    kl_warmup_frac: float = 0.3


def beta_at(step: int, total_steps: int, cfg: CVAECfg) -> float:
    """Linear KL warmup: 0 -> beta_max over the first kl_warmup_frac of training.

    Warming beta from zero lets the decoder first learn to use the latent before
    the KL pressure pulls the posterior toward the prior — the standard guard
    against posterior collapse with a strong pretrained decoder.
    """
    warm = max(int(cfg.kl_warmup_frac * total_steps), 1)
    if step >= warm:
        return cfg.beta_max
    return cfg.beta_max * step / warm


class _GaussianMLP(nn.Module):
    """Maps a conditioning vector to (mu, logvar) of a diagonal Gaussian."""

    def __init__(self, d_in: int, hidden: int, d_w: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * d_w),
        )

    def forward(self, c: torch.Tensor):
        mu, logvar = self.net(c).chunk(2, dim=-1)
        # Clamp logvar for numerical stability (exp(logvar) feeds the KL).
        logvar = logvar.clamp(-8.0, 8.0)
        return mu, logvar


class CVAEHeads(nn.Module):
    """Prior p(w|src), posterior q(w|src,tgt), and w -> memory-token decoder."""

    def __init__(self, cfg: CVAECfg):
        super().__init__()
        self.cfg = cfg
        self.prior = _GaussianMLP(cfg.d_pool, cfg.hidden, cfg.d_w)
        self.posterior = _GaussianMLP(2 * cfg.d_pool, cfg.hidden, cfg.d_w)
        self.to_memory = nn.Linear(cfg.d_w, cfg.n_latent_tokens * cfg.mem_dim)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def prior_params(self, z_src_pool: torch.Tensor):
        return self.prior(z_src_pool)

    def posterior_params(self, z_src_pool: torch.Tensor, z_tgt_pool: torch.Tensor):
        return self.posterior(torch.cat([z_src_pool, z_tgt_pool], dim=-1))

    def sample_prior(self, z_src_pool: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.prior_params(z_src_pool)
        return self.reparam(mu, logvar)

    def latent_tokens(self, w: torch.Tensor) -> torch.Tensor:
        """w [B, d_w] -> [B, n_latent_tokens, mem_dim] (extra cross-attn memory)."""
        B = w.size(0)
        return self.to_memory(w).view(B, self.cfg.n_latent_tokens, self.cfg.mem_dim)

    def kl(self, qmu, qlv, pmu, plv) -> torch.Tensor:
        """KL(q || p) for diagonal Gaussians, with free bits.

        Per-dim KL is floored at `free_bits` before summing so the latent keeps a
        minimum capacity (dims at the floor contribute a constant => no gradient
        pressure toward further collapse). Returns the batch-mean of the per-sample
        summed KL.
        """
        kl_dim = 0.5 * (
            plv - qlv + (torch.exp(qlv) + (qmu - pmu) ** 2) / torch.exp(plv) - 1.0
        )
        if self.cfg.free_bits > 0:
            kl_dim = kl_dim.clamp_min(self.cfg.free_bits)
        return kl_dim.sum(dim=-1).mean()


def build_cvae(gen_cfg, mem_dim: int, embed_dim: int) -> CVAEHeads:
    """Construct CVAE heads from a GenerationCfg + the source-side memory dim."""
    cfg = CVAECfg(
        d_pool=embed_dim,
        d_w=gen_cfg.cvae_d_w,
        n_latent_tokens=gen_cfg.cvae_n_latent_tokens,
        hidden=gen_cfg.cvae_hidden,
        mem_dim=mem_dim,
        beta_max=gen_cfg.cvae_beta_max,
        free_bits=gen_cfg.cvae_free_bits,
        kl_warmup_frac=gen_cfg.cvae_kl_warmup_frac,
    )
    return CVAEHeads(cfg)


def load_cvae(ckpt: dict, embed_dim: int, device) -> Optional[CVAEHeads]:
    """Reconstruct frozen CVAE heads from a generation checkpoint, or None.

    Returns None when the checkpoint predates the CVAE (no `cvae_state`), so the
    inference paths transparently fall back to deterministic conditioning.
    """
    if "cvae_state" not in ckpt:
        return None
    c = ckpt.get("cvae_cfg", {})
    cfg = CVAECfg(
        d_pool=embed_dim,
        d_w=c.get("d_w", 32),
        n_latent_tokens=c.get("n_latent_tokens", 4),
        hidden=c.get("hidden", 256),
        mem_dim=c["mem_dim"],
    )
    heads = CVAEHeads(cfg)
    heads.load_state_dict(ckpt["cvae_state"])
    heads.eval().to(device)
    for p in heads.parameters():
        p.requires_grad_(False)
    return heads
