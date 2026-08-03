"""Per-token projection and expansion heads + the wrapper holding both pairs.

Both heads are position-wise (no token mixing inside the head). The encoder
already did the contextualization; the heads' job is to map between encoder
hidden space and the shared space, whose width is `ModelCfg.embed_dim`.

Projection head:
    d_in (768 or 960) -> d_hidden -> d_mid -> d_out (embed_dim)
    Linear, LayerNorm, GELU, Dropout. Output L2-normalized along the last dim.

Expansion head:
    embed_dim -> d_mid -> d_hidden -> d_in (768 or 960)
    Mirrored architecture; separate weights. Output NOT normalized.

No dimension is defaulted here: every caller passes the config's values, so a
sweep over embed_dim cannot silently pick up a stale constant.

The reconstruction loop expand(project(h)) ~= h ties them together via an
auxiliary MSE term during retrieval training; see losses.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """d_in -> d_hidden -> d_mid -> d_out, with LayerNorm + L2-normalized output.

    Applied position-wise to a [B, L, d_in] tensor.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_mid: int,
        d_out: int,
        dropout: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_mid)
        self.norm2 = nn.LayerNorm(d_mid)
        self.fc3 = nn.Linear(d_mid, d_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        # Corpus-mean token vector, subtracted at the input to de-anisotropize the
        # encoder features (AMPLIFY-350M tokens are ~0.89 aligned with one rogue
        # mean direction; that dominates every dot product and collapses FILIP
        # max-sim). Set via MiniEmbedFilip.set_feature_means; zero => no-op. A
        # buffer, so it saves into the checkpoint and every caller of this head
        # (retrieval, generation memory front-end) centers identically.
        self.register_buffer("mean_in", torch.zeros(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean_in
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=-1)


class ExpansionHead(nn.Module):
    """d_in (embed_dim) -> d_mid -> d_hidden -> d_out, mirroring ProjectionHead.

    Separate weights from the corresponding projection (not weight-tied).
    Output not normalized; the consumer (decoder cross-attention) operates in
    encoder hidden space, not on a sphere.
    """

    def __init__(
        self,
        d_in: int,
        d_mid: int,
        d_hidden: int,
        d_out: int,
        dropout: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_mid)
        self.norm1 = nn.LayerNorm(d_mid)
        self.fc2 = nn.Linear(d_mid, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.fc3(x)
        return x


class MiniEmbedFilip(nn.Module):
    """Full trainable retrieval model: projection + expansion heads + temperature."""

    def __init__(
        self,
        text_hidden: int,
        protein_hidden: int,
        proj_d_hidden: int,
        proj_d_mid: int,
        embed_dim: int,
        proj_dropout: float,
        expand_d_mid: int,
        expand_d_hidden: int,
        expand_dropout: float,
        init_temperature: float,
        max_temperature: float,
    ):
        super().__init__()
        self.text_proj = ProjectionHead(
            text_hidden, proj_d_hidden, proj_d_mid, embed_dim, proj_dropout
        )
        self.protein_proj = ProjectionHead(
            protein_hidden, proj_d_hidden, proj_d_mid, embed_dim, proj_dropout
        )
        self.text_expand = ExpansionHead(
            embed_dim, expand_d_mid, expand_d_hidden, text_hidden, expand_dropout
        )
        self.protein_expand = ExpansionHead(
            embed_dim, expand_d_mid, expand_d_hidden, protein_hidden, expand_dropout
        )

        init_logit_scale = math.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        self.max_logit_scale = math.log(max_temperature)

    @torch.no_grad()
    def set_feature_means(self, mean_p: torch.Tensor | None = None,
                          mean_t: torch.Tensor | None = None) -> None:
        """Install the corpus-mean token vectors into the projection heads.

        The whole pipeline then operates in centered space: project() subtracts
        the mean, and center() produces the matching centered reconstruction
        targets for the expansion heads.
        """
        if mean_p is not None:
            self.protein_proj.mean_in.copy_(mean_p.to(self.protein_proj.mean_in))
        if mean_t is not None:
            self.text_proj.mean_in.copy_(mean_t.to(self.text_proj.mean_in))

    def center(self, h_p: torch.Tensor, h_t: torch.Tensor):
        """Mean-centered encoder features — the reconstruction target space.

        Expansion reconstructs h - mean (not raw h): the mean is constant across
        the corpus, so it carries no per-token signal, and keeping the generation
        conditioning memory centered de-anisotropizes the decoder's cross-attention
        the same way it de-anisotropizes FILIP.
        """
        return h_p - self.protein_proj.mean_in, h_t - self.text_proj.mean_in

    def project(self, h_p: torch.Tensor, h_t: torch.Tensor):
        """Returns (z_p, z_t) per-token, L2-normalized. Heads center h internally."""
        return self.protein_proj(h_p), self.text_proj(h_t)

    def expand(self, z_p: torch.Tensor, z_t: torch.Tensor):
        """Returns (h_p_hat, h_t_hat) per-token, in *centered* encoder hidden space."""
        return self.protein_expand(z_p), self.text_expand(z_t)

    def forward(self, h_p: torch.Tensor, h_t: torch.Tensor):
        z_p, z_t = self.project(h_p, h_t)
        h_p_hat, h_t_hat = self.expand(z_p, z_t)
        return {"z_p": z_p, "z_t": z_t, "h_p_hat": h_p_hat, "h_t_hat": h_t_hat}

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.max_logit_scale)


# ---------------------------------------------------------------------------
# Construction / checkpoint loading
# ---------------------------------------------------------------------------
# Every phase needs the same model built from the same eleven config fields, and
# most of them need it loaded frozen from a checkpoint. These two are the single
# place that knows how — a shape or a field added to MiniEmbedFilip is a one-line
# change here rather than a seven-file sweep.
def build_retrieval(cfg=None) -> "MiniEmbedFilip":
    """Build an untrained retrieval model from a `Cfg` (default: `default_cfg()`)."""
    if cfg is None:
        from config import default_cfg
        cfg = default_cfg()
    return MiniEmbedFilip(
        text_hidden=cfg.model.text_hidden,
        protein_hidden=cfg.model.protein_hidden,
        proj_d_hidden=cfg.model.proj_d_hidden,
        proj_d_mid=cfg.model.proj_d_mid,
        embed_dim=cfg.model.embed_dim,
        proj_dropout=cfg.model.proj_dropout,
        expand_d_mid=cfg.model.expand_d_mid,
        expand_d_hidden=cfg.model.expand_d_hidden,
        expand_dropout=cfg.model.expand_dropout,
        init_temperature=cfg.retrieval.init_temperature,
        max_temperature=cfg.retrieval.max_temperature,
    )


def load_retrieval(ckpt_path: str, device, cfg=None,
                   *, freeze: bool = True) -> "MiniEmbedFilip":
    """Load a `train_retrieval`-format checkpoint ({"epoch", "model_state", ...}).

    `freeze` (the default) also puts the model in eval mode and clears
    requires_grad — what every downstream consumer wants, since generation,
    inference and the round-trip eval all treat the retrieval heads as fixed.
    Pass `freeze=False` to keep training it (the hard-negative phase).
    """
    m = build_retrieval(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state["model_state"])
    m.to(device)
    if freeze:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m
