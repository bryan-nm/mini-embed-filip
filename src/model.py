"""Per-token projection and expansion heads + the wrapper holding both pairs.

Both heads are position-wise (no token mixing inside the head). The encoder
already did the contextualization; the heads' job is to map between encoder
hidden space and the 32-d shared space.

Projection head:
    d_in (768 or 640) -> d_hidden -> d_mid -> d_out (32)
    Linear, LayerNorm, GELU, Dropout. Output L2-normalized along the last dim.

Expansion head:
    d_out (32) -> d_mid -> d_hidden -> d_in (768 or 640)
    Mirrored architecture; separate weights. Output NOT normalized.

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
        d_hidden: int = 512,
        d_mid: int = 256,
        d_out: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_mid)
        self.norm2 = nn.LayerNorm(d_mid)
        self.fc3 = nn.Linear(d_mid, d_out)
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
        return F.normalize(x, p=2, dim=-1)


class ExpansionHead(nn.Module):
    """d_in (32) -> d_mid -> d_hidden -> d_out, mirroring ProjectionHead.

    Separate weights from the corresponding projection (not weight-tied).
    Output not normalized; the consumer (decoder cross-attention) operates in
    encoder hidden space, not on a sphere.
    """

    def __init__(
        self,
        d_in: int = 32,
        d_mid: int = 256,
        d_hidden: int = 512,
        d_out: int = 768,
        dropout: float = 0.1,
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

    def project(self, h_p: torch.Tensor, h_t: torch.Tensor):
        """Returns (z_p, z_t) per-token, L2-normalized in 32-d."""
        return self.protein_proj(h_p), self.text_proj(h_t)

    def expand(self, z_p: torch.Tensor, z_t: torch.Tensor):
        """Returns (h_p_hat, h_t_hat) per-token, in encoder hidden space."""
        return self.protein_expand(z_p), self.text_expand(z_t)

    def forward(self, h_p: torch.Tensor, h_t: torch.Tensor):
        z_p, z_t = self.project(h_p, h_t)
        h_p_hat, h_t_hat = self.expand(z_p, z_t)
        return {"z_p": z_p, "z_t": z_t, "h_p_hat": h_p_hat, "h_t_hat": h_t_hat}

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.max_logit_scale)
