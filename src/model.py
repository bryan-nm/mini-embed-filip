"""Per-token projection and expansion heads + the wrapper holding both pairs.

Both heads are position-wise (no token mixing inside the head). The encoder
already did the contextualization; the heads' job is to map between encoder
hidden space and the 64-d shared space.

Projection head:
    d_in (768 or 960) -> d_hidden -> d_mid -> d_out (64)
    Linear, LayerNorm, GELU, Dropout. Output L2-normalized along the last dim.

Expansion head:
    d_out (64) -> d_mid -> d_hidden -> d_in (768 or 960)
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


class MemoryMap(nn.Module):
    """Cross-modal residual map g: embed_dim -> embed_dim in the shared space.

    Nudges a source token's aligned projection toward the *target* modality's
    region of the shared space (text->protein for text2protein), so the decoder
    conditions on a protein-ward object instead of a text-ward one. Applied
    per-token; the L_t token structure (and thus the interpretable FILIP array) is
    preserved.

    The final layer is zero-initialized and the input is unit-norm, so at init
    g(z) = normalize(z) = z — an exact identity. Training (the frozen map phase)
    moves it off identity. Output is re-normalized so it stays a citizen of the
    unit-sphere aligned space (matching ProjectionHead's normalized output), which
    both the FILIP-contrastive aux and the aligned-mode cross-attention rely on.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d = self.fc2(self.drop(self.norm(self.act(self.fc1(z)))))
        return F.normalize(z + d, p=2, dim=-1)


class MiniEmbedFilip(nn.Module):
    """Full trainable retrieval model: projection + expansion heads + temperature.

    When `with_maps` is set it also holds the two cross-modal memory maps
    (text_map, protein_map). They are absent by default so retrieval /
    reconstruction checkpoints are byte-identical to before; the map phase and the
    generation paths that use --memory-map opt in.
    """

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
        with_maps: bool = False,
        map_hidden: int = 128,
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

        self.with_maps = with_maps
        if with_maps:
            self.text_map = MemoryMap(embed_dim, map_hidden, proj_dropout)
            self.protein_map = MemoryMap(embed_dim, map_hidden, proj_dropout)

        init_logit_scale = math.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        self.max_logit_scale = math.log(max_temperature)

    def map_source(self, z: torch.Tensor, direction: str) -> torch.Tensor:
        """Apply the cross-modal map for a generation direction's SOURCE side.

        text2protein conditions on text -> text_map (text->protein-ward);
        protein2text conditions on protein -> protein_map. Raises if the maps were
        not built (with_maps=False), so a --memory-map run fails loudly against a
        map-less checkpoint rather than silently conditioning on the raw z.
        """
        if not self.with_maps:
            raise RuntimeError(
                "map_source called but this model was built with_maps=False "
                "(no trained memory map in the checkpoint; run train_memory_map)")
        return self.text_map(z) if direction == "text2protein" else self.protein_map(z)

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
