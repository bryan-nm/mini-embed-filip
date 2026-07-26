"""Cross-attention adapters for the generation decoders.

Supported architectures: Mixtral (ProtGPT3, the default text->protein decoder),
Jamba (Dayhoff), ProGen2, and BioGPT.

Decoders are loaded as their pretrained `ForCausalLM` classes. We inject
a `CrossAttentionAdapter` into a subset of decoder blocks (every Nth, by
default) and freeze everything else. Optionally, LoRA is added on top of the
existing self-attention QKV projections and FFN.

Cross-attention "memory" is set on the model before each forward call via
`set_cross_memory(model, memory, mask)`, which stores the tensors on each
adapter block. The adapter blocks read those during their forward. This
stateful approach avoids monkey-patching the underlying transformer's
forward signature.

The injection points differ slightly:
  ProGen2 block:  parallel attn + MLP, both reading from ln_1(h). We append a
                  cross-attention "residual update" after the parallel block.
  BioGPT block:   standard pre-norm self-attn + FFN. We insert cross-attention
                  between them, also as a residual update.
  Mixtral/Jamba:  the layer is left intact and a forward hook applies the
                  cross-attention residual to its output (see
                  `_make_cross_attn_hook`).

The user-facing API:
  load_decoder_with_cross_attn(direction, path, cross_attn_every, memory_dim,
                               lora_cfg, device)
      -> (model, tokenizer, adapter_blocks)
  set_cross_memory(adapter_blocks, memory, mask) -> None
  target_prefix_ids(model, tokenizer) -> List[int]   # decoder control tokens
  decode_target(model, tokenizer, ids) -> str        # inverse of the above
  count_trainable(model) -> int

`memory_dim` is the dimension of the per-token expansion-head output (e.g.
640 for protein encoder memory, 768 for text encoder memory). It is *not*
the decoder hidden dim; the cross-attention K/V projections handle the
re-projection internally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cross-attention adapter
# ---------------------------------------------------------------------------
class CrossAttentionAdapter(nn.Module):
    """Trainable cross-attention: queries come from the decoder hidden state,
    keys/values come from external encoder memory (per-token expansion-head
    output). Residual added to the decoder hidden state.

    Standard multi-head attention with separate Q/K/V/O linear projections.
    Pre-LN on the decoder side; the encoder memory is consumed as-is.

    Stateful: `self.memory` and `self.memory_mask` are set externally before
    each forward call via `set_cross_memory(...)`. If `self.memory is None`
    the adapter passes the input through unchanged (useful for initial
    layer-shape sanity checks).
    """

    def __init__(self, dec_hidden: int, mem_dim: int, n_heads: int,
                 dropout: float = 0.0, score_space: str = "head"):
        super().__init__()
        assert dec_hidden % n_heads == 0
        assert score_space in ("head", "aligned")
        self.dec_hidden = dec_hidden
        self.mem_dim = mem_dim
        self.n_heads = n_heads
        self.head_dim = dec_hidden // n_heads
        self.scale = self.head_dim ** -0.5
        self.score_space = score_space

        self.ln_q = nn.LayerNorm(dec_hidden)
        self.v_proj = nn.Linear(mem_dim, dec_hidden, bias=False)
        self.o_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.drop = nn.Dropout(dropout)

        if score_space == "head":
            # Learned multi-head attention in the decoder head space (original).
            self.q_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
            self.k_proj = nn.Linear(mem_dim, dec_hidden, bias=False)
        else:
            # "aligned" (Strong): single-head cosine max-sim in the shared mem_dim
            # (embed_dim) space. The decoder hidden state is projected into the
            # shared space and scored directly against the aligned memory — no
            # k_proj, the keys ARE the retrieval vectors — with a learned FILIP
            # temperature. The attention weights are then the interpretable
            # [T, L_mem] alignment array. Values stay learned (v_proj), so the
            # decoder still pulls a usable dec_hidden-d signal while the *weights*
            # remain the alignment.
            #
            # q_align is a small MLP with the same body as the retrieval
            # ProjectionHead (dec_hidden -> dec_hidden -> dec_hidden//2 -> mem_dim,
            # Linear/GELU/LayerNorm, dropout after the first block), so both sides
            # of the max-sim reach the shared space through the same *kind* of
            # nonlinear map. The final L2-normalize lives in the forward. ln_q
            # (above) is the decoder-side input norm, analogous to the projection
            # head's mean-centering. The array's interpretability depends only on
            # cosine-scoring against the real keys, not on this map's form.
            mid = dec_hidden // 2
            self.q_align = nn.Sequential(
                nn.Linear(dec_hidden, dec_hidden),
                nn.GELU(),
                nn.LayerNorm(dec_hidden),
                nn.Dropout(dropout),
                nn.Linear(dec_hidden, mid),
                nn.GELU(),
                nn.LayerNorm(mid),
                nn.Linear(mid, mem_dim),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            self.max_logit_scale = math.log(100.0)

        # Initialize output projection to zero so the adapter starts as a no-op
        # — important for not destabilizing the frozen decoder at step 0.
        nn.init.zeros_(self.o_proj.weight)

        # Set externally before each forward; cleared after.
        self.memory: Optional[torch.Tensor] = None      # [B, L_mem, mem_dim]
        self.memory_mask: Optional[torch.Tensor] = None  # [B, L_mem] bool
        # Last attention weights (aligned mode) for interpretability / --dump-attn.
        self.last_attn: Optional[torch.Tensor] = None    # [B, T, L_mem]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.memory is None:
            return hidden_states  # pass-through if no memory is set
        if self.score_space == "aligned":
            return self._forward_aligned(hidden_states)

        # Run the adapter in its own parameter dtype (fp32) and cast the residual
        # back to the decoder's dtype. This keeps a bf16-loaded frozen decoder and
        # an fp32 memory source compatible with fp32 trainable adapters, both
        # under autocast (training) and without it (generation).
        in_dtype = hidden_states.dtype
        w_dtype = self.q_proj.weight.dtype
        B, T, D = hidden_states.shape
        h = self.ln_q(hidden_states.to(w_dtype))
        memory = self.memory.to(w_dtype)

        # [B, T, D] -> [B, n_heads, T, head_dim]
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale          # [B, n_heads, T, L_mem]
        if self.memory_mask is not None:
            mask = self.memory_mask[:, None, None, :]                  # [B, 1, 1, L_mem]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)
        attn_out = attn_weights @ v                                    # [B, n_heads, T, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return hidden_states + self.o_proj(attn_out).to(in_dtype)

    def _forward_aligned(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Single-head cosine max-sim cross-attention in the shared aligned space.

        scores[t, m] = temp * <norm(q_align(h_t)), norm(mem_m)>, where q_align is
        the ProjectionHead-style MLP. The attention weights [B, T, L_mem] are the
        FILIP alignment between decoder positions and source tokens; stored on
        self.last_attn for inspection.
        """
        in_dtype = hidden_states.dtype
        w_dtype = self.q_align[0].weight.dtype        # first Linear of the MLP
        h = self.ln_q(hidden_states.to(w_dtype))
        memory = self.memory.to(w_dtype)                              # [B, L, mem_dim]

        q = F.normalize(self.q_align(h), p=2, dim=-1)                 # [B, T, mem_dim]
        k = F.normalize(memory, p=2, dim=-1)                          # [B, L, mem_dim]
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        scores = (q @ k.transpose(-1, -2)) * scale                    # [B, T, L]
        if self.memory_mask is not None:
            scores = scores.masked_fill(~self.memory_mask[:, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)                              # [B, T, L]
        self.last_attn = attn.detach()
        attn = self.drop(attn)
        v = self.v_proj(memory)                                       # [B, L, D]
        attn_out = attn @ v                                           # [B, T, D]
        return hidden_states + self.o_proj(attn_out).to(in_dtype)

    def warm_start_q_align_from(self, proj_head: nn.Module) -> None:
        """Warm-start the aligned-mode query MLP from a retrieval ProjectionHead.

        `q_align` maps decoder hidden states into the shared space, to be cosine-
        scored against the aligned memory. The retrieval projection head of the
        DECODER's own modality (protein_proj for text2protein) already defines how
        that modality's tokens carve the shared sphere — and in retrieval it is the
        matching *protein* token that sits near a given text key. So we transplant
        that head's body — the layers from d_hidden through the d_out projection
        (norm1, fc2, norm2, fc3) — and leave only the input layer (`q_align[0]`:
        dec_hidden -> d_hidden) freshly initialized, since the ProtGPT3 decoder
        hidden space differs from the encoder hidden space the head was trained on.
        From step 0 the query then lands in a sensible region of the shared space
        instead of cold-starting a random map through the frozen decoder.

        `o_proj` stays zero (the adapter is still a no-op at step 0); this only sets
        where the *alignment* points once o_proj opens under gradient. The head's
        input mean-centering (`mean_in`) is intentionally dropped — it belongs to
        the encoder feature space, and the fresh `q_align[0]` learns its own input
        map.

        Requires the transplant layers to match shape, i.e. dec_hidden ==
        proj_d_hidden and dec_hidden // 2 == proj_d_mid (holds for ProtGPT3-112M:
        512/256). Raises a clear error otherwise rather than silently skipping, so
        a mismatched decoder/head pairing fails loudly.
        """
        if self.score_space != "aligned":
            raise ValueError("warm_start_q_align_from applies only to aligned mode")
        # q_align layout: [0] fc1, [1] GELU, [2] norm1, [3] Dropout,
        #                 [4] fc2, [5] GELU, [6] norm2, [7] fc3
        pairs = [
            (self.q_align[2], proj_head.norm1),
            (self.q_align[4], proj_head.fc2),
            (self.q_align[6], proj_head.norm2),
            (self.q_align[7], proj_head.fc3),
        ]
        for dst, src in pairs:
            if dst.weight.shape != src.weight.shape:
                raise ValueError(
                    "aligned warm-start dim mismatch: q_align body layer "
                    f"{tuple(dst.weight.shape)} != projection-head layer "
                    f"{tuple(src.weight.shape)}. Warm-start requires dec_hidden == "
                    "proj_d_hidden and dec_hidden//2 == proj_d_mid (e.g. ProtGPT3-112M "
                    "hidden=512 with proj 512/256). Rebuild q_align at the head's dims "
                    "for other sizes.")
        with torch.no_grad():
            for dst, src in pairs:
                dst.weight.copy_(src.weight.to(dst.weight))
                dst.bias.copy_(src.bias.to(dst.bias))


# ---------------------------------------------------------------------------
# Minimal LoRA wrapper (avoids hard dep on peft)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear, freezes it, and adds a low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # B initialized to zero -> adapter starts as a no-op
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Low-rank update runs in the LoRA params' dtype (fp32), then casts back —
        # so an fp32 adapter on a bf16-loaded frozen base stays dtype-consistent.
        lx = self.drop(x).to(self.lora_A.weight.dtype)
        lora = self.scaling * self.lora_B(self.lora_A(lx))
        return base_out + lora.to(base_out.dtype)


def _replace_linear_with_lora(module: nn.Module, attr: str,
                              rank: int, alpha: int, dropout: float) -> bool:
    target = getattr(module, attr, None)
    if isinstance(target, nn.Linear):
        setattr(module, attr, LoRALinear(target, rank, alpha, dropout))
        return True
    return False


# ---------------------------------------------------------------------------
# ProGen2 block wrapper
# ---------------------------------------------------------------------------
class _ProGenBlockWithCrossAttn(nn.Module):
    """Wraps a ProGenBlock; appends cross-attention after the original parallel
    attn + MLP block.
    """

    def __init__(self, inner: nn.Module, cross_attn: CrossAttentionAdapter):
        super().__init__()
        self.inner = inner
        self.cross_attn = cross_attn

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, use_cache=False, output_attentions=False):
        outputs = self.inner(
            hidden_states=hidden_states, layer_past=layer_past,
            attention_mask=attention_mask, head_mask=head_mask,
            use_cache=use_cache, output_attentions=output_attentions,
        )
        # outputs[0] is the updated hidden state
        new_hidden = self.cross_attn(outputs[0])
        return (new_hidden,) + outputs[1:]


# ---------------------------------------------------------------------------
# BioGPT block wrapper
# ---------------------------------------------------------------------------
class _BioGptBlockWithCrossAttn(nn.Module):
    """Wraps a BioGptDecoderLayer; appends cross-attention to its output."""

    def __init__(self, inner: nn.Module, cross_attn: CrossAttentionAdapter):
        super().__init__()
        self.inner = inner
        self.cross_attn = cross_attn

    def forward(self, *args, **kwargs):
        outputs = self.inner(*args, **kwargs)
        if isinstance(outputs, tuple):
            new_hidden = self.cross_attn(outputs[0])
            return (new_hidden,) + outputs[1:]
        return self.cross_attn(outputs)


# ---------------------------------------------------------------------------
# Loading and injection
# ---------------------------------------------------------------------------
@dataclass
class LoRACfg:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_self_attn: bool = True
    target_ffn: bool = True


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _unwrap(model: nn.Module) -> nn.Module:
    """The raw model behind a DDP wrapper (a no-op for an unwrapped model)."""
    return getattr(model, "module", model)


def _decoder_arch(model: nn.Module) -> str:
    """Identify the decoder family from a loaded model, so injection/unfreezing
    work regardless of which checkpoint a direction is pointed at."""
    model = _unwrap(model)
    mt = getattr(getattr(model, "config", None), "model_type", "")
    if mt in ("jamba", "mixtral"):
        return mt
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "progen"
    if hasattr(model, "biogpt"):
        return "biogpt"
    raise ValueError(f"Unsupported decoder architecture (model_type={mt!r})")


def _decoder_blocks(model: nn.Module):
    """The ordered list of decoder blocks/layers for the model's architecture."""
    arch = _decoder_arch(model)
    model = _unwrap(model)
    if arch == "progen":
        return model.transformer.h            # ProGen2
    if arch == "biogpt":
        return model.biogpt.layers            # BioGPT
    return model.model.layers                 # Jamba / Mixtral (ProtGPT3)


def unfreeze_decoder_blocks(model: nn.Module, n: int, where: str = "top") -> int:
    """Unfreeze `n` decoder blocks in place (full fine-tune of those blocks, on
    top of the adapters/LoRA). Returns # params unfrozen.

    Gives the decoder real capacity to incorporate the cross-attention memory
    when small adapters alone can't overcome the frozen prior. Keep `n` small and
    the LR low (see the separate optimizer group in train_generation) to avoid
    wrecking the pretrained protein/text prior.

    `where` selects which end of the stack to unfreeze:
      "top"    — the last `n` blocks (nearest the output logits): capacity to
                 turn a conditioned hidden state into the right next-token dist.
      "bottom" — the first `n` blocks (nearest the embeddings): capacity to make
                 the early representation receptive to the injected memory so the
                 conditioning propagates up through the (frozen) rest of the stack.
    Which wins is empirical; A/B them.
    """
    if n <= 0:
        return 0
    if where not in ("top", "bottom"):
        raise ValueError(f"unfreeze `where` must be 'top' or 'bottom', got {where!r}")
    blocks = list(_decoder_blocks(model))
    n = min(n, len(blocks))
    chosen = blocks[-n:] if where == "top" else blocks[:n]
    count = 0
    for block in chosen:
        for p in block.parameters():
            p.requires_grad_(True)
            count += p.numel()
    return count


def _progen_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg,
                   cross_attn_mode: str = "head") -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters + LoRA into a ProGen2 model in-place."""
    base = model.transformer                              # ProGenModel
    cfg = base.h[0].attn  # type: ignore[attr-defined]
    dec_hidden = base.embed_dim
    n_heads = model.config.n_head

    adapters: List[CrossAttentionAdapter] = []
    for i, block in enumerate(base.h):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0,
                                   score_space=cross_attn_mode)
        wrapped = _ProGenBlockWithCrossAttn(block, ca)
        base.h[i] = wrapped
        adapters.append(ca)

    if lora_cfg.target_self_attn:
        for block in base.h:
            inner = block.inner if isinstance(block, _ProGenBlockWithCrossAttn) else block
            _replace_linear_with_lora(inner.attn, "qkv_proj",
                                      lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)
            _replace_linear_with_lora(inner.attn, "out_proj",
                                      lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)
    if lora_cfg.target_ffn:
        for block in base.h:
            inner = block.inner if isinstance(block, _ProGenBlockWithCrossAttn) else block
            for attr in ("fc_in", "fc_out"):
                _replace_linear_with_lora(inner.mlp, attr,
                                          lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)

    return adapters


def _biogpt_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg,
                   cross_attn_mode: str = "head") -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters + LoRA into a BioGPT model in-place."""
    base = model.biogpt                                   # BioGptModel
    dec_hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads

    adapters: List[CrossAttentionAdapter] = []
    for i, block in enumerate(base.layers):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0,
                                   score_space=cross_attn_mode)
        wrapped = _BioGptBlockWithCrossAttn(block, ca)
        base.layers[i] = wrapped
        adapters.append(ca)

    if lora_cfg.target_self_attn:
        for block in base.layers:
            inner = block.inner if isinstance(block, _BioGptBlockWithCrossAttn) else block
            for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
                _replace_linear_with_lora(inner.self_attn, attr,
                                          lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)
    if lora_cfg.target_ffn:
        for block in base.layers:
            inner = block.inner if isinstance(block, _BioGptBlockWithCrossAttn) else block
            _replace_linear_with_lora(inner, "fc1",
                                      lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)
            _replace_linear_with_lora(inner, "fc2",
                                      lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout)

    return adapters


# ---------------------------------------------------------------------------
# Jamba (Dayhoff) / Mixtral (ProtGPT3) injection — via forward hooks
# ---------------------------------------------------------------------------
def _make_cross_attn_hook(adapter: CrossAttentionAdapter):
    """Forward hook that applies the cross-attention residual to a layer output.

    These decoders are hooked *in place* rather than wrapped in a new module.
    For Jamba that is mandatory: JambaModel picks the per-layer attention mask
    with `isinstance(layer, JambaMambaDecoderLayer)`, so wrapping a mamba layer
    would route the wrong mask to it. A forward hook leaves the layer's class
    intact. Modern decoder layers return a bare hidden-state tensor; older ones
    return a tuple.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (adapter(output[0]),) + tuple(output[1:])
        return adapter(output)
    return hook


def _load_protein_tokenizer(path: str):
    """Load Dayhoff's tokenizer, working around transformers-5.

    The repo ships a custom slow char tokenizer (`ProteinTokenizer`, written for
    transformers 4.42). transformers 5's `AutoTokenizer` routes it through the
    fast backend and fails to instantiate, but the class itself works when
    constructed directly. Try Auto first (so a future model with a normal
    tokenizer still works), then fall back to importing the repo's
    `tokenizers.py` (same trust model as trust_remote_code).
    """
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        import importlib.util
        import os
        tok_file = os.path.join(path, "tokenizers.py")
        if not os.path.exists(tok_file):
            raise
        spec = importlib.util.spec_from_file_location("_dayhoff_tokenizer", tok_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ProteinTokenizer()


def _jamba_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg,
                  cross_attn_mode: str = "head") -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters (via hooks) + LoRA into a Jamba model.

    Cross-attention is added after every Nth decoder layer (mamba or attention
    alike — the adapter only needs the [B, T, hidden] output). LoRA targets the
    attention layers' self-attn projections and the dense MLP layers; the MoE
    blocks' fused expert Parameters (not nn.Linear) and the SSM mixers are left
    frozen.
    """
    base = model.model                                   # JambaModel
    dec_hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads

    adapters: List[CrossAttentionAdapter] = []
    for i, layer in enumerate(base.layers):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0,
                                   score_space=cross_attn_mode)
        layer.register_forward_hook(_make_cross_attn_hook(ca))
        adapters.append(ca)
    # Register the adapters on the model so their params are tracked by the
    # optimizer / state_dict / .to(device). The hooks above hold the same module
    # objects, so set_cross_memory() reaches them.
    model.cross_attn_adapters = nn.ModuleList(adapters)

    if lora_cfg.target_self_attn:
        for layer in base.layers:
            attn = getattr(layer, "self_attn", None)     # attention layers only
            if attn is not None:
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _replace_linear_with_lora(attn, attr, lora_cfg.rank,
                                              lora_cfg.alpha, lora_cfg.dropout)
    if lora_cfg.target_ffn:
        for layer in base.layers:
            ff = getattr(layer, "feed_forward", None)
            # Dense MLP layers (gate/up/down Linear) are LoRA-able; MoE blocks
            # use fused expert Parameter tensors and are left frozen.
            if ff is not None and hasattr(ff, "gate_proj"):
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    _replace_linear_with_lora(ff, attr, lora_cfg.rank,
                                              lora_cfg.alpha, lora_cfg.dropout)

    return adapters


# ---------------------------------------------------------------------------
# Mixtral (ProtGPT3) injection — via forward hooks
# ---------------------------------------------------------------------------
def _mixtral_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg,
                    cross_attn_mode: str = "head") -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters (via hooks) + LoRA into a Mixtral model.

    ProtGPT3 is a Mixtral-architecture sparse MoE. Its decoder layers return a
    bare hidden-state tensor, so the same forward-hook injection as Jamba works;
    hooks also keep the adapters inside the layer `__call__`, which is what makes
    them recompute correctly under gradient checkpointing.

    LoRA targets the self-attention projections only. The MoE feed-forward stores
    its experts as fused Parameter tensors (`experts.gate_up_proj` [E, 2*d_ff, d],
    `experts.down_proj` [E, d, d_ff]) rather than nn.Linear, so there is nothing
    for `_replace_linear_with_lora` to wrap; those stay frozen, as do the router
    weights (perturbing routing on a frozen backbone destabilizes the prior).
    """
    base = model.model                                   # MixtralModel
    dec_hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads

    adapters: List[CrossAttentionAdapter] = []
    for i, layer in enumerate(base.layers):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0,
                                   score_space=cross_attn_mode)
        layer.register_forward_hook(_make_cross_attn_hook(ca))
        adapters.append(ca)
    # Register on the model so the optimizer / state_dict / .to(device) / DDP all
    # see the adapter params. The hooks hold the same module objects, so
    # set_cross_memory() reaches them.
    model.cross_attn_adapters = nn.ModuleList(adapters)

    if lora_cfg.target_self_attn:
        for layer in base.layers:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _replace_linear_with_lora(layer.self_attn, attr, lora_cfg.rank,
                                          lora_cfg.alpha, lora_cfg.dropout)

    return adapters


# ProtGPT3 was pretrained with a generation-direction marker as the first token
# after BOS: "1" = N-to-C (the normal reading direction), "2" = C-to-N (residues
# emitted in reverse order). It is an ordinary vocab token, not a special token,
# so it must be added explicitly to every training target and to the generation
# prompt — without it the decoder is off-distribution from token 0 — and stripped
# back off when decoding (with the residues re-reversed for a "2" sequence).
PROTGPT3_FORWARD_TOKEN = "1"        # N-to-C
PROTGPT3_REVERSE_TOKEN = "2"        # C-to-N
PROTGPT3_DIRECTION_TOKENS = (PROTGPT3_FORWARD_TOKEN, PROTGPT3_REVERSE_TOKEN)


def _protgpt3_marker_id(model: nn.Module, tokenizer, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    # convert_tokens_to_ids falls back to the UNK id for an unknown token, and
    # this tokenizer's UNK id sits outside the model's embedding table — so bail
    # out rather than feed the decoder an out-of-range id.
    vocab_size = _unwrap(model).config.vocab_size
    if tid is None or not (0 <= tid < vocab_size):
        raise ValueError(
            f"Mixtral decoder's tokenizer has no usable {token!r} direction token "
            f"(got id {tid!r}, vocab_size={vocab_size})")
    return tid


def target_prefix_ids(model: nn.Module, tokenizer) -> List[int]:
    """Forward (N-to-C) control tokens between BOS and the target body.

    Empty for every decoder except ProtGPT3 (see PROTGPT3_FORWARD_TOKEN).
    """
    if _decoder_arch(model) != "mixtral":
        return []
    return [_protgpt3_marker_id(model, tokenizer, PROTGPT3_FORWARD_TOKEN)]


def reverse_prefix_ids(model: nn.Module, tokenizer) -> List[int]:
    """Reverse (C-to-N) control tokens, for direction augmentation.

    ProtGPT3 only — raises for a decoder without a reverse marker, so a
    --direction-augment run fails loudly rather than teaching a directionless
    decoder to model reversed sequences as if forward.
    """
    if _decoder_arch(model) != "mixtral":
        raise ValueError(
            "direction augmentation requires a decoder with a C-to-N reverse marker "
            "(ProtGPT3/Mixtral); this decoder has none")
    return [_protgpt3_marker_id(model, tokenizer, PROTGPT3_REVERSE_TOKEN)]


def decode_target(model: nn.Module, tokenizer, ids) -> str:
    """Decode one row of generated ids into an N-to-C target string.

    ProtGPT3's char-level WordLevel tokenizer decodes to space-separated residues
    ("M K T"), and its direction marker survives `skip_special_tokens` because it
    is a normal vocab token. Undo both. Critically, if the sequence was generated
    C-to-N (leading "2"), the emitted residues are in reverse order, so re-reverse
    them — otherwise a downstream re-encode (roundtrip / best-of-N) sees a
    backwards protein. The caller always gets a bare N-to-C sequence.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if _decoder_arch(model) != "mixtral":
        return text.strip()
    seq = "".join(text.split())
    reverse = seq[:1] == PROTGPT3_REVERSE_TOKEN
    while seq[:1] in PROTGPT3_DIRECTION_TOKENS:
        seq = seq[1:]
    return seq[::-1] if reverse else seq


def load_decoder_with_cross_attn(
    direction: str,
    path: str,
    cross_attn_every: int,
    mem_dim: int,
    lora_cfg: LoRACfg,
    device: torch.device,
    cross_attn_mode: str = "head",
) -> Tuple[nn.Module, object, List[CrossAttentionAdapter]]:
    """Load the appropriate decoder, freeze it, inject adapters + LoRA.

    Dispatch is by architecture (read from the checkpoint's config) first, then
    by `direction`. This lets a direction be re-pointed at a different model
    (e.g. text2protein: ProGen2 -> Dayhoff/Jamba -> ProtGPT3/Mixtral) without
    code changes. `cross_attn_mode` ("head" | "aligned") selects the adapter
    scoring space (see CrossAttentionAdapter); "aligned" requires aligned memory.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_type = getattr(AutoConfig.from_pretrained(path, trust_remote_code=True),
                         "model_type", "")

    if model_type == "mixtral":
        # ProtGPT3: native Mixtral sparse MoE, no remote modeling code. Drop
        # router-logits output — we take CE from the logits and never use the MoE
        # load-balancing aux loss, and computing it on a frozen router is waste.
        model = AutoModelForCausalLM.from_pretrained(
            path, output_router_logits=False, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
        _freeze(model)
        adapters = _mixtral_inject(model, cross_attn_every, mem_dim, lora_cfg,
                                   cross_attn_mode)
    elif model_type == "jamba":
        # Dayhoff-3b: native Jamba (hybrid Mamba/attention MoE), no remote
        # modeling code. Force the pure-PyTorch SSM path — the fused
        # mamba-ssm/causal-conv1d CUDA kernels aren't available on Mac/XPU — and
        # drop router-logits output (we compute CE from logits and never use the
        # MoE load-balancing aux loss).
        model = AutoModelForCausalLM.from_pretrained(
            path, use_mamba_kernels=False, output_router_logits=False,
            dtype=torch.bfloat16)
        tokenizer = _load_protein_tokenizer(path)
        _freeze(model)
        adapters = _jamba_inject(model, cross_attn_every, mem_dim, lora_cfg,
                                 cross_attn_mode)
    elif direction == "text2protein":
        # ProGen2 — custom code via auto_map
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # transformers >=5 removed ModuleUtilsMixin.get_head_mask, which ProGen2's
        # custom modeling code calls. Patch in a trivial implementation since we
        # don't use head masking anyway.
        if not hasattr(model.transformer, "get_head_mask"):
            def _noop_head_mask(self, head_mask, num_hidden_layers, *args, **kwargs):
                return [None] * num_hidden_layers
            import types as _types
            model.transformer.get_head_mask = _types.MethodType(
                _noop_head_mask, model.transformer
            )
        # transformers 5 generate() expects `num_hidden_layers` on the config;
        # ProGen2 uses `n_layer`. Add the alias.
        if not hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = model.config.n_layer
        # Re-materialize plain-attribute tensors that transformers 5 leaves on
        # the meta device. ProGenAttention.scale_attn = sqrt(head_dim) as a
        # bare attribute; rebuild it on the right device/dtype.
        for block in model.transformer.h:
            head_dim = block.attn.head_dim
            block.attn.scale_attn = torch.sqrt(
                torch.tensor(head_dim, dtype=torch.float32)
            ).to(torch.get_default_dtype())
        _freeze(model)
        adapters = _progen_inject(model, cross_attn_every, mem_dim, lora_cfg,
                                  cross_attn_mode)
    elif direction == "protein2text":
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        _freeze(model)
        adapters = _biogpt_inject(model, cross_attn_every, mem_dim, lora_cfg,
                                  cross_attn_mode)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    model.to(device)
    return model, tokenizer, adapters


def set_cross_memory(adapters: List[CrossAttentionAdapter],
                     memory: torch.Tensor, mask: torch.Tensor) -> None:
    """Set per-token encoder memory on every cross-attention adapter."""
    for a in adapters:
        a.memory = memory
        a.memory_mask = mask


def clear_cross_memory(adapters: List[CrossAttentionAdapter]) -> None:
    for a in adapters:
        a.memory = None
        a.memory_mask = None


def warm_start_q_align(adapters: List[CrossAttentionAdapter],
                       proj_head: nn.Module) -> int:
    """Warm-start every aligned-mode adapter's query MLP from `proj_head`.

    `proj_head` should be the retrieval projection head of the DECODER's own
    modality (protein_proj for text2protein, text_proj for protein2text). Returns
    the number of adapters warm-started. No-op for adapters not in aligned mode.
    """
    n = 0
    for a in adapters:
        if getattr(a, "score_space", None) == "aligned":
            a.warm_start_q_align_from(proj_head)
            n += 1
    return n


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
