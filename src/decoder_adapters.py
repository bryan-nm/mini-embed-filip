"""Cross-attention adapters for ProGen2 and BioGPT decoders.

Both decoders are loaded as their pretrained `ForCausalLM` classes. We inject
a `CrossAttentionAdapter` into a subset of decoder blocks (every Nth, by
default) and freeze everything else. Optionally, LoRA is added on top of the
existing self-attention QKV projections and FFN.

Cross-attention "memory" is set on the model before each forward call via
`set_cross_memory(model, memory, mask)`, which stores the tensors on each
adapter block. The adapter blocks read those during their forward. This
stateful approach avoids monkey-patching the underlying transformer's
forward signature.

The injection points differ slightly:
  ProGen2 block: parallel attn + MLP, both reading from ln_1(h). We append a
                 cross-attention "residual update" after the parallel block.
  BioGPT block:  standard pre-norm self-attn + FFN. We insert cross-attention
                 between them, also as a residual update.

The user-facing API:
  load_decoder_with_cross_attn(direction, path, cross_attn_every, memory_dim,
                               lora_cfg, device)
      -> (model, tokenizer, adapter_blocks)
  set_cross_memory(adapter_blocks, memory, mask) -> None
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
                 dropout: float = 0.0):
        super().__init__()
        assert dec_hidden % n_heads == 0
        self.dec_hidden = dec_hidden
        self.mem_dim = mem_dim
        self.n_heads = n_heads
        self.head_dim = dec_hidden // n_heads
        self.scale = self.head_dim ** -0.5

        self.ln_q = nn.LayerNorm(dec_hidden)
        self.q_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.k_proj = nn.Linear(mem_dim, dec_hidden, bias=False)
        self.v_proj = nn.Linear(mem_dim, dec_hidden, bias=False)
        self.o_proj = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.drop = nn.Dropout(dropout)

        # Initialize output projection to zero so the adapter starts as a no-op
        # — important for not destabilizing the frozen decoder at step 0.
        nn.init.zeros_(self.o_proj.weight)

        # Set externally before each forward; cleared after.
        self.memory: Optional[torch.Tensor] = None      # [B, L_mem, mem_dim]
        self.memory_mask: Optional[torch.Tensor] = None  # [B, L_mem] bool

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.memory is None:
            return hidden_states  # pass-through if no memory is set

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


def _decoder_arch(model: nn.Module) -> str:
    """Identify the decoder family from a loaded model, so injection/unfreezing
    work regardless of which checkpoint a direction is pointed at."""
    mt = getattr(getattr(model, "config", None), "model_type", "")
    if mt == "jamba":
        return "jamba"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "progen"
    if hasattr(model, "biogpt"):
        return "biogpt"
    raise ValueError(f"Unsupported decoder architecture (model_type={mt!r})")


def _decoder_blocks(model: nn.Module):
    """The ordered list of decoder blocks/layers for the model's architecture."""
    arch = _decoder_arch(model)
    if arch == "progen":
        return model.transformer.h            # ProGen2
    if arch == "biogpt":
        return model.biogpt.layers            # BioGPT
    return model.model.layers                 # Jamba (Dayhoff)


def unfreeze_top_blocks(model: nn.Module, direction: str, n: int) -> int:
    """Unfreeze the top `n` decoder blocks in place (full fine-tune of those
    blocks, on top of the adapters/LoRA). Returns # params unfrozen.

    Gives the decoder real capacity to incorporate the cross-attention memory
    when small adapters alone can't overcome the frozen prior. Keep `n` small
    and the LR low to avoid wrecking the pretrained protein/text prior.
    `direction` is accepted for backward compatibility; the block list is
    resolved from the model architecture.
    """
    if n <= 0:
        return 0
    blocks = list(_decoder_blocks(model))
    n = min(n, len(blocks))
    count = 0
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)
            count += p.numel()
    return count


def _progen_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg
                   ) -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters + LoRA into a ProGen2 model in-place."""
    base = model.transformer                              # ProGenModel
    cfg = base.h[0].attn  # type: ignore[attr-defined]
    dec_hidden = base.embed_dim
    n_heads = model.config.n_head

    adapters: List[CrossAttentionAdapter] = []
    for i, block in enumerate(base.h):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0)
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


def _biogpt_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg
                   ) -> List[CrossAttentionAdapter]:
    """Inject cross-attention adapters + LoRA into a BioGPT model in-place."""
    base = model.biogpt                                   # BioGptModel
    dec_hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads

    adapters: List[CrossAttentionAdapter] = []
    for i, block in enumerate(base.layers):
        if i % every != 0:
            continue
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0)
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
# Jamba (Dayhoff) injection — via forward hooks
# ---------------------------------------------------------------------------
def _make_cross_attn_hook(adapter: CrossAttentionAdapter):
    """Forward hook that applies the cross-attention residual to a layer output.

    Jamba is used *unwrapped* (not replaced by a module) because JambaModel
    selects the per-layer attention mask with `isinstance(layer,
    JambaMambaDecoderLayer)`; wrapping a mamba layer would route the wrong mask
    to it. A forward hook leaves the layer's class intact. Jamba decoder layers
    return a bare hidden-state tensor; older/other layers may return a tuple.
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


def _jamba_inject(model, every: int, mem_dim: int, lora_cfg: LoRACfg
                  ) -> List[CrossAttentionAdapter]:
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
        ca = CrossAttentionAdapter(dec_hidden, mem_dim, n_heads, dropout=0.0)
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


def load_decoder_with_cross_attn(
    direction: str,
    path: str,
    cross_attn_every: int,
    mem_dim: int,
    lora_cfg: LoRACfg,
    device: torch.device,
) -> Tuple[nn.Module, object, List[CrossAttentionAdapter]]:
    """Load the appropriate decoder, freeze it, inject adapters + LoRA.

    Dispatch is by architecture (read from the checkpoint's config) first, then
    by `direction`. This lets a direction be re-pointed at a different model
    (e.g. text2protein: ProGen2 -> Dayhoff/Jamba) without code changes.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_type = getattr(AutoConfig.from_pretrained(path, trust_remote_code=True),
                         "model_type", "")

    if model_type == "jamba":
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
        adapters = _jamba_inject(model, cross_attn_every, mem_dim, lora_cfg)
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
        adapters = _progen_inject(model, cross_attn_every, mem_dim, lora_cfg)
    elif direction == "protein2text":
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        _freeze(model)
        adapters = _biogpt_inject(model, cross_attn_every, mem_dim, lora_cfg)
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


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
