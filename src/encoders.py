"""Load and run the two frozen encoders, returning per-token hidden states.

BioLinkBERT-base: standard HuggingFace BertModel; we return its full last
hidden state along with a "valid token" mask that excludes [PAD], [CLS], [SEP].

AMPLIFY-350M: custom AMPLIFY model loaded via trust_remote_code. We return
the post-layer-norm final hidden state plus a mask that excludes <pad>, <bos>,
<eos>.

The AMPLIFY source imports `xformers.ops` at module top. On non-CUDA hosts
(Mac/CPU, Intel XPU) xformers is typically unavailable, so we install a stub
before loading. The stubbed SwiGLU is weight-compatible with
`xformers.ops.SwiGLU(_pack_weights=True)` (parameters `w12` and `w3`).
"""
from __future__ import annotations

import sys
import types
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# xformers stub for non-CUDA loading of AMPLIFY
# ---------------------------------------------------------------------------
def install_xformers_stub_if_missing() -> bool:
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401
        return False
    except Exception:
        pass

    class _StubSwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, bias=False):
            super().__init__()
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def _stub_mea(*args, **kwargs):
        raise RuntimeError(
            "xformers.memory_efficient_attention should not be called on non-CUDA devices."
        )

    xf = types.ModuleType("xformers")
    xf_ops = types.ModuleType("xformers.ops")
    xf_ops.SwiGLU = _StubSwiGLU
    xf_ops.memory_efficient_attention = _stub_mea
    xf.ops = xf_ops
    sys.modules["xformers"] = xf
    sys.modules["xformers.ops"] = xf_ops
    return True


# ---------------------------------------------------------------------------
# Text encoder: BioLinkBERT-base
# ---------------------------------------------------------------------------
def load_text_encoder(path: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _text_valid_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     tokenizer, mask_specials: bool) -> torch.Tensor:
    """Bool mask: True at positions we want to keep for FILIP / uniformity."""
    valid = attention_mask.bool()
    if not mask_specials:
        return valid
    special_ids = set()
    for tok_id in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                   tokenizer.pad_token_id, tokenizer.bos_token_id,
                   tokenizer.eos_token_id):
        if tok_id is not None:
            special_ids.add(int(tok_id))
    if not special_ids:
        return valid
    spec = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        spec = spec | (input_ids == sid)
    return valid & ~spec


def text_encoder_max_len(model) -> int:
    """The hard architectural limit on input length for this text encoder.

    BERT-family models cap at config.max_position_embeddings; long-context
    variants (Longformer, BigBird, etc.) advertise their own cap. We respect
    whichever is smaller: the model's cap or the caller's requested max_len.
    """
    return int(getattr(model.config, "max_position_embeddings", 512))


@torch.no_grad()
def encode_text_batch(
    model, tokenizer, texts: List[str], device: torch.device,
    max_len: int, mask_specials: bool = True,
):
    """Returns (h_t, valid_mask) where h_t is [B, L, 768] and mask is [B, L].

    `max_len` is silently capped at the model's max_position_embeddings —
    BioLinkBERT-base cannot index past 512 tokens regardless of what the
    config requests. Swap to a long-context text encoder if you need more.
    """
    effective_max = min(max_len, text_encoder_max_len(model))
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=effective_max,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    h = out.last_hidden_state                            # [B, L, 768]
    valid = _text_valid_mask(enc["input_ids"], enc["attention_mask"],
                             tokenizer, mask_specials)
    return h, valid


# ---------------------------------------------------------------------------
# Protein encoder: AMPLIFY-350M
# ---------------------------------------------------------------------------
def _view_safe(out: torch.Tensor) -> torch.Tensor:
    """Return `out` ([B, H, M, K]) with a stride that survives AMPLIFY's reshape.

    AMPLIFY-350M's `_att_block` does `sdpa(...).transpose(1, 2).view(...)`. A
    `view` requires the post-transpose tensor to be contiguous, but a plain
    [B, H, M, K] attention output is contiguous, so its `.transpose(1, 2)` is
    NOT — and the view raises "view size is not compatible ... use .reshape()".
    (AMPLIFY-120M uses `.reshape` here and is immune; the 350M remote code
    regressed to `.view`.) We return a tensor whose `.transpose(1, 2)` IS
    contiguous, so AMPLIFY's view succeeds; harmless under the 120M `.reshape`.
    """
    return out.transpose(1, 2).contiguous().transpose(1, 2)


def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kwargs):
    """Plain matmul+softmax attention (q,k,v: [B, H, M, K]; additive attn_mask).

    Drop-in for `torch.nn.functional.scaled_dot_product_attention` used by
    AMPLIFY's XPU branch. The Aurora IPEX build has no xetla fused-attention
    kernel ("Your IPEX version currently doesn't support xetla"), so AMPLIFY
    falls back to an IPEX SDPA path that reads out of bounds (GPU "NotPresent"
    page-fault segfault) once the sequence length exceeds ~512 with the dense
    [B, H, L, L] additive mask AMPLIFY builds. Routing attention through
    ordinary GEMM + softmax avoids that kernel entirely and works at any length.
    """
    scale = (query.size(-1) ** -0.5) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask                      # additive (-inf at pads)
    attn = torch.softmax(scores, dim=-1)
    if dropout_p and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return _view_safe(torch.matmul(attn, value))


def _view_safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, **kwargs):
    """Real torch SDPA, returned in a layout AMPLIFY-350M's `view` accepts.

    On CPU (and any non-XPU/non-CUDA host) the fused kernel is fine, but the
    350M remote code still does `sdpa(...).transpose(1, 2).view(...)`, which can
    raise whenever the backend's output stride leaves that transpose
    non-contiguous. Delegate to the stock kernel and fix up the stride
    defensively so the layout never depends on the chosen SDPA backend. See
    `_view_safe`.
    """
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale,
    )
    return _view_safe(out)


def load_protein_encoder(path: str, device: torch.device):
    install_xformers_stub_if_missing()
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().to(device)

    # AMPLIFY stores its RoPE cache as a plain attribute (not a registered
    # buffer); meta-tensor loader leaves it on meta. Re-materialize on device.
    amp_mod = sys.modules[type(model).__module__]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    model.freqs_cis = amp_mod.precompute_freqs_cis(
        head_dim, model.config.max_length
    ).to(device)

    # AMPLIFY's non-CUDA attention path (everything except `x.is_cuda`) has two
    # problems we patch by swapping the module-global `scaled_dot_product_attention`
    # AMPLIFY imported:
    #   1. On XPU the Aurora IPEX build (no xetla) segfaults for seqlen > ~512 —
    #      route through plain matmul+softmax (`_manual_sdpa`) at any length.
    #   2. AMPLIFY-350M's `_att_block` reshapes the attention output with `.view`
    #      on `sdpa(...).transpose(1, 2)`. That transpose is non-contiguous for a
    #      plain [B,H,M,K] output (e.g. our matmul on XPU, or some CPU/IPEX SDPA
    #      backends), so the view raises (the 120M used `.reshape` and is immune).
    #      Both replacements return a view-safe layout; CPU's is defensive.
    # CUDA is left alone: it takes AMPLIFY's xformers branch, not SDPA.
    if device.type == "xpu":
        amp_mod.scaled_dot_product_attention = _manual_sdpa
    elif device.type != "cuda":
        amp_mod.scaled_dot_product_attention = _view_safe_sdpa

    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _amplify_additive_mask(pad_mask_1_0: torch.Tensor) -> torch.Tensor:
    """HF-style 1/0 mask -> AMPLIFY additive (0 / -inf)."""
    am = torch.zeros_like(pad_mask_1_0, dtype=torch.float)
    return am.masked_fill(pad_mask_1_0 == 0, float("-inf"))


def _protein_valid_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        tokenizer, mask_specials: bool) -> torch.Tensor:
    valid = attention_mask.bool()
    if not mask_specials:
        return valid
    special_ids = set()
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id",
                 "mask_token_id"):
        sid = getattr(tokenizer, attr, None)
        if sid is not None:
            special_ids.add(int(sid))
    if not special_ids:
        return valid
    spec = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        spec = spec | (input_ids == sid)
    return valid & ~spec


@torch.no_grad()
def encode_protein_batch(
    model, tokenizer, seqs: List[str], device: torch.device,
    max_len: int, mask_specials: bool = True,
):
    """Returns (h_p, valid_mask) where h_p is [B, L, 960] and mask is [B, L]."""
    enc = tokenizer(
        seqs, padding=True, truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    pad_mask = enc["attention_mask"].to(device)
    additive = _amplify_additive_mask(pad_mask)

    out = model(input_ids=input_ids, attention_mask=additive,
                output_hidden_states=True)
    last = out.hidden_states[-1]                          # [B, L, 960]
    if getattr(model.config, "layer_norm_before_last_layer", False):
        last = model.layer_norm_2(last)

    valid = _protein_valid_mask(input_ids, pad_mask, tokenizer, mask_specials)
    return last, valid
