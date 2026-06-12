"""Inference utility: generate from a prompt in either direction.

Loads retrieval + decoder + trained adapters, encodes the input, builds the
cross-attention memory, and runs autoregressive decoding. Two optional layers on
top (Features 1 & 3):

  - CVAE latent: if the decoder checkpoint carries `cvae_state`, the conditioning
    memory is augmented with latent tokens sampled from the prior p(w|source).
  - Best-of-N: with --num-candidates > 1, generate N candidates (diverse via
    latent samples and/or temperature), re-encode each, and select the one with
    the best contrastive round-trip margin against a reference panel.

Usage:
  python -m src.generate --direction text2protein \\
      --retrieval-ckpt checkpoints/retrieval/epoch04.pt \\
      --decoder-ckpt   checkpoints/generation/text2protein/epoch02.pt \\
      --input "DNA helicase from S. cerevisiae that..."

  # best-of-8 with contrastive selection:
  python -m src.generate --direction text2protein --retrieval-ckpt ... \\
      --decoder-ckpt ... --input "..." --num-candidates 8 --selection margin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
from src.best_of_n import pad_stack, select_best_of_n
from src.cvae import load_cvae
from src.data import load_pairs
from src.decoder_adapters import (
    LoRACfg, load_decoder_with_cross_attn,
    set_cross_memory, clear_cross_memory,
)
from src.encoders import (
    encode_protein_batch, encode_text_batch,
    load_protein_encoder, load_text_encoder,
)
from src.model import MiniEmbedFilip


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_retrieval(ckpt_path: str, device: torch.device) -> MiniEmbedFilip:
    cfg = default_cfg()
    m = MiniEmbedFilip(
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
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state["model_state"])
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["text2protein", "protein2text"], required=True)
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="auto")
    # Best-of-N (Feature 3)
    ap.add_argument("--num-candidates", type=int, default=1,
                    help="generate N candidates and select the best by round-trip margin")
    ap.add_argument("--selection", choices=["margin", "pos"], default="margin")
    ap.add_argument("--panel-size", type=int, default=256,
                    help="reference negatives sampled from --panel-csv for margin selection")
    ap.add_argument("--panel-csv", default=cfg.data.csv_path)
    args = ap.parse_args()

    device = pick_device(args.device)
    retrieval = load_retrieval(args.retrieval_ckpt, device)
    N = max(args.num_candidates, 1)

    # Direction-specific handles: source side (input) + target side (re-encode
    # candidates). The panel is the SAME modality as the source.
    if args.direction == "text2protein":
        src_model, src_tok = load_text_encoder(cfg.model.text_encoder_path, device)
        src_proj, src_expand = retrieval.text_proj, retrieval.text_expand
        src_max = cfg.data.max_text_tokens
        def enc_src(strs):
            return encode_text_batch(src_model, src_tok, strs, device, src_max)
        decoder_path, mem_dim = cfg.generation.decoder_path, cfg.model.text_hidden
        tgt_proj = retrieval.protein_proj
        tgt_max = cfg.data.max_protein_tokens
    else:
        src_model, src_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        src_proj, src_expand = retrieval.protein_proj, retrieval.protein_expand
        src_max = cfg.data.max_protein_tokens
        def enc_src(strs):
            return encode_protein_batch(src_model, src_tok, strs, device, src_max)
        decoder_path, mem_dim = TEXT_DECODER_PATH, cfg.model.protein_hidden
        tgt_proj = retrieval.text_proj
        tgt_max = cfg.data.max_text_tokens

    # Source -> 32-d (retrieval candidate) + per-token expansion memory.
    h_src, m_src = enc_src([args.input])
    with torch.no_grad():
        z_src = src_proj(h_src.float())                 # [1, L, 32]
        mem = src_expand(z_src)                         # [1, L, mem_dim]

    # Decoder + adapters + (optional) CVAE.
    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
    )
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    cae = ckpt.get("cross_attn_every", cfg.generation.cross_attn_every)
    decoder, target_tok, adapters = load_decoder_with_cross_attn(
        args.direction, decoder_path, cae, mem_dim, lora_cfg, device,
    )
    decoder.load_state_dict(ckpt["adapter_state"], strict=False)
    decoder.eval()
    cvae = load_cvae(ckpt, cfg.model.embed_dim, device)
    if target_tok.pad_token is None:
        target_tok.pad_token = target_tok.eos_token
    bos = target_tok.bos_token_id if target_tok.bos_token_id is not None else target_tok.eos_token_id
    pad_id = target_tok.pad_token_id if target_tok.pad_token_id is not None else target_tok.eos_token_id

    # Build the N-way batched conditioning memory. Diversity comes from N latent
    # samples (CVAE) and/or sampling temperature.
    mem_b = mem.expand(N, -1, -1).contiguous()
    mask_b = m_src.expand(N, -1).contiguous()
    if cvae is not None:
        with torch.no_grad():
            z_src_pool = (z_src * m_src.unsqueeze(-1)).sum(1) / m_src.sum(1, keepdim=True).clamp_min(1)
            w = cvae.sample_prior(z_src_pool.expand(N, -1))    # N independent samples
            w_tok = cvae.latent_tokens(w)                      # [N, k, mem_dim]
        mem_b = torch.cat([mem_b, w_tok], dim=1)
        kmask = torch.ones(N, cvae.cfg.n_latent_tokens, dtype=torch.bool, device=device)
        mask_b = torch.cat([mask_b, kmask], dim=1)

    set_cross_memory(adapters, mem_b, mask_b)
    input_ids = torch.full((N, 1), bos, device=device, dtype=torch.long)
    with torch.no_grad():
        generated = decoder.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            pad_token_id=pad_id,
            use_cache=True,
        )
    clear_cross_memory(adapters)
    cands = [target_tok.decode(row, skip_special_tokens=True).strip() for row in generated]

    if N == 1:
        best, scores = cands[0], None
    else:
        # Re-encode candidates -> 32-d; build a panel of other sources; select.
        best, scores = _select(
            args, cfg, device, retrieval, cands, z_src, m_src, tgt_proj,
        )

    print(f"[generate] direction={args.direction}")
    print(f"[generate] input: {args.input[:120]}{'...' if len(args.input) > 120 else ''}")
    if scores is not None:
        order = sorted(range(len(cands)), key=lambda i: scores[i].item(), reverse=True)
        print(f"[generate] {N} candidates, selection={args.selection}; "
              f"best score={scores[order[0]].item():.4f}")
        for rank, i in enumerate(order):
            tag = "  *" if i == order[0] else "   "
            print(f"{tag} cand[{i}] score={scores[i].item():.4f} "
                  f"len={len(cands[i])} {cands[i][:80]}")
    print(f"[generate] output:")
    print(best)


def _select(args, cfg, device, retrieval, cands, z_src, m_src, tgt_proj):
    """Re-encode candidates + reference panel, return (best_str, scores[N])."""
    N = len(cands)
    # Target encoder (the generated modality) + panel encoder (source modality).
    if args.direction == "text2protein":
        tgt_model, tgt_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        enc_tgt = lambda strs: encode_protein_batch(
            tgt_model, tgt_tok, strs, device, cfg.data.max_protein_tokens)
        panel_proj = retrieval.text_proj
        panel_model, panel_tok = load_text_encoder(cfg.model.text_encoder_path, device)
        enc_panel = lambda strs: encode_text_batch(
            panel_model, panel_tok, strs, device, cfg.data.max_text_tokens)
        empty = "M"
    else:
        tgt_model, tgt_tok = load_text_encoder(cfg.model.text_encoder_path, device)
        enc_tgt = lambda strs: encode_text_batch(
            tgt_model, tgt_tok, strs, device, cfg.data.max_text_tokens)
        panel_proj = retrieval.protein_proj
        panel_model, panel_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        enc_panel = lambda strs: encode_protein_batch(
            panel_model, panel_tok, strs, device, cfg.data.max_protein_tokens)
        empty = "the protein"

    with torch.no_grad():
        enc_in = [c if c.strip() else empty for c in cands]
        h_c, m_c = enc_tgt(enc_in)
        z_c = tgt_proj(h_c.float())                      # [N, Lc, 32]
        z_cands = [z_c[i][m_c[i]].cpu() for i in range(N)]

        z_panel = z_panel_mask = None
        if args.selection == "margin" and args.panel_size > 0:
            pairs = load_pairs(args.panel_csv, id_col=cfg.data.csv_id_col,
                               protein_col=cfg.data.csv_protein_col,
                               text_col=cfg.data.csv_text_col, pfam_col=cfg.data.csv_pfam_col,
                               subset_size=args.panel_size)
            strs = [(p.text if args.direction == "text2protein" else p.protein) for p in pairs]
            h_pn, m_pn = enc_panel(strs)
            z_pn = panel_proj(h_pn.float())
            z_panel, z_panel_mask = pad_stack(
                [z_pn[i][m_pn[i]].cpu() for i in range(len(strs))], 32, device)

    zc, mc = pad_stack(z_cands, 32, device)
    zs, ms = pad_stack([z_src[0][m_src[0]].cpu()], 32, device)
    best_idx, scores = select_best_of_n(
        zc, mc, zs, ms, z_panel, z_panel_mask, mode=args.selection)
    return cands[best_idx], scores.cpu()


if __name__ == "__main__":
    main()
