"""Inference utility: generate from a prompt in either direction.

Loads retrieval + decoder + trained adapters, encodes the input, builds the
cross-attention memory, and runs autoregressive decoding. With
--num-candidates > 1 it generates N candidates (diversified by sampling
temperature), re-encodes each, and selects the one with the best contrastive
round-trip margin against a reference panel (see src/best_of_n.py).

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
from src.data import load_pairs
from src.decoder_adapters import (
    decode_target, load_decoder_with_cross_attn,
    set_cross_memory, clear_cross_memory, target_prefix_ids,
)
from src.encoders import (
    encode_protein_batch, encode_text_batch,
    load_protein_encoder, load_text_encoder,
)
from src.gen_ckpt import load_generation_ckpt
from src.model import load_retrieval


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


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["text2protein", "protein2text"], required=True)
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the config cap for the GENERATED modality "
                         "(max_protein_tokens for text2protein, max_text_tokens for "
                         "protein2text), so a bare run never truncates its own output")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-candidates", type=int, default=1,
                    help="generate N candidates and select the best by round-trip margin")
    ap.add_argument("--selection", choices=["margin", "pos"], default="margin")
    ap.add_argument("--panel-size", type=int, default=256,
                    help="reference negatives sampled from --panel-csv for margin selection")
    ap.add_argument("--panel-csv", default=cfg.data.csv_path)
    args = ap.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = (cfg.data.max_protein_tokens
                               if args.direction == "text2protein"
                               else cfg.data.max_text_tokens)

    device = pick_device(args.device)
    # The decoder checkpoint's own architecture header decides how the decoder is
    # rebuilt and how the conditioning memory is formed.
    ckpt, meta = load_generation_ckpt(args.decoder_ckpt)
    print(f"[generate] decoder ckpt architecture: {meta.describe()}")
    retrieval = load_retrieval(args.retrieval_ckpt, device, cfg)
    N = max(args.num_candidates, 1)

    # Direction-specific handles: source side (input) + target side (re-encode
    # candidates). The panel is the SAME modality as the source.
    if args.direction == "text2protein":
        src_model, src_tok = load_text_encoder(cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
        src_proj, src_expand = retrieval.text_proj, retrieval.text_expand
        src_max = cfg.data.max_text_tokens
        def enc_src(strs):
            return encode_text_batch(src_model, src_tok, strs, device, src_max)
        decoder_path = cfg.generation.decoder_path
        tgt_proj = retrieval.protein_proj
    else:
        src_model, src_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        src_proj, src_expand = retrieval.protein_proj, retrieval.protein_expand
        src_max = cfg.data.max_protein_tokens
        def enc_src(strs):
            return encode_protein_batch(src_model, src_tok, strs, device, src_max)
        decoder_path = TEXT_DECODER_PATH
        tgt_proj = retrieval.text_proj

    h_src, m_src = enc_src([args.input])
    with torch.no_grad():
        z_src = src_proj(h_src.float())                 # [1, L, embed_dim]
        # "expanded" lifts z back to encoder-hidden space; "aligned" uses z directly.
        mem = z_src if meta.aligned_memory else src_expand(z_src)

    decoder, target_tok, adapters = load_decoder_with_cross_attn(
        args.direction, decoder_path, meta.cross_attn_every,
        meta.mem_dim(cfg, args.direction), device,
        cross_attn_mode=meta.cross_attn_mode,
    )
    decoder.load_state_dict(ckpt["adapter_state"], strict=False)
    decoder.eval()
    if target_tok.pad_token is None:
        target_tok.pad_token = target_tok.eos_token
    bos = target_tok.bos_token_id if target_tok.bos_token_id is not None else target_tok.eos_token_id
    pad_id = target_tok.pad_token_id if target_tok.pad_token_id is not None else target_tok.eos_token_id

    # N-way batched conditioning memory; candidate diversity comes from the
    # sampling temperature.
    mem_b = mem.expand(N, -1, -1).contiguous()
    mask_b = m_src.expand(N, -1).contiguous()
    set_cross_memory(adapters, mem_b, mask_b)
    # Seed with BOS + the decoder's control tokens (ProtGPT3's direction marker),
    # matching how training targets were built.
    prompt = [bos] + target_prefix_ids(decoder, target_tok)
    input_ids = torch.tensor([prompt] * N, device=device, dtype=torch.long)
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
    cands = [decode_target(decoder, target_tok, row) for row in generated]

    if N == 1:
        best, scores = cands[0], None
    else:
        # Re-encode candidates; build a panel of other sources; select.
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
    print("[generate] output:")
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
        panel_model, panel_tok = load_text_encoder(cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
        enc_panel = lambda strs: encode_text_batch(
            panel_model, panel_tok, strs, device, cfg.data.max_text_tokens)
        empty = "M"
    else:
        tgt_model, tgt_tok = load_text_encoder(cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
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
        z_c = tgt_proj(h_c.float())                      # [N, Lc, embed_dim]
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
                [z_pn[i][m_pn[i]].cpu() for i in range(len(strs))],
                cfg.model.embed_dim, device)

    zc, mc = pad_stack(z_cands, cfg.model.embed_dim, device)
    zs, ms = pad_stack([z_src[0][m_src[0]].cpu()], cfg.model.embed_dim, device)
    best_idx, scores = select_best_of_n(
        zc, mc, zs, ms, z_panel, z_panel_mask, mode=args.selection)
    return cands[best_idx], scores.cpu()


if __name__ == "__main__":
    main()
