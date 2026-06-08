"""Inference utility: generate from a prompt in either direction.

Loads retrieval + decoder + trained adapters, encodes the input, builds the
cross-attention memory, and runs autoregressive decoding.

Usage:
  python -m src.generate --direction text2protein \\
      --retrieval-ckpt checkpoints/retrieval/epoch04.pt \\
      --decoder-ckpt   checkpoints/generation/text2protein/epoch02.pt \\
      --input "DNA helicase from S. cerevisiae that..."

  python -m src.generate --direction protein2text \\
      --retrieval-ckpt checkpoints/retrieval/epoch04.pt \\
      --decoder-ckpt   checkpoints/generation/protein2text/epoch02.pt \\
      --input "MVRLFYNPIKYLFY..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
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
    args = ap.parse_args()

    device = pick_device(args.device)
    retrieval = load_retrieval(args.retrieval_ckpt, device)

    # Encoder front-end for the INPUT side
    if args.direction == "text2protein":
        text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
        h, m = encode_text_batch(text_model, text_tok, [args.input],
                                 device, cfg.data.max_text_tokens)
        with torch.no_grad():
            z = retrieval.text_proj(h.float())
            mem = retrieval.text_expand(z)
        decoder_path = cfg.generation.decoder_path
        mem_dim = cfg.model.text_hidden
    else:
        prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        h, m = encode_protein_batch(prot_model, prot_tok, [args.input],
                                    device, cfg.data.max_protein_tokens)
        with torch.no_grad():
            z = retrieval.protein_proj(h.float())
            mem = retrieval.protein_expand(z)
        decoder_path = TEXT_DECODER_PATH
        mem_dim = cfg.model.protein_hidden

    # Decoder
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

    # Generate
    set_cross_memory(adapters, mem, m)
    bos = target_tok.bos_token_id or target_tok.eos_token_id
    input_ids = torch.tensor([[bos]], device=device, dtype=torch.long)

    with torch.no_grad():
        generated = decoder.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            pad_token_id=target_tok.pad_token_id or target_tok.eos_token_id,
            use_cache=True,
        )
    clear_cross_memory(adapters)

    text = target_tok.decode(generated[0], skip_special_tokens=True)
    print(f"[generate] direction={args.direction}")
    print(f"[generate] input: {args.input[:120]}{'...' if len(args.input) > 120 else ''}")
    print(f"[generate] output:")
    print(text)


if __name__ == "__main__":
    main()
