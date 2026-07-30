"""Batch text->protein generation for a list of accessions, with best-of-N.

Takes an accession list (one primary_Accession per line, like inspect.pbs),
pulls each accession's caption + true protein from the SwissProt CSV, and
generates `--num-outputs` protein sequences per accession. Each output is the
winner of an independent best-of-`--num-candidates` round: sample N candidates
from the decoder, re-encode them with the protein encoder, and keep the one with
the best contrastive round-trip margin against the source caption (see
src/best_of_n.py). Candidate diversity comes purely from sampling temperature —
no CVAE (a decoder checkpoint carrying `cvae_state` is rejected, since dropping
its latent tokens would not match how it was trained).

Writes one CSV row per generated sequence: accession, prompt (the caption used),
the true protein it was paired with, and the generated sequence. Rows are
flushed as each accession finishes, so a wall-time kill still leaves usable
output.

Single process — best-of-N batches the N candidates into one decoder call.

Usage:
  python -m src.generate_set \\
      --retrieval-ckpt checkpoints/retrieval/epoch50.pt \\
      --decoder-ckpt   checkpoints/generation/text2protein/epoch29.pt \\
      --csv /path/to/swissprot.csv --id-file protein_ids.txt \\
      --num-candidates 10 --num-outputs 5 --out generated_proteins.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.best_of_n import pad_stack, select_best_of_n
from src.data import load_pairs
from src.decoder_adapters import (
    LoRACfg, decode_target, load_decoder_with_cross_attn,
    set_cross_memory, clear_cross_memory, target_prefix_ids,
)
from src.encoders import (
    encode_protein_batch, encode_text_batch,
    load_protein_encoder, load_text_encoder,
)
from src.generate import load_retrieval, pick_device
from src.inspect import lookup_pairs_in_csv


def build_panel(cfg, args, enc_text, text_proj, device, exclude: set):
    """Reference negatives for margin selection: captions of *other* proteins.

    A candidate that scores high against its own caption but equally high
    against arbitrary captions isn't conditioned — it's generic. Subtracting the
    panel max is what makes the selector prefer specificity. Accessions in the
    worklist are excluded so a source's own caption is never a negative.
    """
    if args.num_candidates <= 1 or args.selection != "margin" or args.panel_size <= 0:
        return None, None
    pairs = load_pairs(
        args.panel_csv, id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.panel_size + len(exclude) + 64,
    )
    texts = [p.text for p in pairs if p.uid not in exclude][:args.panel_size]
    if not texts:
        return None, None
    with torch.no_grad():
        h, m = enc_text(texts)
        z = text_proj(h.float())
        panel, panel_mask = pad_stack(
            [z[i][m[i]].float().cpu() for i in range(len(texts))],
            cfg.model.embed_dim, device)
    print(f"[gen-set] margin panel: {len(texts)} captions from {args.panel_csv}",
          flush=True)
    return panel, panel_mask


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--csv", default=cfg.data.csv_path,
                    help="CSV holding the caption + true protein per accession")
    ap.add_argument("--id-file", required=True,
                    help="one primary_Accession per line")
    ap.add_argument("--out", default="generated_proteins.csv")
    # Best-of-N: N candidates per output, --num-outputs independent rounds.
    ap.add_argument("--num-candidates", type=int, default=10,
                    help="N: candidates generated per output before selection")
    ap.add_argument("--num-outputs", type=int, default=5,
                    help="sequences to emit per accession (best-of-N rounds)")
    ap.add_argument("--selection", choices=["margin", "pos"], default="margin")
    ap.add_argument("--panel-size", type=int, default=256,
                    help="reference captions sampled from --panel-csv for margin selection")
    ap.add_argument("--panel-csv", default=None,
                    help="defaults to --csv")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    args = ap.parse_args()
    if args.panel_csv is None:
        args.panel_csv = args.csv
    N = max(args.num_candidates, 1)
    R = max(args.num_outputs, 1)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # --- worklist: accession -> (true protein, caption) ---
    ids = [ln.strip() for ln in open(args.id_file) if ln.strip()]
    if not ids:
        raise SystemExit(f"no accessions in {args.id_file}")
    found = lookup_pairs_in_csv(args.csv, ids, cfg)
    missing = [a for a in ids if a not in found]
    if missing:
        print(f"[gen-set] WARNING: not found in CSV: {missing}", flush=True)
    worklist = [(a, found[a][0], found[a][1]) for a in dict.fromkeys(ids) if a in found]
    if not worklist:
        raise SystemExit("nothing to generate")
    print(f"[gen-set] {len(worklist)} accessions x {R} outputs, best-of-{N} "
          f"(selection={args.selection}) on {device}", flush=True)

    # --- models: decoder architecture flags come from the ckpt, as elsewhere ---
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    if "cvae_state" in ckpt:
        raise SystemExit(
            f"{args.decoder_ckpt} was trained with a CVAE (cvae_state present); "
            "this script conditions without latent tokens. Use src.generate / "
            "src.roundtrip_eval for CVAE checkpoints.")
    cae = ckpt.get("cross_attn_every", cfg.generation.cross_attn_every)
    aligned_mem = ckpt.get("memory_space", "expanded") == "aligned"
    memory_map = ckpt.get("memory_map", False)
    cross_attn_mode = ckpt.get("cross_attn_mode", "head")

    retrieval = load_retrieval(args.retrieval_ckpt, device, with_maps=memory_map)
    text_model, text_tok = load_text_encoder(
        cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)

    def enc_text(strs):
        return encode_text_batch(text_model, text_tok, strs, device,
                                 cfg.data.max_text_tokens)

    def enc_prot(strs):
        return encode_protein_batch(prot_model, prot_tok, strs, device,
                                    cfg.data.max_protein_tokens)

    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
        target_self_attn=ckpt.get("lora_targets_self_attn", True),
        target_ffn=ckpt.get("lora_targets_ffn", True),
    )
    mem_dim = cfg.model.embed_dim if aligned_mem else cfg.model.text_hidden
    decoder, dtok, adapters = load_decoder_with_cross_attn(
        "text2protein", cfg.generation.decoder_path, cae, mem_dim, lora_cfg,
        device, cross_attn_mode=cross_attn_mode,
    )
    decoder.load_state_dict(ckpt["adapter_state"], strict=False)
    decoder.eval()
    if dtok.pad_token is None:
        dtok.pad_token = dtok.eos_token
    dtok.padding_side = "left"                 # correct for batched decoding
    bos = dtok.bos_token_id if dtok.bos_token_id is not None else dtok.eos_token_id
    pad_id = dtok.pad_token_id if dtok.pad_token_id is not None else dtok.eos_token_id
    prompt_ids = [bos] + target_prefix_ids(decoder, dtok)

    panel, panel_mask = build_panel(
        cfg, args, enc_text, retrieval.text_proj, device, {a for a, _, _ in worklist})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_rows = 0
    with open(out_path, "w", newline="") as fh, torch.no_grad():
        w = _csv.writer(fh)
        w.writerow(["accession", "replicate", "prompt", "true_protein",
                    "generated_protein", "select_score", "true_len", "gen_len"])
        for k, (acc, true_prot, caption) in enumerate(worklist):
            # Caption -> embed_dim -> cross-attn memory, matching the ckpt's
            # memory_space ("expanded" lifts back to encoder-hidden space;
            # "aligned" conditions on the projection, optionally memory-mapped).
            h_src, m_src = enc_text([caption])
            z_src = retrieval.text_proj(h_src.float())          # [1, L, 32]
            if not aligned_mem:
                mem = retrieval.text_expand(z_src)
            elif memory_map:
                mem = retrieval.map_source(z_src, "text2protein")
            else:
                mem = z_src
            mem_b = mem.expand(N, -1, -1).contiguous()
            mask_b = m_src.expand(N, -1).contiguous()
            zs, ms = pad_stack([z_src[0][m_src[0]].float().cpu()],
                               cfg.model.embed_dim, device)

            for rep in range(R):
                # One best-of-N round: N sampled candidates, one batched call.
                set_cross_memory(adapters, mem_b, mask_b)
                input_ids = torch.tensor([prompt_ids] * N, device=device,
                                         dtype=torch.long)
                gen = decoder.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-6), top_p=args.top_p,
                    pad_token_id=pad_id, use_cache=True,
                )
                clear_cross_memory(adapters)
                cands = [decode_target(decoder, dtok, row) for row in gen]

                if N > 1:
                    # Re-encode candidates into the retrieval space and score
                    # them against their own source ("M" guards empty rows).
                    h_c, m_c = enc_prot([c if c.strip() else "M" for c in cands])
                    z_c = retrieval.protein_proj(h_c.float())
                    zc, mc = pad_stack(
                        [z_c[i][m_c[i]].float().cpu() for i in range(N)],
                        cfg.model.embed_dim, device)
                    best, scores = select_best_of_n(
                        zc, mc, zs, ms, panel, panel_mask, mode=args.selection)
                    score = float(scores[best].item())
                else:
                    best, score = 0, float("nan")

                w.writerow([acc, rep, caption, true_prot, cands[best],
                            f"{score:.4f}", len(true_prot), len(cands[best])])
                n_rows += 1
            fh.flush()
            el = time.time() - t0
            print(f"[gen-set] {k + 1}/{len(worklist)} {acc}: {R} sequences "
                  f"({el / (k + 1):.1f}s/accession)", flush=True)

    print(f"[gen-set] wrote {n_rows} rows to {out_path} in {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
