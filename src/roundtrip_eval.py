"""Round-trip evaluation for either generation direction.

Teacher-forced CE can't tell you whether a *generated* output actually means its
source input. This measures it by closing the loop through the retrieval space:

  text2protein:  caption --decode--> protein_gen --re-encode--> z_gen ; retrieve caption
  protein2text:  protein --decode--> text_gen    --re-encode--> z_gen ; retrieve protein

In both cases we score the FILIP matrix between every generated output and every
source input and report retrieval recall: does a generated output retrieve back
the input it was generated from, against all other inputs? High R@K means the
conditioning genuinely steered generation. We also report the CEILING (true
output re-encoded -> source), since a perfect generator can't beat the retrieval
model's own true-pair match.

Distributed like precompute (embarrassingly parallel): one rank per tile, each
generates+encodes a contiguous slice and writes a shard; rank 0 merges, scores
the full NxN FILIP matrix, and writes outputs.

Outputs (under --out-dir, default eval/<direction>/):
  roundtrip_metrics.json     R@K (both directions) + ceiling + config
  roundtrip_pairs.tsv        id, rank, score, lengths, true & generated output
  roundtrip_sequences.fasta  protein targets only (text2protein): >{id}|true and
                             >{id}|generated, cleaned to A-Z, ready for a folding
                             tool (folding/comparison is out of scope here)

Usage:
  mpiexec ... python -m src.roundtrip_eval --direction text2protein \\
      --retrieval-ckpt checkpoints/retrieval/epoch50.pt \\
      --decoder-ckpt   checkpoints/generation/text2protein/epoch29.pt --device xpu
  mpiexec ... python -m src.roundtrip_eval --direction protein2text \\
      --retrieval-ckpt checkpoints/retrieval/epoch50.pt \\
      --decoder-ckpt   checkpoints/generation/protein2text/epoch29.pt --device xpu
  # re-score existing shards (single process):
  python -m src.roundtrip_eval --direction ... --retrieval-ckpt ... --decoder-ckpt ... --score-only
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
from src.data import load_pairs, load_splits
from src.decoder_adapters import (
    LoRACfg, load_decoder_with_cross_attn, set_cross_memory, clear_cross_memory,
)
from src.dist import init_distributed, barrier, cleanup
from src.encoders import (
    encode_protein_batch, encode_text_batch,
    load_protein_encoder, load_text_encoder,
)
from src.losses import filip_score_matrix_chunked
from src.model import MiniEmbedFilip


_AA_ONLY = re.compile(r"[^A-Z]")


def _shard_range(n: int, rank: int, world: int) -> tuple[int, int]:
    per, rem = divmod(n, world)
    start = rank * per + min(rank, rem)
    return start, start + per + (1 if rank < rem else 0)


def _target_is_protein(direction: str) -> bool:
    return direction == "text2protein"


def load_retrieval(ckpt_path: str, device) -> MiniEmbedFilip:
    cfg = default_cfg()
    m = MiniEmbedFilip(
        text_hidden=cfg.model.text_hidden, protein_hidden=cfg.model.protein_hidden,
        proj_d_hidden=cfg.model.proj_d_hidden, proj_d_mid=cfg.model.proj_d_mid,
        embed_dim=cfg.model.embed_dim, proj_dropout=cfg.model.proj_dropout,
        expand_d_mid=cfg.model.expand_d_mid, expand_d_hidden=cfg.model.expand_d_hidden,
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


# ---------------------------------------------------------------------------
# Generation phase: each rank generates + encodes its slice -> shard file
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_shard(args, cfg, env, sel_indices, pairs) -> None:
    device = env.device
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    start, end = _shard_range(len(sel_indices), env.rank, env.world_size)
    my = sel_indices[start:end]
    if env.is_main:
        print(f"[rt] {len(sel_indices)} samples over {env.world_size} ranks; "
              f"direction={args.direction}", flush=True)
    print(f"[rt][rank {env.rank}] generating {len(my)} outputs on {device}", flush=True)

    # Decoder identity depends on direction (no models needed yet).
    if _target_is_protein(args.direction):
        decoder_path, mem_dim = cfg.generation.decoder_path, cfg.model.text_hidden
    else:
        decoder_path, mem_dim = TEXT_DECODER_PATH, cfg.model.protein_hidden

    # Rank-0-first model load (trust_remote_code module-cache race for ProGen2).
    def _load_models():
        retr = load_retrieval(args.retrieval_ckpt, device)
        tmodel, ttok = load_text_encoder(cfg.model.text_encoder_path, device)
        pmodel, ptok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        lora_cfg = LoRACfg(
            rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
            dropout=cfg.generation.lora_dropout,
        )
        # Build the decoder with the SAME cross_attn_every the ckpt was trained
        # with (stored in the ckpt), so adapter counts line up.
        ck = torch.load(args.decoder_ckpt, map_location="cpu")
        cae = ck.get("cross_attn_every", args.cross_attn_every)
        dec, dtok, adapters = load_decoder_with_cross_attn(
            args.direction, decoder_path, cae, mem_dim, lora_cfg, device,
        )
        dec.load_state_dict(ck["adapter_state"], strict=False)
        dec.eval()
        if dtok.pad_token is None:
            dtok.pad_token = dtok.eos_token
        dtok.padding_side = "left"   # correct for batched decoder generation
        return retr, tmodel, ttok, pmodel, ptok, dec, dtok, adapters

    if env.is_main:
        models = _load_models()
    barrier()
    if not env.is_main:
        models = _load_models()
    retrieval, text_model, text_tok, prot_model, prot_tok, decoder, dtok, adapters = models

    # Direction-specific encode/project handles (source = conditioning input,
    # target = generated modality).
    def enc_text(strs):
        return encode_text_batch(text_model, text_tok, strs, device,
                                 cfg.data.max_text_tokens, mask_specials=True)

    def enc_prot(strs):
        return encode_protein_batch(prot_model, prot_tok, strs, device,
                                    cfg.data.max_protein_tokens, mask_specials=True)

    if _target_is_protein(args.direction):                 # text -> protein
        enc_src, src_proj, src_expand = enc_text, retrieval.text_proj, retrieval.text_expand
        enc_tgt, tgt_proj = enc_prot, retrieval.protein_proj
        get_src, get_tgt, empty_tgt = (lambda p: p.text), (lambda p: p.protein), "M"
    else:                                                  # protein -> text
        enc_src, src_proj, src_expand = enc_prot, retrieval.protein_proj, retrieval.protein_expand
        enc_tgt, tgt_proj = enc_text, retrieval.text_proj
        get_src, get_tgt, empty_tgt = (lambda p: p.protein), (lambda p: p.text), "protein"

    bos = dtok.bos_token_id if dtok.bos_token_id is not None else dtok.eos_token_id
    pad_id = dtok.pad_token_id if dtok.pad_token_id is not None else dtok.eos_token_id

    records = []
    bs = args.batch_size
    t0 = time.time()
    last_log = t0
    for s in range(0, len(my), bs):
        chunk = my[s:s + bs]
        srcs = [get_src(pairs[i]) for i in chunk]
        true_tgts = [get_tgt(pairs[i]) for i in chunk]
        uids = [pairs[i].uid for i in chunk]

        # Source -> 32-d (retrieval candidate) + expanded cross-attn memory.
        h_src, mask_src = enc_src(srcs)
        z_src = src_proj(h_src.float())
        mem = src_expand(z_src)

        set_cross_memory(adapters, mem, mask_src)
        input_ids = torch.full((len(chunk), 1), bos, device=device, dtype=torch.long)
        gen = decoder.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6),
            top_p=args.top_p, pad_token_id=pad_id,
        )
        clear_cross_memory(adapters)
        gen_tgts = [dtok.decode(row, skip_special_tokens=True).strip() for row in gen]

        # Re-encode generated + true targets -> 32-d. Guard empty generations.
        enc_in = [t if t.strip() else empty_tgt for t in gen_tgts]
        h_gen, mask_gen = enc_tgt(enc_in)
        z_gen = tgt_proj(h_gen.float())
        h_true, mask_true = enc_tgt(true_tgts)
        z_true = tgt_proj(h_true.float())

        for b in range(len(chunk)):
            records.append({
                "uid": uids[b], "true_target": true_tgts[b], "gen_target": gen_tgts[b],
                "z_gen": z_gen[b][mask_gen[b]].float().cpu(),    # generated target
                "z_true": z_true[b][mask_true[b]].float().cpu(),  # true target (ceiling)
                "z_src": z_src[b][mask_src[b]].float().cpu(),     # source (candidate)
            })

        if env.is_main and (s + bs >= len(my) or (time.time() - last_log) > 20):
            done = min(s + bs, len(my))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[rt][rank 0] {done}/{len(my)}  {rate:.2f} gen/s "
                  f"eta={(len(my)-done)/max(rate,1e-6)/60:.1f} min", flush=True)
            last_log = time.time()

    torch.save({"start": start, "direction": args.direction, "records": records},
               shards_dir / f"shard.{env.rank:05d}.pt")
    print(f"[rt][rank {env.rank}] wrote {len(records)} records", flush=True)


# ---------------------------------------------------------------------------
# Scoring phase (one rank): merge shards -> FILIP retrieval -> outputs
# ---------------------------------------------------------------------------
def _pad_stack(seqs, dim=32):
    """List of [l, dim] -> ([N, Lmax, dim], bool mask [N, Lmax])."""
    n = len(seqs)
    lmax = max(max((t.size(0) for t in seqs), default=1), 1)
    out = torch.zeros(n, lmax, dim)
    mask = torch.zeros(n, lmax, dtype=torch.bool)
    for i, t in enumerate(seqs):
        l = t.size(0)
        if l:
            out[i, :l] = t
            mask[i, :l] = True
    return out, mask


def _recall(S: torch.Tensor, ks):
    """S [Nq, Nc] with aligned diagonal positives. Returns recall dict + ranks."""
    n = S.size(0)
    tgt = torch.arange(n, device=S.device)
    true_scores = S.gather(1, tgt[:, None])
    ranks = (S > true_scores).sum(dim=1) + 1
    out = {f"R@{k}": (ranks <= k).float().mean().item() for k in ks if k <= n}
    out["median_rank"] = float(ranks.median().item())
    out["mean_pos_score"] = float(true_scores.mean().item())
    return out, ranks


def score_and_write(args, device) -> None:
    shards = sorted(glob.glob(str(Path(args.shards_dir) / "shard.*.pt")))
    if not shards:
        raise RuntimeError(f"No shards in {args.shards_dir}; run generation first.")
    recs = []
    for sp in shards:
        recs.extend(torch.load(sp, map_location="cpu")["records"])
    n = len(recs)
    print(f"[rt][score] merged {n} records from {len(shards)} shards", flush=True)

    Z_gen, mask_gen = _pad_stack([r["z_gen"] for r in recs])
    Z_src, mask_src = _pad_stack([r["z_src"] for r in recs])
    Z_gen, mask_gen = Z_gen.to(device), mask_gen.to(device)
    Z_src, mask_src = Z_src.to(device), mask_src.to(device)

    # NxN FILIP (generated target rows x source cols), chunked.
    S = filip_score_matrix_chunked(Z_gen, Z_src, mask_gen, mask_src,
                                   chunk_rows=args.filip_chunk_rows)
    ks = [1, 5, 10]
    gen2src, ranks_g = _recall(S, ks)              # generated target -> its source
    src2gen, _ = _recall(S.t().contiguous(), ks)   # source -> its generated target

    # Ceiling: true targets -> sources, same scorer/candidate set.
    Z_true, mask_true = _pad_stack([r["z_true"] for r in recs])
    S_true = filip_score_matrix_chunked(Z_true.to(device), Z_src, mask_true.to(device),
                                        mask_src, chunk_rows=args.filip_chunk_rows)
    ceiling, _ = _recall(S_true, ks)

    tgt = "protein" if _target_is_protein(args.direction) else "text"
    src = "text" if _target_is_protein(args.direction) else "protein"
    metrics = {
        "n": n, "direction": args.direction,
        "retrieval_ckpt": args.retrieval_ckpt, "decoder_ckpt": args.decoder_ckpt,
        "num_samples": args.num_samples, "split": args.split,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        f"generated_{tgt}_to_{src}": gen2src,
        f"{src}_to_generated_{tgt}": src2gen,
        f"true_{tgt}_to_{src}_ceiling": ceiling,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "roundtrip_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ranks_cpu = ranks_g.cpu().tolist()
    pos = S.gather(1, torch.arange(n, device=S.device)[:, None]).squeeze(1).cpu().tolist()
    with open(out_dir / "roundtrip_pairs.tsv", "w") as ftsv:
        ftsv.write("id\trank\tscore\ttrue_len\tgen_len\ttrue_target\tgenerated_target\n")
        def _clean(s):  # keep each record on one TSV row
            return " ".join(s.split())
        for i, r in enumerate(recs):
            t_s, g_s = _clean(r["true_target"]), _clean(r["gen_target"])
            ftsv.write(f"{r['uid']}\t{ranks_cpu[i]}\t{pos[i]:.4f}\t"
                       f"{len(t_s)}\t{len(g_s)}\t{t_s}\t{g_s}\n")

    # FASTA only when the target modality is protein (text isn't folded).
    if _target_is_protein(args.direction):
        with open(out_dir / "roundtrip_sequences.fasta", "w") as ffa:
            for r in recs:
                ffa.write(f">{r['uid']}|true\n{_AA_ONLY.sub('', r['true_target'].upper())}\n")
                ffa.write(f">{r['uid']}|generated\n{_AA_ONLY.sub('', r['gen_target'].upper())}\n")

    print(f"[rt][score] wrote outputs to {out_dir}/", flush=True)
    print(f"[rt][score] generated {tgt}->{src}   {gen2src}", flush=True)
    print(f"[rt][score] {src}->generated {tgt}   {src2gen}", flush=True)
    print(f"[rt][score] CEILING true {tgt}->{src} {ceiling}", flush=True)


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="text2protein",
                    choices=["text2protein", "protein2text"])
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--cross-attn-every", type=int, default=cfg.generation.cross_attn_every)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--filip-chunk-rows", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--score-only", action="store_true",
                    help="skip generation; merge+score existing shards (single process)")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path(cfg.generation.ckpt_dir).parent.parent / "eval" / args.direction)
    if args.shards_dir is None:
        args.shards_dir = str(Path(args.out_dir) / "shards")

    torch.manual_seed(args.seed)

    if args.score_only:
        dev = init_distributed(args.device, group_size=1, init_pg=False).device
        score_and_write(args, dev)
        cleanup()
        return

    env = init_distributed(args.device, group_size=1, init_pg=False)

    pairs = load_pairs(
        cfg.data.csv_path, id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
    )
    splits = load_splits(str(Path(args.cache_dir) / "splits.json"))
    sel = list(splits[args.split])
    if args.num_samples > 0:
        sel = sel[:args.num_samples]

    generate_shard(args, cfg, env, sel, pairs)
    barrier()
    if env.is_main:
        score_and_write(args, env.device)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
