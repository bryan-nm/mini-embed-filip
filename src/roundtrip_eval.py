"""Round-trip evaluation for text2protein generation.

The teacher-forced CE the trainer reports cannot tell you whether a *generated*
protein actually means its source caption. This does:

  caption  --(text enc -> text_proj -> text_expand -> ProGen2 decoder)-->  protein_gen
  protein_gen  --(SaAMPLIFY -> protein_proj)-->  z_p_gen  (per-token, 32-d)
  caption      --(BioLinkBERT -> text_proj)  -->  z_t      (per-token, 32-d)

Then it scores the FILIP late-interaction matrix between every generated
protein and every source caption and reports retrieval recall: does a generated
protein retrieve back the caption it was generated from, against all other
captions? High R@K means the conditioning genuinely steered generation.

Distributed like precompute (embarrassingly parallel): one rank per tile,
each generates+encodes a contiguous slice and writes a shard; rank 0 then
merges the shards, scores the full NxN FILIP matrix, and writes outputs.

Outputs (under --out-dir):
  roundtrip_metrics.json    R@K both directions + config
  roundtrip_pairs.tsv       id, rank, score, lengths, true & generated sequence
  roundtrip_sequences.fasta  >{id}|true and >{id}|generated, cleaned to A-Z,
                             ready to hand to a structure-prediction tool
                             (folding/comparison is out of scope for this repo)

Usage:
  mpiexec ... python -m src.roundtrip_eval \\
      --retrieval-ckpt checkpoints/retrieval/epoch50.pt \\
      --decoder-ckpt   checkpoints/generation/text2protein/epoch29.pt \\
      --num-samples 1000 --device xpu
  # re-score an existing set of shards without regenerating:
  python -m src.roundtrip_eval --retrieval-ckpt ... --decoder-ckpt ... --score-only
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

from config import default_cfg
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
    print(f"[rt][rank {env.rank}] generating {len(my)} proteins on {device}", flush=True)

    # Rank-0-first model load (trust_remote_code module-cache race for ProGen2).
    def _load_models():
        retr = load_retrieval(args.retrieval_ckpt, device)
        tmodel, ttok = load_text_encoder(cfg.model.text_encoder_path, device)
        pmodel, ptok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        lora_cfg = LoRACfg(
            rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
            dropout=cfg.generation.lora_dropout,
        )
        # Build the decoder with the SAME cross_attn_every the checkpoint was
        # trained with (stored in the ckpt), so adapter counts line up.
        ck = torch.load(args.decoder_ckpt, map_location="cpu")
        cae = ck.get("cross_attn_every", args.cross_attn_every)
        dec, dtok, adapters = load_decoder_with_cross_attn(
            args.direction, cfg.generation.decoder_path, cae,
            cfg.model.text_hidden, lora_cfg, device,
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

    bos = dtok.bos_token_id if dtok.bos_token_id is not None else dtok.eos_token_id
    pad_id = dtok.pad_token_id if dtok.pad_token_id is not None else dtok.eos_token_id

    records = []
    bs = args.batch_size
    t0 = time.time()
    for s in range(0, len(my), bs):
        chunk = my[s:s + bs]
        captions = [pairs[i].text for i in chunk]
        true_prots = [pairs[i].protein for i in chunk]
        uids = [pairs[i].uid for i in chunk]

        # Caption -> 32-d (retrieval candidate) + expanded cross-attn memory.
        h_t, mask_t = encode_text_batch(text_model, text_tok, captions, device,
                                        cfg.data.max_text_tokens, mask_specials=True)
        z_t = retrieval.text_proj(h_t.float())                  # [B, Lt, 32], normalized
        mem = retrieval.text_expand(z_t)                        # [B, Lt, 768]

        set_cross_memory(adapters, mem, mask_t)
        input_ids = torch.full((len(chunk), 1), bos, device=device, dtype=torch.long)
        gen = decoder.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6),
            top_p=args.top_p, pad_token_id=pad_id,
        )
        clear_cross_memory(adapters)
        gen_prots = [dtok.decode(row, skip_special_tokens=True).strip() for row in gen]

        # Re-encode generated proteins -> 32-d.  Guard empty generations.
        enc_in = [p if p.strip() else "M" for p in gen_prots]
        h_p, mask_p = encode_protein_batch(prot_model, prot_tok, enc_in, device,
                                           cfg.data.max_protein_tokens, mask_specials=True)
        z_p = retrieval.protein_proj(h_p.float())               # [B, Lp, 32], normalized

        # Also encode the TRUE proteins -> the round-trip ceiling (a perfect
        # generator can't beat the retrieval model's own true-protein match).
        h_pt, mask_pt = encode_protein_batch(prot_model, prot_tok, true_prots, device,
                                             cfg.data.max_protein_tokens, mask_specials=True)
        z_pt = retrieval.protein_proj(h_pt.float())

        for b in range(len(chunk)):
            records.append({
                "uid": uids[b], "true_protein": true_prots[b], "gen_protein": gen_prots[b],
                "zp": z_p[b][mask_p[b]].float().cpu(),          # [lp, 32] generated
                "zp_true": z_pt[b][mask_pt[b]].float().cpu(),   # [lp, 32] ground truth
                "zt": z_t[b][mask_t[b]].float().cpu(),          # [lt, 32] caption
            })

        if env.is_main and (s + bs >= len(my) or (time.time() - t0) > 20):
            done = min(s + bs, len(my))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[rt][rank 0] {done}/{len(my)}  {rate:.2f} prot/s "
                  f"eta={(len(my)-done)/max(rate,1e-6)/60:.1f} min", flush=True)
            t0_log = time.time()  # noqa: F841

    torch.save({"start": start, "records": records},
               shards_dir / f"shard.{env.rank:05d}.pt")
    print(f"[rt][rank {env.rank}] wrote {len(records)} records", flush=True)


# ---------------------------------------------------------------------------
# Scoring phase (one rank): merge shards -> FILIP retrieval -> outputs
# ---------------------------------------------------------------------------
def _pad_stack(seqs, dim=32):
    """List of [l, dim] -> ([N, Lmax, dim], bool mask [N, Lmax])."""
    n = len(seqs)
    lmax = max((t.size(0) for t in seqs), default=1)
    lmax = max(lmax, 1)
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
    true_scores = S.gather(1, tgt[:, None])                     # [N, 1]
    ranks = (S > true_scores).sum(dim=1) + 1                    # strict-greater rank
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

    Z_p, mask_p = _pad_stack([r["zp"] for r in recs])
    Z_t, mask_t = _pad_stack([r["zt"] for r in recs])
    Z_p, mask_p = Z_p.to(device), mask_p.to(device)
    Z_t, mask_t = Z_t.to(device), mask_t.to(device)

    # NxN FILIP score (generated protein rows x source caption cols), chunked.
    S = filip_score_matrix_chunked(Z_p, Z_t, mask_p, mask_t, chunk_rows=args.filip_chunk_rows)

    ks = [1, 5, 10]
    p2t, ranks_p2t = _recall(S, ks)            # generated protein -> its caption
    t2p, _ = _recall(S.t().contiguous(), ks)   # caption -> its generated protein

    # Ceiling: true proteins -> their captions, same scorer/candidate set.
    Z_pt, mask_pt = _pad_stack([r["zp_true"] for r in recs])
    S_true = filip_score_matrix_chunked(Z_pt.to(device), Z_t, mask_pt.to(device), mask_t,
                                        chunk_rows=args.filip_chunk_rows)
    ceiling, _ = _recall(S_true, ks)

    metrics = {
        "n": n, "direction": args.direction,
        "retrieval_ckpt": args.retrieval_ckpt, "decoder_ckpt": args.decoder_ckpt,
        "num_samples": args.num_samples, "split": args.split,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generated_protein_to_text": p2t, "text_to_generated_protein": t2p,
        "true_protein_to_text_ceiling": ceiling,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "roundtrip_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Paired sequences: TSV (with the round-trip rank per item) + clean FASTA.
    ranks_cpu = ranks_p2t.cpu().tolist()
    pos_scores = S.gather(1, torch.arange(n, device=S.device)[:, None]).squeeze(1).cpu().tolist()
    with open(out_dir / "roundtrip_pairs.tsv", "w") as ftsv, \
         open(out_dir / "roundtrip_sequences.fasta", "w") as ffa:
        ftsv.write("id\tp2t_rank\tp2t_score\ttrue_len\tgen_len\ttrue_protein\tgenerated_protein\n")
        for i, r in enumerate(recs):
            true_s, gen_s = r["true_protein"], r["gen_protein"]
            ftsv.write(f"{r['uid']}\t{ranks_cpu[i]}\t{pos_scores[i]:.4f}\t"
                       f"{len(true_s)}\t{len(gen_s)}\t{true_s}\t{gen_s}\n")
            ffa.write(f">{r['uid']}|true\n{_AA_ONLY.sub('', true_s.upper())}\n")
            ffa.write(f">{r['uid']}|generated\n{_AA_ONLY.sub('', gen_s.upper())}\n")

    print(f"[rt][score] wrote outputs to {out_dir}/", flush=True)
    print(f"[rt][score] generated protein->text  {p2t}", flush=True)
    print(f"[rt][score] text->generated protein  {t2p}", flush=True)
    print(f"[rt][score] CEILING true protein->text {ceiling}", flush=True)


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="text2protein", choices=["text2protein"],
                    help="only text2protein round-trip is implemented")
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(Path(cfg.generation.ckpt_dir).parent.parent
                                             / "eval" / "text2protein"))
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
    if args.shards_dir is None:
        args.shards_dir = str(Path(args.out_dir) / "shards")

    torch.manual_seed(args.seed)

    if args.score_only:
        dev = init_distributed(args.device, group_size=1, init_pg=False).device
        score_and_write(args, dev)
        cleanup()
        return

    env = init_distributed(args.device, group_size=1, init_pg=False)

    # Same split as training (cache splits.json is keyed on the full dataset).
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
