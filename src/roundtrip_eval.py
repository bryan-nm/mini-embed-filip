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
  roundtrip_metrics.json     R@K + mAP (both directions) + ceiling + config
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
from src.best_of_n import pad_stack, select_best_of_n
from src.data import group_ids_from_accessions, load_pairs, load_splits
from src.decoder_adapters import (
    decode_target, load_decoder_with_cross_attn, set_cross_memory,
    clear_cross_memory, target_prefix_ids,
)
from src.dist import init_distributed, barrier, cleanup
from src.encoders import (
    encode_protein_batch, encode_text_batch,
    load_protein_encoder, load_text_encoder,
)
from src.evaluate import average_precision
from src.gen_ckpt import load_generation_ckpt
from src.losses import filip_score_matrix_chunked
from src.model import load_retrieval


_AA_ONLY = re.compile(r"[^A-Z]")


def _shard_range(n: int, rank: int, world: int) -> tuple[int, int]:
    per, rem = divmod(n, world)
    start = rank * per + min(rank, rem)
    return start, start + per + (1 if rank < rem else 0)


def _target_is_protein(direction: str) -> bool:
    return direction == "text2protein"


# ---------------------------------------------------------------------------
# Generation phase: each rank generates + encodes its slice -> shard file
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_shard(args, cfg, env, sel_indices, pairs, panel_indices=None) -> None:
    device = env.device
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    start, end = _shard_range(len(sel_indices), env.rank, env.world_size)
    my = sel_indices[start:end]
    if env.is_main:
        print(f"[rt] {len(sel_indices)} samples over {env.world_size} ranks; "
              f"direction={args.direction}", flush=True)
    print(f"[rt][rank {env.rank}] generating {len(my)} outputs on {device}", flush=True)

    decoder_path = (cfg.generation.decoder_path if _target_is_protein(args.direction)
                    else TEXT_DECODER_PATH)

    # Rank-0-first model load (trust_remote_code module-cache race for ProGen2).
    # The decoder is rebuilt from the checkpoint's own GenMeta header, so adapter
    # counts, k/v dims and scoring space line up with what it was trained as.
    def _load_models():
        ck, meta = load_generation_ckpt(args.decoder_ckpt)
        retr = load_retrieval(args.retrieval_ckpt, device)
        tmodel, ttok = load_text_encoder(
            cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
        pmodel, ptok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        dec, dtok, adapters = load_decoder_with_cross_attn(
            args.direction, decoder_path, meta.cross_attn_every,
            meta.mem_dim(cfg, args.direction), device,
            cross_attn_mode=meta.cross_attn_mode,
        )
        dec.load_state_dict(ck["adapter_state"], strict=False)
        dec.eval()
        if dtok.pad_token is None:
            dtok.pad_token = dtok.eos_token
        dtok.padding_side = "left"   # correct for batched decoder generation
        return retr, tmodel, ttok, pmodel, ptok, dec, dtok, adapters, meta

    if env.is_main:
        models = _load_models()
    barrier()
    if not env.is_main:
        models = _load_models()
    (retrieval, text_model, text_tok, prot_model, prot_tok, decoder, dtok,
     adapters, meta) = models
    if env.is_main:
        print(f"[rt] decoder ckpt architecture: {meta.describe()}", flush=True)

    # Direction-specific encode/project handles (source = conditioning input,
    # target = generated modality).
    def enc_text(strs):
        return encode_text_batch(text_model, text_tok, strs, device,
                                 cfg.data.max_text_tokens, mask_specials=True,
                                 mask_field_labels=True)

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
    # BOS + any decoder control tokens (ProtGPT3's direction marker), matching the
    # prompt the adapters were trained against.
    prompt = [bos] + target_prefix_ids(decoder, dtok)
    N = max(args.num_candidates, 1)

    # Reference panel for margin selection: source-modality embeddings of train
    # rows (disjoint from the scored set, so selection doesn't peek at the metric).
    z_panel = panel_mask = None
    if N > 1 and args.selection == "margin" and panel_indices:
        with torch.no_grad():
            h_pn, m_pn = enc_src([get_src(pairs[i]) for i in panel_indices])
            z_pn = src_proj(h_pn.float())
            z_panel, panel_mask = pad_stack(
                [z_pn[i][m_pn[i]].cpu() for i in range(len(panel_indices))],
                cfg.model.embed_dim, device)
        if env.is_main:
            print(f"[rt] best-of-{N} selection={args.selection} "
                  f"panel={len(panel_indices)}", flush=True)

    records = []
    bs = args.batch_size
    t0 = time.time()
    last_log = t0
    for s in range(0, len(my), bs):
        chunk = my[s:s + bs]
        srcs = [get_src(pairs[i]) for i in chunk]
        true_tgts = [get_tgt(pairs[i]) for i in chunk]
        uids = [pairs[i].uid for i in chunk]
        B = len(chunk)

        # Source -> embed_dim retrieval candidate + cross-attn memory. "aligned"
        # conditions on the projection directly; "expanded" lifts it back to
        # encoder-hidden space.
        h_src, mask_src = enc_src(srcs)
        z_src = src_proj(h_src.float())
        mem = z_src if meta.aligned_memory else src_expand(z_src)

        # N candidates per source: replicate the memory; temperature supplies the
        # diversity.
        mem_b = mem.repeat_interleave(N, dim=0)
        mask_b = mask_src.repeat_interleave(N, dim=0)
        set_cross_memory(adapters, mem_b, mask_b)
        input_ids = torch.tensor([prompt] * (B * N), device=device, dtype=torch.long)
        gen = decoder.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6),
            top_p=args.top_p, pad_token_id=pad_id, use_cache=True,
        )
        clear_cross_memory(adapters)
        cand_tgts = [decode_target(decoder, dtok, row) for row in gen]

        # Re-encode all B*N candidates -> embed_dim. Guard empty generations.
        enc_in = [t if t.strip() else empty_tgt for t in cand_tgts]
        h_gen, mask_gen = enc_tgt(enc_in)
        z_gen = tgt_proj(h_gen.float())
        h_true, mask_true = enc_tgt(true_tgts)
        z_true = tgt_proj(h_true.float())

        for b in range(B):
            # Select the best of this source's N candidates by round-trip margin.
            js = list(range(b * N, b * N + N))
            cand_z = [z_gen[j][mask_gen[j]].float().cpu() for j in js]
            if N > 1:
                zc, mc = pad_stack(cand_z, cfg.model.embed_dim, device)
                zs, ms = pad_stack([z_src[b][mask_src[b]].float().cpu()],
                                   cfg.model.embed_dim, device)
                best_j, _ = select_best_of_n(
                    zc, mc, zs, ms, z_panel, panel_mask, mode=args.selection)
            else:
                best_j = 0
            records.append({
                "uid": uids[b], "true_target": true_tgts[b],
                "gen_target": cand_tgts[js[best_j]],
                "z_gen": cand_z[best_j],                          # selected candidate
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
def rng_snapshot(device):
    """Save CPU + accelerator RNG so an in-loop eval's sampling doesn't perturb
    the training stream. Without this, turning the eval on (or changing its
    period) silently shifts every subsequent training batch — the augmentation
    coin flips, the input dropout, the rollout sampling — and runs stop being
    reproducible against a no-eval baseline."""
    dev_state = None
    try:
        if device.type == "xpu":
            dev_state = torch.xpu.get_rng_state(device)
        elif device.type == "cuda":
            dev_state = torch.cuda.get_rng_state(device)
    except Exception:
        dev_state = None
    return torch.get_rng_state(), dev_state


def rng_restore(device, state) -> None:
    cpu_state, dev_state = state
    torch.set_rng_state(cpu_state)
    if dev_state is None:
        return
    try:
        if device.type == "xpu":
            torch.xpu.set_rng_state(dev_state, device)
        elif device.type == "cuda":
            torch.cuda.set_rng_state(dev_state, device)
    except Exception:
        pass


def grouped_recall(S: torch.Tensor, groups: torch.Tensor, ks):
    """Accession-grouped retrieval recall + mAP.

    S [Nq, Nc] with query i and candidate i index-aligned. `groups` [N] holds
    each row's accession id; EVERY candidate sharing the query's accession is a
    positive, so a generated output re-embedding onto another row of its own
    source protein counts as a hit rather than a miss. At one caption per
    accession this reduces to diagonal recall; it stays multi-positive so a
    multi-caption corpus is scored correctly without a code change.

    mAP reuses `src.evaluate.average_precision` — the same ProTrek AP definition
    retrieval training reports — over the same multi-positive relevance set used
    for R@K here, so round-trip mAP is directly comparable to the retrieval
    model's own mAP (and to the ceiling row below).
    """
    n = S.size(0)
    same = groups[:, None] == groups[None, :]                  # [Nq, Nc] positives
    neg_inf = torch.finfo(S.dtype).min
    best_pos = S.masked_fill(~same, neg_inf).max(dim=1).values  # [Nq]
    # rank = 1 + (# non-positive candidates strictly outscoring the best positive)
    ranks = ((S > best_pos[:, None]) & (~same)).sum(dim=1) + 1
    out = {f"R@{k}": (ranks <= k).float().mean().item() for k in ks if k <= n}
    out["mAP"] = average_precision(S, same).mean().item()
    out["median_rank"] = float(ranks.median().item())
    out["mean_pos_score"] = float(best_pos.mean().item())
    return out, ranks


# ---------------------------------------------------------------------------
# In-loop round-trip: the generate -> re-encode -> record half
# ---------------------------------------------------------------------------
# `train_generation` and `train_rl` both run this metric inside their training
# loop, and they used to carry a copy each. The copies drifted (one guarded a
# missing R@10, the other crashed on it), so the shared piece lives here next to
# the scorer it feeds. The two callers still differ in where the SOURCE comes
# from — a packed-cache DataLoader for SFT, live encoder calls for RL — so that
# stays theirs: they hand this an iterable of already-projected source batches.
@torch.no_grad()
def generate_roundtrip_records(
    core, tokenizer, adapters, batches, *,
    prompt, pad_id, eos_id, tgt_proj, encode_generated, empty_target,
    max_new_tokens: int, temperature: float, top_p: float,
):
    """Generate free-running from each batch's memory and build round-trip records.

    `batches` yields dicts with:
        z_src      [B, L, D] aligned source projection (also the retrieval query)
        mem        [B, L, D] cross-attention memory (aligned z, or expanded)
        mask_src   [B, L] bool
        uids       list[str] of length B
        z_true     [B, Lt, D] optional — the true target's projection (ceiling row)
        mask_true  [B, Lt] bool, required when z_true is given

    Returns (records, gen_body_lens). Each record holds the fp16 CPU per-token
    embeddings the scorer needs; `gen_body_lens` is the generated body length in
    decoder tokens (prompt and EOS excluded), which catches a length distribution
    collapsing onto the cap — invisible in CE, obvious here.
    """
    recs, gen_lens = [], []
    for batch in batches:
        z_src, mem, mask_src = batch["z_src"], batch["mem"], batch["mask_src"]
        B = z_src.size(0)

        set_cross_memory(adapters, mem, batch.get("mem_mask", mask_src))
        input_ids = torch.tensor([prompt] * B, device=z_src.device, dtype=torch.long)
        # use_cache=True explicitly: a checkpointed trainer sets config.use_cache
        # False, and generating without a KV cache costs more than the epoch this
        # is measuring.
        gen = core.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-6),
            top_p=top_p, pad_token_id=pad_id, use_cache=True)
        clear_cross_memory(adapters)

        gen_strs = [decode_target(core, tokenizer, row) for row in gen]
        h_gen, m_gen = encode_generated(
            [g if g.strip() else empty_target for g in gen_strs])
        z_gen = tgt_proj(h_gen.float())

        # Body length: tokens before the first EOS, or the whole tail if none.
        tail = gen[:, len(prompt):]
        is_eos = tail == eos_id
        hit = is_eos.any(dim=1)
        first_eos = torch.where(
            hit, is_eos.float().argmax(dim=1).long(),
            torch.full((tail.size(0),), tail.size(1), device=tail.device,
                       dtype=torch.long))
        gen_lens.extend(first_eos.cpu().tolist())

        # fp16 on the wire: every rank receives the whole pool in the gather.
        for b in range(B):
            rec = {"uid": batch["uids"][b],
                   "z_gen": z_gen[b][m_gen[b]].half().cpu(),
                   "z_src": z_src[b][mask_src[b]].half().cpu()}
            if batch.get("z_true") is not None:
                rec["z_true"] = batch["z_true"][b][batch["mask_true"][b]].half().cpu()
            recs.append(rec)
    return recs, gen_lens


def gather_roundtrip_payloads(payload: dict, env) -> list:
    """All-gather one rank's eval payload; returns the per-rank list.

    Every rank receives every payload, so callers can agree on a decision (e.g.
    "some rank failed, disable the eval") without a second collective.
    """
    if env.distributed and torch.distributed.is_initialized():
        parts = [None] * env.world_size
        torch.distributed.all_gather_object(parts, payload)
        return parts
    return [payload]


@torch.no_grad()
def score_roundtrip_records(recs, embed_dim: int, device, chunk_rows: int = 4,
                            ks=(1, 5, 10), need_ceiling: bool = False):
    """Score a merged list of round-trip records -> (metrics, ceiling).

    Each record carries the per-token aligned embeddings of one row: `z_gen` (the
    generated target, re-encoded), `z_src` (its source) and, when `need_ceiling`,
    `z_true` (the true target). Any dtype is accepted (in-loop callers ship fp16
    over the wire); everything is upcast to fp32 here.

    This is the scoring half of the round-trip metric, shared by the offline tool
    and both in-loop trainers (`train_rl`, `train_generation`) so a curve logged
    during training is directly comparable to a `python -m src.roundtrip_eval`
    number computed on the same pool. `ceiling` is the true target scored against
    the same sources with the same scorer: it depends only on the frozen retrieval
    model and the (fixed) pool, so an in-loop caller computes it once and reuses it.
    """
    groups = torch.as_tensor(
        group_ids_from_accessions([r["uid"] for r in recs]), device=device)
    Z_src, m_src = pad_stack([r["z_src"].float() for r in recs], embed_dim, device)
    Z_gen, m_gen = pad_stack([r["z_gen"].float() for r in recs], embed_dim, device)

    S = filip_score_matrix_chunked(Z_gen, Z_src, m_gen, m_src, chunk_rows=chunk_rows)
    gen2src, _ = grouped_recall(S, groups, ks)              # generated target -> its source
    src2gen, _ = grouped_recall(S.t().contiguous(), groups, ks)
    del S

    ceiling = None
    if need_ceiling:
        Z_true, m_true = pad_stack([r["z_true"].float() for r in recs], embed_dim, device)
        S_true = filip_score_matrix_chunked(Z_true, Z_src, m_true, m_src,
                                            chunk_rows=chunk_rows)
        ceiling, _ = grouped_recall(S_true, groups, ks)
        del S_true

    return {"n": len(recs), "gen2src": gen2src, "src2gen": src2gen}, ceiling


def score_and_write(args, device) -> None:
    embed_dim = default_cfg().model.embed_dim
    shards = sorted(glob.glob(str(Path(args.shards_dir) / "shard.*.pt")))
    if not shards:
        raise RuntimeError(f"No shards in {args.shards_dir}; run generation first.")
    recs = []
    for sp in shards:
        recs.extend(torch.load(sp, map_location="cpu")["records"])
    n = len(recs)
    print(f"[rt][score] merged {n} records from {len(shards)} shards", flush=True)

    Z_gen, mask_gen = pad_stack([r["z_gen"] for r in recs], embed_dim)
    Z_src, mask_src = pad_stack([r["z_src"] for r in recs], embed_dim)
    Z_gen, mask_gen = Z_gen.to(device), mask_gen.to(device)
    Z_src, mask_src = Z_src.to(device), mask_src.to(device)

    # Accession ids align rows/cols of every matrix below, so the same vector
    # marks positives for gen->src, src->gen, and the ceiling.
    groups = torch.as_tensor(
        group_ids_from_accessions([r["uid"] for r in recs]), device=device)
    n_groups = int(groups.unique().numel())
    print(f"[rt][score] {n} rows over {n_groups} distinct proteins", flush=True)

    # NxN FILIP (generated target rows x source cols), chunked.
    S = filip_score_matrix_chunked(Z_gen, Z_src, mask_gen, mask_src,
                                   chunk_rows=args.filip_chunk_rows)
    ks = [1, 5, 10]
    gen2src, ranks_g = grouped_recall(S, groups, ks)              # generated target -> its source
    src2gen, _ = grouped_recall(S.t().contiguous(), groups, ks)   # source -> its generated target

    # Ceiling: true targets -> sources, same scorer/candidate set.
    Z_true, mask_true = pad_stack([r["z_true"] for r in recs], embed_dim)
    S_true = filip_score_matrix_chunked(Z_true.to(device), Z_src, mask_true.to(device),
                                        mask_src, chunk_rows=args.filip_chunk_rows)
    ceiling, _ = grouped_recall(S_true, groups, ks)

    tgt = "protein" if _target_is_protein(args.direction) else "text"
    src = "text" if _target_is_protein(args.direction) else "protein"
    metrics = {
        "n": n, "direction": args.direction,
        "retrieval_ckpt": args.retrieval_ckpt, "decoder_ckpt": args.decoder_ckpt,
        "num_samples": args.num_samples, "split": args.split,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "num_candidates": args.num_candidates, "selection": args.selection,
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
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the config cap for the GENERATED modality, so the "
                         "round-trip never scores a truncated output against a whole source")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--filip-chunk-rows", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    # Best-of-N: generate N per source, keep the best round-trip margin.
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--selection", choices=["margin", "pos"], default="margin")
    ap.add_argument("--panel-size", type=int, default=256,
                    help="train rows sampled as the selection reference panel")
    ap.add_argument("--score-only", action="store_true",
                    help="skip generation; merge+score existing shards (single process)")
    args = ap.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (cfg.data.max_protein_tokens
                               if _target_is_protein(args.direction)
                               else cfg.data.max_text_tokens)
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
    if not _target_is_protein(args.direction):
        # protein2text: the source is the protein, so on a multi-caption corpus
        # generation would repeat once per caption of it. Keep one row per
        # protein. A no-op at one caption per accession.
        seen, dedup = set(), []
        for i in sel:
            a = pairs[i].uid
            if a not in seen:
                seen.add(a)
                dedup.append(i)
        sel = dedup
    if args.num_samples > 0:
        sel = sel[:args.num_samples]

    # Margin-selection panel: train rows, disjoint from the scored split.
    panel_indices = None
    if args.num_candidates > 1 and args.selection == "margin" and args.panel_size > 0:
        panel_indices = list(splits["train"])[:args.panel_size]

    generate_shard(args, cfg, env, sel, pairs, panel_indices=panel_indices)
    barrier()
    if env.is_main:
        score_and_write(args, env.device)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
