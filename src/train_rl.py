"""GRPO-style RL fine-tuning of the text->protein generator (SEPARATE TRACK).

Teacher-forced objectives (plain CE, target-input dropout, contrastive) can drive
the *teacher-forced* content-gain metric up without improving free-running
generation, because the true prefix is a confound: the model can predict the next
residue from its own prefix, or validate memory against it, instead of generating
FROM the conditioning. Round-trip retrieval — the honest, prefix-free metric —
stays on the floor.

This module optimizes that honest metric directly. It is intentionally a distinct
entry point from `train_generation.py`: it imports the same frozen building blocks
(retrieval heads, encoders, decoder+adapters) but shares no training code, so the
working protein2text SFT pipeline is untouched. It only trains the text2protein
direction.

Algorithm (GRPO-style, no value network):
  for each step:
    1. sample B source captions; build the aligned cross-attn memory (frozen).
    2. ROLLOUT (no grad): sample G proteins per caption -> B*G sequences.
    3. REWARD (no grad): re-encode each generated protein through the frozen
       protein encoder + projection, FILIP-score it against its own caption
       (the same 0.5*(p2t+t2p) max-sim the retrieval model uses). This is r_i.
    4. group-relative advantage A_i = (r_i - mean_g) / (std_g + eps), per caption.
    5. POLICY LOSS (grad): recompute log pi(a_t) over the generated tokens with a
       single teacher-forced forward; maximize sum_t A_i * log pi, plus an optional
       KL penalty to the frozen SFT reference (guards reward hacking / collapse)
       and an optional entropy bonus.

Every --eval-every steps (default 200, aligned with --save-every) the loop also runs
the held-out round-trip eval in-process: generate a FIXED val subset free-running,
re-encode, and score accession-grouped R@K + mAP over the whole eval pool with the
same code path as `python -m src.roundtrip_eval`. The training reward only compares a
generation to its OWN source, which a policy can raise by drifting toward one generic
high-scoring output; the pooled metric is what catches that. Rows are sharded across
ranks, so the cost is ~one extra rollout's generation per eval (<1% of wall time at
the default cadence). Results stream to stdout and to <ckpt_dir>/rl_roundtrip.jsonl.

Init from an SFT checkpoint (the cleanest-LM one — the unfrozen-only run, NOT the
contrastive one whose no-memory behaviour is damaged). The saved checkpoint mirrors
train_generation's adapter_state + meta, so roundtrip_eval / ablate_memory load it
unchanged.

Usage:
  mpiexec ... python -m src.train_rl \\
      --retrieval-ckpt checkpoints/retrieval/epoch50.pt \\
      --init-ckpt      checkpoints/generation/text2protein/epoch09.pt \\
      --device xpu --steps 2000 --prompts-per-rank 8 --group-size 8 \\
      --lr 1e-5 --kl-coef 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
from src.data import dense_group_ids, group_ids_from_accessions, load_pairs, load_splits
from src.decoder_adapters import (
    LoRACfg, clear_cross_memory, count_trainable, decode_target,
    load_decoder_with_cross_attn, set_cross_memory, target_prefix_ids,
    unfreeze_decoder_blocks,
)
from src.dist import barrier, cleanup, init_distributed
from src.encoders import (
    encode_protein_batch, encode_text_batch, load_protein_encoder, load_text_encoder,
)
from src.losses import filip_score_matrix_chunked
from src.roundtrip_eval import (
    load_retrieval, rng_restore, rng_snapshot, score_roundtrip_records,
)


# ---------------------------------------------------------------------------
# Reward: per-pair FILIP score (mirrors losses.filip_score_matrix, diagonal only)
# ---------------------------------------------------------------------------
def filip_per_pair(z_p: torch.Tensor, z_t: torch.Tensor,
                   mask_p: torch.Tensor, mask_t: torch.Tensor,
                   neg_inf: float = -1e4) -> torch.Tensor:
    """0.5*(p2t + t2p) max-sim for index-ALIGNED pairs. z_p [N,Lp,D], z_t [N,Lt,D]
    (both L2-normalized per token) -> [N]. Same definition as filip_score_matrix,
    but only the aligned pairs (no NxN matrix)."""
    sim = torch.einsum("nld,nmd->nlm", z_p, z_t)              # [N, Lp, Lt]
    mp, mt = mask_p[:, :, None], mask_t[:, None, :]
    sim = sim.masked_fill(~mp, neg_inf).masked_fill(~mt, neg_inf)
    max_p = sim.max(dim=2).values.masked_fill(~mask_p, 0.0)   # [N, Lp]
    s_p2t = max_p.sum(1) / mask_p.sum(1).clamp_min(1)
    max_t = sim.max(dim=1).values.masked_fill(~mask_t, 0.0)   # [N, Lt]
    s_t2p = max_t.sum(1) / mask_t.sum(1).clamp_min(1)
    # Clamp to the valid cosine-mean range. Normal scores already sit in [-1, 1];
    # this only catches the degenerate case where a generation encodes to ZERO
    # valid tokens (all special / field-label), for which the max over an
    # all-invalid axis leaks the -1e4 sentinel (~-5000 reward). Flooring it at -1
    # ("matched nothing") keeps a single junk sample from blowing up the group
    # std and washing out the real advantages — and makes true collapse legible.
    return (0.5 * (s_p2t + s_t2p)).clamp(-1.0, 1.0)           # [N]


# ---------------------------------------------------------------------------
# Log-prob / masking helpers over rolled-out sequences
# ---------------------------------------------------------------------------
def _gen_valid_mask(gen_tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """[N, T] True for generated positions up to AND INCLUDING the first EOS (the
    EOS action counts; everything after is padding). Robust to pad_id == eos_id,
    which ProtGPT3 uses — we key off first-EOS, not token identity."""
    T = gen_tokens.size(1)
    is_eos = gen_tokens == eos_id
    has = is_eos.any(dim=1)
    first = torch.where(has, is_eos.float().argmax(dim=1).long(),
                        torch.full((gen_tokens.size(0),), T - 1, device=gen_tokens.device))
    ar = torch.arange(T, device=gen_tokens.device)
    return ar[None, :] <= first[:, None]


def _token_logprobs(logits: torch.Tensor, seqs: torch.Tensor,
                    prompt_len: int) -> torch.Tensor:
    """log pi(a_t) for the generated tokens. logits [N, L, V], seqs [N, L] ->
    [N, T] where T = L - prompt_len. Position P-1+j predicts token at P+j."""
    P = prompt_len
    tgt = seqs[:, P:]                                    # [N, T]
    pred = logits[:, P - 1:P - 1 + tgt.size(1), :]       # [N, T, V]
    logp = torch.log_softmax(pred.float(), dim=-1)
    return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Model loading — mirror the SFT checkpoint's architecture so the policy IS the
# SFT model, continued. Also builds a frozen reference copy for the KL penalty.
# ---------------------------------------------------------------------------
def _build_decoder_from_ckpt(direction, decoder_path, ck, cfg, device):
    """Construct a decoder+adapters matching the flags stored in the SFT ckpt and
    load its adapter_state. Returns (decoder, tokenizer, adapters, meta)."""
    meta = {
        "cross_attn_every": ck.get("cross_attn_every", cfg.generation.cross_attn_every),
        "memory_space": ck.get("memory_space", "expanded"),
        "memory_map": ck.get("memory_map", False),
        "cross_attn_mode": ck.get("cross_attn_mode", "head"),
        "unfreeze_top": ck.get("unfreeze_top", 0),
        "unfreeze_where": ck.get("unfreeze_where", "top"),
        "warm_start_qalign": ck.get("warm_start_qalign", False),
        "lora_targets_self_attn": ck.get("lora_targets_self_attn", True),
        "lora_targets_ffn": ck.get("lora_targets_ffn", True),
    }
    if meta["memory_space"] != "aligned":
        raise SystemExit("train_rl currently supports --memory-space aligned checkpoints only")
    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
        target_self_attn=meta["lora_targets_self_attn"],
        target_ffn=meta["lora_targets_ffn"],
    )
    mem_dim = cfg.model.embed_dim   # aligned
    dec, dtok, adapters = load_decoder_with_cross_attn(
        direction, decoder_path, meta["cross_attn_every"], mem_dim, lora_cfg, device,
        cross_attn_mode=meta["cross_attn_mode"],
    )
    dec.load_state_dict(ck["adapter_state"], strict=False)
    if dtok.pad_token is None:
        dtok.pad_token = dtok.eos_token
    return dec, dtok, adapters, meta


def cosine_warmup_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# In-loop round-trip eval — the held-out version of the training reward
# ---------------------------------------------------------------------------
@torch.no_grad()
def roundtrip_eval_inloop(core, dtok, adapters, retrieval, pairs, eval_idx, prompt,
                          pad_id, enc_src, src_proj, enc_tgt, tgt_proj, get_src,
                          get_tgt, empty_tgt, direction, memory_map, embed_dim,
                          args, env, device, need_ceiling: bool):
    """Free-running round-trip retrieval over a FIXED val subset.

    Identical measurement to `python -m src.roundtrip_eval`: same generation path
    (cross-attn memory from the source, sampled decode), same FILIP scorer, same
    `grouped_recall` -> accession-grouped R@K + mAP. Run in-process, so every
    model is already resident and the only real cost is one extra `generate`.

    This is the honest counterpart to the training reward: the reward scores a
    generation against ITS OWN source only, which a degenerate policy can raise by
    drifting toward a generic high-scoring output. R@K/mAP score it against the
    whole eval pool, so that drift shows up as a flat or falling curve while the
    reward climbs.

    Rows are sharded round-robin across ranks (`eval_idx` is identical on every
    rank), so wall time is one SMALL-batch generation rather than one large one;
    the per-token embeddings are all-gathered (fp16 on the wire) and rank 0 scores
    the full N x N matrix.

    `need_ceiling` also re-encodes the TRUE targets for the ceiling row. The
    retrieval model is frozen and the eval set fixed, so that number never moves —
    pass True on the first call only and reuse the result.

    Returns (metrics, ceiling) on rank 0; (None, None) elsewhere.
    """
    my = eval_idx[env.rank::env.world_size]
    ks = (1, 5, 10)
    recs = []
    bs = max(args.eval_batch_size, 1)
    for s in range(0, len(my), bs):
        chunk = my[s:s + bs]
        h_src, mask_src = enc_src([get_src(pairs[i]) for i in chunk])
        z_src = src_proj(h_src.float())
        mem = retrieval.map_source(z_src, direction) if memory_map else z_src

        set_cross_memory(adapters, mem, mask_src)
        input_ids = torch.tensor([prompt] * len(chunk), device=device, dtype=torch.long)
        gen = core.generate(
            input_ids, max_new_tokens=args.eval_max_new_tokens,
            do_sample=args.eval_temperature > 0,
            temperature=max(args.eval_temperature, 1e-6),
            top_p=args.eval_top_p, pad_token_id=pad_id, use_cache=True)
        clear_cross_memory(adapters)

        gen_strs = [decode_target(core, dtok, row) for row in gen]
        h_gen, mask_gen = enc_tgt([g if g.strip() else empty_tgt for g in gen_strs])
        z_gen = tgt_proj(h_gen.float())
        if need_ceiling:
            h_true, mask_true = enc_tgt([get_tgt(pairs[i]) for i in chunk])
            z_true = tgt_proj(h_true.float())

        # Keep only the valid tokens, fp16 on CPU: the gather payload is ~200 KB
        # per row at fp32, and every rank receives the whole eval set.
        for b, i in enumerate(chunk):
            rec = {"uid": pairs[i].uid,
                   "z_gen": z_gen[b][mask_gen[b]].half().cpu(),
                   "z_src": z_src[b][mask_src[b]].half().cpu()}
            if need_ceiling:
                rec["z_true"] = z_true[b][mask_true[b]].half().cpu()
            recs.append(rec)

    if env.distributed and torch.distributed.is_initialized():
        buf = [None] * env.world_size
        torch.distributed.all_gather_object(buf, recs)
        recs = [r for part in buf for r in part]
    if not env.is_main:
        return None, None

    # Scoring is shared with the offline tool and the SFT trainer (accession ids
    # align rows/cols, so a protein's sibling captions count as positives).
    return score_roundtrip_records(
        recs, embed_dim, device, chunk_rows=args.eval_filip_chunk_rows,
        ks=ks, need_ceiling=need_ceiling)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="text2protein",
                    choices=["text2protein", "protein2text"],
                    help="text2protein: condition on caption, generate protein, reward = "
                         "FILIP(gen protein, caption). protein2text: the mirror.")
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--init-ckpt", required=True,
                    help="SFT text2protein checkpoint to start the policy from "
                         "(use the cleanest-LM one, not the contrastive one)")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--ckpt-dir", default=None,
                    help="output dir (default: <ckpt_dir>/<direction>_rl)")
    ap.add_argument("--resume", default=None,
                    help="continue a prior RL run: path to a stepNNNNNN.pt, or 'auto' to pick "
                         "the latest in --ckpt-dir. Restores policy weights, optimizer state, "
                         "and the step counter; the KL reference stays anchored to --init-ckpt "
                         "(the original SFT model), NOT the resumed policy")
    ap.add_argument("--steps", type=int, default=2000,
                    help="TOTAL RL update-step budget (not per-job). A resumed run continues "
                         "toward this total, so the LR schedule stays continuous across restarts")
    ap.add_argument("--prompts-per-rank", type=int, default=8,
                    help="B: distinct captions sampled per rank per step")
    ap.add_argument("--group-size", type=int, default=8,
                    help="G: proteins generated per caption (GRPO group). Rollout = B*G")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    # ---- length band on the reward (see the rollout loop for the rationale) ----
    ap.add_argument("--length-lambda", type=float, default=0.0,
                    help="weight on the two-sided length penalty (0 = off, reward is pure "
                         "FILIP). Only within-group spread survives GRPO's standardizer, so "
                         "size this against the logged rstd (within-group std of the FILIP "
                         "reward): lambda ~ 4.5*rstd makes a 2x length error cost about one "
                         "group-std at the default tolerance. Turn it up only if the length "
                         "distribution is still wrong once the SFT-side fixes are in")
    ap.add_argument("--length-tolerance", type=float, default=0.25,
                    help="dead zone: no penalty while L_gen is in "
                         "[L_true/(1+t), L_true*(1+t)] — log-symmetric, so 2x and 0.5x cost "
                         "the same. A caption constrains domain composition, not exact "
                         "length, so the band must be wide enough not to punish legitimate "
                         "variation")
    # ---- contrastive (margin) reward ----
    ap.add_argument("--reward-contrastive", action="store_true",
                    help="reward = FILIP(gen, own source) - beta * max_j FILIP(gen, other "
                         "source in batch), instead of the raw positive score. The raw score "
                         "is absolute and negative-free, so it can be raised by generating a "
                         "'hub' output that scores well against EVERY caption; GRPO's "
                         "standardizer cannot see this (it removes a per-prompt constant, and "
                         "hubness is a property of the sample). Retrieval trains a symmetric "
                         "InfoNCE and best_of_n selects on this same margin, so this makes the "
                         "RL objective, the inference-time selector, and the R@K/mAP eval "
                         "measure the same thing. Negatives are the other --prompts-per-rank "
                         "sources in the batch (7 at the default B=8) — a small panel")
    ap.add_argument("--reward-margin-beta", type=float, default=1.0,
                    help="weight on the hardest-negative term for --reward-contrastive. 0 "
                         "reproduces the raw positive score; 1.0 is the best_of_n margin")
    ap.add_argument("--lr", type=float, default=1e-5, help="adapter/head LR")
    ap.add_argument("--unfreeze-lr", type=float, default=None,
                    help="separate LR for the (SFT-)unfrozen decoder blocks; defaults to --lr")
    ap.add_argument("--kl-coef", type=float, default=0.05,
                    help="KL-to-SFT-reference penalty (0 = off, skips loading the reference). "
                         "Guards against reward hacking / policy collapse")
    ap.add_argument("--entropy-coef", type=float, default=0.0,
                    help="entropy bonus on the generated tokens (maintains exploration)")
    ap.add_argument("--adv-eps", type=float, default=1e-4,
                    help="epsilon in the group-relative advantage normalizer")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    # ---- held-out round-trip eval (same metric as src.roundtrip_eval) ----
    ap.add_argument("--eval-every", type=int, default=200,
                    help="run the held-out round-trip eval every N steps (0 = off). "
                         "Defaults to --save-every so every checkpoint has a metric")
    ap.add_argument("--eval-samples", type=int, default=1000,
                    help="val rows in the eval pool. Fixed across the run (--eval-seed), "
                         "so the curve is comparable step to step. Also the candidate-pool "
                         "size, so R@K/mAP are NOT comparable across different values")
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="per-rank generation batch during eval. The pool is sharded across "
                         "ranks, so this only bites on small worlds")
    ap.add_argument("--eval-temperature", type=float, default=None,
                    help="eval sampling temperature (default: --temperature, i.e. measure the "
                         "same distribution the policy is being trained on). 0 = greedy")
    ap.add_argument("--eval-top-p", type=float, default=None,
                    help="eval nucleus cutoff (default: --top-p)")
    ap.add_argument("--eval-max-new-tokens", type=int, default=None,
                    help="default: --max-new-tokens")
    ap.add_argument("--eval-filip-chunk-rows", type=int, default=4,
                    help="row chunk for the N x N FILIP scoring on rank 0; bounds the "
                         "transient (~chunk_rows*N*L_gen*L_src floats) on top of the "
                         "resident fp32 policy + KL reference")
    ap.add_argument("--eval-seed", type=int, default=1234,
                    help="picks the fixed eval subset and seeds eval sampling; the RL rollout "
                         "RNG stream is saved/restored around each eval")
    ap.add_argument("--eval-at-start", dest="eval_at_start", action="store_true",
                    help="also eval before the first step (baseline for the init/resumed policy)")
    ap.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    ap.set_defaults(eval_at_start=True)
    args = ap.parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 (group-relative advantage needs a group)")
    if args.eval_temperature is None:
        args.eval_temperature = args.temperature
    if args.eval_top_p is None:
        args.eval_top_p = args.top_p
    if args.eval_max_new_tokens is None:
        args.eval_max_new_tokens = args.max_new_tokens

    direction = args.direction
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else \
        Path(cfg.generation.ckpt_dir) / f"{direction}_rl"

    # Resolve --resume: 'auto' picks the latest stepNNNNNN.pt in ckpt_dir (fresh
    # start if none exist yet, so a re-queued PBS job just continues).
    resume_path = None
    if args.resume == "auto":
        existing = sorted(ckpt_dir.glob("step*.pt"))
        resume_path = existing[-1] if existing else None
    elif args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"--resume checkpoint not found: {resume_path}")

    env = init_distributed(args.device, group_size=1)
    device = env.device
    torch.manual_seed(args.seed + env.rank)   # per-rank RNG -> diverse rollouts

    # ---- frozen reward-side models + policy/reference decoders (rank-0-first) ----
    # text2protein decodes with the protein LM (ProtGPT3); protein2text with the
    # text LM (BioGPT). Both encoders are always loaded (source uses one, the reward
    # re-encodes the generated target with the other).
    decoder_path = cfg.generation.decoder_path if direction == "text2protein" else TEXT_DECODER_PATH

    def _load():
        ck = torch.load(args.init_ckpt, map_location="cpu")
        retr = load_retrieval(args.retrieval_ckpt, device,
                              with_maps=ck.get("memory_map", False))
        tmodel, ttok = load_text_encoder(
            cfg.model.text_encoder_path, device, cfg.data.caption_field_labels)
        pmodel, ptok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        dec, dtok, adapters, meta = _build_decoder_from_ckpt(
            direction, decoder_path, ck, cfg, device)
        ref = ref_adapters = None
        if args.kl_coef > 0:
            ref, _, ref_adapters, _ = _build_decoder_from_ckpt(
                direction, decoder_path, ck, cfg, device)
        return ck, retr, tmodel, ttok, pmodel, ptok, dec, dtok, adapters, meta, ref, ref_adapters

    if env.is_main:
        loaded = _load()
    barrier()
    if not env.is_main:
        loaded = _load()
    (ck, retrieval, text_model, text_tok, prot_model, prot_tok,
     decoder, dtok, adapters, meta, ref_decoder, ref_adapters) = loaded

    # Unfreeze the SAME blocks the SFT run trained (so the RL policy has the same
    # trainable set), and upcast to fp32 when doing so — bf16 master weights don't
    # train (updates fall below the bf16 ULP) and destabilize AdamW; see
    # train_generation for the full rationale.
    n_unfrozen = unfreeze_decoder_blocks(decoder, meta["unfreeze_top"], where=meta["unfreeze_where"])
    if n_unfrozen > 0:
        decoder.float()
    if ref_decoder is not None:
        for p in ref_decoder.parameters():
            p.requires_grad_(False)
        ref_decoder.eval()
        # Match the policy's dtype so KL(policy || ref) is ~0 at step 0 (same
        # weights) rather than an artifact of fp32-vs-bf16 precision.
        if n_unfrozen > 0:
            ref_decoder.float()

    # Resume: overlay the prior RL policy weights onto the (init-built) decoder,
    # AFTER unfreeze + fp32 upcast so shapes/dtypes match, and BEFORE the DDP wrap
    # so DDP broadcasts the resumed weights. The reference stays anchored to
    # --init-ckpt (loaded above), NOT this resumed policy. Optimizer state and the
    # step counter are restored further down.
    resume_ck = None
    start_step = 0
    if resume_path is not None:
        resume_ck = torch.load(resume_path, map_location=device)
        _, unexpected = decoder.load_state_dict(resume_ck["adapter_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"--resume unexpected keys: {unexpected}")
        start_step = int(resume_ck.get("step", 0))
        if env.is_main:
            print(f"[rl] resumed policy from {resume_path} at step {start_step} "
                  f"(KL reference stays anchored to {args.init_ckpt})", flush=True)

    # Frozen encode+project handles, per direction: the SOURCE builds the cross-attn
    # memory; the TARGET is what we generate and re-encode for the reward. The FILIP
    # reward is symmetric, so filip_per_pair(z_gen, z_src) is unchanged either way.
    def enc_text(strs):
        return encode_text_batch(text_model, text_tok, strs, device,
                                 cfg.data.max_text_tokens, mask_specials=True,
                                 mask_field_labels=True)

    def enc_prot(strs):
        return encode_protein_batch(prot_model, prot_tok, strs, device,
                                    cfg.data.max_protein_tokens, mask_specials=True)

    if direction == "text2protein":
        enc_src, src_proj = enc_text, retrieval.text_proj
        enc_tgt, tgt_proj = enc_prot, retrieval.protein_proj
        get_src, get_tgt, empty_tgt = (lambda p: p.text), (lambda p: p.protein), "M"
    else:
        enc_src, src_proj = enc_prot, retrieval.protein_proj
        enc_tgt, tgt_proj = enc_text, retrieval.text_proj
        get_src, get_tgt, empty_tgt = (lambda p: p.protein), (lambda p: p.text), "protein"

    # ---- data: sample captions from the train split ----
    pairs = load_pairs(
        cfg.data.csv_path, id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col)
    splits = load_splits(str(Path(args.cache_dir) / "splits.json"))
    train_idx = list(splits["train"])
    if env.is_main:
        print(f"[rl] direction={direction} | {len(train_idx)} train rows | policy trainable="
              f"{count_trainable(decoder):,} | unfrozen={n_unfrozen:,} | "
              f"fp32={n_unfrozen > 0} | KL={'on' if ref_decoder is not None else 'off'}",
              flush=True)

    # Fixed held-out eval pool: same rows every eval (and across restarts), drawn
    # from val with a dedicated generator so it's independent of --seed and of the
    # rollout stream. Identical on every rank — the eval shards it by rank.
    eval_idx: list = []
    if args.eval_every > 0 and args.eval_samples > 0:
        val_idx = list(splits["val"])
        g_ev = torch.Generator().manual_seed(args.eval_seed)
        perm = torch.randperm(len(val_idx), generator=g_ev).tolist()
        eval_idx = [val_idx[i] for i in perm[:args.eval_samples]]
        if env.is_main:
            print(f"[rl] round-trip eval: {len(eval_idx)} val rows every "
                  f"{args.eval_every} steps (temp={args.eval_temperature} "
                  f"top_p={args.eval_top_p} max_new={args.eval_max_new_tokens}); "
                  f"~{len(eval_idx) / max(env.world_size, 1):.1f} rows/rank", flush=True)

    # Per-rank shuffled caption stream (reshuffles when exhausted).
    g = torch.Generator().manual_seed(args.seed + 1234 + env.rank)
    order = [train_idx[i] for i in torch.randperm(len(train_idx), generator=g).tolist()]
    cursor = 0

    def next_prompts(n):
        nonlocal cursor, order
        if cursor + n > len(order):
            order = [train_idx[i] for i in torch.randperm(len(train_idx), generator=g).tolist()]
            cursor = 0
        batch = order[cursor:cursor + n]
        cursor += n
        return batch

    # ---- DDP + optimizer (two LR groups: heads vs unfrozen backbone) ----
    if env.distributed:
        ddp_ids = [device.index] if device.type in ("xpu", "cuda") else None
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=ddp_ids, find_unused_parameters=True)
    core = decoder.module if env.distributed else decoder
    # eval() everywhere: disables backbone dropout so the rollout distribution and
    # the recomputed log pi(a_t) match exactly (on-policy), and there's no dropout
    # noise in the reward. Regularization comes from the KL penalty, not dropout.
    decoder.eval()

    adapter_ids = {id(p) for a in adapters for p in a.parameters()}
    head_params, backbone_params = [], []
    for name, p in core.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in adapter_ids or "lora_" in name or "cross_attn" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    backbone_lr = args.unfreeze_lr if args.unfreeze_lr is not None else args.lr
    param_groups = [{"params": head_params, "base_lr": args.lr, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "base_lr": backbone_lr, "lr": backbone_lr})
    optim = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.0)
    train_params = head_params + backbone_params
    warmup = max(int(args.warmup_frac * args.steps), 1)
    # Resume the AdamW moments so updates stay well-conditioned across restarts
    # (the param-group structure is identical, since unfreeze matches). The custom
    # base_lr keys survive state_dict; the loop overwrites lr each step regardless.
    if resume_ck is not None and "optimizer_state" in resume_ck:
        optim.load_state_dict(resume_ck["optimizer_state"])

    bos = dtok.bos_token_id if dtok.bos_token_id is not None else dtok.eos_token_id
    pad_id = dtok.pad_token_id if dtok.pad_token_id is not None else dtok.eos_token_id
    eos_id = dtok.eos_token_id if dtok.eos_token_id is not None else pad_id
    prompt = [bos] + target_prefix_ids(core, dtok)
    P = len(prompt)
    B, G = args.prompts_per_rank, args.group_size

    ckpt_dir.mkdir(parents=True, exist_ok=True) if env.is_main else None
    barrier()

    def save(step):
        if not env.is_main:
            return
        trainable = {n for n, p in core.named_parameters() if p.requires_grad}
        adapter_state = {k: v for k, v in core.state_dict().items() if k in trainable}
        payload = {"step": step, "adapter_state": adapter_state,
                   "optimizer_state": optim.state_dict(),
                   "rl": True, "direction": direction, "init_ckpt": args.init_ckpt,
                   "kl_coef": args.kl_coef, "lr": args.lr,
                   # Rollout budget + reward shaping this policy was trained under:
                   # a checkpoint's length behaviour is only interpretable against them.
                   "rl_max_new_tokens": args.max_new_tokens,
                   "length_lambda": args.length_lambda,
                   "length_tolerance": args.length_tolerance,
                   # Which reward this policy climbed: raw positive FILIP vs the
                   # in-batch margin. They are on different scales, so a saved
                   # reward curve is meaningless without it.
                   "reward_contrastive": args.reward_contrastive,
                   "reward_margin_beta": (args.reward_margin_beta
                                          if args.reward_contrastive else None),
                   **meta}
        path = ckpt_dir / f"step{step:06d}.pt"
        torch.save(payload, path)
        print(f"[rl][ckpt] saved {path}", flush=True)

    # Held-out round-trip eval. The ceiling (true target -> source, same scorer and
    # candidate pool) depends only on the frozen retrieval model and the fixed eval
    # set, so it's computed once and reused — the "done" flag is tracked on EVERY
    # rank, not inferred from rank 0's returned ceiling (only rank 0 scores), or the
    # non-main ranks would keep encoding true targets forever.
    eval_state = {"ceiling": None, "done": False, "seconds": 0.0}

    def run_eval(step_id: int) -> None:
        if not eval_idx:
            return
        barrier()
        t_ev = time.time()
        rng = rng_snapshot(device)
        torch.manual_seed(args.eval_seed + env.rank)
        try:
            metrics, ceiling = roundtrip_eval_inloop(
                core, dtok, adapters, retrieval, pairs, eval_idx, prompt, pad_id,
                enc_src, src_proj, enc_tgt, tgt_proj, get_src, get_tgt, empty_tgt,
                direction, meta["memory_map"], cfg.model.embed_dim, args, env, device,
                need_ceiling=not eval_state["done"])
        finally:
            rng_restore(device, rng)
        if ceiling is not None:
            eval_state["ceiling"] = ceiling
        eval_state["done"] = True
        dt = time.time() - t_ev
        eval_state["seconds"] += dt
        if env.is_main:
            g2s, s2g = metrics["gen2src"], metrics["src2gen"]
            ceil = eval_state["ceiling"] or {}
            print(f"[rl][rt] step={step_id} n={metrics['n']} "
                  f"gen->src R@1={g2s['R@1']:.4f} R@10={g2s['R@10']:.4f} "
                  f"mAP={g2s['mAP']:.4f} medrank={g2s['median_rank']:.0f} | "
                  f"src->gen R@1={s2g['R@1']:.4f} mAP={s2g['mAP']:.4f} | "
                  f"ceiling R@1={ceil.get('R@1', float('nan')):.4f} "
                  f"mAP={ceil.get('mAP', float('nan')):.4f} | {dt:.1f}s", flush=True)
            with open(ckpt_dir / "rl_roundtrip.jsonl", "a") as f:
                f.write(json.dumps({"step": step_id, "seconds": round(dt, 2),
                                    "ceiling": eval_state["ceiling"], **metrics}) + "\n")
        barrier()

    # =====================================================================
    # RL loop
    # =====================================================================
    if args.eval_at_start:
        run_eval(start_step)
    t0 = time.time()
    eval_state["seconds"] = 0.0     # s/step below is reported EXCLUDING eval time
    for step in range(start_step, args.steps):
        frac = cosine_warmup_factor(step, args.steps, warmup)
        for grp in optim.param_groups:
            grp["lr"] = grp["base_lr"] * frac
        optim.zero_grad(set_to_none=True)

        prompt_idx = next_prompts(B)
        srcs = [get_src(pairs[i]) for i in prompt_idx]
        if args.reward_contrastive:
            # Per-batch group ids so the margin reward can drop false negatives.
            acc_g = torch.as_tensor(
                group_ids_from_accessions([pairs[i].uid for i in prompt_idx]),
                device=device)
            txt_g = torch.as_tensor(dense_group_ids(srcs), device=device)

        # ---- build aligned source memory (frozen), replicate G times ----
        with torch.no_grad():
            h_src, mask_src = enc_src(srcs)
            z_src = src_proj(h_src.float())                       # [B, Lt, D] normalized
            mem = retrieval.map_source(z_src, direction) if meta["memory_map"] else z_src
            mem_g = mem.repeat_interleave(G, dim=0)               # [B*G, Lt, D]
            mask_g = mask_src.repeat_interleave(G, dim=0)

        # ---- ROLLOUT: sample B*G proteins (no grad) ----
        with torch.no_grad():
            set_cross_memory(adapters, mem_g, mask_g)
            input_ids = torch.tensor([prompt] * (B * G), device=device, dtype=torch.long)
            gen = core.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=max(args.temperature, 1e-6),
                top_p=args.top_p, pad_token_id=pad_id, use_cache=True)
            clear_cross_memory(adapters)
        seqs = gen                                               # [B*G, P+T]
        gen_tokens = seqs[:, P:]
        gen_valid = _gen_valid_mask(gen_tokens, eos_id)          # [B*G, T]

        # ---- REWARD: re-encode generated proteins, FILIP vs source caption ----
        with torch.no_grad():
            gen_strs = [decode_target(core, dtok, row) for row in seqs]
            enc_in = [s if s.strip() else empty_tgt for s in gen_strs]
            h_gen, mask_gen = enc_tgt(enc_in)
            empty_rows = mask_gen.sum(dim=1) == 0             # gens with no valid tokens
            n_empty = int(empty_rows.sum())
            z_gen = tgt_proj(h_gen.float())                      # [B*G, Lp, D]
            z_src_g = z_src.repeat_interleave(G, dim=0)
            mask_src_g = mask_src.repeat_interleave(G, dim=0)
            if args.reward_contrastive:
                # [B*G, B]: every rollout scored against EVERY source in the batch.
                # z_src is already computed, so the negatives cost one wider FILIP
                # call (B columns instead of 1) and no extra encoder work.
                # Clamped for the same reason filip_per_pair clamps: a generation
                # that encodes to zero valid tokens leaks the -1e4 max sentinel,
                # which here would cancel between pos and neg and read as an
                # average sample rather than a collapsed one.
                S = filip_score_matrix_chunked(
                    z_gen, z_src, mask_gen, mask_src,
                    chunk_rows=args.eval_filip_chunk_rows).clamp(-1.0, 1.0)
                own = torch.arange(B, device=device).repeat_interleave(G)   # [B*G]
                filip_pos = S.gather(1, own[:, None]).squeeze(1)            # [B*G]

                # False negatives: next_prompts draws captions independently and the
                # augmented corpus carries ~8.87 captions per protein, so two prompts
                # in one batch can be siblings of the same accession (or byte-identical
                # generic captions). Penalizing a generation for matching its own
                # protein's other caption is backwards — mask those columns, mirroring
                # retrieval's mask_false_negatives. If every column of a row ends up
                # blocked (tiny B, all siblings) the negative term is the constant
                # -1.0, and a per-prompt constant is invisible to GRPO anyway.
                same = ((acc_g[:, None] == acc_g[None, :])
                        | (txt_g[:, None] == txt_g[None, :]))               # [B, B]
                hard_neg = S.masked_fill(same[own], -1.0).max(dim=1).values  # [B*G]
                filip_r = filip_pos - args.reward_margin_beta * hard_neg
                # A generation that encodes to zero valid tokens clamps to -1 in
                # EVERY column, so pos and neg cancel and the margin lands at
                # -1 + beta -> 0 at the default beta: mid-pack, and above a merely
                # generic sample. The floor has to be applied to the margin itself,
                # not inherited from the clamp. -(1 + beta) is the true range floor:
                # unambiguously worst, and bounded, so one junk rollout cannot blow
                # up the group std and wash out the real advantages.
                filip_r = torch.where(
                    empty_rows,
                    torch.full_like(filip_r, -(1.0 + args.reward_margin_beta)),
                    filip_r)
            else:
                filip_pos = filip_per_pair(z_gen, z_src_g, mask_gen, mask_src_g)
                hard_neg = None
                filip_r = filip_pos                                          # [B*G]

            # ---- two-sided length band (off at --length-lambda 0) ----
            # FILIP is not monotone in length, but it does leave one exploitable
            # channel: s_t2p is a max over protein positions, so DUPLICATING an
            # already-well-matching region is nearly free (the copy adds no new
            # per-position maxima to dilute s_p2t, and can only raise s_t2p).
            #
            # A one-sided correction can't fix that: score/L and score - lambda*L
            # are both monotone, so their optimum sits at zero or at the cap. This
            # penalizes |log(L_gen/L_true)| only OUTSIDE a +/-tolerance dead zone,
            # quadratically, so the optimum IS the band. The dead zone matters
            # because a caption fixes domain composition, not length — several
            # lengths are legitimately correct, and a hard target would train the
            # model to guess one reference protein's exact length.
            #
            # L_true is privileged training signal (unavailable at inference); what
            # the policy internalizes is "caption => approximate length". Lengths
            # are string lengths: residues for text2protein, characters for
            # protein2text, and only their ratio enters.
            len_gen = torch.tensor([max(len(s), 1) for s in gen_strs],
                                   device=device, dtype=torch.float32)
            len_true = torch.tensor(
                [max(len(get_tgt(pairs[i])), 1) for i in prompt_idx],
                device=device, dtype=torch.float32).repeat_interleave(G)
            len_log_ratio = torch.log(len_gen / len_true)            # [B*G]
            len_pen = (len_log_ratio.abs()
                       - math.log(1.0 + args.length_tolerance)).clamp_min(0.0) ** 2
            reward = filip_r - args.length_lambda * len_pen

            # Group-relative advantage (per caption). Note a per-prompt constant is
            # invisible here — only WITHIN-group spread survives standardization,
            # which is what sizes --length-lambda (see its help text).
            r = reward.view(B, G)
            adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + args.adv_eps)
            adv = adv.reshape(B * G)                             # [B*G]

        # ---- POLICY LOSS: recompute log pi over generated tokens (grad) ----
        attn = torch.cat([torch.ones(B * G, P, device=device, dtype=torch.long),
                          gen_valid.long()], dim=1)
        set_cross_memory(adapters, mem_g, mask_g)
        out = decoder(input_ids=seqs, attention_mask=attn)
        logp_tok = _token_logprobs(out.logits, seqs, P)          # [B*G, T]
        clear_cross_memory(adapters)

        denom = gen_valid.sum().clamp_min(1)
        pg_loss = -((adv[:, None] * logp_tok) * gen_valid).sum() / denom
        loss = pg_loss

        kl_val = torch.zeros((), device=device)
        if ref_decoder is not None:
            with torch.no_grad():
                set_cross_memory(ref_adapters, mem_g, mask_g)
                ref_out = ref_decoder(input_ids=seqs, attention_mask=attn)
                ref_logp_tok = _token_logprobs(ref_out.logits, seqs, P)
                clear_cross_memory(ref_adapters)
            # k3 KL estimator (positive, low variance); grad flows via logp_tok.
            log_ratio = ref_logp_tok - logp_tok
            kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
            kl_val = (kl_tok * gen_valid).sum() / denom
            loss = loss + args.kl_coef * kl_val

        ent_val = torch.zeros((), device=device)
        if args.entropy_coef > 0:
            ent_val = (-(logp_tok) * gen_valid).sum() / denom
            loss = loss - args.entropy_coef * ent_val

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        if torch.isfinite(total_norm):
            optim.step()
        elif env.is_main:
            print(f"[rl][warn] non-finite grad norm at step {step}; skipping", flush=True)

        if env.is_main and (step % args.log_every == 0 or step == args.steps - 1):
            gen_len = gen_valid.sum(1).float().mean().item()
            # rstd is the within-group std of the FILIP reward — the scale GRPO
            # actually sees, and what --length-lambda should be sized against.
            rstd = filip_r.view(B, G).std(dim=1).mean().item()
            len_ratio = (len_gen / len_true).median().item()
            # margin= is the headroom the contrastive reward buys: pos minus the
            # hardest in-batch negative. Flat-or-falling margin while pos climbs is
            # the hubness failure the raw reward can't see.
            margin_msg = ""
            if hard_neg is not None:
                margin_msg = (f"pos={filip_pos.mean().item():.4f} "
                              f"neg={hard_neg.mean().item():.4f} "
                              f"margin={(filip_pos - hard_neg).mean().item():.4f} ")
            print(f"[rl] step={step}/{args.steps} lr={optim.param_groups[0]['lr']:.2e} "
                  f"reward={reward.mean().item():.4f} (max {reward.max().item():.4f}) "
                  f"filip={filip_r.mean().item():.4f} {margin_msg}rstd={rstd:.4f} "
                  f"lenratio={len_ratio:.2f} lenpen={len_pen.mean().item():.4f} "
                  f"adv|={adv.abs().mean().item():.3f} kl={kl_val.item():.4f} "
                  f"ent={ent_val.item():.3f} genlen={gen_len:.0f} empty={n_empty}/{B*G} "
                  f"pg={pg_loss.item():.4f} "
                  f"{(time.time()-t0-eval_state['seconds'])/(step-start_step+1):.1f}s/step",
                  flush=True)

        if (step + 1) % args.save_every == 0:
            save(step + 1)
            barrier()

        # After the save, so each checkpoint on disk has a matching metric line.
        if args.eval_every > 0 and ((step + 1) % args.eval_every == 0
                                    or step + 1 == args.steps):
            run_eval(step + 1)

    save(args.steps)
    barrier()
    cleanup()
    if env.is_main:
        print("[rl] done", flush=True)


if __name__ == "__main__":
    main()
