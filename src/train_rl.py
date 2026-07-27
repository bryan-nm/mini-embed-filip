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
from src.data import load_pairs, load_splits
from src.decoder_adapters import (
    LoRACfg, clear_cross_memory, count_trainable, decode_target,
    load_decoder_with_cross_attn, set_cross_memory, target_prefix_ids,
    unfreeze_decoder_blocks,
)
from src.dist import barrier, cleanup, init_distributed
from src.encoders import (
    encode_protein_batch, encode_text_batch, load_protein_encoder, load_text_encoder,
)
from src.roundtrip_eval import load_retrieval


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
    return 0.5 * (s_p2t + s_t2p)                              # [N]


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
                    help="output dir (default: <ckpt_dir>/text2protein_rl)")
    ap.add_argument("--steps", type=int, default=2000, help="number of RL update steps")
    ap.add_argument("--prompts-per-rank", type=int, default=8,
                    help="B: distinct captions sampled per rank per step")
    ap.add_argument("--group-size", type=int, default=8,
                    help="G: proteins generated per caption (GRPO group). Rollout = B*G")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
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
    args = ap.parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 (group-relative advantage needs a group)")

    direction = args.direction
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else \
        Path(cfg.generation.ckpt_dir) / f"{direction}_rl"

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
        get_src, empty_tgt = (lambda p: p.text), "M"
    else:
        enc_src, src_proj = enc_prot, retrieval.protein_proj
        enc_tgt, tgt_proj = enc_text, retrieval.text_proj
        get_src, empty_tgt = (lambda p: p.protein), "protein"

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
                   "rl": True, "direction": direction, "init_ckpt": args.init_ckpt,
                   "kl_coef": args.kl_coef, "lr": args.lr, **meta}
        path = ckpt_dir / f"step{step:06d}.pt"
        torch.save(payload, path)
        print(f"[rl][ckpt] saved {path}", flush=True)

    # =====================================================================
    # RL loop
    # =====================================================================
    t0 = time.time()
    for step in range(args.steps):
        frac = cosine_warmup_factor(step, args.steps, warmup)
        for grp in optim.param_groups:
            grp["lr"] = grp["base_lr"] * frac
        optim.zero_grad(set_to_none=True)

        prompt_idx = next_prompts(B)
        srcs = [get_src(pairs[i]) for i in prompt_idx]

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
            z_gen = tgt_proj(h_gen.float())                      # [B*G, Lp, D]
            z_src_g = z_src.repeat_interleave(G, dim=0)
            mask_src_g = mask_src.repeat_interleave(G, dim=0)
            reward = filip_per_pair(z_gen, z_src_g, mask_gen, mask_src_g)   # [B*G]
            # Group-relative advantage (per caption).
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
            print(f"[rl] step={step}/{args.steps} lr={optim.param_groups[0]['lr']:.2e} "
                  f"reward={reward.mean().item():.4f} (max {reward.max().item():.4f}) "
                  f"adv|={adv.abs().mean().item():.3f} kl={kl_val.item():.4f} "
                  f"ent={ent_val.item():.3f} genlen={gen_len:.0f} "
                  f"pg={pg_loss.item():.4f} {(time.time()-t0)/(step+1):.1f}s/step", flush=True)

        if (step + 1) % args.save_every == 0:
            save(step + 1)
            barrier()

    save(args.steps)
    barrier()
    cleanup()
    if env.is_main:
        print("[rl] done", flush=True)


if __name__ == "__main__":
    main()
