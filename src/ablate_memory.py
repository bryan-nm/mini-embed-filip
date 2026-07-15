"""Conditioning ablation: how much is the cross-attention memory worth, and why?

Runs the generation validation cross-entropy over the *same* val batches under
four conditions, so the total trained gain can be decomposed:

  conditioned      — the trained encoder->project->expand memory (+ CVAE prior
                     mean) set on the adapters, exactly as in training. LoRA on.
  shuffled-memory  — same, but each target is paired with a DIFFERENT row's source
                     memory (batch roll). Memory is present but its CONTENT is
                     wrong. LoRA on.
  memory-off       — no memory set, so every adapter takes its pass-through branch
                     (`CrossAttentionAdapter.forward` returns the hidden state
                     unchanged when `self.memory is None`). LoRA on.
  base             — memory off AND LoRA disabled (every LoRALinear scaling -> 0):
                     the pristine frozen ProtGPT3, conditioning on the target prefix
                     alone. This is the true unconditional baseline.

Decomposition (in CE nats; lower is better):
  lora gain      = base        - memory_off   how much LoRA domain-adaptation alone
                                              explains (text-independent).
  presence gain  = memory_off  - shuffled     benefit from memory merely being
                                              present, regardless of content.
  content gain   = shuffled    - conditioned  the ACTUAL conditioning signal: does
                                              the RIGHT caption help vs a wrong one.

Reading it:
  * content gain ~ 0  => the decoder ignores which caption it is given; whatever
    val improvement happened was LoRA (and/or a degenerate memory-presence effect),
    not conditioning. This is the failure mode we suspect.
  * content gain > 0, presence gain ~ 0 => healthy: the model uses memory content.
  * presence gain > 0 while content gain ~ 0 => suspicious: the model exploits
    memory being present but not its content (check source<->target alignment and
    the bf16 residual cast in the adapter).

Single process — the val split is small, so this needs no MPI launch; point
--device at one GPU (or cpu). Use a larger --val-subset than training's default
(100 is too noisy to separate, say, 8.08 from 8.30).

Usage:
  python -m src.ablate_memory --direction text2protein \\
      --retrieval-ckpt checkpoints/reconstruction/epoch09.pt \\
      --decoder-ckpt checkpoints/generation/text2protein/epoch11.pt \\
      --device xpu --val-subset 2000
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
from src.cvae import load_cvae
from src.data import (
    build_or_load_splits,
    group_ids_from_accessions,
    load_pairs,
    load_splits,
)
from src.decoder_adapters import (
    LoRACfg,
    LoRALinear,
    clear_cross_memory,
    load_decoder_with_cross_attn,
    set_cross_memory,
    target_prefix_ids,
)
from src.losses import masked_mean
from src.train_generation import (
    GenerationDataset,
    make_collate,
    load_retrieval_model,
    pick_device,
)


@contextlib.contextmanager
def _lora_disabled(model, enabled: bool):
    """When `enabled` is False, temporarily zero every LoRALinear's scaling so the
    low-rank update contributes nothing, restoring it on exit. Lets a single loaded
    model produce both the with-LoRA and pristine-backbone perplexities."""
    if enabled:
        yield
        return
    saved = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            saved.append((m, m.scaling))
            m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in saved:
            m.scaling = s


def _run_val(decoder, adapters, val_loader, device, *, memory_mode,
             src_proj, src_expand, cvae, n_latent):
    """One full pass over val_loader. Returns (tw_ppl, mm_ppl, tw_ce, mm_ce).

    memory_mode:
      "conditioned" — set the row's own source memory on the adapters.
      "shuffled"    — set another row's memory (batch roll): present but wrong.
      "off"         — never set memory; adapters pass through.

    token-weighted (tw) is the correct estimate; mean-of-means (mm) mirrors the
    training-time [val] print so the conditioned number is directly comparable to
    the training log.
    """
    ce_sum = 0.0            # sum of per-token CE over all non-pad target tokens
    tok_total = 0           # number of non-pad target tokens
    batch_means = []        # per-batch mean CE (matches the training val print)

    with torch.no_grad():
        for batch in val_loader:
            h = batch["h"].to(device).float()
            mask = batch["m"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["labels"].to(device)

            if memory_mode == "off":
                clear_cross_memory(adapters)   # adapters -> pass-through
            else:
                if memory_mode == "shuffled" and h.size(0) > 1:
                    # Pair each target with the previous row's source conditioning.
                    # Rolling h+mask *before* project/expand means the deterministic
                    # memory AND the CVAE prior latent both come from the wrong row,
                    # while input_ids/labels (the target) stay put.
                    h = torch.roll(h, shifts=1, dims=0)
                    mask = torch.roll(mask, shifts=1, dims=0)
                z = src_proj(h)
                mem = src_expand(z)
                if cvae is not None:
                    # Inference-time conditioning = prior mean (no target available),
                    # exactly as the training val loop does it.
                    pmu, _ = cvae.prior_params(masked_mean(z, mask))
                    mem = torch.cat([mem, cvae.latent_tokens(pmu)], dim=1)
                    k_mask = torch.ones(mem.size(0), n_latent, dtype=torch.bool,
                                        device=device)
                    mask = torch.cat([mask, k_mask], dim=1)
                set_cross_memory(adapters, mem, mask)

            out = decoder(input_ids=input_ids, attention_mask=attn_mask)
            clear_cross_memory(adapters)

            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            mean_ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            sum_ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100,
                                     reduction="sum")
            n_tok = (flat_labels != -100).sum().item()

            batch_means.append(mean_ce.item())
            ce_sum += sum_ce.item()
            tok_total += n_tok

    tw_ce = ce_sum / max(tok_total, 1)
    mm_ce = sum(batch_means) / max(len(batch_means), 1)
    return math.exp(tw_ce), math.exp(mm_ce), tw_ce, mm_ce


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["text2protein", "protein2text"], required=True)
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True,
                    help="a generation epochNN.pt (its adapter_state / cvae_state / "
                         "cross_attn_every are loaded)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--batch-size", type=int, default=cfg.generation.batch_size)
    ap.add_argument("--val-subset", type=int, default=2000,
                    help="evaluate on the first N val pairs (0 = full val split)")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[ablate] direction={args.direction} device={device}")

    # 1) Frozen retrieval front-end (projection + expansion heads).
    retrieval = load_retrieval_model(args.retrieval_ckpt, device)
    if args.direction == "text2protein":
        src_proj, src_expand = retrieval.text_proj, retrieval.text_expand
        decoder_path = cfg.generation.decoder_path
        mem_dim = cfg.model.text_hidden
    else:
        src_proj, src_expand = retrieval.protein_proj, retrieval.protein_expand
        decoder_path = TEXT_DECODER_PATH
        mem_dim = cfg.model.protein_hidden

    # 2) Decoder + adapters, restored from the generation checkpoint. cross_attn_every
    # must match what the checkpoint was trained with (stored in it) so adapter
    # counts line up; adapter_state loads strict=False against the frozen backbone.
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    cae = ckpt.get("cross_attn_every", cfg.generation.cross_attn_every)
    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
    )
    decoder, target_tok, adapters = load_decoder_with_cross_attn(
        args.direction, decoder_path, cae, mem_dim, lora_cfg, device)
    missing, unexpected = decoder.load_state_dict(ckpt["adapter_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in adapter_state: {unexpected}")
    decoder.eval()
    print(f"[ablate] loaded decoder ckpt {args.decoder_ckpt} "
          f"(cross_attn_every={cae}, {len(adapters)} adapters)")

    # CVAE heads (frozen) if the checkpoint carries them, mirroring generation.
    cvae = load_cvae(ckpt, cfg.model.embed_dim, device)
    n_latent = cvae.cfg.n_latent_tokens if cvae is not None else 0
    print(f"[ablate] CVAE: {'on (prior mean)' if cvae is not None else 'off'}")

    # 3) Val data — the SAME group-aware split the trainer used (read splits.json
    # from the cache; build it if somehow absent).
    pairs = load_pairs(
        cfg.data.csv_path, id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col, text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col, subset_size=args.subset_size)
    splits_path = Path(args.cache_dir) / "splits.json"
    if splits_path.exists():
        splits = load_splits(str(splits_path))
    else:
        group_ids = group_ids_from_accessions([p.uid for p in pairs])
        splits = build_or_load_splits(
            str(splits_path), len(pairs), cfg.data.splits, args.seed, group_ids=group_ids)

    val_indices = splits["val"]
    if args.val_subset and args.val_subset > 0:
        val_indices = val_indices[:args.val_subset]

    val_ds = GenerationDataset(args.direction, args.cache_dir, pairs, val_indices,
                               cfg.model.protein_hidden, cfg.model.text_hidden,
                               need_target=False)
    prefix_ids = target_prefix_ids(decoder, target_tok)
    collate = make_collate(target_tok, cfg.generation.max_target_tokens, prefix_ids)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=cfg.generation.num_workers)
    print(f"[ablate] val pairs evaluated: {len(val_indices)} "
          f"({len(val_loader)} batches)")

    # 4) Four passes over the identical batches (val_loader is shuffle=False, so
    # every pass sees the same batches in the same order — comparisons are exact).
    def run(memory_mode, lora_on, label):
        with _lora_disabled(decoder, lora_on):
            tw, mm, tw_ce, _ = _run_val(
                decoder, adapters, val_loader, device, memory_mode=memory_mode,
                src_proj=src_proj, src_expand=src_expand, cvae=cvae, n_latent=n_latent)
        print(f"  {label:<24} ce={tw_ce:.4f}  ppl={tw:.3f}", flush=True)
        return mm, tw_ce

    print("\n[ablate] running passes...")
    cond_mm, cond_ce = run("conditioned", True, "conditioned")
    _, shuf_ce = run("shuffled", True, "shuffled-memory")
    _, off_ce = run("off", True, "memory-off (LoRA on)")
    _, base_ce = run("off", False, "base (LoRA off)")

    # Decompose the total base->conditioned improvement into its sources. CE is in
    # nats and additive, so the three gains sum to (base_ce - cond_ce).
    lora_gain = base_ce - off_ce        # unconditional LoRA domain adaptation
    presence_gain = off_ce - shuf_ce    # memory present at all (content-independent)
    content_gain = shuf_ce - cond_ce    # the RIGHT caption vs a wrong one — real conditioning
    total = base_ce - cond_ce

    print("\n[ablate] ===== decomposition (CE nats, positive = improvement) =====")
    print(f"  conditioned mean-of-means ppl = {cond_mm:.3f}   (compare to training [val] print)")
    print(f"  lora gain      (base    -> off)      = {lora_gain:+.4f}")
    print(f"  presence gain  (off     -> shuffled) = {presence_gain:+.4f}")
    print(f"  content gain   (shuffled-> cond)     = {content_gain:+.4f}   <-- the conditioning signal")
    print(f"  total          (base    -> cond)     = {total:+.4f}")
    if total > 1e-6:
        print(f"                 shares: lora={100*lora_gain/total:.0f}%  "
              f"presence={100*presence_gain/total:.0f}%  content={100*content_gain/total:.0f}%")

    print("\n[ablate] ===== verdict =====")
    if content_gain < 0.02:
        print("  content gain ~ 0: the decoder IGNORES which caption it is given. Any")
        print("  val improvement is LoRA domain-adaptation (and/or a content-free memory-")
        print("  presence effect), NOT conditioning.")
        if presence_gain > 0.05:
            print("  presence gain is non-trivial while content gain is ~0 => the model")
            print("  exploits memory being present but not its content. Suspicious: check")
            print("  source<->target alignment and the bf16 residual cast in the adapter.")
    elif content_gain < 0.15:
        print("  content gain is small but non-zero: conditioning is landing weakly.")
    else:
        print("  content gain is clear: the RIGHT caption meaningfully lowers perplexity —")
        print("  the conditioning is working.")


if __name__ == "__main__":
    main()
