"""Per-direction generation trainer.

Reads a trained retrieval checkpoint (projection + expansion heads), runs the
appropriate encoder→project→expand front-end to produce per-token cross-
attention memory, then trains the decoder's cross-attention adapters + LoRA
on cross-entropy of the target sequence.

What's frozen vs trainable:
  frozen:   encoder, projection head, expansion head, decoder backbone
  trains:   cross-attention adapter layers + LoRA on decoder self-attn / FFN

Directions:
  text2protein:  text -> BioLinkBERT -> proj -> expand -> Dayhoff-170M -> protein
  protein2text:  protein -> AMPLIFY-350M -> proj -> expand -> BioGPT     -> text

Usage:
  python -m src.train_generation --direction text2protein \\
      --retrieval-ckpt checkpoints/retrieval/epoch04.pt --use-cache
  python -m src.train_generation --direction protein2text \\
      --retrieval-ckpt checkpoints/retrieval/epoch04.pt --use-cache
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
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, Cfg, TEXT_DECODER_PATH
from src.data import (
    PackedPerTokenCache,
    Pair,
    build_or_load_splits,
    group_ids_from_accessions,
    load_pairs,
    load_row_protein_idx,
    load_splits,
)
from src.dist import (
    init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
)
from src.decoder_adapters import (
    LoRACfg,
    clear_cross_memory,
    count_trainable,
    load_decoder_with_cross_attn,
    set_cross_memory,
    unfreeze_top_blocks,
)
from src.cvae import build_cvae, beta_at
from src.losses import masked_mean
from src.model import MiniEmbedFilip


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset: one side per-token cache, other side raw target text for the decoder
# ---------------------------------------------------------------------------
class GenerationDataset(Dataset):
    """For direction='text2protein':
       inputs from text cache (per-token text encoder hidden states),
       targets are raw protein sequences (tokenized later by the protein
       decoder's tokenizer).

       For direction='protein2text':
       inputs from protein cache, targets are raw text annotations.

    Protein-side access always goes through `row_protein_idx` (CSV row -> unique
    protein row), because the protein cache is deduplicated. When `need_target` is
    set (CVAE training), the opposite modality's per-token hidden states are also
    returned so the posterior q(w|source,target) can pool the target embedding.
    """

    def __init__(self, direction: str, cache_dir: str, pairs, indices,
                 protein_dim: int, text_dim: int, need_target: bool = False,
                 row_protein_idx=None):
        if direction not in ("text2protein", "protein2text"):
            raise ValueError(direction)
        self.direction = direction
        self.pairs = pairs
        self.indices = list(indices)
        self.need_target = need_target
        self.text_cache = PackedPerTokenCache(cache_dir, "text", text_dim)
        self.protein_cache = PackedPerTokenCache(cache_dir, "protein", protein_dim)
        if row_protein_idx is None:
            row_protein_idx = load_row_protein_idx(cache_dir)
        self.row_protein_idx = row_protein_idx

    def __len__(self):
        return len(self.indices)

    def _protein(self, idx: int):
        return self.protein_cache.get(int(self.row_protein_idx[idx]))

    def _text(self, idx: int):
        return self.text_cache.get(idx)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        if self.direction == "text2protein":
            h, m = self._text(idx)                       # source = text
            target = self.pairs[idx].protein
        else:
            h, m = self._protein(idx)                    # source = protein
            target = self.pairs[idx].text
        item = {"h": h, "m": m, "target": target, "idx": idx}
        if self.need_target:
            # Opposite modality (the generated side) for the CVAE posterior.
            h_t, m_t = (self._protein(idx) if self.direction == "text2protein"
                        else self._text(idx))
            item["h_tgt"] = h_t
            item["m_tgt"] = m_t
        return item


def make_collate(target_tokenizer, max_target_tokens: int):
    # ProGen2's tokenizer ships without a pad token; fall back to EOS.
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    pad_id = target_tokenizer.pad_token_id
    bos_id = target_tokenizer.bos_token_id
    eos_id = target_tokenizer.eos_token_id

    # We add BOS/EOS explicitly rather than relying on the tokenizer's
    # `add_special_tokens`: Dayhoff's char tokenizer never adds them (its
    # build_inputs_with_special_tokens is a no-op), which would leave the model
    # with no EOS to learn termination and a BOS-mismatch vs inference (which
    # seeds generation with BOS). Doing it here is uniform across decoders.
    body_cap = max_target_tokens - (1 if bos_id is not None else 0) - (1 if eos_id is not None else 0)

    def collate(batch):
        B = len(batch)
        L_h = max(b["h"].size(0) for b in batch)
        d_h = batch[0]["h"].size(-1)
        h_pad = torch.zeros(B, L_h, d_h, dtype=torch.bfloat16)
        m_pad = torch.zeros(B, L_h, dtype=torch.bool)
        for i, b in enumerate(batch):
            l = b["h"].size(0)
            h_pad[i, :l] = b["h"]
            m_pad[i, :l] = b["m"]

        # Tokenize bodies without specials, then wrap with [BOS] ... [EOS].
        seqs = []
        for b in batch:
            body = target_tokenizer(
                b["target"], add_special_tokens=False, truncation=True,
                max_length=body_cap,
            )["input_ids"]
            ids = ([bos_id] if bos_id is not None else []) + body + \
                  ([eos_id] if eos_id is not None else [])
            seqs.append(ids)

        L = max(len(s) for s in seqs)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(B, L, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, :len(s)] = 1

        # Labels = input_ids; pad positions become -100 (ignored by CE).
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        out = {
            "h": h_pad, "m": m_pad,
            "input_ids": input_ids, "attn_mask": attn_mask, "labels": labels,
        }

        # Optional target-side per-token hidden states (CVAE posterior). Padded
        # to the batch max; pooled through the frozen target projection later.
        if "h_tgt" in batch[0]:
            L_t = max(b["h_tgt"].size(0) for b in batch)
            d_t = batch[0]["h_tgt"].size(-1)
            ht_pad = torch.zeros(B, L_t, d_t, dtype=torch.bfloat16)
            mt_pad = torch.zeros(B, L_t, dtype=torch.bool)
            for i, b in enumerate(batch):
                lt = b["h_tgt"].size(0)
                ht_pad[i, :lt] = b["h_tgt"]
                mt_pad[i, :lt] = b["m_tgt"]
            out["h_tgt"] = ht_pad
            out["m_tgt"] = mt_pad

        return out

    return collate


# ---------------------------------------------------------------------------
def load_retrieval_model(ckpt_path: str, device: torch.device) -> MiniEmbedFilip:
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


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["text2protein", "protein2text"], required=True)
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--device", default=cfg.generation.device)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--ckpt-dir", default=cfg.generation.ckpt_dir)
    ap.add_argument("--batch-size", type=int, default=cfg.generation.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.generation.epochs)
    ap.add_argument("--lr", type=float, default=cfg.generation.lr)
    ap.add_argument("--cross-attn-every", type=int, default=cfg.generation.cross_attn_every)
    ap.add_argument("--unfreeze-top", type=int, default=0,
                    help="fully fine-tune the top N decoder blocks (0 = adapters/LoRA only)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute layer activations in backward to fit large decoders "
                         "(e.g. the 3B Dayhoff/Jamba); ~30%% slower, much less memory. "
                         "NOTE: incompatible with the Jamba MoE on XPU — routing isn't "
                         "bit-reproducible on recompute, so non-reentrant checkpoint aborts")
    ap.add_argument("--max-target-tokens", type=int, default=cfg.generation.max_target_tokens,
                    help="cap on the generated/teacher-forced target length; lower it to "
                         "cut decoder activation memory (truncates long targets)")
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate on the first N val pairs each epoch (0 = full val split)")
    # Generation-side CVAE (Feature 1).
    ap.add_argument("--use-cvae", action="store_true", default=cfg.generation.use_cvae,
                    help="train a conditional VAE latent injected as extra cross-attn memory tokens")
    ap.add_argument("--cvae-d-w", type=int, default=cfg.generation.cvae_d_w)
    ap.add_argument("--cvae-n-latent-tokens", type=int,
                    default=cfg.generation.cvae_n_latent_tokens)
    ap.add_argument("--cvae-beta-max", type=float, default=cfg.generation.cvae_beta_max)
    args = ap.parse_args()

    # Fold CVAE CLI overrides back into the config block build_cvae reads.
    cfg.generation.use_cvae = args.use_cvae
    cfg.generation.cvae_d_w = args.cvae_d_w
    cfg.generation.cvae_n_latent_tokens = args.cvae_n_latent_tokens
    cfg.generation.cvae_beta_max = args.cvae_beta_max
    cfg.generation.max_target_tokens = args.max_target_tokens

    env = init_distributed(args.device, group_size=1)
    device = env.device
    torch.manual_seed(args.seed)
    if env.is_main:
        print(f"[gen] direction={args.direction} world_size={env.world_size} device={device}")

    # 1) Retrieval model (frozen) — provides projection + expansion
    retrieval = load_retrieval_model(args.retrieval_ckpt, device)
    if env.is_main:
        print(f"[gen] loaded retrieval model from {args.retrieval_ckpt}")

    # 2) Decoder + adapters
    decoder_path = (
        cfg.generation.decoder_path
        if args.direction == "text2protein"
        else TEXT_DECODER_PATH
    )
    # Memory dim = encoder hidden dim of the INPUT modality (what the expansion
    # head outputs). For text2protein, input is text -> text_expand -> 768-d.
    mem_dim = cfg.model.text_hidden if args.direction == "text2protein" else cfg.model.protein_hidden

    # Gradient checkpointing recomputes the forward during backward, but dropout
    # RNG is not reliably preserved across recomputation on XPU. A non-zero LoRA
    # dropout then makes the recomputed hidden states differ from the original,
    # which — through Jamba's MoE router — changes per-expert token counts and
    # trips the non-reentrant checkpoint metadata check. Zero it for checkpointed
    # runs (negligible regularization on a frozen-backbone finetune); the
    # cross-attn adapter dropout is already 0.
    if args.grad_checkpointing and cfg.generation.lora_dropout > 0:
        if env.is_main:
            print(f"[gen] grad-checkpointing: forcing LoRA dropout "
                  f"{cfg.generation.lora_dropout} -> 0 (dropout RNG not preserved on recompute)")
        cfg.generation.lora_dropout = 0.0

    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
        target_self_attn=cfg.generation.lora_targets_self_attn,
        target_ffn=cfg.generation.lora_targets_ffn,
    )
    # trust_remote_code (ProGen2) compiles the custom modeling file into a
    # *shared* transformers_modules cache. If all ranks do this at once they
    # race on the write and some import a half-defined module ("has no attribute
    # ProGenForCausalLM"). Rank 0 populates the cache first; the barrier then
    # releases the others to import it read-only (concurrent reads are safe).
    def _load_decoder():
        return load_decoder_with_cross_attn(
            args.direction, decoder_path, args.cross_attn_every, mem_dim, lora_cfg, device,
        )

    decoder = target_tok = adapters = None
    if env.is_main:
        decoder, target_tok, adapters = _load_decoder()
    barrier()
    if not env.is_main:
        decoder, target_tok, adapters = _load_decoder()
    assert decoder is not None and adapters is not None  # always loaded on both branches

    # Optional partial unfreeze of the top decoder blocks (all ranks, identically,
    # before DDP captures requires_grad state).
    n_unfrozen = unfreeze_top_blocks(decoder, args.direction, args.unfreeze_top)
    if env.is_main and args.unfreeze_top > 0:
        print(f"[gen] unfroze top {args.unfreeze_top} decoder blocks "
              f"({n_unfrozen:,} params now trainable)")
    if env.is_main:
        n_train = count_trainable(decoder)
        print(f"[gen] decoder trainable params (cross-attn + LoRA): {n_train:,}")
        print(f"[gen] num cross-attention adapters: {len(adapters)}")

    # Gradient checkpointing: recompute layer activations in backward instead of
    # storing them across the deep stack. Essential for the 3B Jamba decoder,
    # whose pure-PyTorch Mamba scan materializes large fp32 [B, d_inner, L,
    # d_state] tensors per layer that otherwise OOM the tile.
    #   - use_reentrant=False is REQUIRED: the backbone is frozen (only adapters/
    #     LoRA train), and reentrant checkpointing demands an input that requires
    #     grad, which a frozen layer input doesn't have.
    #   - Incompatible with the KV/SSM cache, so disable use_cache for training.
    # Applied before the DDP wrap so `decoder` is still the raw model. For the
    # Jamba path the cross-attn adapters are forward hooks inside the layer
    # __call__, so they fall inside the checkpointed region and recompute
    # correctly; the ProGen2/BioGPT paths replace the block module (those decoders
    # are small and don't need this).
    if args.grad_checkpointing:
        decoder.config.use_cache = False
        if hasattr(decoder, "gradient_checkpointing_enable"):
            decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if env.is_main:
                print("[gen] gradient checkpointing ON "
                      "(use_reentrant=False, use_cache=False)")
        elif env.is_main:
            print("[gen] WARNING: decoder has no gradient_checkpointing_enable(); "
                  "skipping (--grad-checkpointing had no effect)")

    # Direction-specific frozen retrieval handles: source side (project+expand for
    # the cross-attn memory) and target projection (for the CVAE posterior).
    if args.direction == "text2protein":
        src_proj, src_expand, tgt_proj = (
            retrieval.text_proj, retrieval.text_expand, retrieval.protein_proj)
    else:
        src_proj, src_expand, tgt_proj = (
            retrieval.protein_proj, retrieval.protein_expand, retrieval.text_proj)

    # 2b) Generation-side CVAE heads (Feature 1). Trained alongside the adapters;
    # injected as extra cross-attention memory tokens.
    cvae = None
    n_latent = 0
    if args.use_cvae:
        cvae = build_cvae(cfg.generation, mem_dim, cfg.model.embed_dim).to(device)
        n_latent = cvae.cfg.n_latent_tokens
        if env.distributed:
            broadcast_parameters(cvae)            # all ranks start from rank-0 weights
        if env.is_main:
            n_cvae = sum(p.numel() for p in cvae.parameters())
            print(f"[gen] CVAE enabled: d_w={cvae.cfg.d_w} "
                  f"latent_tokens={n_latent} beta_max={cvae.cfg.beta_max} "
                  f"({n_cvae:,} params)")

    # 3) Data
    pairs = load_pairs(
        cfg.data.csv_path,
        id_col=cfg.data.csv_id_col,
        protein_col=cfg.data.csv_protein_col,
        text_col=cfg.data.csv_text_col,
        pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.subset_size,
    )
    splits_path = Path(args.cache_dir) / "splits.json"
    # Group-aware (by-accession) split, matching retrieval. Rank 0 owns creation;
    # everyone reads after a barrier (no write race). Reuses the retrieval split
    # when the cache's splits.json is already present and valid.
    if env.is_main:
        group_ids = group_ids_from_accessions([p.uid for p in pairs])
        splits = build_or_load_splits(
            str(splits_path), len(pairs), cfg.data.splits, args.seed, group_ids=group_ids)
        print(f"[gen] splits: {len(pairs)} rows over {splits['n_groups']} proteins")
    barrier()
    splits = load_splits(str(splits_path))

    # `val` is a random permutation slice, so the first N is a deterministic
    # random sample; keeps the per-epoch rank-0 eval cheap at full scale.
    val_indices = splits["val"]
    if args.val_subset and args.val_subset > 0:
        val_indices = val_indices[:args.val_subset]

    # Only train needs the target side (CVAE posterior); val conditions on the
    # prior mean, so it never loads the target modality.
    train_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                                 splits["train"],
                                 cfg.model.protein_hidden, cfg.model.text_hidden,
                                 need_target=args.use_cvae)
    val_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                               val_indices,
                               cfg.model.protein_hidden, cfg.model.text_hidden,
                               need_target=False)

    collate = make_collate(target_tok, cfg.generation.max_target_tokens)
    if env.distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=env.world_size, rank=env.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=collate,
                                  drop_last=True, num_workers=cfg.generation.num_workers)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate, drop_last=True,
                                  num_workers=cfg.generation.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate,
                            num_workers=cfg.generation.num_workers)
    if len(train_loader) == 0:
        raise RuntimeError(
            f"train_loader has 0 batches (per-rank train_size≈"
            f"{len(train_ds)//max(env.world_size,1)}, bs={args.batch_size}). "
            f"Lower --batch-size or use fewer ranks."
        )
    if env.is_main:
        print(f"[gen] split sizes: train={len(splits['train'])} "
              f"val={len(splits['val'])} (eval on {len(val_indices)})")
        print(f"[gen] batches/epoch/rank: train={len(train_loader)} val={len(val_loader)}")

    # Data-parallel DDP over the decoder. Many decoder blocks have no cross-attn
    # adapter and stay frozen, so allow unused params.
    if env.distributed:
        ddp_ids = [device.index] if device.type in ("xpu", "cuda") else None
        decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=True)
    core = decoder.module if env.distributed else decoder

    # 4) Optim — decoder adapters/LoRA (+ unfrozen blocks) and the CVAE heads.
    train_params = [p for p in decoder.parameters() if p.requires_grad]
    if cvae is not None:
        train_params += list(cvae.parameters())
    optim = torch.optim.AdamW(
        train_params, lr=args.lr, weight_decay=cfg.generation.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup = max(int(cfg.generation.warmup_frac * total_steps), 1)

    # 5) Train loop
    ckpt_dir = Path(args.ckpt_dir) / args.direction
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    global_step = 0
    log = []
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        decoder.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            lr = cosine_warmup_lr(global_step, total_steps, warmup, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad(set_to_none=True)

            h = batch["h"].to(device).float()
            mask = batch["m"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["labels"].to(device)

            # Encoder front-end (frozen): project then expand the source side.
            with torch.no_grad():
                z = src_proj(h)
                mem = src_expand(z)

            kl = torch.zeros((), device=device)
            if cvae is not None:
                # Posterior sees pooled source + target (both frozen embeddings);
                # sampled latent -> extra cross-attn memory tokens. KL pulls the
                # posterior toward the learned conditional prior p(w|source).
                with torch.no_grad():
                    z_src_pool = masked_mean(z, mask)
                    h_tgt = batch["h_tgt"].to(device).float()
                    m_tgt = batch["m_tgt"].to(device)
                    z_tgt_pool = masked_mean(tgt_proj(h_tgt), m_tgt)
                qmu, qlv = cvae.posterior_params(z_src_pool, z_tgt_pool)
                pmu, plv = cvae.prior_params(z_src_pool)
                w = cvae.reparam(qmu, qlv)
                w_tok = cvae.latent_tokens(w)                # [B, k, mem_dim], carries grad
                mem = torch.cat([mem, w_tok], dim=1)
                k_mask = torch.ones(mem.size(0), n_latent, dtype=torch.bool, device=device)
                mask = torch.cat([mask, k_mask], dim=1)
                kl = cvae.kl(qmu, qlv, pmu, plv)

            set_cross_memory(adapters, mem, mask)
            out = decoder(input_ids=input_ids, attention_mask=attn_mask)

            logits = out.logits                              # [B, L, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            beta = beta_at(global_step, total_steps, cvae.cfg) if cvae is not None else 0.0
            loss = ce + beta * kl
            loss.backward()
            # Clear AFTER backward, not before: with gradient checkpointing the
            # decoder layers (and their cross-attn adapter hooks) are recomputed
            # during backward, so the memory must still be set then. Clearing it
            # before backward made the recomputed forward take the adapters'
            # pass-through branch, tripping the checkpoint tensor-count check.
            # (Harmless without checkpointing — next step's set_cross_memory
            # overwrites it.)
            clear_cross_memory(adapters)
            # DDP syncs the decoder grads; the CVAE heads live outside its forward,
            # so average their grads manually (mirrors the retrieval trainer).
            if cvae is not None:
                average_gradients(cvae)
            torch.nn.utils.clip_grad_norm_(train_params, cfg.generation.grad_clip)
            optim.step()
            global_step += 1

            if env.is_main and ((it + 1) % cfg.generation.log_every == 0 or it == 0):
                with torch.no_grad():
                    ppl = torch.exp(ce).item()
                msg = (f"[{args.direction}] epoch={epoch} step={it+1}/{len(train_loader)} "
                       f"lr={lr:.2e} ce={ce.item():.4f} ppl={ppl:.2f}")
                if cvae is not None:
                    msg += f" kl={kl.item():.4f} beta={beta:.3f}"
                print(msg, flush=True)

        dt = time.time() - t0
        if env.is_main:
            print(f"[{args.direction}] epoch={epoch} done in {dt:.1f}s")

        # Val pass + checkpoint on rank 0 only (uses the unwrapped decoder).
        if env.is_main:
            core.eval()
            val_losses = []
            if cvae is not None:
                cvae.eval()
            with torch.no_grad():
                for batch in val_loader:
                    h = batch["h"].to(device).float()
                    mask = batch["m"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attn_mask"].to(device)
                    labels = batch["labels"].to(device)
                    z = src_proj(h)
                    mem = src_expand(z)
                    # Inference-time conditioning: prior mean (no target available).
                    if cvae is not None:
                        pmu, _ = cvae.prior_params(masked_mean(z, mask))
                        mem = torch.cat([mem, cvae.latent_tokens(pmu)], dim=1)
                        k_mask = torch.ones(mem.size(0), n_latent, dtype=torch.bool, device=device)
                        mask = torch.cat([mask, k_mask], dim=1)
                    set_cross_memory(adapters, mem, mask)
                    out = core(input_ids=input_ids, attention_mask=attn_mask)
                    clear_cross_memory(adapters)
                    logits = out.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    val_losses.append(loss.item())
            if cvae is not None:
                cvae.train()
            val_ce = sum(val_losses) / max(len(val_losses), 1)
            print(f"[val] epoch={epoch} ce={val_ce:.4f} ppl={math.exp(val_ce):.2f}")
            log.append({"epoch": epoch, "val_ce": val_ce, "val_ppl": math.exp(val_ce)})

            # Save every trainable tensor (cross-attn + LoRA + any unfrozen
            # blocks), keyed by requires_grad so partial-unfreeze weights persist.
            trainable = {n for n, p in core.named_parameters() if p.requires_grad}
            adapter_state = {k: v for k, v in core.state_dict().items() if k in trainable}
            payload = {"epoch": epoch, "adapter_state": adapter_state,
                       "cross_attn_every": args.cross_attn_every,
                       "unfreeze_top": args.unfreeze_top}
            # CVAE heads (Feature 1); absent => downstream loads run without a latent.
            if cvae is not None:
                payload["cvae_state"] = cvae.state_dict()
                payload["cvae_cfg"] = {
                    "d_w": cvae.cfg.d_w, "n_latent_tokens": cvae.cfg.n_latent_tokens,
                    "hidden": cvae.cfg.hidden, "mem_dim": cvae.cfg.mem_dim,
                }
            path = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save(payload, path)
            print(f"[ckpt] saved {path}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[gen] done")
    cleanup()


if __name__ == "__main__":
    main()
