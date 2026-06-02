"""Per-direction generation trainer.

Reads a trained retrieval checkpoint (projection + expansion heads), runs the
appropriate encoder→project→expand front-end to produce per-token cross-
attention memory, then trains the decoder's cross-attention adapters + LoRA
on cross-entropy of the target sequence.

What's frozen vs trainable:
  frozen:   encoder, projection head, expansion head, decoder backbone
  trains:   cross-attention adapter layers + LoRA on decoder self-attn / FFN

Directions:
  text2protein:  text -> BioLinkBERT -> proj -> expand -> ProGen2 -> protein
  protein2text:  protein -> SaAMPLIFY -> proj -> expand -> BioGPT  -> text

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

from config import default_cfg, Cfg
from src.data import (
    PackedPerTokenCache,
    Pair,
    load_pairs,
    load_splits,
    make_splits,
    save_splits,
    splits_are_valid,
)
from src.dist import init_distributed, barrier, cleanup
from src.decoder_adapters import (
    LoRACfg,
    clear_cross_memory,
    count_trainable,
    load_decoder_with_cross_attn,
    set_cross_memory,
)
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
    """

    def __init__(self, direction: str, cache_dir: str, pairs, indices,
                 protein_dim: int, text_dim: int):
        self.direction = direction
        self.pairs = pairs
        self.indices = list(indices)
        if direction == "text2protein":
            self.input_cache = PackedPerTokenCache(cache_dir, "text", text_dim)
        elif direction == "protein2text":
            self.input_cache = PackedPerTokenCache(cache_dir, "protein", protein_dim)
        else:
            raise ValueError(direction)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        h, m = self.input_cache.get(idx)                 # bf16, bool
        pair = self.pairs[idx]
        target = pair.protein if self.direction == "text2protein" else pair.text
        return {"h": h, "m": m, "target": target, "idx": idx}


def make_collate(target_tokenizer, max_target_tokens: int):
    # ProGen2's tokenizer ships without a pad token; fall back to EOS.
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    pad_id = target_tokenizer.pad_token_id
    bos_id = target_tokenizer.bos_token_id
    eos_id = target_tokenizer.eos_token_id

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

        # Tokenize targets. Add BOS/EOS, pad to max.
        tok = target_tokenizer(
            [b["target"] for b in batch],
            padding=True, truncation=True, max_length=max_target_tokens,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"]
        attn_mask = tok["attention_mask"]

        # Labels = input_ids shifted; positions with pad become -100 (ignored by CE).
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        return {
            "h": h_pad, "m": m_pad,
            "input_ids": input_ids, "attn_mask": attn_mask, "labels": labels,
        }

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
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate on the first N val pairs each epoch (0 = full val split)")
    args = ap.parse_args()

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
        else "/Users/bryan/Documents/models/biogpt"
    )
    # Memory dim = encoder hidden dim of the INPUT modality (what the expansion
    # head outputs). For text2protein, input is text -> text_expand -> 768-d.
    mem_dim = cfg.model.text_hidden if args.direction == "text2protein" else cfg.model.protein_hidden

    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
        target_self_attn=cfg.generation.lora_targets_self_attn,
        target_ffn=cfg.generation.lora_targets_ffn,
    )
    decoder, target_tok, adapters = load_decoder_with_cross_attn(
        args.direction, decoder_path, args.cross_attn_every, mem_dim, lora_cfg, device,
    )
    if env.is_main:
        n_train = count_trainable(decoder)
        print(f"[gen] decoder trainable params (cross-attn + LoRA): {n_train:,}")
        print(f"[gen] num cross-attention adapters: {len(adapters)}")

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
    # Rank 0 owns split creation; everyone reads after a barrier (no write race).
    if env.is_main:
        if splits_path.exists():
            splits = load_splits(str(splits_path))
            if not splits_are_valid(splits, len(pairs), args.seed, cfg.data.splits):
                print("[gen] splits stale; rebuilding")
                splits = make_splits(len(pairs), cfg.data.splits, args.seed)
                save_splits(splits, str(splits_path))
        else:
            splits = make_splits(len(pairs), cfg.data.splits, args.seed)
            save_splits(splits, str(splits_path))
    barrier()
    splits = load_splits(str(splits_path))

    # `val` is a random permutation slice, so the first N is a deterministic
    # random sample; keeps the per-epoch rank-0 eval cheap at full scale.
    val_indices = splits["val"]
    if args.val_subset and args.val_subset > 0:
        val_indices = val_indices[:args.val_subset]

    train_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                                 splits["train"],
                                 cfg.model.protein_hidden, cfg.model.text_hidden)
    val_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                               val_indices,
                               cfg.model.protein_hidden, cfg.model.text_hidden)

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

    # 4) Optim
    optim = torch.optim.AdamW(
        (p for p in decoder.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=cfg.generation.weight_decay,
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

            # Encoder front-end (frozen): project then expand the appropriate side.
            with torch.no_grad():
                if args.direction == "text2protein":
                    z = retrieval.text_proj(h)
                    mem = retrieval.text_expand(z)
                else:
                    z = retrieval.protein_proj(h)
                    mem = retrieval.protein_expand(z)

            set_cross_memory(adapters, mem, mask)
            out = decoder(input_ids=input_ids, attention_mask=attn_mask)
            clear_cross_memory(adapters)

            logits = out.logits                              # [B, L, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in decoder.parameters() if p.requires_grad),
                cfg.generation.grad_clip,
            )
            optim.step()
            global_step += 1

            if env.is_main and ((it + 1) % cfg.generation.log_every == 0 or it == 0):
                with torch.no_grad():
                    ppl = torch.exp(loss).item()
                print(
                    f"[{args.direction}] epoch={epoch} step={it+1}/{len(train_loader)} "
                    f"lr={lr:.2e} ce={loss.item():.4f} ppl={ppl:.2f}",
                    flush=True,
                )

        dt = time.time() - t0
        if env.is_main:
            print(f"[{args.direction}] epoch={epoch} done in {dt:.1f}s")

        # Val pass + checkpoint on rank 0 only (uses the unwrapped decoder).
        if env.is_main:
            core.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    h = batch["h"].to(device).float()
                    mask = batch["m"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attn_mask"].to(device)
                    labels = batch["labels"].to(device)
                    if args.direction == "text2protein":
                        mem = retrieval.text_expand(retrieval.text_proj(h))
                    else:
                        mem = retrieval.protein_expand(retrieval.protein_proj(h))
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
            val_ce = sum(val_losses) / max(len(val_losses), 1)
            print(f"[val] epoch={epoch} ce={val_ce:.4f} ppl={math.exp(val_ce):.2f}")
            log.append({"epoch": epoch, "val_ce": val_ce, "val_ppl": math.exp(val_ce)})

            # Save adapter-only checkpoint (frozen base weights stay on disk)
            adapter_state = {k: v for k, v in core.state_dict().items()
                             if "lora_" in k or "cross_attn" in k}
            path = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save({"epoch": epoch, "adapter_state": adapter_state}, path)
            print(f"[ckpt] saved {path}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[gen] done")
    cleanup()


if __name__ == "__main__":
    main()
