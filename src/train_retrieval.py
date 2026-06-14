"""Two-phase FILIP retrieval trainer.

Phase R1 (warmup): positive-pair FILIP + token uniformity + reconstruction.
Phase R2 (main):   FILIP-based InfoNCE + small align aux + reconstruction.

By default reads the packed per-token cache built by `src/precompute.py`.
`--no-cache` falls back to running the encoders live; useful for smoke tests
on a host that hasn't built the cache yet.

Usage:
  python -m src.train_retrieval --use-cache
  python -m src.train_retrieval --no-cache --subset-size 256 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, Cfg
from src.data import (
    PackedPerTokenDataset,
    RawPairsDataset,
    build_or_load_splits,
    group_ids_from_accessions,
    load_pairs,
    load_splits,
    packed_collate,
    raw_collate,
    read_cache_fingerprint,
    fingerprint_matches,
    cache_fingerprint,
)
from src.dist import (
    init_distributed, barrier, cleanup, grouped_all_gather, grouped_all_gather_ids,
    broadcast_parameters, average_gradients,
)
from src.evaluate import evaluate_split
from src.losses import phase_r1_loss, phase_r2_loss_grouped
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


def autocast_ctx(device: torch.device):
    if device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Build data loaders
# ---------------------------------------------------------------------------
def build_loaders(cfg: Cfg, splits_path: Path, env, pairs=None, val_subset: int = 0):
    if cfg.retrieval.use_cache:
        # Validate cache fingerprint
        expected = cache_fingerprint(
            cfg.model.text_encoder_path, cfg.model.protein_encoder_path,
            cfg.data.max_text_tokens, cfg.data.max_protein_tokens,
            cfg.retrieval.mask_text_special_tokens, cfg.retrieval.mask_protein_special_tokens,
        )
        saved = read_cache_fingerprint(cfg.retrieval.cache_dir)
        if not fingerprint_matches(saved, expected):
            raise RuntimeError(
                f"Cache fingerprint mismatch at {cfg.retrieval.cache_dir}.\n"
                f"  expected: {expected}\n  found:    {saved}\n"
                f"Rebuild with `python -m src.precompute`."
            )
        with open(Path(cfg.retrieval.cache_dir) / "pair_ids.json") as f:
            accessions = json.load(f)
        n = len(accessions)
        if env.is_main:
            print(f"[train] cache at {cfg.retrieval.cache_dir}; n={n}")
    else:
        if pairs is None:
            raise RuntimeError("pairs required when use_cache=False")
        accessions = [p.uid for p in pairs]
        n = len(accessions)
        if env.is_main:
            print(f"[train] live mode; n={n}")

    # Group-aware (by-accession) split: every caption of a protein stays on one
    # side of the train/val/test boundary. row_group_ids (per global row) is also
    # used by the R2 loss (false-negative masking) and the val recall, so it is
    # built on every rank. Only rank 0 creates/repairs the split file; everyone
    # reads it after a barrier (avoids a write race across ranks).
    row_group_np = group_ids_from_accessions(accessions)
    if env.is_main:
        splits = build_or_load_splits(
            str(splits_path), n, cfg.data.splits, cfg.data.seed, group_ids=row_group_np)
        print(f"[train] splits: {n} rows over {splits['n_groups']} proteins")
    barrier()
    splits = load_splits(str(splits_path))
    row_group_ids = torch.as_tensor(row_group_np, dtype=torch.long)

    # Subsample validation: `val` is already a random permutation slice, so the
    # first N is a deterministic random sample. Keeps the per-epoch rank-0 eval
    # cheap on the full-scale dataset. val_subset<=0 means use the whole split.
    val_indices = splits["val"]
    if val_subset and val_subset > 0:
        val_indices = val_indices[:val_subset]

    if env.is_main:
        print(f"[train] split sizes: train={len(splits['train'])} "
              f"val={len(splits['val'])} (eval on {len(val_indices)}) "
              f"test={len(splits['test'])}")

    if cfg.retrieval.use_cache:
        train_ds = PackedPerTokenDataset(
            cfg.retrieval.cache_dir, splits["train"],
            cfg.model.protein_hidden, cfg.model.text_hidden,
        )
        val_ds = PackedPerTokenDataset(
            cfg.retrieval.cache_dir, val_indices,
            cfg.model.protein_hidden, cfg.model.text_hidden,
        )
        collate = packed_collate
        bs = cfg.retrieval.batch_size
    else:
        train_ds = RawPairsDataset(pairs, splits["train"])
        val_ds = RawPairsDataset(pairs, val_indices)
        collate = raw_collate
        bs = cfg.retrieval.live_batch_size

    # Distributed: shard the train set across ranks. drop_last keeps a constant
    # per-rank batch size B, which the grouped all-gather (fixed-shape gather)
    # depends on. Validation runs on rank 0 only (unsharded loader).
    if env.distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=env.world_size, rank=env.rank,
            shuffle=True, seed=cfg.data.seed, drop_last=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=train_sampler,
            num_workers=cfg.retrieval.num_workers, collate_fn=collate, drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=cfg.retrieval.num_workers, collate_fn=collate,
            drop_last=len(train_ds) >= 2 * bs,
        )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg.retrieval.num_workers, collate_fn=collate,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            f"train_loader has 0 batches (per-rank train_size≈{len(train_ds)//max(env.world_size,1)}, "
            f"bs={bs}). Lower --batch-size or use fewer ranks."
        )
    if env.is_main:
        print(f"[train] batches/epoch/rank: train={len(train_loader)} val={len(val_loader)} bs={bs}")
    return train_loader, val_loader, train_sampler, row_group_ids


# ---------------------------------------------------------------------------
# Step: get (h_p, h_t, mask_p, mask_t) ready on device
# ---------------------------------------------------------------------------
def fetch_batch(batch, device, encoders, cfg: Cfg):
    if "h_p" in batch:
        h_p = batch["h_p"].to(device).float()
        h_t = batch["h_t"].to(device).float()
        mask_p = batch["mask_p"].to(device)
        mask_t = batch["mask_t"].to(device)
    else:
        from src.encoders import encode_protein_batch, encode_text_batch
        text_model, text_tok, prot_model, prot_tok = encoders
        h_t, mask_t = encode_text_batch(
            text_model, text_tok, batch["text"], device,
            cfg.data.max_text_tokens, cfg.retrieval.mask_text_special_tokens,
        )
        h_p, mask_p = encode_protein_batch(
            prot_model, prot_tok, batch["protein"], device,
            cfg.data.max_protein_tokens, cfg.retrieval.mask_protein_special_tokens,
        )
    return h_p, h_t, mask_p, mask_t


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=cfg.retrieval.device)
    ap.add_argument("--use-cache", dest="use_cache", action="store_true",
                    default=cfg.retrieval.use_cache)
    ap.add_argument("--no-cache", dest="use_cache", action="store_false")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--ckpt-dir", default=cfg.retrieval.ckpt_dir)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--phase1-epochs", type=int, default=cfg.retrieval.phase1_epochs)
    ap.add_argument("--phase2-epochs", type=int, default=cfg.retrieval.phase2_epochs)
    ap.add_argument("--lr", type=float, default=cfg.retrieval.lr)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--group-size", type=int, default=16,
                    help="ranks per contrastive subgroup; global negatives = group_size*batch_size")
    ap.add_argument("--filip-chunk-rows", type=int, default=0,
                    help=">0 chunks the FILIP score matrix over the protein axis to cap memory")
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate on the first N val pairs each epoch (0 = full val split)")
    ap.add_argument("--phase1-uniformity-weight", type=float,
                    default=cfg.retrieval.phase1_uniformity_weight,
                    help="R1 token-spread weight; raise toward 0.3 if R1 collapses")
    ap.add_argument("--r2-uniformity-weight", type=float,
                    default=cfg.retrieval.r2_uniformity_weight,
                    help="R2 token-spread weight; counters per-token crowding under contrastive")
    ap.add_argument("--align-aux-weight", type=float, default=cfg.retrieval.align_aux_weight,
                    help="R2 positive-pair maintenance weight")
    ap.add_argument("--recon-weight", type=float, default=cfg.retrieval.recon_weight,
                    help="autoencoder reconstruction weight (both phases)")
    args = ap.parse_args()

    cfg.retrieval.use_cache = args.use_cache
    cfg.retrieval.cache_dir = args.cache_dir
    cfg.retrieval.ckpt_dir = args.ckpt_dir
    cfg.data.subset_size = args.subset_size
    cfg.retrieval.phase1_epochs = args.phase1_epochs
    cfg.retrieval.phase2_epochs = args.phase2_epochs
    cfg.retrieval.lr = args.lr
    cfg.retrieval.phase1_uniformity_weight = args.phase1_uniformity_weight
    cfg.retrieval.r2_uniformity_weight = args.r2_uniformity_weight
    cfg.retrieval.align_aux_weight = args.align_aux_weight
    cfg.retrieval.recon_weight = args.recon_weight
    cfg.data.seed = args.seed
    if args.batch_size is not None:
        if cfg.retrieval.use_cache:
            cfg.retrieval.batch_size = args.batch_size
        else:
            cfg.retrieval.live_batch_size = args.batch_size

    torch.manual_seed(cfg.data.seed)
    env = init_distributed(args.device, group_size=args.group_size)
    device = env.device
    if env.is_main:
        print(f"[train] world_size={env.world_size} group_size={env.group_size} "
              f"device={device} use_cache={cfg.retrieval.use_cache}")

    ckpt_dir = Path(cfg.retrieval.ckpt_dir)
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    splits_dir = Path(cfg.retrieval.cache_dir) if cfg.retrieval.use_cache else Path("data")
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / "splits.json"

    pairs = None
    encoders = None
    if not cfg.retrieval.use_cache:
        pairs = load_pairs(
            cfg.data.csv_path,
            id_col=cfg.data.csv_id_col,
            protein_col=cfg.data.csv_protein_col,
            text_col=cfg.data.csv_text_col,
            pfam_col=cfg.data.csv_pfam_col,
            subset_size=cfg.data.subset_size,
        )
        from src.encoders import load_protein_encoder, load_text_encoder
        text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
        prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
        encoders = (text_model, text_tok, prot_model, prot_tok)

    train_loader, val_loader, train_sampler, row_group_ids = build_loaders(
        cfg, splits_path, env, pairs=pairs, val_subset=args.val_subset)

    model = MiniEmbedFilip(
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
    ).to(device)
    if env.is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] trainable params: {n_params:,}")

    # NOTE: the retrieval model is NOT wrapped in DDP. Its contrastive loss uses
    # logit_scale and subgroup-gathered embeddings *outside* the module forward,
    # which DDP's reducer mishandles ("logit_scale marked ready twice"). Instead
    # we broadcast initial weights and average gradients manually each step.
    core = model
    if env.distributed:
        broadcast_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.retrieval.lr,
        weight_decay=cfg.retrieval.weight_decay,
    )
    total_epochs = cfg.retrieval.phase1_epochs + cfg.retrieval.phase2_epochs
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = max(int(cfg.retrieval.warmup_frac * total_steps), 1)

    log = []
    global_step = 0
    for epoch in range(total_epochs):
        in_phase1 = epoch < cfg.retrieval.phase1_epochs
        phase_name = "R1-warm" if in_phase1 else "R2-NCE"
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, cfg.retrieval.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            h_p, h_t, mask_p, mask_t = fetch_batch(batch, device, encoders, cfg)
            with autocast_ctx(device):
                out = model(h_p, h_t)
                if in_phase1:
                    losses = phase_r1_loss(
                        out, h_p, h_t, mask_p, mask_t,
                        uniformity_weight=cfg.retrieval.phase1_uniformity_weight,
                        uniformity_t=cfg.retrieval.uniformity_t,
                        recon_weight=cfg.retrieval.recon_weight,
                    )
                else:
                    # Bounded contrastive negatives: gather z/masks within the
                    # subgroup (local slice keeps grad). When not distributed
                    # this reduces to standard in-batch InfoNCE over B.
                    z_p_all = grouped_all_gather(out["z_p"], env)
                    z_t_all = grouped_all_gather(out["z_t"], env)
                    mp_all = grouped_all_gather(mask_p.to(out["z_p"].dtype), env) > 0.5
                    mt_all = grouped_all_gather(mask_t.to(out["z_t"].dtype), env) > 0.5
                    local_offset = env.group_rank * out["z_p"].size(0)
                    # Accession ids for false-negative masking: this rank's
                    # anchors + the gathered subgroup columns (same order as z_*_all).
                    groups_local = row_group_ids[batch["idx"]].to(device)
                    groups_all = grouped_all_gather_ids(groups_local, env)
                    losses = phase_r2_loss_grouped(
                        out, h_p, h_t, mask_p, mask_t, core.logit_scale,
                        z_p_all=z_p_all, z_t_all=z_t_all,
                        mask_p_all=mp_all, mask_t_all=mt_all,
                        local_offset=local_offset,
                        align_aux_weight=cfg.retrieval.align_aux_weight,
                        recon_weight=cfg.retrieval.recon_weight,
                        uniformity_weight=cfg.retrieval.r2_uniformity_weight,
                        uniformity_t=cfg.retrieval.uniformity_t,
                        chunk_rows=args.filip_chunk_rows,
                        groups=groups_local, groups_all=groups_all,
                    )

            losses["loss"].backward()
            average_gradients(model)            # manual data-parallel sync
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.retrieval.grad_clip)
            optimizer.step()
            core.clamp_temperature()
            global_step += 1

            if env.is_main and ((it + 1) % cfg.retrieval.log_every == 0 or it == 0):
                tau = 1.0 / core.logit_scale.exp().item()
                print(
                    f"[{phase_name}] epoch={epoch} step={it+1}/{steps_per_epoch} "
                    f"lr={lr:.2e} loss={losses['loss'].item():.4f} "
                    f"align={losses['align'].item():.4f} "
                    f"unif={losses['unif'].item():.4f} "
                    f"recon={losses['recon'].item():.4f} "
                    f"nce={losses['nce'].item():.4f} "
                    f"acc@1={losses['acc'].item():.3f} "
                    f"filip_pos={losses['filip_pos'].item():.3f} "
                    f"tau={tau:.4f}",
                    flush=True,
                )

        dt = time.time() - t0
        if env.is_main:
            print(f"[{phase_name}] epoch={epoch} done in {dt:.1f}s")

        # Evaluation + checkpointing on rank 0 only (uses the unwrapped model).
        if env.is_main:
            if cfg.retrieval.eval_every_epoch:
                metrics = evaluate_split(
                    core, val_loader, device, encoders,
                    cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
                    row_group_ids=row_group_ids,
                )
                short = {k: round(v, 4) for k, v in metrics.items()
                         if k in ("R@1", "R@5", "R@10", "gap_l2",
                                  "mean_cross_token_cos", "mean_pos_token_cos",
                                  "uniformity_p_tokens")}
                print(f"[val] epoch={epoch}  {short}")
                log.append({"epoch": epoch, "phase": phase_name, **metrics})

            ckpt = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model_state": core.state_dict()}, ckpt)
            print(f"[ckpt] saved {ckpt}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[train] done")
    cleanup()


if __name__ == "__main__":
    main()
