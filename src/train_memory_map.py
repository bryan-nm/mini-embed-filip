"""Cross-modal memory-map phase (Medium).

Takes a trained retrieval checkpoint, freezes the projection + expansion heads and
the temperature, and trains ONLY the two memory maps (text_map, protein_map) with
a FILIP-contrastive aux:

  text2protein arm:  text_map(z_text)    must retrieve the true protein's z
  protein2text arm:  protein_map(z_prot) must retrieve the true text's z

The maps' inputs are the frozen projections computed under no_grad, so gradient
flows only through the maps — the retrieval geometry (and R@K) is mathematically
unchanged. The map is a per-token residual that nudges a source token toward the
*target* modality's region of the shared space, giving generation a protein-ward
(resp. text-ward) conditioning object while keeping the L_t token structure (and
the interpretable FILIP array) intact.

Writes the same {"epoch", "model_state"} format as train_retrieval, but with the
two map submodules present, so the generation paths can load it and optionally
apply the map via --memory-map. The bypass arm (--memory-map off) ignores it.

Usage:
  python -m src.train_memory_map --ckpt checkpoints/retrieval/epoch50.pt \\
      --device xpu
  # local smoke test:
  python -m src.train_memory_map --ckpt checkpoints/retrieval/epoch02.pt \\
      --subset-size 512 --device cpu --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.dist import (
    init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
    grouped_all_gather, grouped_all_gather_ids,
)
from src.losses import filip_score_matrix, mask_false_negatives
from src.model import MiniEmbedFilip
# Reuse the retrieval data plumbing + small LR/device helpers.
from src.train_retrieval import (
    build_loaders, pick_device, autocast_ctx, cosine_warmup_lr,
)


def load_retrieval_with_maps(ckpt_path: str, cfg, device: torch.device) -> MiniEmbedFilip:
    """Build the retrieval model WITH the memory maps and load a checkpoint.

    A base retrieval checkpoint has no map keys, so they load at their zero-init
    identity (strict=False); a resumed map checkpoint fills them. Any missing key
    that is NOT a map submodule, or any unexpected key, is a real mismatch.
    """
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
        with_maps=True,
        map_hidden=cfg.model.map_hidden,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = m.load_state_dict(state["model_state"], strict=False)
    bad = [k for k in missing if not (k.startswith("text_map.") or k.startswith("protein_map."))]
    if bad or unexpected:
        raise RuntimeError(
            f"map-phase load mismatch: missing(non-map)={bad} unexpected={unexpected}")
    return m.to(device)


def map_loss(model, z_p, z_t, mask_p, mask_t, *, z_p_all, z_t_all, mask_p_all,
             mask_t_all, local_offset, scale, groups, groups_all, text_groups,
             text_groups_all):
    """Symmetric FILIP-contrastive loss for the two maps.

    Anchors are the *mapped* local source tokens (they carry grad through the
    maps); columns are the frozen gathered target projections. Same false-negative
    masking as retrieval (columns sharing an anchor's protein or caption are
    dropped from the denominator). Returns (loss, accuracy).
    """
    g_t = model.text_map(z_t)                         # [B, L_t, D] (grad via map)
    g_p = model.protein_map(z_p)                      # [B, L_p, D]
    B = z_p.size(0)
    target = torch.arange(B, device=z_p.device) + local_offset

    # text->protein: mapped text retrieves the true protein among all proteins.
    mat_tp = filip_score_matrix(g_t, z_p_all, mask_t, mask_p_all)   # [B, G]
    # protein->text: mapped protein retrieves the true text among all texts.
    mat_pt = filip_score_matrix(g_p, z_t_all, mask_p, mask_t_all)   # [B, G]

    logits_tp = mask_false_negatives(scale * mat_tp, target, groups, groups_all,
                                     text_groups, text_groups_all)
    logits_pt = mask_false_negatives(scale * mat_pt, target, groups, groups_all,
                                     text_groups, text_groups_all)
    loss = 0.5 * (F.cross_entropy(logits_tp, target) + F.cross_entropy(logits_pt, target))
    with torch.no_grad():
        acc = 0.5 * ((logits_tp.argmax(-1) == target).float().mean()
                     + (logits_pt.argmax(-1) == target).float().mean())
    return loss, acc


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained retrieval checkpoint")
    ap.add_argument("--device", default=cfg.memory_map.device)
    ap.add_argument("--cache-dir", default=cfg.memory_map.cache_dir)
    ap.add_argument("--ckpt-dir", default=cfg.memory_map.ckpt_dir)
    ap.add_argument("--batch-size", type=int, default=cfg.memory_map.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.memory_map.epochs)
    ap.add_argument("--lr", type=float, default=cfg.memory_map.lr)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--group-size", type=int, default=1,
                    help="contrastive subgroup size for the gathered negative pool")
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate map retrieval acc on the first N val pairs (0 = full)")
    args = ap.parse_args()

    # Point the shared retrieval loaders at the map cache/batch settings.
    cfg.retrieval.use_cache = True
    cfg.retrieval.cache_dir = args.cache_dir
    cfg.retrieval.batch_size = args.batch_size
    cfg.data.subset_size = args.subset_size
    cfg.data.seed = args.seed

    torch.manual_seed(args.seed)
    env = init_distributed(args.device, group_size=args.group_size)
    device = env.device
    if env.is_main:
        print(f"[map] world_size={env.world_size} device={device} "
              f"group_size={env.group_size} ckpt={args.ckpt}")

    ckpt_dir = Path(args.ckpt_dir)
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    splits_path = Path(args.cache_dir) / "splits.json"

    train_loader, val_loader, train_sampler, row_group_ids, row_text_ids = build_loaders(
        cfg, splits_path, env, pairs=None, val_subset=args.val_subset)

    model = load_retrieval_with_maps(args.ckpt, cfg, device)
    # Freeze everything, then unfreeze only the two maps. eval() keeps projection
    # dropout off so the z the maps see matches what generation reads.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    map_params = []
    for head in (model.text_map, model.protein_map):
        for p in head.parameters():
            p.requires_grad_(True)
            map_params.append(p)
    if env.is_main:
        n_train = sum(p.numel() for p in map_params)
        print(f"[map] training memory maps only: {n_train:,} params")

    if env.distributed:
        broadcast_parameters(model)

    optimizer = torch.optim.AdamW(
        map_params, lr=args.lr, weight_decay=cfg.memory_map.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(int(cfg.memory_map.warmup_frac * total_steps), 1)

    log = []
    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            h_p = batch["h_p"].to(device).float()
            h_t = batch["h_t"].to(device).float()
            mask_p = batch["mask_p"].to(device)
            mask_t = batch["mask_t"].to(device)

            with autocast_ctx(device):
                # Projection is frozen: z carries no grad; the maps do.
                with torch.no_grad():
                    z_p, z_t = model.project(h_p, h_t)
                # Gather frozen columns for a bounded negative pool (identity when
                # group_size==1). local_offset = this rank's slot in the gather.
                z_p_all = grouped_all_gather(z_p, env)
                z_t_all = grouped_all_gather(z_t, env)
                mp_all = grouped_all_gather(mask_p.to(z_p.dtype), env) > 0.5
                mt_all = grouped_all_gather(mask_t.to(z_t.dtype), env) > 0.5
                local_offset = env.group_rank * z_p.size(0)
                groups_local = row_group_ids[batch["idx"]].to(device)
                groups_all = grouped_all_gather_ids(groups_local, env)
                text_local = row_text_ids[batch["idx"]].to(device)
                text_all = grouped_all_gather_ids(text_local, env)
                scale = model.logit_scale.exp()
                loss, acc = map_loss(
                    model, z_p, z_t, mask_p, mask_t,
                    z_p_all=z_p_all, z_t_all=z_t_all, mask_p_all=mp_all, mask_t_all=mt_all,
                    local_offset=local_offset, scale=scale,
                    groups=groups_local, groups_all=groups_all,
                    text_groups=text_local, text_groups_all=text_all)

            loss.backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(map_params, cfg.memory_map.grad_clip)
            optimizer.step()
            global_step += 1

            if env.is_main and ((it + 1) % cfg.memory_map.log_every == 0 or it == 0):
                print(f"[map] epoch={epoch} step={it+1}/{steps_per_epoch} "
                      f"lr={lr:.2e} loss={loss.item():.4f} acc={acc.item():.3f}",
                      flush=True)

        dt = time.time() - t0
        if env.is_main:
            print(f"[map] epoch={epoch} done in {dt:.1f}s")

        # Rank-0 validation: map retrieval acc on val (local-batch negatives).
        if env.is_main:
            val_acc, nb = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    h_p = batch["h_p"].to(device).float()
                    h_t = batch["h_t"].to(device).float()
                    mask_p = batch["mask_p"].to(device)
                    mask_t = batch["mask_t"].to(device)
                    z_p, z_t = model.project(h_p, h_t)
                    B = z_p.size(0)
                    target = torch.arange(B, device=device)
                    scale = model.logit_scale.exp()
                    g_t = model.text_map(z_t)
                    mat = scale * filip_score_matrix(g_t, z_p, mask_t, mask_p)  # [B, B]
                    val_acc += (mat.argmax(-1) == target).float().mean().item()
                    nb += 1
            val_acc /= max(nb, 1)
            print(f"[val] epoch={epoch} map_retrieval_acc={val_acc:.4f}")
            log.append({"epoch": epoch, "val_map_acc": val_acc})

            ckpt = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt)
            print(f"[ckpt] saved {ckpt}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[map] done")
    cleanup()


if __name__ == "__main__":
    main()
