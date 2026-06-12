"""Expansion-only reconstruction phase (Feature 2).

Takes a trained retrieval checkpoint, freezes the projection heads + temperature,
and trains ONLY the expansion heads on per-token reconstruction MSE. Retrieval
metrics depend only on the projection, so R@K is mathematically unchanged; this
sharpens the generation conditioning memory expand(project(h)) up to the ceiling
the frozen 32-d code permits — a free lunch at zero retrieval cost.

Writes the SAME checkpoint format as train_retrieval ({"epoch", "model_state"}),
so generation / inference / round-trip load the result transparently.

Usage:
  python -m src.train_reconstruction --ckpt checkpoints/retrieval/epoch50.pt \\
      --device xpu
  # local smoke test:
  python -m src.train_reconstruction --ckpt checkpoints/retrieval/epoch02.pt \\
      --subset-size 512 --device cpu --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.dist import (
    init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
)
from src.evaluate import evaluate_split
from src.losses import reconstruction_loss
from src.model import MiniEmbedFilip
# Reuse the retrieval data plumbing (cache fingerprint check, by-accession splits,
# packed per-token loaders) and the small LR/device helpers.
from src.train_retrieval import (
    build_loaders, pick_device, autocast_ctx, cosine_warmup_lr,
)


def load_retrieval_model(ckpt_path: str, cfg, device: torch.device) -> MiniEmbedFilip:
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
    return m.to(device)


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained retrieval checkpoint")
    ap.add_argument("--device", default=cfg.recon.device)
    ap.add_argument("--cache-dir", default=cfg.recon.cache_dir)
    ap.add_argument("--ckpt-dir", default=cfg.recon.ckpt_dir)
    ap.add_argument("--batch-size", type=int, default=cfg.recon.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.recon.epochs)
    ap.add_argument("--lr", type=float, default=cfg.recon.lr)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--group-size", type=int, default=1)
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate recon MSE + R@K on the first N val pairs (0 = full)")
    args = ap.parse_args()

    # Point the shared retrieval loaders at the recon cache/batch settings.
    cfg.retrieval.use_cache = True
    cfg.retrieval.cache_dir = args.cache_dir
    cfg.retrieval.batch_size = args.batch_size
    cfg.data.subset_size = args.subset_size
    cfg.data.seed = args.seed

    torch.manual_seed(args.seed)
    env = init_distributed(args.device, group_size=args.group_size)
    device = env.device
    if env.is_main:
        print(f"[recon] world_size={env.world_size} device={device} ckpt={args.ckpt}")

    ckpt_dir = Path(args.ckpt_dir)
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    splits_path = Path(args.cache_dir) / "splits.json"

    train_loader, val_loader, train_sampler, row_group_ids = build_loaders(
        cfg, splits_path, env, pairs=None, val_subset=args.val_subset)

    model = load_retrieval_model(args.ckpt, cfg, device)
    # Freeze everything, then unfreeze only the expansion heads. eval() keeps
    # projection dropout off so the reconstructed z matches what generation reads.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    expand_params = []
    for head in (model.text_expand, model.protein_expand):
        for p in head.parameters():
            p.requires_grad_(True)
            expand_params.append(p)
    if env.is_main:
        n_train = sum(p.numel() for p in expand_params)
        print(f"[recon] training expansion heads only: {n_train:,} params")

    if env.distributed:
        broadcast_parameters(model)

    optimizer = torch.optim.AdamW(
        expand_params, lr=args.lr, weight_decay=cfg.recon.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(int(cfg.recon.warmup_frac * total_steps), 1)

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
                # Projection is frozen: compute z without grad, reconstruct with grad.
                with torch.no_grad():
                    z_p, z_t = model.project(h_p, h_t)
                h_p_hat, h_t_hat = model.expand(z_p, z_t)
                l_p = reconstruction_loss(h_p_hat, h_p, mask_p)
                l_t = reconstruction_loss(h_t_hat, h_t, mask_t)
                loss = 0.5 * (l_p + l_t)

            loss.backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(expand_params, cfg.recon.grad_clip)
            optimizer.step()
            global_step += 1

            if env.is_main and ((it + 1) % cfg.recon.log_every == 0 or it == 0):
                print(
                    f"[recon] epoch={epoch} step={it+1}/{steps_per_epoch} "
                    f"lr={lr:.2e} recon={loss.item():.4f} "
                    f"recon_p={l_p.item():.4f} recon_t={l_t.item():.4f}",
                    flush=True,
                )

        dt = time.time() - t0
        if env.is_main:
            print(f"[recon] epoch={epoch} done in {dt:.1f}s")

        # Rank-0 validation: recon MSE on val + a R@K check (should be unchanged
        # vs the input retrieval checkpoint, since projection is frozen).
        if env.is_main:
            val_p, val_t, nb = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    h_p = batch["h_p"].to(device).float()
                    h_t = batch["h_t"].to(device).float()
                    mask_p = batch["mask_p"].to(device)
                    mask_t = batch["mask_t"].to(device)
                    z_p, z_t = model.project(h_p, h_t)
                    h_p_hat, h_t_hat = model.expand(z_p, z_t)
                    val_p += reconstruction_loss(h_p_hat, h_p, mask_p).item()
                    val_t += reconstruction_loss(h_t_hat, h_t, mask_t).item()
                    nb += 1
            val_p /= max(nb, 1)
            val_t /= max(nb, 1)
            metrics = evaluate_split(
                model, val_loader, device, None,
                cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
                row_group_ids=row_group_ids,
            )
            rk = {k: round(metrics[k], 4) for k in ("R@1", "R@5", "R@10") if k in metrics}
            print(f"[val] epoch={epoch} recon_p={val_p:.4f} recon_t={val_t:.4f} "
                  f"R@K(unchanged)={rk}")
            log.append({"epoch": epoch, "val_recon_p": val_p, "val_recon_t": val_t,
                        **{k: metrics[k] for k in ("R@1", "R@5", "R@10") if k in metrics}})

            ckpt = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt)
            print(f"[ckpt] saved {ckpt}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[recon] done")
    cleanup()


if __name__ == "__main__":
    main()
