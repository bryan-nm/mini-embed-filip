"""Hard-negative mining phase: resume retrieval, ramp in a decoy loss.

Takes a trained retrieval checkpoint and continues it with the SAME objective it
finished on (phase R2: FILIP InfoNCE over subgroup-gathered negatives + align aux
+ reconstruction + optional token uniformity), then gradually adds one new term:

    L = L_R2  +  w(step) * L_decoy

`L_decoy` asks each protein to rank its true caption above its own *decoys* —
captions identical to the truth except for one templated field swapped in from a
different caption (built by `src/precompute_decoys.py`, see `src/decoys.py`).
In-batch negatives are separable from topic alone; a decoy is not. Ranking it
correctly requires the shared space to encode which *field values* belong to this
protein, which is exactly the failure mode that survives a good R@K.

w ramps linearly from 0 so the decoy term never lands on the model as a step
change: the first stretch of the run is pure R2 (re-warming the optimizer state
at the new LR), then w rises to `--decoy-weight` and holds. Watch `d_acc` (the
fraction of anchors already ranked above their hardest decoy) against the usual
R@K — the point of the ramp is to buy d_acc without paying R@K.

The checkpoint format is unchanged, so reconstruction / memory-map / generation /
export all consume the result exactly as they consume a retrieval checkpoint.

Usage:
  python -m src.train_hard_negatives --resume checkpoints/retrieval/epoch20.pt \\
      --device xpu --batch-size 16 --group-size 16 --epochs 3 \\
      --align-aux-weight 0 --r2-uniformity-weight 0
  # local smoke test on a small cache:
  python -m src.train_hard_negatives --resume checkpoints/retrieval/epoch02.pt \\
      --device cpu --batch-size 8 --group-size 1 --epochs 1 --val-subset 128
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
from src.compute_means import load_means
from src.data import PackedDecoyDataset, decoy_collate, read_cache_fingerprint
from src.decoys import read_decoy_fingerprint
from src.dist import (
    average_gradients, barrier, broadcast_parameters, cleanup,
    grouped_all_gather, grouped_all_gather_ids, init_distributed,
)
from src.evaluate import evaluate_split
from src.losses import (
    filip_paired_scores, filip_paired_scores_chunked, hard_decoy_loss,
    phase_r2_loss_grouped,
)
from src.model import MiniEmbedFilip, build_retrieval
from src.train_retrieval import (
    autocast_ctx, build_loaders, cosine_warmup_lr, resolve_resume_path,
)


# ---------------------------------------------------------------------------
def decoy_weight_at(step: int, total_steps: int, weight: float,
                    start_frac: float, ramp_frac: float) -> float:
    """Linear ramp: 0 until `start_frac`, rising to `weight` over `ramp_frac`.

    Expressed as fractions of the run so the schedule is invariant to epoch count
    and world size. A zero-length ramp turns the term on as a step change at
    `start_frac` (allowed, but the ramp exists for a reason).
    """
    if weight <= 0.0:
        return 0.0
    start = start_frac * total_steps
    if step < start:
        return 0.0
    ramp = ramp_frac * total_steps
    if ramp <= 0:
        return weight
    return weight * min(1.0, (step - start) / ramp)


def build_model(cfg, state: dict, device: torch.device) -> MiniEmbedFilip:
    """Rebuild the retrieval model matching a checkpoint's state dict.

    Stays trainable (this phase fine-tunes the projection heads), unlike
    `model.load_retrieval`, which freezes.
    """
    model = build_retrieval(cfg)
    model.load_state_dict(state)
    return model.to(device)


def check_decoy_cache(args, env) -> dict:
    """Validate the decoy cache against the base cache; return its stats."""
    decoy_fp = read_decoy_fingerprint(args.decoy_cache_dir)
    if not decoy_fp:
        raise FileNotFoundError(
            f"No decoy_fingerprint.json in {args.decoy_cache_dir}. Build the decoy "
            f"cache first:\n"
            f"    mpiexec ... python -m src.precompute_decoys --device xpu")
    base_fp = read_cache_fingerprint(args.cache_dir)
    if decoy_fp.get("base") != base_fp:
        raise RuntimeError(
            f"Decoy cache was built against a different base cache.\n"
            f"  decoy['base']: {decoy_fp.get('base')}\n"
            f"  {args.cache_dir}: {base_fp}\n"
            f"Decoy and true-caption scores are only comparable when both were "
            f"encoded by the same text encoder with the same caps and masking.")
    stats_path = Path(args.decoy_cache_dir) / "decoy_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if env.is_main:
        rows = stats.get("rows_total", 0)
        usable = stats.get("rows_with_usable_decoy", 0)
        print(f"[hardneg] decoy cache {args.decoy_cache_dir}: "
              f"{stats.get('n_decoys_kept', '?')} usable decoys, "
              f"{usable}/{rows} rows covered "
              f"({100.0 * usable / max(rows, 1):.1f}%)")
        if rows and usable < 0.5 * rows:
            print("[hardneg] WARNING: fewer than half the rows have a usable decoy; "
                  "the decoy loss will be averaged over a thin slice of each batch. "
                  "Check decoy_stats.json (sparse captions, or --max-swap-char too "
                  "tight for long captions).")
    return stats


@torch.no_grad()
def evaluate_decoys(model, loader, device, margin: float, chunk_rows: int) -> dict:
    """Val-split decoy discrimination: accuracy and score margin.

    The metric that says whether mining worked. `d_acc` is the fraction of val
    anchors whose true caption outscores every one of its decoys; `d_margin` is
    the mean FILIP-score lead over the hardest decoy. R@K can sit still while
    both move, which is the whole point of this phase.
    """
    model.eval()
    tot_acc = tot_margin = tot_score = 0.0
    n = 0
    for batch in loader:
        if not bool(batch["decoy_valid"].any()):
            continue
        h_p = batch["h_p"].to(device).float()
        h_t = batch["h_t"].to(device).float()
        mask_p = batch["mask_p"].to(device)
        mask_t = batch["mask_t"].to(device)
        h_d = batch["h_d"].to(device).float()
        mask_d = batch["mask_d"].to(device)
        valid = batch["decoy_valid"].to(device)

        z_p, z_t = model.project(h_p, h_t)
        B, K, L_d, _ = h_d.shape
        z_d = model.text_proj(h_d.view(B * K, L_d, -1)).view(B, K, L_d, -1)
        # Positive score in the same form the training loss uses: the anchor
        # against its own true caption (a K=1 paired score).
        pos = _paired_positive(z_p, mask_p, z_t, mask_t, chunk_rows)
        out = hard_decoy_loss(z_p, mask_p, z_d, mask_d, valid, pos,
                              model.logit_scale, margin=margin,
                              chunk_rows=chunk_rows)
        rows = float(out["decoy_rows"].item())
        if rows == 0:
            continue
        tot_acc += float(out["decoy_acc"].item()) * rows
        tot_margin += float(out["decoy_margin"].item()) * rows
        tot_score += float(out["decoy_score"].item()) * rows
        n += rows
    if n == 0:
        return {}
    return {"val_decoy_acc": tot_acc / n, "val_decoy_margin": tot_margin / n,
            "val_decoy_score": tot_score / n, "val_decoy_rows": n}


def _paired_positive(z_p, mask_p, z_t, mask_t, chunk_rows: int) -> torch.Tensor:
    """Each anchor's FILIP score against its own true caption -> [B].

    Eval-side stand-in for the training loop's `filip_pos_vec`, which comes free
    off the contrastive matrix. Here there is no contrastive matrix, so we score
    the positives as a K=1 paired batch — same arithmetic, so `val_decoy_margin`
    is on the same scale as the training-time `d_margin`.
    """
    z1 = z_t.unsqueeze(1)                      # [B, 1, L_t, D]
    m1 = mask_t.unsqueeze(1)                   # [B, 1, L_t]
    if chunk_rows > 0:
        return filip_paired_scores_chunked(z_p, z1, mask_p, m1, chunk_rows).squeeze(1)
    return filip_paired_scores(z_p, z1, mask_p, m1).squeeze(1)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = default_cfg()
    hn = cfg.hard_neg
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None,
                    help="retrieval checkpoint to continue: a path to an epochNN.pt, "
                         "or 'auto' to pick the latest in --retrieval-ckpt-dir")
    ap.add_argument("--continue-from", default=None,
                    help="restart an INTERRUPTED hard-negative run from its own "
                         "checkpoint ('auto' = latest in --ckpt-dir). Restores the "
                         "epoch, global_step, optimizer and log, so the decoy ramp "
                         "picks up where it stopped instead of replaying from zero. "
                         "Use this after a node/fabric failure; --resume starts a "
                         "fresh run from a retrieval checkpoint.")
    ap.add_argument("--retrieval-ckpt-dir", default=cfg.retrieval.ckpt_dir,
                    help="searched by --resume auto")
    ap.add_argument("--device", default=hn.device)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--decoy-cache-dir", default=hn.decoy_cache_dir)
    ap.add_argument("--ckpt-dir", default=hn.ckpt_dir)
    ap.add_argument("--batch-size", type=int, default=hn.batch_size)
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--epochs", type=int, default=hn.epochs)
    ap.add_argument("--lr", type=float, default=hn.lr)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--group-size", type=int, default=16,
                    help="ranks per contrastive subgroup; global negatives = group_size*batch_size")
    ap.add_argument("--filip-chunk-rows", type=int, default=0,
                    help=">0 chunks the FILIP score matrices over the anchor axis")
    ap.add_argument("--val-subset", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=hn.num_workers,
                    help="DataLoader workers. This phase reads 1+max_decoys text "
                         "rows per sample, so 0 serializes ~3x the random reads "
                         "into the training step; raise until data_wait falls")
    ap.add_argument("--load-optimizer", action="store_true",
                    help="restore the retrieval run's optimizer state instead of "
                         "starting fresh (default: fresh, since this phase runs its "
                         "own LR schedule at a lower base LR)")

    # R2 objective knobs — pass the SAME values the retrieval run finished with,
    # otherwise 'resume with its existing objective' silently isn't.
    ap.add_argument("--align-aux-weight", type=float, default=cfg.retrieval.align_aux_weight)
    ap.add_argument("--recon-weight", type=float, default=cfg.retrieval.recon_weight)
    ap.add_argument("--r2-uniformity-weight", type=float,
                    default=cfg.retrieval.r2_uniformity_weight)

    # Decoy term.
    ap.add_argument("--decoy-weight", type=float, default=hn.decoy_weight)
    ap.add_argument("--decoy-start-frac", type=float, default=hn.decoy_start_frac)
    ap.add_argument("--decoy-ramp-frac", type=float, default=hn.decoy_ramp_frac)
    ap.add_argument("--decoy-margin", type=float, default=hn.decoy_margin,
                    help="0 = (K+1)-way softmax; >0 = hinge on the raw FILIP gap")
    ap.add_argument("--max-decoys", type=int, default=hn.max_decoys,
                    help="decoys scored per anchor per step (a random subset when "
                         "the row has more)")
    args = ap.parse_args()

    cfg.retrieval.use_cache = True
    cfg.retrieval.cache_dir = args.cache_dir
    cfg.retrieval.batch_size = args.batch_size
    cfg.retrieval.num_workers = args.num_workers
    cfg.data.subset_size = args.subset_size
    cfg.data.seed = args.seed
    cfg.retrieval.align_aux_weight = args.align_aux_weight
    cfg.retrieval.recon_weight = args.recon_weight
    cfg.retrieval.r2_uniformity_weight = args.r2_uniformity_weight

    torch.manual_seed(args.seed)
    env = init_distributed(args.device, group_size=args.group_size)
    device = env.device
    if env.is_main:
        print(f"[hardneg] world_size={env.world_size} group_size={env.group_size} "
              f"device={device}")
        print(f"[hardneg] R2 objective: align_aux={args.align_aux_weight} "
              f"recon={args.recon_weight} r2_unif={args.r2_uniformity_weight}")
        print(f"[hardneg] decoy term: weight={args.decoy_weight} "
              f"start={args.decoy_start_frac} ramp={args.decoy_ramp_frac} "
              f"margin={args.decoy_margin} max_decoys={args.max_decoys} "
              f"({'hinge' if args.decoy_margin > 0 else 'softmax'})")

    ckpt_dir = Path(args.ckpt_dir)
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    check_decoy_cache(args, env)
    splits_path = Path(args.cache_dir) / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} missing. This phase must reuse the retrieval run's "
            f"splits — rebuilding them here could move val/test rows into train.")

    def make_ds(cache_dir, indices):
        return PackedDecoyDataset(
            cache_dir, args.decoy_cache_dir, indices,
            cfg.model.protein_hidden, cfg.model.text_hidden,
            max_decoys=args.max_decoys)

    train_loader, val_loader, train_sampler, row_group_ids, row_text_ids = build_loaders(
        cfg, splits_path, env, pairs=None, val_subset=args.val_subset,
        dataset_factory=make_ds, collate_fn=decoy_collate)

    # --- resume: either start a fresh phase from a retrieval checkpoint, or
    # continue an interrupted one from this phase's own checkpoint ---
    if args.continue_from:
        resume_path = resolve_resume_path(args.continue_from, ckpt_dir)
    elif args.resume:
        resume_path = resolve_resume_path(args.resume, Path(args.retrieval_ckpt_dir))
    else:
        raise SystemExit(
            "pass --resume <retrieval checkpoint> to start this phase, or "
            "--continue-from <hard-negative checkpoint|auto> to restart an "
            "interrupted run")
    ckpt = torch.load(resume_path, map_location="cpu")
    model = build_model(cfg, ckpt["model_state"], device)
    core = model
    if env.is_main:
        print(f"[hardneg] loaded {resume_path} (epoch {ckpt.get('epoch')}, "
              f"global_step {ckpt.get('global_step')})")

    # The checkpoint's projection heads carry the corpus means as buffers. Fall
    # back to the cache's mean files only if they are absent/zero — an uncentered
    # projection collapses FILIP max-sim, so this is worth failing loudly over.
    if float(core.protein_proj.mean_in.norm()) == 0.0:
        mean_p, mean_t = load_means(args.cache_dir)
        if mean_p is None or mean_t is None:
            raise RuntimeError(
                f"Checkpoint has zero feature means and {args.cache_dir} has no "
                f"protein_mean.pt / text_mean.pt. Run "
                f"`python -m src.compute_means --cache-dir {args.cache_dir}`.")
        core.set_feature_means(mean_p, mean_t)
        if env.is_main:
            print("[hardneg] checkpoint had no feature means; loaded them from cache")

    if env.distributed:
        broadcast_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=hn.weight_decay)
    # Continuing this phase always restores the optimizer (an interrupted run must
    # not restart Adam's moments mid-schedule); starting the phase restores it only
    # on request, since a fresh LR schedule at a lower LR is the sane default.
    if (args.continue_from or args.load_optimizer) and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if env.is_main:
            print("[hardneg] restored optimizer state")

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(int(hn.warmup_frac * total_steps), 1)

    log = []
    start_epoch = 0
    global_step = 0
    if args.continue_from:
        start_epoch = int(ckpt["epoch"]) + 1
        # global_step drives BOTH the LR cosine and the decoy ramp, so restoring it
        # is what makes the restart continuous rather than a replay.
        global_step = int(ckpt.get("global_step", start_epoch * steps_per_epoch))
        log = ckpt.get("train_log", [])
        if start_epoch >= args.epochs:
            raise SystemExit(
                f"--continue-from is at epoch {ckpt['epoch']}, already at/past "
                f"--epochs {args.epochs}; raise --epochs to keep training")
        prev = ckpt.get("hard_neg_args", {})
        drift = {k: (prev.get(k), getattr(args, k)) for k in
                 ("decoy_weight", "decoy_start_frac", "decoy_ramp_frac",
                  "decoy_margin", "max_decoys", "epochs", "lr", "batch_size",
                  "group_size", "align_aux_weight", "recon_weight",
                  "r2_uniformity_weight")
                 if k in prev and prev[k] != getattr(args, k)}
        if drift and env.is_main:
            print("[hardneg] WARNING: these flags differ from the interrupted run; "
                  "the schedule will not match what it would have done:")
            for k, (was, now) in drift.items():
                print(f"[hardneg]   {k}: {was} -> {now}")

    if env.is_main:
        ramp_end = int((args.decoy_start_frac + args.decoy_ramp_frac) * total_steps)
        w_now = decoy_weight_at(global_step, total_steps, args.decoy_weight,
                                args.decoy_start_frac, args.decoy_ramp_frac)
        print(f"[hardneg] {total_steps} steps ({args.epochs} epochs x "
              f"{steps_per_epoch}); decoy weight reaches {args.decoy_weight} at "
              f"step {ramp_end}")
        if args.continue_from:
            print(f"[hardneg] continuing at epoch {start_epoch}, "
                  f"global_step {global_step}, decoy weight now {w_now:.3f}")

    # Baseline before any decoy gradient: the number every later epoch is judged
    # against. Without it the first val report already sits partway up the ramp,
    # and "did hard negatives help" has no reference point. Skipped when continuing
    # an interrupted run — the baseline is already in the restored log, and
    # re-measuring it here would record a mid-ramp model under the "baseline" label.
    if env.is_main and not args.continue_from:
        base = evaluate_split(
            core, val_loader, device, None,
            cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
            row_group_ids=row_group_ids, row_text_ids=row_text_ids)
        base.update(evaluate_decoys(core, val_loader, device, args.decoy_margin,
                                    args.filip_chunk_rows or 8))
        short = {k: round(v, 4) for k, v in base.items()
                 if k in ("R@1", "R@5", "R@10", "mAP",
                          "val_decoy_acc", "val_decoy_margin")}
        print(f"[val] baseline (resumed checkpoint, no decoy gradient yet)  {short}")
        log.append({"epoch": -1, "phase": "baseline", "decoy_weight_end": 0.0, **base})
    barrier()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            # Seeded by epoch, so a continued run reshuffles exactly as an
            # uninterrupted one would have at this epoch.
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        # Split wall-clock into "waiting for the loader" vs "doing the step". This
        # phase reads 1+max_decoys text rows per sample, and the decoy math is <1%
        # of the R2 contrastive math, so a slow epoch is almost always data_wait.
        # If data_wait dominates: raise --num-workers, then lower --max-decoys.
        # If step dominates: it is the R2 [B, group_size*batch_size] FILIP matrix,
        # not the decoys — lower --group-size or --filip-chunk-rows.
        # No explicit device sync is added: average_gradients' all-reduce and the
        # .item() in the log line already force one, so the split is meaningful.
        t_data = t_step = 0.0
        t_mark = time.time()
        for it, batch in enumerate(train_loader):
            t_data += time.time() - t_mark
            t_mark = time.time()
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            w_decoy = decoy_weight_at(global_step, total_steps, args.decoy_weight,
                                      args.decoy_start_frac, args.decoy_ramp_frac)
            optimizer.zero_grad(set_to_none=True)

            h_p = batch["h_p"].to(device).float()
            h_t = batch["h_t"].to(device).float()
            mask_p = batch["mask_p"].to(device)
            mask_t = batch["mask_t"].to(device)

            with autocast_ctx(device):
                out = model(h_p, h_t)
                h_p_c, h_t_c = core.center(h_p, h_t)

                # --- the retrieval run's own objective, unchanged ---
                z_p_all = grouped_all_gather(out["z_p"], env)
                z_t_all = grouped_all_gather(out["z_t"], env)
                mp_all = grouped_all_gather(mask_p.to(out["z_p"].dtype), env) > 0.5
                mt_all = grouped_all_gather(mask_t.to(out["z_t"].dtype), env) > 0.5
                local_offset = env.group_rank * out["z_p"].size(0)
                groups_local = row_group_ids[batch["idx"]].to(device)
                groups_all = grouped_all_gather_ids(groups_local, env)
                text_local = row_text_ids[batch["idx"]].to(device)
                text_all = grouped_all_gather_ids(text_local, env)
                losses = phase_r2_loss_grouped(
                    out, h_p_c, h_t_c, mask_p, mask_t, core.logit_scale,
                    z_p_all=z_p_all, z_t_all=z_t_all,
                    mask_p_all=mp_all, mask_t_all=mt_all,
                    local_offset=local_offset,
                    align_aux_weight=cfg.retrieval.align_aux_weight,
                    recon_weight=cfg.retrieval.recon_weight,
                    uniformity_weight=cfg.retrieval.r2_uniformity_weight,
                    uniformity_t=cfg.retrieval.uniformity_t,
                    chunk_rows=args.filip_chunk_rows,
                    groups=groups_local, groups_all=groups_all,
                    text_groups=text_local, text_groups_all=text_all,
                )
                total = losses["loss"]

                # --- ramped hard-decoy term ---
                # Skipped entirely while w == 0 (no decoy projection, no scores),
                # so the pre-ramp stretch costs exactly what plain R2 costs. The
                # emptiness test is done on the CPU-side collate tensor so it
                # doesn't force a device sync every step.
                # Skipping is rank-local (a rank can draw a batch of captions that
                # all lack decoys) but safe for `average_gradients`: the decoy term
                # only touches text_proj and logit_scale, both of which already
                # carry R2 gradients, so the set of grad-bearing parameters — and
                # therefore the flattened all-reduce layout — is identical on every
                # rank either way.
                has_decoys = bool(batch["decoy_valid"].any())
                if w_decoy > 0.0 and has_decoys:
                    valid = batch["decoy_valid"].to(device)
                    h_d = batch["h_d"].to(device).float()
                    mask_d = batch["mask_d"].to(device)
                    B, K, L_d, d_t = h_d.shape
                    z_d = core.text_proj(h_d.view(B * K, L_d, d_t)).view(B, K, L_d, -1)
                    dec = hard_decoy_loss(
                        out["z_p"], mask_p, z_d, mask_d, valid,
                        losses["filip_pos_vec"], core.logit_scale,
                        margin=args.decoy_margin, chunk_rows=args.filip_chunk_rows)
                    total = total + w_decoy * dec["decoy"]
                else:
                    zero = torch.zeros((), device=device)
                    dec = {"decoy": zero, "decoy_acc": zero, "decoy_margin": zero,
                           "decoy_score": zero, "decoy_rows": zero}

            total.backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hn.grad_clip)
            optimizer.step()
            core.clamp_temperature()
            global_step += 1
            t_step += time.time() - t_mark

            if env.is_main and ((it + 1) % hn.log_every == 0 or it == 0):
                tau = 1.0 / core.logit_scale.exp().item()
                n_done = it + 1
                frac_data = t_data / max(t_data + t_step, 1e-9)
                print(
                    f"[hardneg] epoch={epoch} step={it+1}/{steps_per_epoch} "
                    f"lr={lr:.2e} loss={total.item():.4f} "
                    f"nce={losses['nce'].item():.4f} "
                    f"acc@1={losses['acc'].item():.3f} "
                    f"recon={losses['recon'].item():.4f} "
                    f"filip_pos={losses['filip_pos'].item():.3f} "
                    f"w_dec={w_decoy:.3f} dec={dec['decoy'].item():.4f} "
                    f"d_acc={dec['decoy_acc'].item():.3f} "
                    f"d_margin={dec['decoy_margin'].item():.4f} "
                    f"d_rows={int(dec['decoy_rows'].item())}/{h_p.size(0)} "
                    f"tau={tau:.4f} "
                    f"| {(t_data + t_step)/n_done:.2f}s/step "
                    f"data_wait={t_data/n_done:.2f}s ({100*frac_data:.0f}%) "
                    f"step={t_step/n_done:.2f}s",
                    flush=True,
                )
            t_mark = time.time()

        dt = time.time() - t0
        if env.is_main:
            frac_data = t_data / max(t_data + t_step, 1e-9)
            print(f"[hardneg] epoch={epoch} done in {dt:.1f}s "
                  f"({steps_per_epoch} steps, {dt/max(steps_per_epoch,1):.2f}s/step; "
                  f"data_wait {100*frac_data:.0f}%, compute {100*(1-frac_data):.0f}%)")
            if frac_data > 0.5:
                print(f"[hardneg] NOTE: over half the epoch was spent waiting on the "
                      f"loader (num_workers={args.num_workers}, max_decoys="
                      f"{args.max_decoys}). The decoy loss is <1% of the step's FILIP "
                      f"math, so this is I/O, not the new objective: raise "
                      f"--num-workers, then lower --max-decoys.")
            t_eval = time.time()
            metrics = evaluate_split(
                core, val_loader, device, None,
                cfg.data.max_protein_tokens, cfg.data.max_text_tokens,
                row_group_ids=row_group_ids, row_text_ids=row_text_ids,
            )
            metrics.update(evaluate_decoys(core, val_loader, device,
                                           args.decoy_margin,
                                           args.filip_chunk_rows or 8))
            eval_dt = time.time() - t_eval
            short = {k: round(v, 4) for k, v in metrics.items()
                     if k in ("R@1", "R@5", "R@10", "mAP", "gap_l2",
                              "val_decoy_acc", "val_decoy_margin")}
            print(f"[val] epoch={epoch}  {short}  eval_time={eval_dt:.1f}s")
            log.append({"epoch": epoch, "phase": "hard-negatives",
                        "decoy_weight_end": decoy_weight_at(
                            global_step, total_steps, args.decoy_weight,
                            args.decoy_start_frac, args.decoy_ramp_frac),
                        "eval_time": eval_dt, **metrics})
            model.train()

            ckpt_out = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": core.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_log": log,
                "hard_neg_args": vars(args),
                "resumed_from": str(resume_path),
            }, ckpt_out)
            print(f"[ckpt] saved {ckpt_out}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[hardneg] done")
    cleanup()


if __name__ == "__main__":
    main()
