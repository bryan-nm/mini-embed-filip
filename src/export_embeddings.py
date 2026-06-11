"""Export retrieval embeddings (z_p, z_t) from the cache — no generation.

Runs the trained projection heads over the precomputed per-token encoder cache
to produce the 32-d FILIP embeddings, keyed by primary accession, for one or
more splits. Read-from-cache only: no encoders, no decoders, nothing generative.

Two output modes:
  --pooling mean  (default)  one mean-pooled [N, D] vector per protein / caption,
                             saved as a per-split .npz — the convenient unit for
                             latent-space structure analysis (clustering, UMAP,
                             modality-gap, family structure).
  --pooling none             the full per-token embeddings, streamed to a packed
                             flat float32 .bin + an int64 offsets .npy per split
                             and modality (large: text can be tens of GB).

Layout note (protein dedup): the protein cache holds one row per *unique*
protein, the text cache one row per CSV row. Splits are by accession, so each
protein lands in exactly one split. We map split rows -> their unique protein
rows via row_protein_idx, so z_p is exported once per protein per split.

Usage:
  python -m src.export_embeddings --ckpt checkpoints/retrieval/epochNN.pt \
      --splits train,test --device xpu
  # full per-token export of just the test proteins:
  python -m src.export_embeddings --ckpt ... --splits test \
      --modalities protein --pooling none

Two input sources:
  --cache-dir (default)  project the precomputed per-token cache for whole
                         train/val/test splits. The projection is tiny and the
                         cost is cache I/O, so this runs SINGLE PROCESS (plain
                         `python`, not mpiexec) — prefer a GPU/XPU device.
  --csv PATH             encode every row of a CSV live (no cache, no splits):
                         this runs the frozen encoders, the expensive part, so
                         it is DISTRIBUTED — launch under `mpiexec` and it shards
                         proteins/rows across ranks exactly like precompute. With
                         no MPI it falls back to one process.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.data import (
    PackedPerTokenCache, dedup_proteins, load_pairs, load_row_protein_idx,
    load_splits,
)
from src.model import MiniEmbedFilip


def _device(name: str) -> torch.device:
    """Resolve a device, importing IPEX so `xpu` registers."""
    if name == "xpu" or name.startswith("xpu"):
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except Exception:
            pass
        return torch.device(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import intel_extension_for_pytorch  # noqa: F401
            if torch.xpu.is_available():
                return torch.device("xpu")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(name)


def load_retrieval(ckpt_path: str, device: torch.device) -> MiniEmbedFilip:
    cfg = default_cfg()
    m = MiniEmbedFilip(
        text_hidden=cfg.model.text_hidden, protein_hidden=cfg.model.protein_hidden,
        proj_d_hidden=cfg.model.proj_d_hidden, proj_d_mid=cfg.model.proj_d_mid,
        embed_dim=cfg.model.embed_dim, proj_dropout=cfg.model.proj_dropout,
        expand_d_mid=cfg.model.expand_d_mid, expand_d_hidden=cfg.model.expand_d_hidden,
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


def _read_batch(cache: PackedPerTokenCache, idxs):
    """Read rows `idxs` from the cache into a padded (H [B, Lmax, d], mask [B, Lmax])."""
    rows = [cache.get(int(i)) for i in idxs]
    lmax = max(h.size(0) for h, _ in rows)
    d = rows[0][0].size(-1)
    H = torch.zeros(len(rows), lmax, d, dtype=torch.float32)
    M = torch.zeros(len(rows), lmax, dtype=torch.bool)
    for j, (h, m) in enumerate(rows):
        L = h.size(0)
        H[j, :L] = h.float()
        M[j, :L] = m
    return H, M


@torch.no_grad()
def project_pooled(cache, proj, idxs, device, batch_size, renormalize, log_tag=""):
    """Mean-pool the per-token projected embeddings over valid tokens -> [N, D]."""
    chunks = []
    n = len(idxs)
    for s in range(0, n, batch_size):
        H, M = _read_batch(cache, idxs[s:s + batch_size])
        z = proj(H.to(device))                                  # [B, L, D], unit per token
        Md = M.to(device)
        denom = Md.sum(1, keepdim=True).clamp_min(1).to(z.dtype)
        pooled = (z * Md.unsqueeze(-1)).sum(1) / denom          # [B, D]
        if renormalize:
            pooled = F.normalize(pooled, dim=-1)
        chunks.append(pooled.float().cpu())
        if log_tag and (s // batch_size) % 50 == 0:
            print(f"  [{log_tag}] {min(s + batch_size, n)}/{n}", flush=True)
    return torch.cat(chunks, 0) if chunks else torch.zeros(0, proj.fc3.out_features)


@torch.no_grad()
def project_per_token(cache, proj, idxs, device, batch_size, out_prefix, log_tag=""):
    """Stream full per-token embeddings to <out_prefix>_z.f32.bin + _offsets.npy.

    Flat layout: row i occupies [offsets[i], offsets[i+1]) of the [*, D] buffer,
    matching the packed cache convention. Streamed so it never holds the whole
    (potentially tens of GB) array in memory.
    """
    n = len(idxs)
    offsets = [0]
    with open(f"{out_prefix}_z.f32.bin", "wb") as fbin:
        for s in range(0, n, batch_size):
            H, M = _read_batch(cache, idxs[s:s + batch_size])
            z = proj(H.to(device)).float().cpu()                # [B, L, D]
            for j in range(z.size(0)):
                zr = z[j][M[j]].contiguous().numpy()            # [L_valid, D]
                zr.tofile(fbin)
                offsets.append(offsets[-1] + zr.shape[0])
            if log_tag and (s // batch_size) % 50 == 0:
                print(f"  [{log_tag}] {min(s + batch_size, n)}/{n}", flush=True)
    np.save(f"{out_prefix}_offsets.npy", np.asarray(offsets, dtype=np.int64))
    return offsets[-1]


def run_cached(args, cfg) -> None:
    """Single-process: project the precomputed cache for the requested splits."""
    device = _device(args.device)
    print(f"[export] cached  device={device} ckpt={args.ckpt} pooling={args.pooling}", flush=True)
    model = load_retrieval(args.ckpt, device)

    cache_dir = Path(args.cache_dir)
    p_cache = PackedPerTokenCache(str(cache_dir), "protein", cfg.model.protein_hidden)
    t_cache = PackedPerTokenCache(str(cache_dir), "text", cfg.model.text_hidden)
    with open(cache_dir / "pair_ids.json") as f:
        pair_ids = json.load(f)                       # accession per CSV row
    with open(cache_dir / "protein_ids.json") as f:
        protein_ids = json.load(f)                    # accession per unique protein
    row_protein_idx = load_row_protein_idx(str(cache_dir))
    splits = load_splits(str(cache_dir / "splits.json"))

    requested = ["train", "val", "test"] if args.splits == "all" else \
        [s.strip() for s in args.splits.split(",") if s.strip()]
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in requested:
        if split not in splits:
            raise SystemExit(f"split {split!r} not in splits.json ({list(splits)})")
        rows = list(splits[split])
        # Protein rows for this split = the unique proteins its captions point to.
        p_idx = sorted({int(row_protein_idx[r]) for r in rows})
        print(f"[export] split={split}: {len(rows)} captions, {len(p_idx)} proteins",
              flush=True)

        if args.pooling == "mean":
            arrays = {}
            if "text" in modalities:
                z_t = project_pooled(t_cache, model.text_proj, rows, device,
                                     args.batch_size, args.renormalize, f"{split}:text")
                arrays["z_t"] = z_t.numpy()
                arrays["acc_t"] = np.array([pair_ids[r] for r in rows], dtype="U16")
                arrays["row_t"] = np.asarray(rows, dtype=np.int64)
            if "protein" in modalities:
                z_p = project_pooled(p_cache, model.protein_proj, p_idx, device,
                                    args.batch_size, args.renormalize, f"{split}:protein")
                arrays["z_p"] = z_p.numpy()
                arrays["acc_p"] = np.array([protein_ids[i] for i in p_idx], dtype="U16")
            path = out_dir / f"{split}_pooled.npz"
            np.savez(path, **arrays)
            print(f"[export] wrote {path}", flush=True)
        else:
            meta = {"split": split, "ckpt": args.ckpt, "embed_dim": cfg.model.embed_dim}
            if "text" in modalities:
                prefix = str(out_dir / f"{split}_text")
                tot = project_per_token(t_cache, model.text_proj, rows, device,
                                       args.batch_size, prefix, f"{split}:text")
                with open(f"{prefix}_accessions.json", "w") as f:
                    json.dump([pair_ids[r] for r in rows], f)
                meta["text_tokens"] = tot
                print(f"[export] wrote {prefix}_z.f32.bin ({tot} tokens) + offsets + accessions",
                      flush=True)
            if "protein" in modalities:
                prefix = str(out_dir / f"{split}_protein")
                tot = project_per_token(p_cache, model.protein_proj, p_idx, device,
                                       args.batch_size, prefix, f"{split}:protein")
                with open(f"{prefix}_accessions.json", "w") as f:
                    json.dump([protein_ids[i] for i in p_idx], f)
                meta["protein_tokens"] = tot
                print(f"[export] wrote {prefix}_z.f32.bin ({tot} tokens) + offsets + accessions",
                      flush=True)
            with open(out_dir / f"{split}_pertoken_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    print("[export] done", flush=True)


# ---------------------------------------------------------------------------
# Live (no cache): distributed encode + project over a CSV's rows
# ---------------------------------------------------------------------------
def _shard_range(n: int, rank: int, world: int):
    per, rem = divmod(n, world)
    start = rank * per + min(rank, rem)
    return start, start + per + (1 if rank < rem else 0)


@torch.no_grad()
def _live_shard(items, ids, start, prefix, encode_fn, proj, shards_dir, env,
                batch_size, pooling, renormalize):
    """Encode + project one rank's slice; write a pooled .npz or a packed shard."""
    tag = f"{env.rank:05d}"
    if pooling == "mean":
        chunks = []
        for s in range(0, len(items), batch_size):
            h, m = encode_fn(items[s:s + batch_size])
            z = proj(h.float())
            md = m.to(z.dtype)
            denom = md.sum(1, keepdim=True).clamp_min(1)
            pooled = (z * md.unsqueeze(-1)).sum(1) / denom
            if renormalize:
                pooled = F.normalize(pooled, dim=-1)
            chunks.append(pooled.float().cpu())
        z = torch.cat(chunks, 0) if chunks else torch.zeros(0, proj.fc3.out_features)
        np.savez(shards_dir / f"{prefix}.{tag}.npz",
                 z=z.numpy(), ids=np.array(ids, dtype="U16"), start=start)
    else:
        offsets = [0]
        with open(shards_dir / f"{prefix}_z.{tag}.f32.bin", "wb") as fbin:
            for s in range(0, len(items), batch_size):
                h, m = encode_fn(items[s:s + batch_size])
                z = proj(h.float()).cpu()
                mm = m.cpu()
                for j in range(z.size(0)):
                    zr = z[j][mm[j]].contiguous().numpy()
                    zr.tofile(fbin)
                    offsets.append(offsets[-1] + zr.shape[0])
        with open(shards_dir / f"{prefix}meta.{tag}.json", "w") as f:
            json.dump({"start": start, "offsets": offsets, "ids": ids}, f)


def _merge_live_pooled(shards_dir, prefix):
    """Concatenate pooled shard .npz files in start order -> (z [N,D], ids [N])."""
    parts = []
    for p in glob.glob(str(shards_dir / f"{prefix}.*.npz")):
        d = np.load(p, allow_pickle=False)
        parts.append((int(d["start"]), d["z"], d["ids"]))
    parts.sort(key=lambda t: t[0])
    z = np.concatenate([p[1] for p in parts], 0) if parts else np.zeros((0, 0), np.float32)
    ids = np.concatenate([p[2] for p in parts], 0) if parts else np.array([], dtype="U16")
    return z, ids


def _merge_live_pertoken(shards_dir, shard_prefix, out_prefix):
    """Concatenate packed per-token shards (start order) -> flat bin + offsets + ids.

    `shard_prefix` is the modality the shards were written under ("text"/
    "protein"); `out_prefix` is the final output basename.
    """
    pieces = []
    for mp in glob.glob(str(shards_dir / f"{shard_prefix}meta.*.json")):
        with open(mp) as f:
            meta = json.load(f)
        tag = Path(mp).name.split(".")[-2]
        pieces.append((meta["start"], tag, meta))
    pieces.sort(key=lambda t: t[0])

    offsets = [0]
    ids = []
    bufsize = 64 * 1024 * 1024
    with open(f"{out_prefix}_z.f32.bin", "wb") as out:
        for _, tag, meta in pieces:
            with open(shards_dir / f"{shard_prefix}_z.{tag}.f32.bin", "rb") as src:
                while True:
                    b = src.read(bufsize)
                    if not b:
                        break
                    out.write(b)
            for c in np.diff(meta["offsets"]):
                offsets.append(offsets[-1] + int(c))
            ids.extend(meta["ids"])
    np.save(f"{out_prefix}_offsets.npy", np.asarray(offsets, dtype=np.int64))
    with open(f"{out_prefix}_accessions.json", "w") as f:
        json.dump(ids, f)
    return offsets[-1]


def run_live(args, cfg) -> None:
    """Distributed: encode every CSV row live, project, and export embeddings.

    No cache, no splits — all rows. Proteins are deduped (encoded once per unique
    accession). Launch under mpiexec to shard across tiles; without MPI it runs
    as a single rank.
    """
    from src.dist import init_distributed, barrier, cleanup
    from src.encoders import (
        load_text_encoder, load_protein_encoder,
        encode_text_batch, encode_protein_batch,
    )

    env = init_distributed(args.device, group_size=1, init_pg=False)
    device = env.device

    pairs = load_pairs(
        args.csv, id_col=cfg.data.csv_id_col, protein_col=cfg.data.csv_protein_col,
        text_col=cfg.data.csv_text_col, pfam_col=cfg.data.csv_pfam_col,
        subset_size=args.subset_size,
    )
    row_protein_idx, unique_seqs, unique_ids = dedup_proteins(pairs)
    n_rows, n_uniq = len(pairs), len(unique_seqs)
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if env.is_main:
        print(f"[export] live  device={device} world={env.world_size} "
              f"rows={n_rows} unique_proteins={n_uniq} pooling={args.pooling}", flush=True)

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / f"{args.name}_shards"
    if env.is_main:
        shards_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    # Rank-0-first load (AMPLIFY's trust_remote_code module cache is written once).
    def _load():
        tm, tt = load_text_encoder(cfg.model.text_encoder_path, device)
        pm, pt = load_protein_encoder(cfg.model.protein_encoder_path, device)
        rm = load_retrieval(args.ckpt, device)
        return tm, tt, pm, pt, rm
    models = _load() if env.is_main else None
    barrier()
    if not env.is_main:
        models = _load()
    text_model, text_tok, prot_model, prot_tok, model = models
    barrier()

    def enc_text(strs):
        return encode_text_batch(text_model, text_tok, strs, device,
                                 cfg.data.max_text_tokens,
                                 cfg.retrieval.mask_text_special_tokens)

    def enc_prot(strs):
        return encode_protein_batch(prot_model, prot_tok, strs, device,
                                    cfg.data.max_protein_tokens,
                                    cfg.retrieval.mask_protein_special_tokens)

    if "protein" in modalities:
        ps, pe = _shard_range(n_uniq, env.rank, env.world_size)
        _live_shard(unique_seqs[ps:pe], unique_ids[ps:pe], ps, "protein",
                    enc_prot, model.protein_proj, shards_dir, env,
                    args.batch_size, args.pooling, args.renormalize)
    if "text" in modalities:
        ts, te = _shard_range(n_rows, env.rank, env.world_size)
        _live_shard([p.text for p in pairs[ts:te]], [p.uid for p in pairs[ts:te]], ts,
                    "text", enc_text, model.text_proj, shards_dir, env,
                    args.batch_size, args.pooling, args.renormalize)
    barrier()

    if env.is_main:
        if args.pooling == "mean":
            arrays = {}
            if "protein" in modalities:
                arrays["z_p"], arrays["acc_p"] = _merge_live_pooled(shards_dir, "protein")
            if "text" in modalities:
                arrays["z_t"], arrays["acc_t"] = _merge_live_pooled(shards_dir, "text")
                arrays["row_t"] = np.arange(n_rows, dtype=np.int64)
            path = out_dir / f"{args.name}_pooled.npz"
            np.savez(path, **arrays)
            print(f"[export] wrote {path}", flush=True)
        else:
            if "text" in modalities:
                tot = _merge_live_pertoken(shards_dir, "text", str(out_dir / f"{args.name}_text"))
                print(f"[export] wrote {args.name}_text_z.f32.bin ({tot} tokens)", flush=True)
            if "protein" in modalities:
                tot = _merge_live_pertoken(shards_dir, "protein", str(out_dir / f"{args.name}_protein"))
                print(f"[export] wrote {args.name}_protein_z.f32.bin ({tot} tokens)", flush=True)
        print("[export] done", flush=True)
    barrier()
    cleanup()


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained retrieval checkpoint")
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--csv", default=None,
                    help="encode this CSV live (no cache, no splits); distributed under mpiexec")
    ap.add_argument("--out-dir", default=str(Path(cfg.retrieval.cache_dir).parent / "embeddings"))
    ap.add_argument("--name", default="live",
                    help="output basename for live mode (e.g. <name>_pooled.npz)")
    ap.add_argument("--splits", default="train,test",
                    help="cache mode: comma list of train/val/test, or 'all'")
    ap.add_argument("--modalities", default="protein,text", help="comma list: protein,text")
    ap.add_argument("--pooling", default="mean", choices=["mean", "none"],
                    help="mean = one vector per item; none = full per-token packed")
    ap.add_argument("--renormalize", action="store_true",
                    help="L2-normalize pooled vectors (pooling=mean)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="rows per batch; default 512 cached (projection only) / "
                         "64 live (the batch runs through the encoders, where "
                         "attention memory is O(batch * seqlen^2))")
    ap.add_argument("--subset-size", type=int, default=0, help="live mode: first N CSV rows")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Live runs the encoders (AMPLIFY attention is materialized full, ~[B,H,L,L],
    # by the non-fused XPU path), so it needs a small batch like precompute; the
    # cached path only feeds the tiny projection head and can go large.
    if args.batch_size is None:
        args.batch_size = 64 if args.csv else 512

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv:
        run_live(args, cfg)
    else:
        run_cached(args, cfg)


if __name__ == "__main__":
    main()
