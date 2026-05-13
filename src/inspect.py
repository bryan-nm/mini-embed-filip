"""Token × token similarity matrix utility.

Used for:
- Per-pair interpretability (residue ↔ word alignment maps)
- Dimensionality sweep diagnostics (compare matrices across embed_dim values)

Returns the full [L_p, L_t] matrix plus enough metadata (masks, token labels)
to plot or analyze.

CLI:
  python -m src.inspect --pair-id P0C9F0 --ckpt checkpoints/retrieval/epoch04.pt
  python -m src.inspect --pair-idx 42 --use-cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.data import (
    PackedPerTokenCache,
    load_pairs,
)
from src.model import MiniEmbedFilip


def _load_model(ckpt_path: str, device: torch.device) -> MiniEmbedFilip:
    cfg = default_cfg()
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
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval().to(device)
    return model


def compute_similarity_matrix_from_cache(
    model: MiniEmbedFilip,
    cache_dir: str,
    pair_idx: int,
    device: torch.device,
) -> dict:
    """Build the [L_p, L_t] FILIP-precursor similarity matrix from cache."""
    cfg = default_cfg()
    p_cache = PackedPerTokenCache(cache_dir, "protein", cfg.model.protein_hidden)
    t_cache = PackedPerTokenCache(cache_dir, "text", cfg.model.text_hidden)

    h_p, m_p = p_cache.get(pair_idx)
    h_t, m_t = t_cache.get(pair_idx)
    h_p = h_p.to(device).float().unsqueeze(0)               # [1, L_p, 640]
    h_t = h_t.to(device).float().unsqueeze(0)               # [1, L_t, 768]
    m_p = m_p.to(device).unsqueeze(0)
    m_t = m_t.to(device).unsqueeze(0)

    with torch.no_grad():
        z_p, z_t = model.project(h_p, h_t)
    S = (z_p[0] @ z_t[0].t()).cpu()                         # [L_p, L_t]

    with open(Path(cache_dir) / "pair_ids.json") as f:
        ids = json.load(f)

    return {
        "uid": ids[pair_idx],
        "S": S,
        "mask_p": m_p[0].cpu(),
        "mask_t": m_t[0].cpu(),
        "z_p": z_p[0].cpu(),
        "z_t": z_t[0].cpu(),
    }


def compute_similarity_matrix_live(
    model: MiniEmbedFilip,
    protein_seq: str,
    text: str,
    device: torch.device,
    mask_text_specials: bool = True,
    mask_protein_specials: bool = True,
    max_protein_tokens: int = 1024,
    max_text_tokens: int = 1024,
) -> dict:
    """Same as cache version, but encodes fresh inputs (used for novel pairs)."""
    cfg = default_cfg()
    from src.encoders import (
        load_protein_encoder, load_text_encoder,
        encode_protein_batch, encode_text_batch,
    )
    text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)

    h_t, m_t = encode_text_batch(text_model, text_tok, [text], device,
                                 max_text_tokens, mask_text_specials)
    h_p, m_p = encode_protein_batch(prot_model, prot_tok, [protein_seq], device,
                                    max_protein_tokens, mask_protein_specials)
    with torch.no_grad():
        z_p, z_t = model.project(h_p.float(), h_t.float())
    S = (z_p[0] @ z_t[0].t()).cpu()

    # Resolve token labels
    text_ids = text_tok([text], padding=True, truncation=True,
                        max_length=max_text_tokens, return_tensors="pt")["input_ids"][0]
    text_tokens = text_tok.convert_ids_to_tokens(text_ids.tolist())
    protein_ids = prot_tok([protein_seq], padding=True, truncation=True,
                           max_length=max_protein_tokens, return_tensors="pt")["input_ids"][0]
    # Protein tokenizer for AMPLIFY is character-level. Convert via tokenizer if available.
    if hasattr(prot_tok, "convert_ids_to_tokens"):
        protein_tokens = prot_tok.convert_ids_to_tokens(protein_ids.tolist())
    else:
        protein_tokens = [str(int(i)) for i in protein_ids.tolist()]

    return {
        "S": S,
        "mask_p": m_p[0].cpu(),
        "mask_t": m_t[0].cpu(),
        "z_p": z_p[0].cpu(),
        "z_t": z_t[0].cpu(),
        "tokens_p": protein_tokens,
        "tokens_t": text_tokens,
    }


def top_k_alignments(out: dict, k: int = 5):
    """For each valid protein position, return its top-k text-token matches."""
    S = out["S"]
    mask_p = out["mask_p"]
    mask_t = out["mask_t"]
    L_p, L_t = S.shape

    results = []
    for i in range(L_p):
        if not mask_p[i]:
            continue
        row = S[i].clone()
        row[~mask_t] = -1e9
        scores, idxs = row.topk(min(k, mask_t.sum().item()))
        results.append({"p_idx": i, "top_text_idxs": idxs.tolist(),
                        "top_scores": scores.tolist()})
    return results


def plot_heatmap(out: dict, path: str = None, max_dim: int = 200):
    """Optional matplotlib heatmap. Truncates very long sequences for display."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return

    S = out["S"]
    mask_p = out["mask_p"]
    mask_t = out["mask_t"]
    # Keep only valid rows/cols
    S = S[mask_p][:, mask_t]
    if S.size(0) > max_dim:
        S = S[:max_dim]
    if S.size(1) > max_dim:
        S = S[:, :max_dim]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(S.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("text token index")
    ax.set_ylabel("protein residue index")
    fig.colorbar(im, ax=ax)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"[inspect] wrote {path}")
    else:
        plt.show()


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--pair-id", type=str, default=None,
                    help="UniProt ID to look up in pair_ids.json")
    ap.add_argument("--pair-idx", type=int, default=None,
                    help="Direct row index into the cache")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--plot", type=str, default=None,
                    help="path to save heatmap PNG")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.ckpt, device)

    if args.pair_idx is not None:
        idx = args.pair_idx
    elif args.pair_id is not None:
        with open(Path(args.cache_dir) / "pair_ids.json") as f:
            ids = json.load(f)
        idx = ids.index(args.pair_id)
    else:
        raise SystemExit("Provide --pair-id or --pair-idx")

    out = compute_similarity_matrix_from_cache(model, args.cache_dir, idx, device)
    print(f"[inspect] uid={out['uid']}  S.shape={tuple(out['S'].shape)}  "
          f"valid_p={int(out['mask_p'].sum())}  valid_t={int(out['mask_t'].sum())}")
    print(f"[inspect] FILIP score (mean of max-sim, both dirs): "
          f"{0.5 * (out['S'].max(dim=1).values[out['mask_p']].mean().item() + out['S'].max(dim=0).values[out['mask_t']].mean().item()):.4f}")

    tops = top_k_alignments(out, args.top_k)
    print(f"[inspect] top-{args.top_k} text matches for the first 5 protein positions:")
    for r in tops[:5]:
        print(f"  p_idx={r['p_idx']:4d}  text_idxs={r['top_text_idxs']}  "
              f"scores={[round(s, 3) for s in r['top_scores']]}")

    if args.plot:
        plot_heatmap(out, args.plot)


if __name__ == "__main__":
    main()
