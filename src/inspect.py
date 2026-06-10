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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg
from src.data import (
    PackedPerTokenCache,
    load_row_protein_idx,
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

    # Protein-dedup layout: the text cache is per CSV row, the protein cache is
    # per unique protein. Map the row index to its protein row before reading.
    row_protein_idx = load_row_protein_idx(cache_dir)
    h_p, m_p = p_cache.get(int(row_protein_idx[pair_idx]))
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


def load_inspect_encoders(device: torch.device):
    """Load the two frozen encoders once, to reuse across many live pairs."""
    cfg = default_cfg()
    from src.encoders import load_protein_encoder, load_text_encoder
    text_model, text_tok = load_text_encoder(cfg.model.text_encoder_path, device)
    prot_model, prot_tok = load_protein_encoder(cfg.model.protein_encoder_path, device)
    return text_model, text_tok, prot_model, prot_tok


def compute_similarity_matrix_live(
    model: MiniEmbedFilip,
    protein_seq: str,
    text: str,
    device: torch.device,
    *,
    encoders=None,
    uid: Optional[str] = None,
    mask_text_specials: bool = True,
    mask_protein_specials: bool = True,
    max_protein_tokens: int = 1024,
    max_text_tokens: int = 1024,
) -> dict:
    """Same as cache version, but encodes fresh inputs (used for novel pairs).

    Pass `encoders` (from `load_inspect_encoders`) to avoid reloading the frozen
    encoders for every protein when inspecting a batch.
    """
    from src.encoders import encode_protein_batch, encode_text_batch
    if encoders is None:
        encoders = load_inspect_encoders(device)
    text_model, text_tok, prot_model, prot_tok = encoders

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
        "uid": uid,
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


def display_matrix(out: dict, max_dim: int = 1024) -> torch.Tensor:
    """The valid-token (specials dropped), max_dim-clipped [P, T] matrix that the
    heatmap renders — also what `write_matrix_npy` saves, so PNG and .npy agree.

    Axes are valid tokens in order; cross-reference labels via the `valid==1`
    rows of the `*_tokens.tsv` files (same order).
    """
    S = out["S"][out["mask_p"]][:, out["mask_t"]]
    return S[:max_dim, :max_dim]


def write_matrix_npy(out: dict, path: str, max_dim: int = 1024) -> None:
    """Save the plotted similarity values as a .npy float array [P_valid, T_valid]."""
    np.save(path, display_matrix(out, max_dim).numpy())
    print(f"[inspect] wrote {path}")


def plot_heatmap(out: dict, path: str = None, max_dim: int = 1024):
    """Optional matplotlib heatmap. Truncates very long sequences for display."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return

    S = display_matrix(out, max_dim)

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


def lookup_pairs_in_csv(csv_path: str, accessions, cfg) -> dict:
    """Stream the CSV once, returning {accession: (protein_seq, text)} for the
    requested accessions (those present). One pass, so it scales to many ids."""
    import csv as _csv
    want = list(dict.fromkeys(accessions))            # de-dup, keep order
    want_set = set(want)
    found: dict = {}
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for col in (cfg.data.csv_id_col, cfg.data.csv_protein_col, cfg.data.csv_text_col):
            if col not in (reader.fieldnames or []):
                raise SystemExit(f"CSV {csv_path} missing column {col!r}; "
                                 f"has {reader.fieldnames}")
        for row in reader:
            a = row[cfg.data.csv_id_col]
            if a in want_set and a not in found:
                found[a] = (row[cfg.data.csv_protein_col], row[cfg.data.csv_text_col])
                if len(found) == len(want_set):
                    break
    return found


def report(out: dict, top_k: int) -> None:
    """Print the per-pair summary + top-k alignments for one similarity result."""
    S, mp, mt = out["S"], out["mask_p"], out["mask_t"]
    uid = out.get("uid")
    filip = 0.5 * (
        S.max(dim=1).values[mp].mean().item() + S.max(dim=0).values[mt].mean().item()
    )
    tag = f"uid={uid}  " if uid else ""
    print(f"[inspect] {tag}S.shape={tuple(S.shape)}  valid_p={int(mp.sum())}  "
          f"valid_t={int(mt.sum())}  FILIP={filip:.4f}")
    for r in top_k_alignments(out, top_k)[:5]:
        print(f"  p_idx={r['p_idx']:4d}  text_idxs={r['top_text_idxs']}  "
              f"scores={[round(s, 3) for s in r['top_scores']]}")


def write_token_tsvs(out: dict, out_dir: str, uid: str, max_dim: int = 1024) -> None:
    """Write index->token TSVs, one file per modality, with two index columns:

      index       raw token position — matches the full `out["S"]` axes and the
                  `top_k_alignments` `text_idxs` (includes specials).
      plot_index  row/col position in the rendered heatmap / `<uid>.npy` (valid
                  tokens only, same `max_dim` clip); -1 if the token isn't shown
                  (a masked special, or clipped past `max_dim`).

    `valid` flags real tokens vs masked specials ([CLS]/[SEP]/<bos>/<eos>/pad).
    Keep `max_dim` in sync with `display_matrix`/`write_matrix_npy` so plot_index
    lines up with the saved array.
    """
    import csv as _csv
    for modality, tok_key, mask_key in (("text", "tokens_t", "mask_t"),
                                        ("protein", "tokens_p", "mask_p")):
        tokens = out.get(tok_key)
        if tokens is None:                        # cache mode carries no labels
            continue
        mask = out[mask_key].tolist()
        path = Path(out_dir) / f"{uid}_{modality}_tokens.tsv"
        with open(path, "w", newline="") as f:
            w = _csv.writer(f, delimiter="\t")
            w.writerow(["index", "token", "valid", "plot_index"])
            valid_ord = 0                         # position among valid tokens
            for i, tok in enumerate(tokens):
                valid = int(i < len(mask) and bool(mask[i]))
                if valid:
                    plot_index = valid_ord if valid_ord < max_dim else -1
                    valid_ord += 1
                else:
                    plot_index = -1
                w.writerow([i, tok, valid, plot_index])


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=5)
    # Cache mode (run-2 cache, dedup-aware)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--pair-id", type=str, default=None,
                    help="accession to look up in the cache's pair_ids.json")
    ap.add_argument("--pair-idx", type=int, default=None,
                    help="direct CSV-row index into the cache")
    # Live mode: a single explicit pair
    ap.add_argument("--protein", type=str, default=None, help="raw protein sequence")
    ap.add_argument("--text", type=str, default=None, help="raw caption text")
    # Live mode: look accessions up in a CSV
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV to resolve --protein-id / --id-file accessions")
    ap.add_argument("--protein-id", nargs="+", default=None,
                    help="one or more primary_Accession to inspect live (needs --csv)")
    ap.add_argument("--id-file", type=str, default=None,
                    help="file with one primary_Accession per line (needs --csv)")
    # Output
    ap.add_argument("--plot", type=str, default=None,
                    help="heatmap PNG path (single-pair modes)")
    ap.add_argument("--plot-dir", type=str, default=None,
                    help="directory for per-accession heatmaps (one <uid>.png each)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.ckpt, device)

    # --- Cache mode (no live encoders needed) ---
    if args.pair_id is not None or args.pair_idx is not None:
        if args.pair_idx is not None:
            idx = args.pair_idx
        else:
            with open(Path(args.cache_dir) / "pair_ids.json") as f:
                idx = json.load(f).index(args.pair_id)
        out = compute_similarity_matrix_from_cache(model, args.cache_dir, idx, device)
        report(out, args.top_k)
        if args.plot:
            write_matrix_npy(out, str(Path(args.plot).with_suffix(".npy")))
            plot_heatmap(out, args.plot)
        return

    # --- Live mode: build the worklist of (uid, protein, text) ---
    worklist = []
    if args.protein and args.text:
        worklist.append(("query", args.protein, args.text))
    elif args.protein_id or args.id_file:
        if not args.csv:
            raise SystemExit("--protein-id / --id-file require --csv")
        ids = list(args.protein_id or [])
        if args.id_file:
            ids += [ln.strip() for ln in open(args.id_file) if ln.strip()]
        if not ids:
            raise SystemExit("no accessions provided")
        found = lookup_pairs_in_csv(args.csv, ids, cfg)
        missing = [a for a in ids if a not in found]
        if missing:
            print(f"[inspect] WARNING: not found in CSV: {missing}")
        worklist = [(a, found[a][0], found[a][1]) for a in ids if a in found]
    else:
        raise SystemExit(
            "Provide one of: --protein + --text, --protein-id/--id-file + --csv, "
            "or --pair-id/--pair-idx (cache).")

    if not worklist:
        raise SystemExit("nothing to inspect")

    encoders = load_inspect_encoders(device)          # loaded once, reused
    if args.plot_dir:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    for uid, protein, text in worklist:
        out = compute_similarity_matrix_live(
            model, protein, text, device, encoders=encoders, uid=uid,
            mask_text_specials=cfg.retrieval.mask_text_special_tokens,
            mask_protein_specials=cfg.retrieval.mask_protein_special_tokens,
            max_protein_tokens=cfg.data.max_protein_tokens,
            max_text_tokens=cfg.data.max_text_tokens,
        )
        report(out, args.top_k)
        if args.plot_dir:
            write_token_tsvs(out, args.plot_dir, uid)
            write_matrix_npy(out, str(Path(args.plot_dir) / f"{uid}.npy"))
            plot_heatmap(out, str(Path(args.plot_dir) / f"{uid}.png"))
        elif args.plot:
            write_matrix_npy(out, str(Path(args.plot).with_suffix(".npy")))
            plot_heatmap(out, args.plot)


if __name__ == "__main__":
    main()
