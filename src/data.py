"""Data loading, splits, and packed per-token cache.

Three concepts:

- `load_pairs` / splits utilities — unchanged from the pooled version; splits
  are fingerprinted by (n, seed, ratios).
- `PackedPerTokenCache` — memmap-backed reader for the per-token encoder
  outputs produced by `src/precompute.py`. Storage layout is a flat bf16
  buffer per modality + a per-row `[N+1]` offsets table.
- Two dataset classes — `PackedPerTokenDataset` for the cached path,
  `RawPairsDataset` for the live-encoder fallback.

Collation pads variable-length rows in the batch to that batch's max length
and returns matching attention masks (the "valid token" masks include the
encoder's actual valid positions; whether specials are kept or stripped is
decided at precompute time).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# CSV captions in the SwissProt-full file are long; bump csv field cap so a
# single quoted caption can be hundreds of KB without DCO error.
csv.field_size_limit(10 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Raw text loading from CSV
# ---------------------------------------------------------------------------
@dataclass
class Pair:
    uid: str
    protein: str
    text: str
    pfam: Optional[str] = None     # raw pfam_label cell (e.g. "['PF13676']"), if available


def load_pairs(
    csv_path: str,
    *,
    id_col: str = "primary_Accession",
    protein_col: str = "protein_sequence",
    text_col: str = "[final]text_caption",
    pfam_col: Optional[str] = "pfam_label",
    subset_size: int = 0,
) -> List[Pair]:
    """Load (uid, protein, text) triples from a CSV file.

    The new SwissProt-full file ships with header row:
      primary_Accession, protein_sequence, [final]text_caption, pfam_label
    Captions contain commas inside quotes, so we use the csv module rather
    than splitting on commas manually.
    """
    pairs: List[Pair] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (id_col, protein_col, text_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns {missing}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            pairs.append(Pair(
                uid=row[id_col],
                protein=row[protein_col],
                text=row[text_col],
                pfam=(row.get(pfam_col) if pfam_col and pfam_col in row else None),
            ))
            if subset_size > 0 and len(pairs) >= subset_size:
                break
    return pairs


# ---------------------------------------------------------------------------
# Splits (fingerprinted)
# ---------------------------------------------------------------------------
def group_ids_from_accessions(accessions: Sequence[str]) -> np.ndarray:
    """Per-row accession strings -> dense int group ids in first-appearance order.

    Rows describing the same protein (same accession) get the same id, so a
    group-aware split keeps every caption of a protein on one side of the
    train/val/test boundary. The augmented SwissProt corpus has ~8.87 captions
    per protein; a row-level split would leak proteins across splits.
    """
    mapping: dict = {}
    out = np.empty(len(accessions), dtype=np.int64)
    for i, a in enumerate(accessions):
        gid = mapping.get(a)
        if gid is None:
            gid = len(mapping)
            mapping[a] = gid
        out[i] = gid
    return out


def dedup_proteins(pairs: List["Pair"]):
    """Collapse rows to unique proteins (keyed by accession, first-appearance).

    Returns:
      row_protein_idx : np.int64 [N_rows]  -- CSV row -> unique-protein index
      unique_seqs     : list[str]          -- one sequence per unique protein
      unique_ids      : list[str]          -- accessions, unique-protein order

    The augmented corpus repeats each protein across ~8.87 caption rows; encoding
    and storing the protein once per unique accession (rather than per row) saves
    ~9x of the protein encoder pass (the precompute bottleneck) and ~1.5 TB of
    cache. Dedup key is the accession, which is also the split group key, so the
    two stay consistent.
    """
    mapping: dict = {}
    row_protein_idx = np.empty(len(pairs), dtype=np.int64)
    unique_seqs: List[str] = []
    unique_ids: List[str] = []
    for i, p in enumerate(pairs):
        j = mapping.get(p.uid)
        if j is None:
            j = len(unique_seqs)
            mapping[p.uid] = j
            unique_seqs.append(p.protein)
            unique_ids.append(p.uid)
        row_protein_idx[i] = j
    return row_protein_idx, unique_seqs, unique_ids


def make_splits(
    n: int,
    ratios: Sequence[float],
    seed: int,
    group_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Train/val/test split over row indices [0, n).

    If `group_ids` is given (one per row), the split is done over *groups*
    (proteins) and then expanded back to rows, so no group straddles two
    splits. Without it, falls back to the legacy per-row split (used only by
    the live smoke-test path / single-caption corpora).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    if group_ids is None:
        perm = rng.permutation(n)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train = perm[:n_train]
        val = perm[n_train : n_train + n_val]
        test = perm[n_train + n_val :]
        n_groups = int(n)
    else:
        g = np.asarray(group_ids)
        assert g.shape[0] == n, "group_ids must have one entry per row"
        uniq = np.unique(g)                       # sorted unique group ids
        gperm = rng.permutation(uniq.shape[0])
        ng_train = int(round(ratios[0] * uniq.shape[0]))
        ng_val = int(round(ratios[1] * uniq.shape[0]))
        train_g = set(uniq[gperm[:ng_train]].tolist())
        val_g = set(uniq[gperm[ng_train : ng_train + ng_val]].tolist())
        train_rows, val_rows, test_rows = [], [], []
        for row, gid in enumerate(g.tolist()):
            if gid in train_g:
                train_rows.append(row)
            elif gid in val_g:
                val_rows.append(row)
            else:
                test_rows.append(row)
        # Shuffle row order within each split so a prefix slice (val_subset) is
        # still a random sample rather than accession-contiguous.
        train = rng.permutation(np.array(train_rows, dtype=np.int64))
        val = rng.permutation(np.array(val_rows, dtype=np.int64))
        test = rng.permutation(np.array(test_rows, dtype=np.int64))
        n_groups = int(uniq.shape[0])

    return {
        "n": int(n),
        "seed": int(seed),
        "ratios": list(ratios),
        "n_groups": n_groups,
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }


def build_or_load_splits(
    splits_path: str,
    n: int,
    ratios: Sequence[float],
    seed: int,
    group_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Return a valid split dict, rebuilding + saving if missing/stale.

    Caller must invoke this on rank 0 only and barrier before other ranks read
    the file (avoids a write race), matching the existing pattern.
    """
    n_groups = int(np.unique(np.asarray(group_ids)).shape[0]) if group_ids is not None else int(n)
    if Path(splits_path).exists():
        sp = load_splits(splits_path)
        if splits_are_valid(sp, n, seed, ratios, n_groups=n_groups):
            return sp
    sp = make_splits(n, ratios, seed, group_ids=group_ids)
    save_splits(sp, splits_path)
    return sp


def save_splits(splits: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def splits_are_valid(
    splits: dict, n: int, seed: int, ratios: Sequence[float],
    n_groups: Optional[int] = None,
) -> bool:
    if not isinstance(splits, dict):
        return False
    if splits.get("n") != int(n) or splits.get("seed") != int(seed):
        return False
    sr = splits.get("ratios")
    if sr is None or len(sr) != len(ratios):
        return False
    if not all(abs(a - b) < 1e-9 for a, b in zip(sr, ratios)):
        return False
    # When a group-aware split is expected, a file without a matching n_groups
    # (e.g. a legacy row-level split) is treated as stale and rebuilt.
    if n_groups is not None and splits.get("n_groups") != int(n_groups):
        return False
    return True


# ---------------------------------------------------------------------------
# Packed per-token cache
# ---------------------------------------------------------------------------
class PackedPerTokenCache:
    """Memmap-backed reader for one modality's packed per-token cache.

    Files in cache_dir (created by precompute.py):
      <prefix>_h.bin       flat bf16, shape == (total_tokens, d)
      <prefix>_offsets.pt  int64 tensor of shape [N+1]; row i lives at
                           [offsets[i], offsets[i+1]) in the flat buffer
      <prefix>_mask.bin    flat uint8, length == total_tokens
                           (1 = valid for FILIP / uniformity, 0 = invalid)
    """

    def __init__(self, cache_dir: str, prefix: str, d: int):
        cache = Path(cache_dir)
        self.prefix = prefix
        self.d = d
        self.offsets: torch.Tensor = torch.load(
            cache / f"{prefix}_offsets.pt", map_location="cpu"
        )
        self.n_rows = int(self.offsets.numel() - 1)
        total_tokens = int(self.offsets[-1].item())

        # bf16 hidden states
        self._h_mm = np.memmap(
            cache / f"{prefix}_h.bin",
            dtype=np.uint16,                      # bf16 isn't directly representable in numpy
            mode="r",
            shape=(total_tokens, d),
        )
        # uint8 valid mask
        self._mask_mm = np.memmap(
            cache / f"{prefix}_mask.bin",
            dtype=np.uint8, mode="r", shape=(total_tokens,),
        )

    def __len__(self) -> int:
        return self.n_rows

    def get(self, idx: int):
        s = int(self.offsets[idx].item())
        e = int(self.offsets[idx + 1].item())
        # Copy out of the memmap into owned numpy, then to torch bf16.
        h_np = np.asarray(self._h_mm[s:e]).copy()              # [L, d] uint16
        m_np = np.asarray(self._mask_mm[s:e]).copy()           # [L]
        # uint16 -> bf16: view-with-dtype after converting to a contiguous tensor.
        h = torch.from_numpy(h_np).view(torch.bfloat16)
        m = torch.from_numpy(m_np).bool()
        return h, m


def load_row_protein_idx(cache_dir: str) -> torch.Tensor:
    """Load the CSV-row -> unique-protein-index map written by precompute."""
    path = Path(cache_dir) / "row_protein_idx.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. This cache predates protein-dedup; rebuild with "
            f"`python -m src.precompute`."
        )
    return torch.load(path, map_location="cpu")


class PackedPerTokenDataset(Dataset):
    """Per-row paired packed reader over the protein-deduplicated cache.

    The text cache has one row per CSV row; the protein cache has one row per
    *unique* protein. `row_protein_idx[row]` maps a CSV row to its protein row.
    Returns a dict of bf16 tensors and bool masks for one (protein, text) pair;
    collation pads to the batch's max length. `idx` is the global CSV row index.
    """

    def __init__(self, cache_dir: str, indices: Sequence[int],
                 protein_dim: int = 640, text_dim: int = 768,
                 row_protein_idx: Optional[torch.Tensor] = None):
        self.indices = list(indices)
        self.protein_cache = PackedPerTokenCache(cache_dir, "protein", protein_dim)
        self.text_cache = PackedPerTokenCache(cache_dir, "text", text_dim)
        if row_protein_idx is None:
            row_protein_idx = load_row_protein_idx(cache_dir)
        self.row_protein_idx = row_protein_idx
        if len(self.text_cache) != int(self.row_protein_idx.numel()):
            raise ValueError(
                f"text cache rows ({len(self.text_cache)}) != row_protein_idx "
                f"length ({int(self.row_protein_idx.numel())})")
        if int(self.row_protein_idx.max()) >= len(self.protein_cache):
            raise ValueError("row_protein_idx references a protein row past the cache")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        h_p, m_p = self.protein_cache.get(int(self.row_protein_idx[idx]))
        h_t, m_t = self.text_cache.get(idx)
        return {"h_p": h_p, "h_t": h_t, "mask_p": m_p, "mask_t": m_t, "idx": idx}


def packed_collate(batch):
    """Pads variable-length rows to the batch's max length in each modality."""
    B = len(batch)

    L_p = max(b["h_p"].size(0) for b in batch)
    L_t = max(b["h_t"].size(0) for b in batch)
    d_p = batch[0]["h_p"].size(-1)
    d_t = batch[0]["h_t"].size(-1)

    h_p = torch.zeros(B, L_p, d_p, dtype=torch.bfloat16)
    h_t = torch.zeros(B, L_t, d_t, dtype=torch.bfloat16)
    mask_p = torch.zeros(B, L_p, dtype=torch.bool)
    mask_t = torch.zeros(B, L_t, dtype=torch.bool)
    idx = torch.empty(B, dtype=torch.long)

    for i, b in enumerate(batch):
        lp = b["h_p"].size(0)
        lt = b["h_t"].size(0)
        h_p[i, :lp] = b["h_p"]
        h_t[i, :lt] = b["h_t"]
        mask_p[i, :lp] = b["mask_p"]
        mask_t[i, :lt] = b["mask_t"]
        idx[i] = b["idx"]
    return {"h_p": h_p, "h_t": h_t, "mask_p": mask_p, "mask_t": mask_t, "idx": idx}


# ---------------------------------------------------------------------------
# Raw fallback (live encoders during training)
# ---------------------------------------------------------------------------
class RawPairsDataset(Dataset):
    def __init__(self, pairs: List[Pair], indices: Sequence[int]):
        self.pairs = pairs
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        p = self.pairs[idx]
        return {"protein": p.protein, "text": p.text, "idx": idx}


def raw_collate(batch):
    return {
        "protein": [b["protein"] for b in batch],
        "text": [b["text"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Cache fingerprint (encoder + tokenizer + length cap config)
# ---------------------------------------------------------------------------
def cache_fingerprint(
    text_encoder_path: str, protein_encoder_path: str,
    max_text_tokens: int, max_protein_tokens: int,
    mask_text_specials: bool, mask_protein_specials: bool,
) -> dict:
    return {
        # Bumped when the on-disk layout changes; v2 stores protein rows
        # deduplicated by accession + a row_protein_idx map. A v1 cache will
        # mismatch here and be flagged for rebuild.
        "format": "v2_protein_dedup",
        "text_encoder_path": text_encoder_path,
        "protein_encoder_path": protein_encoder_path,
        "max_text_tokens": int(max_text_tokens),
        "max_protein_tokens": int(max_protein_tokens),
        "mask_text_specials": bool(mask_text_specials),
        "mask_protein_specials": bool(mask_protein_specials),
    }


def write_cache_fingerprint(cache_dir: str, fp: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cache_dir) / "fingerprint.json", "w") as f:
        json.dump(fp, f, indent=2)


def read_cache_fingerprint(cache_dir: str) -> dict:
    path = Path(cache_dir) / "fingerprint.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fingerprint_matches(saved: dict, expected: dict) -> bool:
    if not saved:
        return False
    for k, v in expected.items():
        if saved.get(k) != v:
            return False
    return True
