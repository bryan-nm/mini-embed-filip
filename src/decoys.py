"""Hard decoy captions: field-swap construction, plan building, packed reader.

A *decoy* for caption i is caption i with exactly one templated field replaced by
the corresponding field of some other caption j:

    truth  PROTEIN NAME: Sho1. FUNCTION: plasma membrane osmosensor... LINEAGE: Fungi...
    decoy  PROTEIN NAME: Sho1. FUNCTION: catalyzes the ATP-dependent...  LINEAGE: Fungi...
                               ^^^^^^^^^ taken verbatim from another caption

Every other token is shared with the truth, so the decoy sits far inside the
in-batch negatives' neighbourhood: ranking it below the true caption requires the
model to actually read the swapped field instead of matching overall topic. That
is the signal `src/train_hard_negatives.py` mines.

Two things live here:

- `parse_caption_fields` — split a caption into (label, value-span) segments.
  Captions do NOT all carry the same fields (SwissProt annotation is sparse:
  most entries have PROTEIN NAME + LINEAGE, far fewer have CATALYTIC ACTIVITY or
  PTM), so both the anchor's swappable set and the donor pool are per-label.
- `build_decoy_plan` — the deterministic (row, field, donor) assignment. Built
  once on one rank and saved; the encode ranks just apply it. The plan stores
  character spans, so applying it is pure string slicing, no re-parsing.

The training-side reader (`PackedDecoyDataset` / `decoy_collate`) lives in
`src/data.py` next to the base packed reader it extends. This module is
deliberately torch-free so `python -m src.decoys --selftest` runs anywhere.

Donor rejection rules (all enforced in `build_decoy_plan`):
  * different value — a byte-identical replacement would leave the decoy equal to
    the truth, i.e. a false negative the loss must never see;
  * different protein — on a multi-caption corpus a sibling caption's FUNCTION
    is usually *also* true of this protein. Inert at one caption per accession
    (the donor is a different caption, hence a different protein), kept so the
    rule still holds if a multi-caption corpus comes back;
  * same split — a donor from val/test would leak held-out caption text into
    training (it is only a fragment, but it is free to avoid).
"""
from __future__ import annotations

import csv
import json
import re
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# csv captions are long; match src/data.py's cap so a single quoted caption can be
# hundreds of KB without a "field larger than field limit" error.
csv.field_size_limit(10 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Caption field parsing
# ---------------------------------------------------------------------------
def compile_label_pattern(labels: Sequence[str]) -> re.Pattern:
    """Regex matching any caption field label at a token boundary.

    Labels are matched longest-first so a label that is a prefix of another can't
    shadow it, and only at the start of the string or after whitespace — the
    uppercase "LABEL:" strings do occasionally appear mid-sentence inside a value
    (e.g. a FUNCTION line quoting "PATHWAY:"), and splitting there would invent a
    field that the caption's author never wrote.
    """
    alts = "|".join(re.escape(l) for l in sorted(labels, key=len, reverse=True))
    return re.compile(rf"(?:^|(?<=\s))({alts})")


def parse_caption_fields(text: str, pattern: re.Pattern,
                         label_to_id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """Caption -> [(label_id, value_start, value_end)], in order of appearance.

    The span covers the field's *value* only: it starts after the label and any
    following whitespace, and ends where the next label begins (trailing
    whitespace trimmed). The label itself therefore survives a swap, and so does
    the caption's separator punctuation — the decoy stays a well-formed caption in
    exactly the corpus's template.

    Empty values are dropped: they are neither swappable (nothing to replace) nor
    donatable (nothing to donate). A label repeated in one caption yields one span
    per occurrence.
    """
    matches = list(pattern.finditer(text))
    spans: List[Tuple[int, int, int]] = []
    n = len(text)
    for i, m in enumerate(matches):
        vs = m.end()
        while vs < n and text[vs].isspace():
            vs += 1
        ve = matches[i + 1].start() if i + 1 < len(matches) else n
        while ve > vs and text[ve - 1].isspace():
            ve -= 1
        if ve > vs:
            spans.append((label_to_id[m.group(1)], vs, ve))
    return spans


def build_decoy_text(owner: str, vs: int, ve: int,
                     donor: str, dvs: int, dve: int) -> str:
    """Apply one field swap: owner caption with [vs, ve) replaced by donor[dvs:dve)."""
    return owner[:vs] + donor[dvs:dve] + owner[ve:]


# ---------------------------------------------------------------------------
# CSV loading (captions only)
# ---------------------------------------------------------------------------
def load_caption_rows(csv_path: str, *, id_col: str, text_col: str,
                      subset_size: int = 0) -> Tuple[List[str], List[str]]:
    """Read (accessions, captions) in CSV row order — the cache's row order.

    Deliberately does not use `src.data.load_pairs`: nothing here needs the
    protein sequences, and at full-corpus scale holding them costs a couple of GB
    per rank for no reason. Row order and `subset_size` semantics are identical,
    and `precompute_decoys` cross-checks the accessions against the base cache's
    `pair_ids.json` before using them.
    """
    uids: List[str] = []
    texts: List[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (id_col, text_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns {missing}. "
                f"Found columns: {reader.fieldnames}")
        for row in reader:
            uids.append(row[id_col])
            texts.append(row[text_col])
            if subset_size > 0 and len(uids) >= subset_size:
                break
    return uids, texts


# ---------------------------------------------------------------------------
# Field index (CSR over spans) + decoy plan
# ---------------------------------------------------------------------------
class FieldIndex:
    """All swappable field spans of the corpus, in a flat CSR layout.

    row_ptr[i]:row_ptr[i+1] indexes this row's spans inside the parallel
    (label, start, end) arrays; `span_row` is the inverse map. Flat numpy arrays
    rather than per-row Python objects: at ~5M rows x ~5 fields the object
    overhead alone would be tens of GB, while this is ~17 bytes per span.
    """

    def __init__(self, row_ptr: np.ndarray, span_label: np.ndarray,
                 span_start: np.ndarray, span_end: np.ndarray,
                 span_row: np.ndarray, n_labels: int):
        self.row_ptr = row_ptr
        self.span_label = span_label
        self.span_start = span_start
        self.span_end = span_end
        self.span_row = span_row
        self.n_labels = n_labels

    @property
    def n_spans(self) -> int:
        return int(self.span_label.shape[0])


def build_field_index(texts: Sequence[str], labels: Sequence[str], *,
                      swap_label_ids: Optional[Sequence[int]] = None,
                      max_value_start: int = 0,
                      log=print, log_every: int = 500_000) -> FieldIndex:
    """Parse every caption once into a `FieldIndex` of swap-eligible spans.

    `swap_label_ids` restricts which fields are eligible (both as a swap target
    and as a donor). `max_value_start` (> 0) drops spans whose value begins past
    that character offset: those sit beyond the text encoder's truncation window,
    so swapping them would produce a decoy that tokenizes *identically* to the
    truth — an unlearnable pair the loss would tear at. It is only a cheap
    character-budget heuristic; `precompute_decoys` re-checks exactly, in token
    space, and drops whatever slips through.
    """
    label_to_id = {l: i for i, l in enumerate(labels)}
    pattern = compile_label_pattern(labels)
    allowed = None if swap_label_ids is None else set(int(i) for i in swap_label_ids)

    row_ptr = array("q", [0])
    span_label = array("b")
    span_start = array("i")
    span_end = array("i")
    span_row = array("q")

    for row, text in enumerate(texts):
        for lid, vs, ve in parse_caption_fields(text, pattern, label_to_id):
            if allowed is not None and lid not in allowed:
                continue
            if max_value_start > 0 and vs > max_value_start:
                continue
            span_label.append(lid)
            span_start.append(vs)
            span_end.append(ve)
            span_row.append(row)
        row_ptr.append(len(span_label))
        if log is not None and log_every and (row + 1) % log_every == 0:
            log(f"[decoy-plan] parsed {row + 1}/{len(texts)} captions, "
                f"{len(span_label)} eligible spans")

    return FieldIndex(
        row_ptr=np.frombuffer(row_ptr, dtype=np.int64).copy(),
        span_label=np.frombuffer(span_label, dtype=np.int8).copy(),
        span_start=np.frombuffer(span_start, dtype=np.int32).copy(),
        span_end=np.frombuffer(span_end, dtype=np.int32).copy(),
        span_row=np.frombuffer(span_row, dtype=np.int64).copy(),
        n_labels=len(labels),
    )


def build_decoy_plan(
    texts: Sequence[str],
    accessions: Sequence[str],
    labels: Sequence[str],
    *,
    decoys_per_row: int = 2,
    seed: int = 1234,
    swap_label_ids: Optional[Sequence[int]] = None,
    max_value_start: int = 0,
    split_of_row: Optional[np.ndarray] = None,
    allow_same_protein_donor: bool = False,
    donor_tries: int = 8,
    log=print,
) -> dict:
    """Assign each row up to `decoys_per_row` (field, donor) swaps.

    Deterministic given (texts, seed): the returned arrays are the whole contract
    between plan-building and encoding, so a re-run reproduces the same decoy
    corpus without re-parsing anything.

    Per row we sample distinct spans (distinct fields where the row has enough of
    them), and for each span sample donors from the pool of *other* rows carrying
    the same label until one passes the rejection rules (see module docstring) or
    `donor_tries` is exhausted. A row that has no eligible field, or whose only
    field has no acceptable donor, simply contributes no decoys — with sparse
    annotation that is a real and expected outcome, not an error.

    Returns a dict of numpy arrays (all length M, ordered by owner row):
      owner_row, label_id, owner_start, owner_end, donor_row, donor_start,
      donor_end, plus `row_ptr` [n_rows+1] (CSR into the decoy list) and `stats`.
    """
    n_rows = len(texts)
    if len(accessions) != n_rows:
        raise ValueError(f"accessions ({len(accessions)}) != texts ({n_rows})")

    log(f"[decoy-plan] parsing {n_rows} captions over {len(labels)} field labels")
    index = build_field_index(texts, labels, swap_label_ids=swap_label_ids,
                              max_value_start=max_value_start, log=log)
    log(f"[decoy-plan] {index.n_spans} eligible spans "
        f"({index.n_spans / max(n_rows, 1):.2f} per caption)")

    # Dense accession ids: donor rejection compares protein identity, and int
    # comparison beats string comparison ~M*donor_tries times.
    acc_ids = np.empty(n_rows, dtype=np.int64)
    seen: Dict[str, int] = {}
    for i, a in enumerate(accessions):
        j = seen.get(a)
        if j is None:
            j = len(seen)
            seen[a] = j
        acc_ids[i] = j

    if split_of_row is None:
        split_of_row = np.zeros(n_rows, dtype=np.int8)
    split_of_row = np.asarray(split_of_row, dtype=np.int8)
    n_splits = int(split_of_row.max()) + 1 if n_rows else 1

    # Donor pools, one per (label, split): span indices a donor may be drawn from.
    # Keeping the split in the key is what makes "same-split donor" a lookup
    # rather than a rejection test that could loop forever on a sparse label.
    span_split = split_of_row[index.span_row]
    pools: Dict[Tuple[int, int], np.ndarray] = {}
    for lid in range(index.n_labels):
        is_l = index.span_label == lid
        for s in range(n_splits):
            pool = np.flatnonzero(is_l & (span_split == s))
            if pool.size:
                pools[(lid, s)] = pool
    log("[decoy-plan] donor pools: " + ", ".join(
        f"{labels[l]}[{s}]={pools[(l, s)].size}" for (l, s) in sorted(pools)))

    rng = np.random.default_rng(seed)
    owner_row = array("q"); label_id = array("b")
    owner_start = array("i"); owner_end = array("i")
    donor_row = array("q"); donor_start = array("i"); donor_end = array("i")
    counts = np.zeros(n_rows, dtype=np.int64)

    n_no_field = 0
    n_no_donor = 0
    draws_per_decoy = 1 + donor_tries
    chunk = 65536
    for c0 in range(0, n_rows, chunk):
        c1 = min(c0 + chunk, n_rows)
        # Bulk-draw the uniforms for this chunk: a per-decision rng call would
        # dominate the runtime at corpus scale (tens of millions of calls).
        u = rng.random((c1 - c0) * decoys_per_row * draws_per_decoy)
        u_at = 0
        for row in range(c0, c1):
            s0, s1 = int(index.row_ptr[row]), int(index.row_ptr[row + 1])
            n_spans = s1 - s0
            if n_spans == 0:
                n_no_field += 1
                u_at += decoys_per_row * draws_per_decoy
                continue
            # Sample spans without replacement while the row has enough of them,
            # so a row's decoys perturb *different* fields.
            order = list(range(s0, s1))
            for k in range(decoys_per_row):
                pick_u = u[u_at]; u_at += 1
                if order:
                    si = order.pop(int(pick_u * len(order)) % len(order))
                else:                                    # fewer fields than decoys
                    si = s0 + int(pick_u * n_spans) % n_spans
                lid = int(index.span_label[si])
                vs, ve = int(index.span_start[si]), int(index.span_end[si])
                value = texts[row][vs:ve]
                pool = pools.get((lid, int(split_of_row[row])))
                hit = -1
                if pool is not None and pool.size:
                    for t in range(donor_tries):
                        j = int(pool[int(u[u_at + t] * pool.size) % pool.size])
                        drow = int(index.span_row[j])
                        if drow == row:
                            continue
                        if not allow_same_protein_donor and acc_ids[drow] == acc_ids[row]:
                            continue
                        dvs, dve = int(index.span_start[j]), int(index.span_end[j])
                        if texts[drow][dvs:dve] == value:
                            continue                     # identical field: not a decoy
                        hit = j
                        break
                u_at += donor_tries
                if hit < 0:
                    n_no_donor += 1
                    continue
                owner_row.append(row); label_id.append(lid)
                owner_start.append(vs); owner_end.append(ve)
                donor_row.append(int(index.span_row[hit]))
                donor_start.append(int(index.span_start[hit]))
                donor_end.append(int(index.span_end[hit]))
                counts[row] += 1
        if log is not None and (c1 % 1_000_000 == 0 or c1 == n_rows):
            log(f"[decoy-plan] planned {c1}/{n_rows} rows, {len(owner_row)} decoys")

    m = len(owner_row)
    per_label = np.bincount(np.frombuffer(label_id, dtype=np.int8).astype(np.int64),
                            minlength=index.n_labels) if m else np.zeros(index.n_labels, np.int64)
    stats = {
        "n_rows": int(n_rows),
        "n_decoys": int(m),
        "decoys_per_row_requested": int(decoys_per_row),
        "rows_with_no_eligible_field": int(n_no_field),
        "rows_with_zero_decoys": int((counts == 0).sum()),
        "decoy_slots_without_donor": int(n_no_donor),
        "decoys_by_label": {labels[i]: int(per_label[i]) for i in range(index.n_labels)},
        "seed": int(seed),
        "max_value_start": int(max_value_start),
        "allow_same_protein_donor": bool(allow_same_protein_donor),
        "n_splits_used_for_donors": int(n_splits),
    }
    log(f"[decoy-plan] {m} decoys for {n_rows} rows "
        f"({stats['rows_with_zero_decoys']} rows with none)")

    return {
        "owner_row": np.frombuffer(owner_row, dtype=np.int64).copy(),
        "label_id": np.frombuffer(label_id, dtype=np.int8).copy(),
        "owner_start": np.frombuffer(owner_start, dtype=np.int32).copy(),
        "owner_end": np.frombuffer(owner_end, dtype=np.int32).copy(),
        "donor_row": np.frombuffer(donor_row, dtype=np.int64).copy(),
        "donor_start": np.frombuffer(donor_start, dtype=np.int32).copy(),
        "donor_end": np.frombuffer(donor_end, dtype=np.int32).copy(),
        "row_ptr": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "labels": list(labels),
        "stats": stats,
    }


def plan_decoy_texts(plan: dict, texts: Sequence[str],
                     start: int, end: int) -> List[str]:
    """Materialize decoys [start, end) of the plan as caption strings."""
    owner = plan["owner_row"]; donor = plan["donor_row"]
    os_, oe = plan["owner_start"], plan["owner_end"]
    ds, de = plan["donor_start"], plan["donor_end"]
    return [
        build_decoy_text(texts[int(owner[i])], int(os_[i]), int(oe[i]),
                         texts[int(donor[i])], int(ds[i]), int(de[i]))
        for i in range(start, end)
    ]


# ---------------------------------------------------------------------------
# Decoy cache fingerprint
# ---------------------------------------------------------------------------
def decoy_fingerprint(base_fp: dict, *, decoys_per_row: int, seed: int,
                      swap_fields: Sequence[str], max_value_start: int,
                      allow_same_protein_donor: bool, n_rows: int) -> dict:
    """Identity of a decoy cache: the base text-encoder fingerprint + plan params.

    The base fingerprint is carried verbatim so a decoy cache built against a
    different text encoder / length cap / masking flag can never be paired with a
    mismatched base cache.
    """
    return {
        "format": "v1_decoy_field_swap",
        "base": dict(base_fp),
        "decoys_per_row": int(decoys_per_row),
        "decoy_seed": int(seed),
        "swap_fields": sorted(str(f) for f in swap_fields),
        "max_value_start": int(max_value_start),
        "allow_same_protein_donor": bool(allow_same_protein_donor),
        "n_rows": int(n_rows),
    }


def write_decoy_fingerprint(cache_dir: str, fp: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cache_dir) / "decoy_fingerprint.json", "w") as f:
        json.dump(fp, f, indent=2)


def read_decoy_fingerprint(cache_dir: str) -> dict:
    path = Path(cache_dir) / "decoy_fingerprint.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Self-test (no corpus, no torch device needed):
#   python -m src.decoys --selftest
# ---------------------------------------------------------------------------
_SELFTEST_LABELS = ("PROTEIN NAME:", "FUNCTION:", "LINEAGE:", "PATHWAY:")

_SELFTEST_CAPTIONS = [
    "PROTEIN NAME: Sho1. FUNCTION: Plasma membrane osmosensor. LINEAGE: Fungi.",
    "PROTEIN NAME: Pbs2. FUNCTION: MAP kinase kinase. LINEAGE: Fungi. PATHWAY: HOG.",
    "PROTEIN NAME: Hog1. LINEAGE: Fungi.",                       # sparse: no FUNCTION
    "PROTEIN NAME: Ldh. FUNCTION: Catalyzes lactate conversion. "
    "LINEAGE: Bacteria. PATHWAY: Glycolysis.",
    "FUNCTION: Plasma membrane osmosensor. LINEAGE: Fungi.",      # dup FUNCTION value
]
_SELFTEST_ACCESSIONS = ["P1", "P2", "P3", "P4", "P1"]             # rows 0 and 4 share P1


def _selftest() -> int:
    labels = list(_SELFTEST_LABELS)
    label_to_id = {l: i for i, l in enumerate(labels)}
    pattern = compile_label_pattern(labels)
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name}: {detail}")
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))

    print("parse_caption_fields")
    spans = parse_caption_fields(_SELFTEST_CAPTIONS[0], pattern, label_to_id)
    got = [(labels[l], _SELFTEST_CAPTIONS[0][s:e]) for l, s, e in spans]
    check("splits all three fields", got == [
        ("PROTEIN NAME:", "Sho1."),
        ("FUNCTION:", "Plasma membrane osmosensor."),
        ("LINEAGE:", "Fungi."),
    ], str(got))

    sparse = parse_caption_fields(_SELFTEST_CAPTIONS[2], pattern, label_to_id)
    check("sparse caption yields only present fields",
          [labels[l] for l, _, _ in sparse] == ["PROTEIN NAME:", "LINEAGE:"],
          str([labels[l] for l, _, _ in sparse]))

    mid = "FUNCTION: Regulates the PATHWAY: label mid-sentence. LINEAGE: Fungi."
    mid_spans = parse_caption_fields(mid, pattern, label_to_id)
    check("mid-value label at a whitespace boundary is still a boundary",
          [labels[l] for l, _, _ in mid_spans] == ["FUNCTION:", "PATHWAY:", "LINEAGE:"],
          str([labels[l] for l, _, _ in mid_spans]))
    glued = "FUNCTION: seeXPATHWAY: y. LINEAGE: Fungi."
    check("label glued to a word is not a boundary",
          [labels[l] for l, _, _ in parse_caption_fields(glued, pattern, label_to_id)]
          == ["FUNCTION:", "LINEAGE:"])

    print("build_decoy_text")
    l, s, e = spans[1]
    swapped = build_decoy_text(_SELFTEST_CAPTIONS[0], s, e,
                               _SELFTEST_CAPTIONS[3], *parse_caption_fields(
                                   _SELFTEST_CAPTIONS[3], pattern, label_to_id)[1][1:])
    check("swap keeps template + labels",
          swapped == "PROTEIN NAME: Sho1. FUNCTION: Catalyzes lactate conversion. "
                     "LINEAGE: Fungi.", swapped)

    print("build_decoy_plan")
    plan = build_decoy_plan(
        _SELFTEST_CAPTIONS, _SELFTEST_ACCESSIONS, labels,
        decoys_per_row=2, seed=7, log=lambda *a, **k: None)
    m = plan["owner_row"].shape[0]
    check("produced decoys", m > 0, f"m={m}")
    check("owner_row sorted", bool(np.all(np.diff(plan["owner_row"]) >= 0)))
    check("row_ptr matches owner_row", plan["row_ptr"][-1] == m
          and bool(np.all(np.bincount(plan["owner_row"], minlength=len(_SELFTEST_CAPTIONS))
                          == np.diff(plan["row_ptr"]))))

    texts = plan_decoy_texts(plan, _SELFTEST_CAPTIONS, 0, m)
    bad_same = [t for t, o in zip(texts, plan["owner_row"])
                if t == _SELFTEST_CAPTIONS[int(o)]]
    check("no decoy equals its own caption", not bad_same, str(bad_same[:2]))

    same_prot = [(int(o), int(d)) for o, d in zip(plan["owner_row"], plan["donor_row"])
                 if _SELFTEST_ACCESSIONS[int(o)] == _SELFTEST_ACCESSIONS[int(d)]]
    check("no donor from the same protein", not same_prot, str(same_prot))

    # Row 0 and row 4 share the FUNCTION value; a donor swap between them would be
    # a no-op and must have been rejected by the value-equality test.
    check("identical field values rejected as donors",
          all(_SELFTEST_CAPTIONS[int(d)][int(ds):int(de)]
              != _SELFTEST_CAPTIONS[int(o)][int(os_):int(oe_)]
              for o, os_, oe_, d, ds, de in zip(
                  plan["owner_row"], plan["owner_start"], plan["owner_end"],
                  plan["donor_row"], plan["donor_start"], plan["donor_end"])))

    check("plan is deterministic under a fixed seed",
          np.array_equal(
              build_decoy_plan(_SELFTEST_CAPTIONS, _SELFTEST_ACCESSIONS, labels,
                               decoys_per_row=2, seed=7,
                               log=lambda *a, **k: None)["donor_row"],
              plan["donor_row"]))

    splits = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    sp_plan = build_decoy_plan(_SELFTEST_CAPTIONS, _SELFTEST_ACCESSIONS, labels,
                               decoys_per_row=2, seed=7, split_of_row=splits,
                               log=lambda *a, **k: None)
    check("donors stay inside the owner's split",
          all(splits[int(o)] == splits[int(d)]
              for o, d in zip(sp_plan["owner_row"], sp_plan["donor_row"])))

    only_fn = build_decoy_plan(_SELFTEST_CAPTIONS, _SELFTEST_ACCESSIONS, labels,
                               decoys_per_row=2, seed=7,
                               swap_label_ids=[label_to_id["FUNCTION:"]],
                               log=lambda *a, **k: None)
    check("swap_label_ids restricts the swapped field",
          bool(np.all(only_fn["label_id"] == label_to_id["FUNCTION:"])))

    truncated = build_decoy_plan(_SELFTEST_CAPTIONS, _SELFTEST_ACCESSIONS, labels,
                                 decoys_per_row=2, seed=7, max_value_start=15,
                                 log=lambda *a, **k: None)
    check("max_value_start drops late fields",
          bool(np.all(truncated["owner_start"] <= 15)))

    print()
    if fails:
        print(f"{len(fails)} check(s) FAILED:")
        for f in fails:
            print("  - " + f)
        return 1
    print("all decoy self-tests passed")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the parser/plan invariant checks on synthetic captions")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.error("nothing to do; pass --selftest")
