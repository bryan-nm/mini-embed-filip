"""Per-direction generation trainer.

Reads a trained retrieval checkpoint (projection + expansion heads), runs the
appropriate encoder→project→expand front-end to produce per-token cross-
attention memory, then trains the decoder's cross-attention adapters on
cross-entropy of the target sequence.

What's frozen vs trainable:
  frozen:   encoder, projection head, expansion head, decoder backbone
  trains:   cross-attention adapter layers (+ any explicitly unfrozen blocks)

Directions:
  text2protein:  text -> BioLinkBERT -> proj -> expand -> ProtGPT3-112M -> protein
  protein2text:  protein -> AMPLIFY-350M -> proj -> expand -> BioGPT     -> text

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
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, TEXT_DECODER_PATH
from src.data import (
    PackedPerTokenCache,
    build_or_load_splits,
    group_ids_from_accessions,
    group_ids_from_texts,
    merged_split_group_ids,
    load_pairs,
    load_row_protein_idx,
    load_splits,
)
from src.dist import init_distributed, barrier, cleanup
from src.decoder_adapters import (
    clear_cross_memory,
    count_trainable,
    coverage_penalty,
    coverage_profile,
    load_decoder_with_cross_attn,
    reverse_prefix_ids,
    set_cross_memory,
    set_retain_attn,
    target_prefix_ids,
    unfreeze_decoder_blocks,
    warm_start_q_align,
)
from src.encoders import (
    encode_protein_batch, encode_text_batch, load_protein_encoder, load_text_encoder,
)
from src.gen_ckpt import GenMeta
from src.model import load_retrieval
from src.roundtrip_eval import (
    gather_roundtrip_payloads, generate_roundtrip_records, rng_restore, rng_snapshot,
    score_roundtrip_records,
)


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

    Protein-side access always goes through `row_protein_idx` (CSV row -> unique
    protein row), because the protein cache is deduplicated. `need_target` also
    returns the opposite modality's per-token hidden states — the round-trip eval
    uses them for its ceiling row.
    """

    def __init__(self, direction: str, cache_dir: str, pairs, indices,
                 protein_dim: int, text_dim: int, need_target: bool = False,
                 row_protein_idx=None):
        if direction not in ("text2protein", "protein2text"):
            raise ValueError(direction)
        self.direction = direction
        self.pairs = pairs
        self.indices = list(indices)
        self.need_target = need_target
        self.text_cache = PackedPerTokenCache(cache_dir, "text", text_dim)
        self.protein_cache = PackedPerTokenCache(cache_dir, "protein", protein_dim)
        if row_protein_idx is None:
            row_protein_idx = load_row_protein_idx(cache_dir)
        self.row_protein_idx = row_protein_idx

    def __len__(self):
        return len(self.indices)

    def _protein(self, idx: int):
        return self.protein_cache.get(int(self.row_protein_idx[idx]))

    def _text(self, idx: int):
        return self.text_cache.get(idx)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        if self.direction == "text2protein":
            h, m = self._text(idx)                       # source = text
            target = self.pairs[idx].protein
        else:
            h, m = self._protein(idx)                    # source = protein
            target = self.pairs[idx].text
        item = {"h": h, "m": m, "target": target, "idx": idx}
        if self.need_target:
            # Opposite modality (the generated side).
            h_t, m_t = (self._protein(idx) if self.direction == "text2protein"
                        else self._text(idx))
            item["h_tgt"] = h_t
            item["m_tgt"] = m_t
        return item


def _corrupt_body_spans(row_ids: torch.Tensor, start: int, length: int,
                        p: float, mean_span: int, aa_ids: torch.Tensor) -> None:
    """In-place: overwrite ~`p` of the `length` body tokens at [start, start+length)
    of the 1-D `row_ids`, in contiguous spans, with random amino-acid tokens.

    Spans have geometric length (mean `mean_span`, capped at 10 and the body
    length); each corrupted position gets an independent random residue from
    `aa_ids`. Operates ONLY on input_ids — never labels — so the decoder is still
    scored on predicting the true residue it can no longer see, which forces it off
    the (near-sufficient) autoregressive prefix and onto the cross-attention
    conditioning. Contiguous spans (vs i.i.d.) defeat local interpolation: a masked
    block can't be filled from its immediate neighbours, so global/conditioning
    info has to carry it. No-op when length<=0 or p<=0.
    """
    if length <= 0 or p <= 0.0:
        return
    n_target = int(round(p * length))
    if n_target <= 0:
        return
    if n_target >= length:
        # p>=1.0: full no-prefix regime. Corrupt every body position directly —
        # span placement can't reliably cover 100% (coupon-collector), so bypass it.
        for off in range(length):
            row_ids[start + off] = int(aa_ids[torch.randint(0, aa_ids.numel(), ())])
        return
    q = 1.0 / max(mean_span, 1)
    max_span = min(10, length)
    covered: set = set()
    guard, guard_cap = 0, 4 * length + 50
    while len(covered) < n_target and guard < guard_cap:
        guard += 1
        u = float(torch.rand(()))
        span = 1 + int(math.log(1.0 - u + 1e-9) / math.log(1.0 - q + 1e-9))
        span = max(1, min(span, max_span))
        s = int(torch.randint(0, length - span + 1, ()).item())
        for off in range(s, s + span):
            if off not in covered:
                covered.add(off)
                row_ids[start + off] = int(aa_ids[torch.randint(0, aa_ids.numel(), ())])
                if len(covered) >= n_target:
                    break


def _per_example_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-example mean next-token CE over non-ignored positions.

    logits [B, L, V], labels [B, L] -> [B]. Uses the same left-shift as the scalar
    CE (predict token t+1 from position t); ignored (-100) positions contribute 0
    and are excluded from the per-example denominator.
    """
    sl = logits[:, :-1, :]
    lb = labels[:, 1:]
    tok = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lb.reshape(-1),
                          ignore_index=-100, reduction="none").view(lb.shape)
    valid = (lb != -100)
    return tok.sum(1) / valid.sum(1).clamp(min=1)


def make_collate(target_tokenizer, max_target_tokens: int, prefix_ids=(),
                 rev_prefix_ids=None, direction_augment: bool = False,
                 target_input_dropout: float = 0.0, dropout_mean_span: int = 5):
    # ProGen2's tokenizer ships without a pad token; fall back to EOS.
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    pad_id = target_tokenizer.pad_token_id
    bos_id = target_tokenizer.bos_token_id
    eos_id = target_tokenizer.eos_token_id
    prefix_ids = list(prefix_ids)
    rev_prefix_ids = list(rev_prefix_ids) if rev_prefix_ids is not None else []
    augment = direction_augment and len(rev_prefix_ids) > 0

    # Amino-acid substitution pool for target-input span dropout: the 20 canonical
    # residues, each a single decoder token (char-level protein vocab). Built once.
    aa_ids = None
    if target_input_dropout > 0.0:
        ids = []
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            toks = target_tokenizer(aa, add_special_tokens=False)["input_ids"]
            if len(toks) == 1:
                ids.append(toks[0])
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(
                "--target-input-dropout needs a decoder vocab where amino acids are "
                "single tokens (a protein decoder); got none. It is a text2protein "
                "feature — leave it 0 for protein2text.")
        aa_ids = torch.tensor(ids, dtype=torch.long)

    # We add BOS/EOS explicitly rather than relying on the tokenizer's
    # `add_special_tokens`: the protein decoders' char tokenizers never add them
    # (ProtGPT3 ships with add_bos_token=False), which would leave the model with
    # no EOS to learn
    # termination and a BOS-mismatch vs inference (which seeds generation with
    # BOS). Doing it here is uniform across decoders. `prefix_ids` carries any
    # decoder control tokens that sit between BOS and the body — for ProtGPT3,
    # the "1" (N-to-C) direction marker it was pretrained with.
    #
    # Direction augmentation (ProtGPT3): when `augment`, each target is factorized
    # forward ("1" + residues) or reverse ("2" + reversed residues) with equal
    # probability. Same protein, same conditioning memory, opposite decode order;
    # decode_target re-reverses a "2" sequence back to N-to-C at inference.
    #
    # EOS is appended only to targets that FIT under body_cap (see collate). A
    # truncated target did not end where we cut it, and appending EOS there would
    # teach exactly that — every over-cap protein in the corpus lands on the same
    # stop position, so the model learns a hard "sequences end at body_cap" mode
    # rather than a length distribution. Truncated rows still supervise the
    # residues they contain; they just never claim the sequence terminated.
    max_prefix = max(len(prefix_ids), len(rev_prefix_ids))
    body_cap = (max_target_tokens - max_prefix
                - (1 if bos_id is not None else 0)
                - (1 if eos_id is not None else 0))

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

        # Tokenize bodies without specials, then wrap with [BOS] [prefix] ... [EOS]
        # — EOS only when the body fit (see the note above body_cap).
        seqs = []
        body_spans = []          # (start, length) of the residue body within each row
        n_trunc = 0
        for b in batch:
            target, pfx = b["target"], prefix_ids
            # Coin-flip C-to-N: reverse the residue string (char == token for the
            # char-level vocab) and use the reverse marker. CPU RNG, seeded, and
            # separate from the device dropout RNG.
            if augment and bool(torch.rand(()) < 0.5):
                target, pfx = target[::-1], rev_prefix_ids
            # Tokenize uncapped so we can SEE the overflow, then cut by hand;
            # tokenizer truncation would hide whether this target actually ended.
            body = target_tokenizer(target, add_special_tokens=False)["input_ids"]
            truncated = len(body) > body_cap
            if truncated:
                body = body[:body_cap]
                n_trunc += 1
            head = ([bos_id] if bos_id is not None else []) + pfx
            tail = [eos_id] if (eos_id is not None and not truncated) else []
            ids = head + body + tail
            seqs.append(ids)
            body_spans.append((len(head), len(body)))

        L = max(len(s) for s in seqs)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(B, L, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, :len(s)] = 1

        # Labels = input_ids; pad positions become -100 (ignored by CE).
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        # Target-input span dropout (train only): corrupt input_ids AFTER cloning
        # labels, so the model still predicts the true (now unseen) residues.
        if aa_ids is not None:
            for i, (start, length) in enumerate(body_spans):
                _corrupt_body_spans(input_ids[i], start, length,
                                    target_input_dropout, dropout_mean_span, aa_ids)

        out = {
            "h": h_pad, "m": m_pad,
            "input_ids": input_ids, "attn_mask": attn_mask, "labels": labels,
            # Rows whose target overflowed body_cap (and so carry no EOS). Logged
            # as a rate during training: it is the direct read on whether
            # --max-target-tokens covers the corpus, and how much of the batch is
            # contributing no termination signal.
            "n_truncated": torch.tensor(n_trunc, dtype=torch.long),
        }

        # Optional target-side per-token hidden states, padded to the batch max.
        if "h_tgt" in batch[0]:
            L_t = max(b["h_tgt"].size(0) for b in batch)
            d_t = batch[0]["h_tgt"].size(-1)
            ht_pad = torch.zeros(B, L_t, d_t, dtype=torch.bfloat16)
            mt_pad = torch.zeros(B, L_t, dtype=torch.bool)
            for i, b in enumerate(batch):
                lt = b["h_tgt"].size(0)
                ht_pad[i, :lt] = b["h_tgt"]
                mt_pad[i, :lt] = b["m_tgt"]
            out["h_tgt"] = ht_pad
            out["m_tgt"] = mt_pad

        return out

    return collate


def cosine_warmup_factor(step: int, total: int, warmup: int) -> float:
    """LR multiplier in [0, 1]: linear warmup then cosine decay. Multiply each
    param group's own base_lr by this so groups can carry different base LRs."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    return base * cosine_warmup_factor(step, total, warmup)


def resolve_resume_path(resume: str, ckpt_dir: Path) -> Path:
    """Resolve --resume into a checkpoint file.

    "auto" picks the highest-numbered `epochNN.pt` in ckpt_dir; anything else is
    treated as a direct path to a checkpoint file.
    """
    if resume == "auto":
        ckpts = sorted(ckpt_dir.glob("epoch*.pt"))
        if not ckpts:
            raise FileNotFoundError(
                f"--resume auto found no epoch*.pt checkpoints in {ckpt_dir}")
        return ckpts[-1]
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    return path


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["text2protein", "protein2text"], required=True)
    ap.add_argument("--retrieval-ckpt", required=True)
    ap.add_argument("--device", default=cfg.generation.device)
    ap.add_argument("--cache-dir", default=cfg.retrieval.cache_dir)
    ap.add_argument("--ckpt-dir", default=cfg.generation.ckpt_dir)
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint: a path to an epochNN.pt file, "
                         "or 'auto' to pick the latest in --ckpt-dir/<direction>")
    ap.add_argument("--batch-size", type=int, default=cfg.generation.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.generation.epochs)
    ap.add_argument("--lr", type=float, default=cfg.generation.lr)
    ap.add_argument("--cross-attn-every", type=int, default=cfg.generation.cross_attn_every)
    ap.add_argument("--unfreeze-top", type=int, default=0,
                    help="fully fine-tune N decoder blocks (0 = cross-attn adapters only); "
                         "which N blocks is set by --unfreeze-where")
    ap.add_argument("--unfreeze-where", choices=["top", "bottom"], default="top",
                    help="which end of the decoder stack --unfreeze-top selects: 'top' = the "
                         "last N blocks (nearest logits, default), 'bottom' = the first N "
                         "(nearest embeddings, lets conditioning become receptive early)")
    ap.add_argument("--unfreeze-lr", type=float, default=None,
                    help="separate (typically lower) LR for the unfrozen decoder blocks; "
                         "defaults to --lr. The cross-attn adapters always use --lr. "
                         "Keep this well below --lr to avoid wrecking the pretrained prior")
    ap.add_argument("--warm-start-qalign", action="store_true",
                    help="initialize the aligned-mode cross-attn query MLP (q_align) from the "
                         "retrieval projection head of the generated modality, so the decoder-"
                         "side query lands in the shared space from step 0 instead of cold-"
                         "starting a random map. Requires --cross-attn-mode aligned")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute layer activations in backward to fit a large decoder; "
                         "~30%% slower, much less memory")
    ap.add_argument("--max-target-tokens", type=int, default=cfg.generation.max_target_tokens,
                    help="cap on the generated/teacher-forced target length; lower it to "
                         "cut decoder activation memory (truncates long targets)")
    ap.add_argument("--memory-space", choices=["expanded", "aligned"],
                    default=cfg.generation.memory_space,
                    help="conditioning memory: 'expanded' cross-attends over "
                         "expand(project(h)) in encoder-hidden space (768/960-d, original); "
                         "'aligned' cross-attends over project(h) directly (the shared "
                         "retrieval space, embed_dim wide, no expansion head)")
    ap.add_argument("--cross-attn-mode", choices=["head", "aligned"],
                    default=cfg.generation.cross_attn_mode,
                    help="cross-attn scoring space (Strong): 'head' learned multi-head "
                         "(original); 'aligned' single-head cosine max-sim in the shared "
                         "space (attention weights = FILIP array). Requires --memory-space aligned")
    ap.add_argument("--direction-augment", action="store_true",
                    default=cfg.generation.direction_augment,
                    help="stochastically factorize each protein target N-to-C ('1') or C-to-N "
                         "('2', residues reversed) per example (ProtGPT3 only). Free augmentation "
                         "+ regularizes conditioning toward direction-invariant signal. Val stays "
                         "forward-only")
    ap.add_argument("--target-input-dropout", type=float, default=0.0,
                    help="text2protein: corrupt this fraction of the target protein's INPUT "
                         "residues (the teacher-forcing context) with random amino acids, in "
                         "contiguous spans, while leaving the labels intact. Handicaps the "
                         "near-sufficient autoregressive prefix so the decoder must lean on the "
                         "cross-attn conditioning. 0 = off; try 0.15-0.3. Val is never corrupted")
    ap.add_argument("--dropout-mean-span", type=int, default=5,
                    help="mean length (geometric, capped at 10) of each corrupted residue span "
                         "for --target-input-dropout")
    ap.add_argument("--contrast-lambda", type=float, default=0.0,
                    help="weight on the contrastive negative-memory hinge (0 = off). Each step "
                         "also runs the decoder on a shuffled (wrong) memory and penalizes the "
                         "correct memory unless it beats the wrong one by --contrast-margin "
                         "nats/token. Directly optimizes 'content gain'. Doubles the forward "
                         "batch to 2B — halve --batch-size if it OOMs")
    ap.add_argument("--contrast-margin", type=float, default=0.5,
                    help="target nats/token by which the correct memory should beat a wrong one "
                         "(the hinge margin for --contrast-lambda)")
    ap.add_argument("--coverage-lambda", type=float, default=0.0,
                    help="weight on the cross-attention coverage aux (0 = off). Cross-attn "
                         "softmaxes over the memory axis, so each decoder position must spend "
                         "its mass on the prompt, but no prompt position is required to RECEIVE "
                         "any — an unattended source token contributes ~0 to every attn_out and "
                         "cannot influence the output. This penalizes source positions whose "
                         "max-over-decoder-positions attention stays below --coverage-tau, "
                         "supplying the direction the retrieval FILIP score's t2p half has. "
                         "Requires --cross-attn-mode aligned; incompatible with "
                         "--grad-checkpointing. Measure coverage before setting this — "
                         "[val][cov] prints even at 0. Sizing: at the measured text2protein "
                         "covpen, 0.5 puts the penalty at ~2% of CE, which is enough to move "
                         "without trading away the LM. Left at 0 because the coverage DEFICIT "
                         "is measured but the BENEFIT is not: nothing yet shows this improves "
                         "multi-clause fidelity or R@K, so enabling it is an A/B to run, not "
                         "a settled default")
    ap.add_argument("--coverage-tau", type=float, default=0.025,
                    help="attention-mass threshold a source position must reach to count as "
                         "covered. Absolute mass, so it must be read against memory length: "
                         "uniform attention over L valid positions gives 1/L, and a tau at or "
                         "below that cannot separate attended from ignored. The penalty is "
                         "normalized by tau, so --coverage-lambda keeps its meaning if you "
                         "retune this. DEFAULT MEASURED ON text2protein (job 8723205, epoch-6 "
                         "SFT): captions average ~96 valid tokens so unif=0.0104, and the "
                         "coverage distribution was p10=0.006 p50=0.026 p90=0.128 with 20% of "
                         "positions at or below uniform. 0.025 is ~2.4x uniform and targets "
                         "that starved lower half; the old 0.1 default penalized 86% of "
                         "positions, which is a blanket pressure rather than a targeted one. "
                         "protein2text is UNMEASURED and wants a different value — its source "
                         "is a protein up to 1024 tokens, so unif is ~10x smaller. Re-measure "
                         "with --coverage-lambda 0 and read [val][cov] before trusting this "
                         "number in that direction")
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate on the first N val pairs each epoch (0 = full val split)")
    # ---- per-epoch round-trip retrieval eval (free-running; see run_roundtrip) ----
    ap.add_argument("--rt-eval", dest="rt_eval", action="store_true",
                    help="once per epoch, generate a fixed held-out pool free-running, "
                         "re-encode it and score accession-grouped R@K/mAP back to the "
                         "sources — the prefix-free counterpart to val CE (default on)")
    ap.add_argument("--no-rt-eval", dest="rt_eval", action="store_false")
    ap.set_defaults(rt_eval=True)
    ap.add_argument("--rt-samples", type=int, default=1000,
                    help="val rows in the round-trip pool. Fixed across epochs and "
                         "restarts (--rt-seed) so the curve is comparable epoch to epoch. "
                         "Also the candidate-pool size, so R@K/mAP are NOT comparable "
                         "across different values. Cost is one sharded generate: rows are "
                         "split across ranks, so this is ~free until it exceeds world_size")
    ap.add_argument("--rt-batch-size", type=int, default=8,
                    help="per-rank generation batch during the round-trip eval")
    ap.add_argument("--rt-max-new-tokens", type=int, default=None,
                    help="default: --max-target-tokens (measure the whole length "
                         "distribution the model was trained on). The main cost knob — "
                         "eval wall time is linear in it")
    ap.add_argument("--rt-temperature", type=float, default=1.0,
                    help="round-trip sampling temperature (0 = greedy)")
    ap.add_argument("--rt-top-p", type=float, default=0.9)
    ap.add_argument("--rt-filip-chunk-rows", type=int, default=4,
                    help="row chunk for the N x N FILIP scoring on rank 0; bounds the "
                         "transient on top of the resident training state")
    ap.add_argument("--rt-seed", type=int, default=1234,
                    help="picks the fixed round-trip pool and seeds its sampling; the "
                         "training RNG stream is saved/restored around each eval, so "
                         "turning this on does not perturb training")
    ap.add_argument("--rt-at-start", dest="rt_at_start", action="store_true",
                    help="also run the round-trip eval before the first epoch: an "
                         "epoch=-1 baseline, and it surfaces any problem with the eval "
                         "in the first minute instead of at the end of epoch 0")
    ap.add_argument("--no-rt-at-start", dest="rt_at_start", action="store_false")
    ap.set_defaults(rt_at_start=True)
    args = ap.parse_args()

    cfg.generation.max_target_tokens = args.max_target_tokens
    cfg.generation.memory_space = args.memory_space
    cfg.generation.cross_attn_mode = args.cross_attn_mode
    cfg.generation.direction_augment = args.direction_augment

    # The architecture header this run trains under. It is validated once here,
    # compared against the checkpoint's on --resume, and written back into every
    # saved checkpoint — so the flags a run used and the flags a consumer rebuilds
    # from can never disagree.
    meta = GenMeta.from_args(args)
    try:
        meta.validate()
    except ValueError as e:
        raise SystemExit(str(e))
    if args.target_input_dropout > 0.0 and args.direction != "text2protein":
        raise SystemExit(
            "--target-input-dropout is a text2protein feature (it corrupts protein-residue "
            "targets with random amino acids); leave it 0 for protein2text")
    if not (0.0 <= args.target_input_dropout <= 1.0):
        raise SystemExit("--target-input-dropout must be in [0, 1] (1.0 = full no-prefix)")
    if args.coverage_lambda > 0.0:
        if meta.cross_attn_mode != "aligned":
            raise SystemExit(
                "--coverage-lambda requires --cross-attn-mode aligned: only the aligned forward "
                "populates last_attn, which the coverage aux reads")
        if args.grad_checkpointing:
            # Under checkpointing the first forward runs in no_grad and the recompute
            # happens inside backward — so last_attn read after the forward carries no
            # graph and the aux would silently contribute nothing. Fail instead.
            raise SystemExit(
                "--coverage-lambda is incompatible with --grad-checkpointing: the aux reads "
                "last_attn after the forward, but under checkpointing that tensor comes from "
                "the no_grad pass and would train as a silent no-op. Drop one of the two")
        if args.coverage_tau <= 0.0:
            raise SystemExit("--coverage-tau must be > 0 (it normalizes the penalty)")
    if args.rt_max_new_tokens is None:
        args.rt_max_new_tokens = args.max_target_tokens

    env = init_distributed(args.device, group_size=1)
    device = env.device
    torch.manual_seed(args.seed)
    if env.is_main:
        print(f"[gen] direction={args.direction} world_size={env.world_size} device={device}")

    # 1) Retrieval model (frozen) — provides projection + expansion (+ maps).
    retrieval = load_retrieval(args.retrieval_ckpt, device, cfg)
    if env.is_main:
        print(f"[gen] loaded retrieval model from {args.retrieval_ckpt} "
              f"(cross_attn_mode={meta.cross_attn_mode})")

    # 2) Decoder + adapters
    decoder_path = (
        cfg.generation.decoder_path
        if args.direction == "text2protein"
        else TEXT_DECODER_PATH
    )
    # Memory dim depends on the conditioning space. "expanded": the encoder hidden
    # dim of the INPUT modality (what the expansion head outputs) — text2protein is
    # text -> text_expand -> 768-d. "aligned": the shared projection space
    # (project(h)), so the adapter k/v are embed_dim->dec_hidden and no expansion
    # head is used.
    aligned_mem = meta.aligned_memory
    mem_dim = meta.mem_dim(cfg, args.direction)
    if env.is_main:
        print(f"[gen] architecture: {meta.describe()} mem_dim={mem_dim}")

    # trust_remote_code (ProGen2) compiles the custom modeling file into a
    # *shared* transformers_modules cache. If all ranks do this at once they
    # race on the write and some import a half-defined module ("has no attribute
    # ProGenForCausalLM"). Rank 0 populates the cache first; the barrier then
    # releases the others to import it read-only (concurrent reads are safe).
    def _load_decoder():
        return load_decoder_with_cross_attn(
            args.direction, decoder_path, meta.cross_attn_every, mem_dim, device,
            cross_attn_mode=meta.cross_attn_mode,
        )

    decoder = target_tok = adapters = None
    if env.is_main:
        decoder, target_tok, adapters = _load_decoder()
    barrier()
    if not env.is_main:
        decoder, target_tok, adapters = _load_decoder()
    assert decoder is not None and adapters is not None  # always loaded on both branches

    # Optional partial unfreeze of the top decoder blocks (all ranks, identically,
    # before DDP captures requires_grad state).
    n_unfrozen = unfreeze_decoder_blocks(decoder, meta.unfreeze_top, where=meta.unfreeze_where)
    if env.is_main and meta.unfreeze_top > 0:
        print(f"[gen] unfroze {meta.unfreeze_where} {meta.unfreeze_top} decoder blocks "
              f"({n_unfrozen:,} params now trainable)")
    # The decoder is loaded in bf16 for frozen-backbone inference (fine when only
    # the fp32 adapters train). Training bf16 *master* weights is broken: an AdamW
    # update at lr≈2e-5 is far below the bf16 ULP at a typical weight magnitude
    # (~8e-5 near |w|≈0.02), so it rounds to zero — the "unfrozen" blocks never
    # actually move (looks implausibly fast, val ppl doesn't budge) — and bf16 Adam
    # moments underflow into inf/NaN that clip_grad_norm_ then smears across every
    # parameter (NaN that never recovers). Upcast to fp32 master weights whenever a
    # backbone block is trained. (Large decoders would instead want autocast(bf16)
    # over fp32 params to keep the compute cheap; the 112M ProtGPT3 is small enough
    # that plain fp32 is fine.)
    if n_unfrozen > 0:
        decoder.float()
        if env.is_main:
            print("[gen] upcast decoder to fp32 for stable fine-tuning of the "
                  "unfrozen blocks (bf16 master weights don't train)")
    if env.is_main:
        print(f"[gen] decoder trainable params: {count_trainable(decoder):,}")
        print(f"[gen] num cross-attention adapters: {len(adapters)}")

    # Coverage aux: keep the attention weights attached to the graph so the
    # penalty can backprop into q_align / logit_scale (and, through the residual,
    # the rest of the trainable stack).
    if args.coverage_lambda > 0.0:
        n_retained = set_retain_attn(adapters, True)
        if n_retained == 0:
            raise SystemExit(
                "--coverage-lambda set but no aligned-mode adapters were injected")
        if env.is_main:
            print(f"[gen] coverage aux ON: lambda={args.coverage_lambda} "
                  f"tau={args.coverage_tau} over {n_retained} adapters")

    # Gradient checkpointing: recompute layer activations in backward instead of
    # storing them across the deep stack.
    #   - use_reentrant=False is REQUIRED: the backbone is frozen (only the
    #     adapters train), and reentrant checkpointing demands an input that
    #     requires grad, which a frozen layer input doesn't have.
    #   - Incompatible with the KV cache, so disable use_cache for training.
    # Applied before the DDP wrap so `decoder` is still the raw model. On the
    # Mixtral path the cross-attn adapters are forward hooks inside the layer
    # __call__, so they fall inside the checkpointed region and recompute
    # correctly; the ProGen2/BioGPT paths replace the block module (those decoders
    # are small and don't need this).
    if args.grad_checkpointing:
        decoder.config.use_cache = False
        if hasattr(decoder, "gradient_checkpointing_enable"):
            decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if env.is_main:
                print("[gen] gradient checkpointing ON "
                      "(use_reentrant=False, use_cache=False)")
        elif env.is_main:
            print("[gen] WARNING: decoder has no gradient_checkpointing_enable(); "
                  "skipping (--grad-checkpointing had no effect)")

    # Direction-specific frozen retrieval handles: source side (project+expand for
    # the cross-attn memory) and the target-modality projection (used by the
    # aligned warm-start and the round-trip eval).
    if args.direction == "text2protein":
        src_proj, src_expand, tgt_proj = (
            retrieval.text_proj, retrieval.text_expand, retrieval.protein_proj)
    else:
        src_proj, src_expand, tgt_proj = (
            retrieval.protein_proj, retrieval.protein_expand, retrieval.text_proj)

    # Optional warm-start of the aligned-mode query MLP from the retrieval head of
    # the GENERATED modality (tgt_proj: protein_proj for text2protein). Done before
    # the DDP wrap so DDP's init-time broadcast syncs the copied weights across
    # ranks, and before the optimizer/resume so param shapes are final. The copy is
    # deterministic from the (identical, frozen) retrieval head on every rank.
    if meta.warm_start_qalign:
        n_ws = warm_start_q_align(adapters, tgt_proj)
        if env.is_main:
            head = "protein_proj" if args.direction == "text2protein" else "text_proj"
            print(f"[gen] warm-started q_align in {n_ws} aligned adapter(s) from {head} "
                  f"(fc1 input layer left freshly initialized)")

    def build_memory(z):
        """Aligned projection -> cross-attn memory. "expanded" lifts z back to
        encoder-hidden space; "aligned" uses z directly. Frozen; call under
        no_grad."""
        return z if aligned_mem else src_expand(z)

    # 2b) Target-modality encoder — ONLY for the per-epoch round-trip eval, which
    # has to re-encode text the model just generated (nothing in the cache covers a
    # sequence that didn't exist yet). This is the one genuinely new resident cost of
    # the eval: AMPLIFY-350M (~1.4 GB fp32) for text2protein, BioLinkBERT-base for
    # protein2text. Everything else the eval needs — the source features AND the true
    # targets for the ceiling row — comes from the packed cache, so no source encoder
    # is ever loaded here. Loaded rank-0-first (AMPLIFY is trust_remote_code, same
    # transformers_modules write race as the decoder above) and up front rather than
    # lazily, so a missing/oversized encoder fails in the first minute of the job
    # instead of at the end of epoch 0.
    tgt_encoder = tgt_enc_tok = None
    if args.rt_eval and args.rt_samples > 0:
        def _load_tgt_encoder():
            if args.direction == "text2protein":
                return load_protein_encoder(cfg.model.protein_encoder_path, device)
            return load_text_encoder(cfg.model.text_encoder_path, device,
                                     cfg.data.caption_field_labels)

        if env.is_main:
            tgt_encoder, tgt_enc_tok = _load_tgt_encoder()
        barrier()
        if not env.is_main:
            tgt_encoder, tgt_enc_tok = _load_tgt_encoder()
        if env.is_main:
            which = ("protein encoder (AMPLIFY)" if args.direction == "text2protein"
                     else "text encoder (BioLinkBERT)")
            print(f"[gen] round-trip eval ON: loaded the frozen {which} to re-encode "
                  f"generated targets")

    def encode_generated(strs):
        """Generated target strings -> (per-token hidden states, valid mask).

        Masking matches the cache the retrieval model was trained on (specials and
        caption field labels dropped), so a generated row and a cached row are
        scored on the same footing.
        """
        if args.direction == "text2protein":
            return encode_protein_batch(tgt_encoder, tgt_enc_tok, strs, device,
                                        cfg.data.max_protein_tokens, mask_specials=True)
        return encode_text_batch(tgt_encoder, tgt_enc_tok, strs, device,
                                 cfg.data.max_text_tokens, mask_specials=True,
                                 mask_field_labels=True)

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
    # Group-aware split over (same protein) ∪ (same caption), matching retrieval —
    # MUST use the identical grouping or build_or_load_splits would treat the
    # retrieval split file as stale (n_groups mismatch) and rebuild it differently.
    # Rank 0 owns creation; everyone reads after a barrier (no write race).
    if env.is_main:
        group_ids = merged_split_group_ids(
            group_ids_from_accessions([p.uid for p in pairs]),
            group_ids_from_texts([p.text for p in pairs]))
        splits = build_or_load_splits(
            str(splits_path), len(pairs), cfg.data.splits, args.seed, group_ids=group_ids)
        print(f"[gen] splits: {len(pairs)} rows over {splits['n_groups']} groups "
              f"(protein ∪ caption)")
    barrier()
    splits = load_splits(str(splits_path))

    # `val` is a random permutation slice, so the first N is a deterministic
    # random sample; keeps the per-epoch rank-0 eval cheap at full scale.
    val_indices = splits["val"]
    if args.val_subset and args.val_subset > 0:
        val_indices = val_indices[:args.val_subset]

    train_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                                 splits["train"],
                                 cfg.model.protein_hidden, cfg.model.text_hidden,
                                 need_target=False)
    val_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                               val_indices,
                               cfg.model.protein_hidden, cfg.model.text_hidden,
                               need_target=False)

    # Decoder control tokens between BOS and the target body (ProtGPT3's "1"
    # direction marker; empty for the other decoders). Resolved from the raw
    # decoder, before the DDP wrap below. --direction-augment additionally samples
    # the "2" (C-to-N) marker per training example; reverse_prefix_ids raises for a
    # decoder without a reverse marker, so a bad --direction-augment fails loudly.
    prefix_ids = target_prefix_ids(decoder, target_tok)
    rev_prefix_ids = reverse_prefix_ids(decoder, target_tok) if meta.direction_augment else None
    if env.is_main and prefix_ids:
        print(f"[gen] decoder target prefix ids: {prefix_ids}"
              + (f" | direction-augment reverse ids: {rev_prefix_ids}" if rev_prefix_ids else ""))

    # Train collate augments direction (if enabled); val stays forward-only so its
    # CE is comparable across runs and reflects the deployment (N-to-C) path.
    train_collate = make_collate(target_tok, cfg.generation.max_target_tokens, prefix_ids,
                                 rev_prefix_ids, direction_augment=meta.direction_augment,
                                 target_input_dropout=args.target_input_dropout,
                                 dropout_mean_span=args.dropout_mean_span)
    # Val is never corrupted, so its CE / the ablation stay comparable across runs.
    val_collate = make_collate(target_tok, cfg.generation.max_target_tokens, prefix_ids)
    if env.distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=env.world_size, rank=env.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=train_collate,
                                  drop_last=True, num_workers=cfg.generation.num_workers)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate, drop_last=True,
                                  num_workers=cfg.generation.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=val_collate,
                            num_workers=cfg.generation.num_workers)

    # Round-trip pool: a FIXED random sample of val, drawn with its own generator so
    # it is independent of --seed, of --val-subset (whose rows are the front of the
    # same permutation and would otherwise couple the two metrics), and stable across
    # restarts. Every rank derives the identical pool, then takes a round-robin shard
    # of it — so the eval's wall time is one SMALL per-rank generate rather than one
    # large one, and it stays flat as long as the pool fits in world_size batches.
    # need_target=True hands the eval the TRUE target's cached per-token features,
    # which is the ceiling row for free (no encoder pass, and identical to what the
    # retrieval model itself reports on these rows).
    rt_pool, rt_loader, rt_uids = [], None, []
    if args.rt_eval and args.rt_samples > 0:
        val_all = list(splits["val"])
        g_rt = torch.Generator().manual_seed(args.rt_seed)
        perm = torch.randperm(len(val_all), generator=g_rt).tolist()
        rt_pool = [val_all[i] for i in perm[:args.rt_samples]]
        my_rt = rt_pool[env.rank::env.world_size]
        rt_uids = [pairs[i].uid for i in my_rt]
        rt_ds = GenerationDataset(args.direction, args.cache_dir, pairs, my_rt,
                                  cfg.model.protein_hidden, cfg.model.text_hidden,
                                  need_target=True)
        # shuffle=False: rows come back in `my_rt` order, which is how rt_uids is
        # aligned to them (the collate carries no ids).
        rt_loader = DataLoader(rt_ds, batch_size=args.rt_batch_size, shuffle=False,
                               collate_fn=val_collate, num_workers=0)
        if env.is_main:
            print(f"[gen] round-trip pool: {len(rt_pool)} val rows "
                  f"(~{len(rt_pool) / max(env.world_size, 1):.1f} rows/rank, "
                  f"bs={args.rt_batch_size}, max_new={args.rt_max_new_tokens}, "
                  f"temp={args.rt_temperature}, top_p={args.rt_top_p})")
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

    # 4) Optim — two LR groups: the cross-attn adapters at --lr, and any unfrozen
    # decoder blocks at --unfreeze-lr (defaults to --lr). The frozen backbone
    # contributes no params. Each group carries its own base_lr; the
    # cosine schedule scales both by the same factor each step (see train loop).
    # With no unfreeze (or --unfreeze-lr unset) this reduces exactly to one LR.
    adapter_param_ids = {id(p) for a in adapters for p in a.parameters()}
    head_params, backbone_params = [], []
    for name, p in core.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in adapter_param_ids or "cross_attn" in name:
            head_params.append(p)          # cross-attn adapters
        else:
            backbone_params.append(p)      # fully-unfrozen decoder blocks

    backbone_lr = args.unfreeze_lr if args.unfreeze_lr is not None else args.lr
    param_groups = [{"params": head_params, "base_lr": args.lr, "lr": args.lr}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "base_lr": backbone_lr, "lr": backbone_lr})
    optim = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=cfg.generation.weight_decay,
    )
    # Captured now, re-applied after any optimizer_state load. torch's
    # load_state_dict REPLACES each param_group with the SAVED dict (splicing the
    # live params back in), so a custom key the current code sets but an older
    # checkpoint never wrote is silently dropped — and the LR schedule reads
    # base_lr every step. See the re-injection in the --resume block.
    base_lrs = [g["base_lr"] for g in param_groups]
    train_params = head_params + backbone_params   # for grad clipping below
    if env.is_main:
        msg = (f"[gen] optim: heads={sum(p.numel() for p in head_params):,} @ lr={args.lr:.2e}")
        msg += (f" | backbone={sum(p.numel() for p in backbone_params):,} @ lr={backbone_lr:.2e}"
                if backbone_params else " | backbone=none")
        print(msg)
    total_steps = args.epochs * len(train_loader)
    warmup = max(int(cfg.generation.warmup_frac * total_steps), 1)

    # 5) Train loop
    ckpt_dir = Path(args.ckpt_dir) / args.direction
    if env.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    # ---- per-epoch round-trip retrieval eval -------------------------------
    # Val CE is teacher-forced: it scores next-token prediction with the TRUE prefix
    # in hand, so it can improve while free-running generation ignores the
    # conditioning entirely. This is the prefix-free counterpart — generate the pool
    # free-running, re-encode each output, and ask whether it retrieves its own source
    # against the whole pool (accession-grouped, so any other row of the same protein
    # counts as a positive). Same generation path, same FILIP scorer and same
    # `score_roundtrip_records` as `python -m src.roundtrip_eval`, so the per-epoch
    # curve and the offline number agree on an identical pool.
    #
    # Cost: the pool is sharded across ranks, so this is ONE small generate per rank
    # (~pool/world_size rows), one encoder pass over those rows, and one N x N FILIP
    # on rank 0. At the job scale in train_*_SFT.pbs (192 ranks) the default 256-row
    # pool is 1-2 rows per rank — a minute or two per epoch against a multi-hour
    # epoch. `--rt-max-new-tokens` is the knob that actually moves that number.
    rt_prompt: list = []
    rt_pad_id = rt_eos_id = None
    rt_empty = "M" if args.direction == "text2protein" else "protein"
    if rt_loader is not None:
        bos = target_tok.bos_token_id
        if bos is None:
            bos = target_tok.eos_token_id
        if bos is None:
            raise SystemExit(
                "--rt-eval needs a decoder tokenizer with a BOS or EOS token to seed "
                "generation; this one has neither. Re-run with --no-rt-eval.")
        # Forward (N-to-C) prefix only, even under --direction-augment: val CE and
        # deployment are both forward, so the eval measures the same path. (A "2"
        # rollout would still decode correctly — decode_target re-reverses it — but
        # it would mix two distributions into one curve.)
        rt_prompt = [bos] + prefix_ids
        rt_pad_id = (target_tok.pad_token_id if target_tok.pad_token_id is not None
                     else target_tok.eos_token_id)
        rt_eos_id = (target_tok.eos_token_id if target_tok.eos_token_id is not None
                     else rt_pad_id)

    # The ceiling (true target -> source, same scorer and pool) depends only on the
    # frozen retrieval model and the fixed pool, so it is computed on the first eval
    # and reused. `done` is tracked on EVERY rank rather than inferred from rank 0's
    # return value, or the off-main ranks would keep projecting true targets forever.
    rt_state = {"ceiling": None, "done": False, "disabled": False}

    @torch.no_grad()
    def run_roundtrip(epoch_id: int):
        """Round-trip eval over the fixed pool. Returns metrics on rank 0, else None."""
        if rt_loader is None or rt_state["disabled"]:
            return None
        barrier()
        t_rt = time.time()
        was_training = decoder.training
        core.eval()
        need_ceiling = not rt_state["done"]
        # Seed the eval's own stream so the sampling noise is the same every epoch
        # (the curve then moves because the model moved), and restore the training
        # stream afterwards so enabling this eval doesn't shift training at all.
        rng = rng_snapshot(device)
        torch.manual_seed(args.rt_seed + env.rank)
        recs, gen_lens, true_lens, failed = [], [], [], ""

        def rt_batches():
            """Cached source rows -> the batch dicts generate_roundtrip_records wants.

            The ceiling row's z_true comes from the TRUE target's cached per-token
            features (need_target=True on this dataset), through the same frozen
            projection — no encoder pass, and exactly what retrieval itself scores
            on these rows. `true_lens` is collected here because only this caller
            has the collated teacher-forcing tensors to measure it from.
            """
            cursor = 0
            for batch in rt_loader:
                h = batch["h"].to(device).float()
                mask = batch["m"].to(device)
                B = h.size(0)
                z = src_proj(h)                     # aligned source tokens (scored below)
                out = {"z_src": z, "mem": build_memory(z), "mask_src": mask,
                       "uids": rt_uids[cursor:cursor + B]}
                if need_ceiling:
                    out["z_true"] = tgt_proj(batch["h_tgt"].to(device).float())
                    out["mask_true"] = batch["m_tgt"].to(device)
                am = batch["attn_mask"]
                last = (am.sum(1) - 1).clamp(min=0)
                last_tok = batch["input_ids"].gather(1, last[:, None]).squeeze(1)
                true_lens.extend(
                    (am.sum(1) - len(rt_prompt) - (last_tok == rt_eos_id).long())
                    .clamp(min=0).tolist())
                cursor += B
                yield out

        try:
            recs, gen_lens = generate_roundtrip_records(
                core, target_tok, adapters, rt_batches(),
                prompt=rt_prompt, pad_id=rt_pad_id, eos_id=rt_eos_id,
                tgt_proj=tgt_proj, encode_generated=encode_generated,
                empty_target=rt_empty, max_new_tokens=args.rt_max_new_tokens,
                temperature=args.rt_temperature, top_p=args.rt_top_p)
        except Exception:
            # A metric must not take down a multi-hour training job — but it also
            # must not deadlock it. Bailing out here would leave the other ranks
            # blocked in the all-gather below until the walltime expires, so the
            # failure is recorded, shipped through the SAME collective, and the eval
            # is then disabled on EVERY rank (each one sees every payload, so they
            # agree without an extra broadcast). Training continues; the traceback is
            # printed in full so the cause is still diagnosable.
            failed = traceback.format_exc()
            print(f"[gen][rt] rank {env.rank}: eval failed at epoch={epoch_id}\n{failed}",
                  flush=True)
        finally:
            rng_restore(device, rng)
            if was_training:
                decoder.train()

        parts = gather_roundtrip_payloads(
            {"recs": recs, "gen": gen_lens, "true": true_lens, "failed": failed}, env)
        bad = [i for i, p in enumerate(parts) if p["failed"]]
        if bad:
            rt_state["disabled"] = True
            if env.is_main:
                print(f"[gen][rt] DISABLED for the rest of the run: the eval raised on "
                      f"{len(bad)}/{len(parts)} rank(s) (first: rank {bad[0]}). Training "
                      f"is unaffected; re-run the metric offline with "
                      f"`python -m src.roundtrip_eval`.", flush=True)
            barrier()
            return None
        recs = [r for p in parts for r in p["recs"]]
        gen_lens = [v for p in parts for v in p["gen"]]
        true_lens = [v for p in parts for v in p["true"]]
        rt_state["done"] = True

        if not env.is_main:
            barrier()
            return None
        metrics, ceiling = score_roundtrip_records(
            recs, cfg.model.embed_dim, device, chunk_rows=args.rt_filip_chunk_rows,
            need_ceiling=need_ceiling)
        if ceiling is not None:
            rt_state["ceiling"] = ceiling
        metrics["ceiling"] = rt_state["ceiling"]
        metrics["gen_len_mean"] = sum(gen_lens) / max(len(gen_lens), 1)
        metrics["true_len_mean"] = sum(true_lens) / max(len(true_lens), 1)
        metrics["seconds"] = round(time.time() - t_rt, 2)
        g2s, s2g = metrics["gen2src"], metrics["src2gen"]
        ceil = metrics["ceiling"] or {}
        # R@K is only reported for k <= pool size, so a tiny --rt-samples has no R@10.
        print(f"[gen][rt] epoch={epoch_id} n={metrics['n']} "
              f"gen->src R@1={g2s['R@1']:.4f} "
              f"R@10={g2s.get('R@10', float('nan')):.4f} "
              f"mAP={g2s['mAP']:.4f} medrank={g2s['median_rank']:.0f} | "
              f"src->gen R@1={s2g['R@1']:.4f} mAP={s2g['mAP']:.4f} | "
              f"ceiling R@1={ceil.get('R@1', float('nan')):.4f} "
              f"mAP={ceil.get('mAP', float('nan')):.4f} | "
              f"len gen={metrics['gen_len_mean']:.0f} true={metrics['true_len_mean']:.0f} "
              f"| {metrics['seconds']:.1f}s", flush=True)
        with open(ckpt_dir / "sft_roundtrip.jsonl", "a") as f:
            # Self-describing rows: R@K/mAP are only comparable at a fixed pool size
            # and sampling config, so each line carries the ones that set them.
            f.write(json.dumps({
                "epoch": epoch_id, **metrics,
                "pool": len(rt_pool), "temperature": args.rt_temperature,
                "top_p": args.rt_top_p, "max_new_tokens": args.rt_max_new_tokens,
            }) + "\n")
        barrier()
        return metrics

    global_step = 0
    log = []
    start_epoch = 0

    # Resume: restore the trainable adapter weights (+ any unfrozen blocks), the
    # optimizer, and the step counter, then continue after the saved epoch. Every
    # rank loads the same file (shared FS) so states stay in sync. adapter_state is
    # a partial state_dict (trainable tensors only), so it loads with strict=False
    # against the full decoder.
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume, ckpt_dir)
        ckpt = torch.load(resume_path, map_location=device)
        # Pre-flag checkpoints read back as the behaviour they actually had
        # (GenMeta's defaults), so an old checkpoint resumes rather than tripping
        # a spurious mismatch.
        GenMeta.from_ckpt(ckpt).assert_matches(meta, f"--resume {resume_path}")
        # `missing` keys are expected (the frozen backbone isn't in adapter_state);
        # only unexpected keys signal a real mismatch.
        _, unexpected = core.load_state_dict(ckpt["adapter_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"--resume unexpected keys in adapter_state: {unexpected}")
        if "optimizer_state" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state"])
            # Restore base_lr, which load_state_dict drops for any checkpoint
            # written before that key existed (KeyError 'base_lr' at the first LR
            # update otherwise). Re-injected from THIS invocation's --lr /
            # --unfreeze-lr, so a resume with a different LR does what you'd expect
            # rather than silently inheriting the old one.
            for g, blr in zip(optim.param_groups, base_lrs):
                g["base_lr"] = blr
        elif env.is_main:
            print("[resume] checkpoint has no optimizer_state; "
                  "optimizer restarts from scratch")
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        if start_epoch >= args.epochs:
            # Report start_epoch, not ckpt['epoch']: the loop is
            # range(start_epoch, epochs), so --epochs must exceed start_epoch.
            raise RuntimeError(
                f"--resume checkpoint finished epoch {ckpt['epoch']}, so training would "
                f"start at epoch {start_epoch}, but --epochs is {args.epochs} and the "
                f"loop runs range({start_epoch}, {args.epochs}) — nothing to do. "
                f"Pass --epochs {args.epochs + 1} or higher to run "
                f"{args.epochs + 1 - start_epoch} more epoch(s).")
        log = ckpt.get("train_log", [])
        if env.is_main:
            print(f"[resume] loaded {resume_path}; resuming at epoch {start_epoch} "
                  f"(global_step={global_step})")

    # Baseline before any training — also a smoke test of the eval path on this
    # config, surfaced in the first minute rather than at the end of epoch 0. Runs
    # after --resume, so a continued job reports where it starts from.
    if args.rt_at_start:
        run_roundtrip(start_epoch - 1)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        decoder.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            frac = cosine_warmup_factor(global_step, total_steps, warmup)
            for g in optim.param_groups:
                g["lr"] = g["base_lr"] * frac
            lr = optim.param_groups[0]["lr"]     # head-group LR, for logging
            optim.zero_grad(set_to_none=True)

            h = batch["h"].to(device).float()
            mask = batch["m"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["labels"].to(device)

            # Encoder front-end (frozen): project the source side, then build the
            # conditioning memory (expand / aligned z / aligned+map — see build_memory).
            with torch.no_grad():
                z = src_proj(h)
                mem = build_memory(z)

            contrast_on = args.contrast_lambda > 0.0 and input_ids.size(0) >= 2
            if contrast_on:
                # Negative memory: pair each row with another example's memory (a
                # batch roll is a derangement for B>=2). Concatenate the correct and
                # shuffled conditioning into ONE forward of size 2B — rows 0:B attend
                # the correct memory, rows B:2B the wrong one — so there is a single
                # forward/backward (DDP-clean, unlike two separate decoder calls).
                B = input_ids.size(0)
                mem_cat = torch.cat([mem, mem.roll(1, 0)], dim=0)
                mask_cat = torch.cat([mask, mask.roll(1, 0)], dim=0)
                set_cross_memory(adapters, mem_cat, mask_cat)
                out = decoder(input_ids=input_ids.repeat(2, 1),
                              attention_mask=attn_mask.repeat(2, 1))
                pos_logits, neg_logits = out.logits[:B], out.logits[B:]
            else:
                set_cross_memory(adapters, mem, mask)
                out = decoder(input_ids=input_ids, attention_mask=attn_mask)
                pos_logits, neg_logits = out.logits, None

            # Primary CE: token-weighted global over the positive forward — numerically
            # identical to the non-contrastive loss, so train/val stay comparable.
            shift_labels = labels[:, 1:].contiguous()
            valid = (shift_labels != -100)
            tok_pos = F.cross_entropy(
                pos_logits[:, :-1, :].reshape(-1, pos_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100, reduction="none",
            ).view(shift_labels.shape)
            ce = tok_pos.sum() / valid.sum().clamp(min=1)

            loss = ce

            cov_pen, cov_mean = None, None
            if args.coverage_lambda > 0.0:
                # rows=B under --contrast-lambda: the back half of the 2B forward
                # attends the ROLLED (wrong) memory, and demanding coverage of a
                # mismatched prompt is exactly backwards.
                cov_pen, cov_mean = coverage_penalty(
                    adapters, attn_mask.bool(), tau=args.coverage_tau,
                    rows=input_ids.size(0) if contrast_on else None)
                loss = loss + args.coverage_lambda * cov_pen

            gap = None
            if contrast_on:
                # Hinge: the correct memory should beat a wrong one by >= margin
                # nats/token, per example. Pushes ce_neg UP and ce_pos DOWN, so the
                # memory *content* has to matter — this is exactly the "content gain"
                # the ablation measures, made an explicit training objective.
                ce_pos_ex = tok_pos.sum(1) / valid.sum(1).clamp(min=1)      # [B]
                ce_neg_ex = _per_example_ce(neg_logits, labels)            # [B]
                gap = (ce_neg_ex - ce_pos_ex).mean()
                hinge = F.relu(args.contrast_margin - (ce_neg_ex - ce_pos_ex)).mean()
                loss = loss + args.contrast_lambda * hinge
            loss.backward()
            # Clear AFTER backward, not before: with gradient checkpointing the
            # decoder layers (and their cross-attn adapter hooks) are recomputed
            # during backward, so the memory must still be set then. Clearing it
            # before backward made the recomputed forward take the adapters'
            # pass-through branch, tripping the checkpoint tensor-count check.
            # (Harmless without checkpointing — next step's set_cross_memory
            # overwrites it.)
            clear_cross_memory(adapters)
            total_norm = torch.nn.utils.clip_grad_norm_(train_params, cfg.generation.grad_clip)
            # Skip the step on a non-finite grad norm instead of letting it poison
            # every weight: clip_grad_norm_ rescales all grads by clip/total_norm, so
            # a single inf/NaN grad makes the coefficient NaN and turns EVERY param
            # NaN on step() — permanently (all later forwards are NaN, train + val).
            # Grads are DDP-synced, so total_norm is identical on all ranks and they
            # skip in lockstep. Next iter's zero_grad clears the bad grads.
            if torch.isfinite(total_norm):
                optim.step()
            elif env.is_main:
                print(f"[warn] non-finite grad norm ({total_norm.item()}) at epoch={epoch} "
                      f"step={it+1}; skipping optimizer step", flush=True)
            global_step += 1

            if env.is_main and ((it + 1) % cfg.generation.log_every == 0 or it == 0):
                with torch.no_grad():
                    ppl = torch.exp(ce).item()
                msg = (f"[{args.direction}] epoch={epoch} step={it+1}/{len(train_loader)} "
                       f"lr={lr:.2e} ce={ce.item():.4f} ppl={ppl:.2f} "
                       f"trunc={int(batch['n_truncated'])}/{input_ids.size(0)} "
                       f"L={input_ids.size(1)}")
                if gap is not None:
                    msg += f" gap={gap.item():+.4f}"   # train-time content gain (ce_neg-ce_pos)
                if cov_pen is not None:
                    # cov = mean max-attention per source position (the quantity tau
                    # thresholds); covpen = normalized squared shortfall, 0 = all covered.
                    msg += f" cov={cov_mean.item():.4f} covpen={cov_pen.item():.4f}"
                print(msg, flush=True)

        dt = time.time() - t0
        if env.is_main:
            print(f"[{args.direction}] epoch={epoch} done in {dt:.1f}s")

        # Round-trip retrieval eval — all ranks (the pool is sharded), so it runs
        # BEFORE the rank-0-only val/checkpoint block, and its metrics go into the
        # same train_log record.
        rt_metrics = run_roundtrip(epoch)

        # Val pass + checkpoint on rank 0 only (uses the unwrapped decoder).
        if env.is_main:
            core.eval()
            val_losses = []
            val_cov, val_covpen, cov_profile = [], [], None
            with torch.no_grad():
                for batch in val_loader:
                    h = batch["h"].to(device).float()
                    mask = batch["m"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attn_mask"].to(device)
                    labels = batch["labels"].to(device)
                    z = src_proj(h)
                    mem = build_memory(z)
                    set_cross_memory(adapters, mem, mask)
                    out = core(input_ids=input_ids, attention_mask=attn_mask)
                    # Read coverage BEFORE clearing — clear_cross_memory drops
                    # last_attn. Reported whenever the attention is available, not
                    # just when the aux is on: with --coverage-lambda 0 this is the
                    # baseline measurement that tells you where to put --coverage-tau.
                    if meta.cross_attn_mode == "aligned":
                        vp, vc = coverage_penalty(adapters, attn_mask.bool(),
                                                  tau=args.coverage_tau)
                        val_cov.append(vc.item())
                        val_covpen.append(vp.item())
                        # Full distribution from the FIRST val batch only: it is the
                        # tau-picking readout, and one batch of source positions is
                        # already O(BS * L) samples.
                        if cov_profile is None:
                            cov_profile = coverage_profile(adapters, attn_mask.bool())
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
            rec = {"epoch": epoch, "val_ce": val_ce, "val_ppl": math.exp(val_ce)}
            if rt_metrics is not None:
                rec["roundtrip"] = rt_metrics
            cov_msg = ""
            if val_cov:
                rec["val_coverage"] = sum(val_cov) / len(val_cov)
                rec["val_coverage_pen"] = sum(val_covpen) / len(val_covpen)
                cov_msg = (f" cov={rec['val_coverage']:.4f} "
                           f"covpen={rec['val_coverage_pen']:.4f}")
            print(f"[val] epoch={epoch} ce={val_ce:.4f} ppl={math.exp(val_ce):.2f}{cov_msg}")
            if cov_profile:
                rec["coverage_profile"] = cov_profile
                fb = " ".join(f"<{t:g}:{f:.3f}"
                              for t, f in cov_profile["frac_below"].items())
                # A tau at or below `unif` cannot separate attended from ignored:
                # that is the mass a position gets when attention spreads evenly.
                print(f"[val][cov] unif={cov_profile['uniform']:.5f} "
                      f"p10={cov_profile['p10']:.4f} p50={cov_profile['p50']:.4f} "
                      f"p90={cov_profile['p90']:.4f} | frac_below {fb} "
                      f"(n={cov_profile['n']})")
            log.append(rec)

            # Save every trainable tensor (cross-attn adapters + any unfrozen
            # blocks), keyed by requires_grad so partial-unfreeze weights persist.
            trainable = {n for n, p in core.named_parameters() if p.requires_grad}
            adapter_state = {k: v for k, v in core.state_dict().items() if k in trainable}
            payload = {"epoch": epoch, "global_step": global_step,
                       "adapter_state": adapter_state,
                       "optimizer_state": optim.state_dict(),
                       "train_log": log,
                       # Training-recipe fields; the architecture header follows.
                       "target_input_dropout": args.target_input_dropout,
                       "dropout_mean_span": args.dropout_mean_span,
                       "contrast_lambda": args.contrast_lambda,
                       "contrast_margin": args.contrast_margin,
                       "coverage_lambda": args.coverage_lambda,
                       "coverage_tau": args.coverage_tau,
                       "lr": args.lr,
                       "unfreeze_lr": args.unfreeze_lr,
                       **meta.to_payload()}
            path = ckpt_dir / f"epoch{epoch:02d}.pt"
            torch.save(payload, path)
            print(f"[ckpt] saved {path}")
        barrier()

    if env.is_main:
        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("[gen] done")
    cleanup()


if __name__ == "__main__":
    main()
