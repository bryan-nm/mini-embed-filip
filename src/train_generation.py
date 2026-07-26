"""Per-direction generation trainer.

Reads a trained retrieval checkpoint (projection + expansion heads), runs the
appropriate encoder→project→expand front-end to produce per-token cross-
attention memory, then trains the decoder's cross-attention adapters + LoRA
on cross-entropy of the target sequence.

What's frozen vs trainable:
  frozen:   encoder, projection head, expansion head, decoder backbone
  trains:   cross-attention adapter layers + LoRA on decoder self-attn / FFN

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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import default_cfg, Cfg, TEXT_DECODER_PATH
from src.data import (
    PackedPerTokenCache,
    Pair,
    build_or_load_splits,
    group_ids_from_accessions,
    group_ids_from_texts,
    merged_split_group_ids,
    load_pairs,
    load_row_protein_idx,
    load_splits,
)
from src.dist import (
    init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
)
from src.decoder_adapters import (
    LoRACfg,
    clear_cross_memory,
    count_trainable,
    load_decoder_with_cross_attn,
    reverse_prefix_ids,
    set_cross_memory,
    target_prefix_ids,
    unfreeze_decoder_blocks,
    warm_start_q_align,
)
from src.cvae import build_cvae, beta_at
from src.losses import masked_mean
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
    protein row), because the protein cache is deduplicated. When `need_target` is
    set (CVAE training), the opposite modality's per-token hidden states are also
    returned so the posterior q(w|source,target) can pool the target embedding.
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
            # Opposite modality (the generated side) for the CVAE posterior.
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
    # (Dayhoff's build_inputs_with_special_tokens is a no-op; ProtGPT3 ships with
    # add_bos_token=False), which would leave the model with no EOS to learn
    # termination and a BOS-mismatch vs inference (which seeds generation with
    # BOS). Doing it here is uniform across decoders. `prefix_ids` carries any
    # decoder control tokens that sit between BOS and the body — for ProtGPT3,
    # the "1" (N-to-C) direction marker it was pretrained with.
    #
    # Direction augmentation (ProtGPT3): when `augment`, each target is factorized
    # forward ("1" + residues) or reverse ("2" + reversed residues) with equal
    # probability. Same protein, same conditioning memory, opposite decode order;
    # decode_target re-reverses a "2" sequence back to N-to-C at inference.
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

        # Tokenize bodies without specials, then wrap with [BOS] [prefix] ... [EOS].
        seqs = []
        body_spans = []          # (start, length) of the residue body within each row
        for b in batch:
            target, pfx = b["target"], prefix_ids
            # Coin-flip C-to-N: reverse the residue string (char == token for the
            # char-level vocab) and use the reverse marker. CPU RNG, seeded, and
            # separate from the device dropout RNG.
            if augment and bool(torch.rand(()) < 0.5):
                target, pfx = target[::-1], rev_prefix_ids
            body = target_tokenizer(
                target, add_special_tokens=False, truncation=True,
                max_length=body_cap,
            )["input_ids"]
            head = ([bos_id] if bos_id is not None else []) + pfx
            ids = head + body + ([eos_id] if eos_id is not None else [])
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
        }

        # Optional target-side per-token hidden states (CVAE posterior). Padded
        # to the batch max; pooled through the frozen target projection later.
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


# ---------------------------------------------------------------------------
def load_retrieval_model(ckpt_path: str, device: torch.device,
                         with_maps: bool = False) -> MiniEmbedFilip:
    """Load a frozen retrieval model. With `with_maps`, also build the memory maps
    and require the checkpoint to carry trained map weights (from train_memory_map).

    The load is strict=False so a map-augmented checkpoint can be read by a
    map-less run (extra map keys ignored) and vice versa; only NON-map missing /
    unexpected keys are a real mismatch. When `with_maps` is set, all-map-keys-
    missing means the checkpoint predates the map phase — fail loudly.
    """
    cfg = default_cfg()
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
        with_maps=with_maps,
        map_hidden=cfg.model.map_hidden,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = m.load_state_dict(state["model_state"], strict=False)
    is_map = lambda k: k.startswith("text_map.") or k.startswith("protein_map.")
    bad = [k for k in missing if not is_map(k)] + [k for k in unexpected if not is_map(k)]
    if bad:
        raise RuntimeError(f"retrieval load mismatch (non-map keys): {bad}")
    if with_maps and any(is_map(k) for k in missing):
        raise RuntimeError(
            f"--memory-map set but retrieval checkpoint {ckpt_path} has no trained "
            f"memory map (run train_memory_map first)")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


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
                    help="fully fine-tune N decoder blocks (0 = adapters/LoRA only); "
                         "which N blocks is set by --unfreeze-where")
    ap.add_argument("--unfreeze-where", choices=["top", "bottom"], default="top",
                    help="which end of the decoder stack --unfreeze-top selects: 'top' = the "
                         "last N blocks (nearest logits, default), 'bottom' = the first N "
                         "(nearest embeddings, lets conditioning become receptive early)")
    ap.add_argument("--unfreeze-lr", type=float, default=None,
                    help="separate (typically lower) LR for the unfrozen decoder blocks; "
                         "defaults to --lr. The adapters/LoRA/CVAE heads always use --lr. "
                         "Keep this well below --lr to avoid wrecking the pretrained prior")
    ap.add_argument("--warm-start-qalign", action="store_true",
                    help="initialize the aligned-mode cross-attn query MLP (q_align) from the "
                         "retrieval projection head of the generated modality, so the decoder-"
                         "side query lands in the shared space from step 0 instead of cold-"
                         "starting a random map. Requires --cross-attn-mode aligned")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute layer activations in backward to fit large decoders "
                         "(e.g. the 3B Dayhoff/Jamba); ~30%% slower, much less memory. "
                         "NOTE: incompatible with the Jamba MoE on XPU — routing isn't "
                         "bit-reproducible on recompute, so non-reentrant checkpoint aborts")
    ap.add_argument("--max-target-tokens", type=int, default=cfg.generation.max_target_tokens,
                    help="cap on the generated/teacher-forced target length; lower it to "
                         "cut decoder activation memory (truncates long targets)")
    ap.add_argument("--memory-space", choices=["expanded", "aligned"],
                    default=cfg.generation.memory_space,
                    help="conditioning memory: 'expanded' cross-attends over "
                         "expand(project(h)) in encoder-hidden space (768/960-d, original); "
                         "'aligned' cross-attends over project(h) directly (64-d shared "
                         "retrieval space, no expansion head)")
    ap.add_argument("--memory-map", action="store_true", default=cfg.generation.memory_map,
                    help="apply the retrieval memory map g to the aligned source projection "
                         "before it becomes cross-attn memory (Medium). Requires "
                         "--memory-space aligned and a map-augmented retrieval ckpt")
    ap.add_argument("--cross-attn-mode", choices=["head", "aligned"],
                    default=cfg.generation.cross_attn_mode,
                    help="cross-attn scoring space (Strong): 'head' learned multi-head "
                         "(original); 'aligned' single-head cosine max-sim in the 64-d shared "
                         "space (attention weights = FILIP array). Requires --memory-space aligned")
    ap.add_argument("--direction-augment", action="store_true",
                    default=cfg.generation.direction_augment,
                    help="stochastically factorize each protein target N-to-C ('1') or C-to-N "
                         "('2', residues reversed) per example (ProtGPT3 only). Free augmentation "
                         "+ regularizes conditioning toward direction-invariant signal. Val stays "
                         "forward-only")
    ap.add_argument("--no-lora", action="store_true",
                    help="disable LoRA on the frozen decoder self-attn/FFN — train ONLY the "
                         "cross-attention adapters. LoRA is caption-blind (unconditional domain "
                         "adaptation); dropping it removes that shortcut so gains must come from "
                         "conditioning. Use to A/B whether LoRA helps or suppresses conditioning")
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
    ap.add_argument("--subset-size", type=int, default=cfg.data.subset_size)
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--val-subset", type=int, default=1000,
                    help="evaluate on the first N val pairs each epoch (0 = full val split)")
    # Generation-side CVAE (Feature 1).
    ap.add_argument("--use-cvae", action="store_true", default=cfg.generation.use_cvae,
                    help="train a conditional VAE latent injected as extra cross-attn memory tokens")
    ap.add_argument("--cvae-d-w", type=int, default=cfg.generation.cvae_d_w)
    ap.add_argument("--cvae-n-latent-tokens", type=int,
                    default=cfg.generation.cvae_n_latent_tokens)
    ap.add_argument("--cvae-beta-max", type=float, default=cfg.generation.cvae_beta_max)
    args = ap.parse_args()

    # Fold CVAE CLI overrides back into the config block build_cvae reads.
    cfg.generation.use_cvae = args.use_cvae
    cfg.generation.cvae_d_w = args.cvae_d_w
    cfg.generation.cvae_n_latent_tokens = args.cvae_n_latent_tokens
    cfg.generation.cvae_beta_max = args.cvae_beta_max
    cfg.generation.max_target_tokens = args.max_target_tokens
    cfg.generation.memory_space = args.memory_space
    cfg.generation.memory_map = args.memory_map
    cfg.generation.cross_attn_mode = args.cross_attn_mode
    cfg.generation.direction_augment = args.direction_augment
    # --no-lora zeroes both LoRA targets; the LoRACfg below reads these, so the
    # injectors then skip LoRA entirely (adapters-only training).
    if args.no_lora:
        cfg.generation.lora_targets_self_attn = False
        cfg.generation.lora_targets_ffn = False

    # "aligned" cross-attn and the memory map both live in the 64-d shared space,
    # so they require aligned memory (the map is a 64->64 residual; aligned scoring
    # keys ARE the aligned vectors). Fail loudly rather than silently mis-condition.
    if args.memory_space != "aligned" and (args.memory_map or args.cross_attn_mode == "aligned"):
        raise SystemExit(
            "--memory-map / --cross-attn-mode aligned require --memory-space aligned")
    if args.target_input_dropout > 0.0 and args.direction != "text2protein":
        raise SystemExit(
            "--target-input-dropout is a text2protein feature (it corrupts protein-residue "
            "targets with random amino acids); leave it 0 for protein2text")
    if not (0.0 <= args.target_input_dropout < 1.0):
        raise SystemExit("--target-input-dropout must be in [0, 1)")
    if args.warm_start_qalign and args.cross_attn_mode != "aligned":
        raise SystemExit(
            "--warm-start-qalign requires --cross-attn-mode aligned (q_align exists only "
            "in aligned mode)")

    env = init_distributed(args.device, group_size=1)
    device = env.device
    torch.manual_seed(args.seed)
    if env.is_main:
        print(f"[gen] direction={args.direction} world_size={env.world_size} device={device}")

    # 1) Retrieval model (frozen) — provides projection + expansion (+ maps).
    retrieval = load_retrieval_model(args.retrieval_ckpt, device, with_maps=args.memory_map)
    if env.is_main:
        print(f"[gen] loaded retrieval model from {args.retrieval_ckpt} "
              f"(memory_map={args.memory_map}, cross_attn_mode={args.cross_attn_mode})")

    # 2) Decoder + adapters
    decoder_path = (
        cfg.generation.decoder_path
        if args.direction == "text2protein"
        else TEXT_DECODER_PATH
    )
    # Memory dim depends on the conditioning space. "expanded": the encoder hidden
    # dim of the INPUT modality (what the expansion head outputs) — text2protein is
    # text -> text_expand -> 768-d. "aligned": the 64-d shared projection space
    # (project(h)), so the adapter k/v are 64->512 and no expansion head is used.
    aligned_mem = args.memory_space == "aligned"
    if aligned_mem:
        mem_dim = cfg.model.embed_dim
    else:
        mem_dim = cfg.model.text_hidden if args.direction == "text2protein" else cfg.model.protein_hidden
    if env.is_main:
        print(f"[gen] memory_space={args.memory_space} mem_dim={mem_dim}")

    # Gradient checkpointing recomputes the forward during backward, but dropout
    # RNG is not reliably preserved across recomputation on XPU. A non-zero LoRA
    # dropout then makes the recomputed hidden states differ from the original,
    # which — through Jamba's MoE router — changes per-expert token counts and
    # trips the non-reentrant checkpoint metadata check. Zero it for checkpointed
    # runs (negligible regularization on a frozen-backbone finetune); the
    # cross-attn adapter dropout is already 0.
    if args.grad_checkpointing and cfg.generation.lora_dropout > 0:
        if env.is_main:
            print(f"[gen] grad-checkpointing: forcing LoRA dropout "
                  f"{cfg.generation.lora_dropout} -> 0 (dropout RNG not preserved on recompute)")
        cfg.generation.lora_dropout = 0.0

    lora_cfg = LoRACfg(
        rank=cfg.generation.lora_rank, alpha=cfg.generation.lora_alpha,
        dropout=cfg.generation.lora_dropout,
        target_self_attn=cfg.generation.lora_targets_self_attn,
        target_ffn=cfg.generation.lora_targets_ffn,
    )
    # trust_remote_code (ProGen2) compiles the custom modeling file into a
    # *shared* transformers_modules cache. If all ranks do this at once they
    # race on the write and some import a half-defined module ("has no attribute
    # ProGenForCausalLM"). Rank 0 populates the cache first; the barrier then
    # releases the others to import it read-only (concurrent reads are safe).
    def _load_decoder():
        return load_decoder_with_cross_attn(
            args.direction, decoder_path, args.cross_attn_every, mem_dim, lora_cfg, device,
            cross_attn_mode=args.cross_attn_mode,
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
    n_unfrozen = unfreeze_decoder_blocks(decoder, args.unfreeze_top, where=args.unfreeze_where)
    if env.is_main and args.unfreeze_top > 0:
        print(f"[gen] unfroze {args.unfreeze_where} {args.unfreeze_top} decoder blocks "
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
            print(f"[gen] upcast decoder to fp32 for stable fine-tuning of the "
                  f"unfrozen blocks (bf16 master weights don't train)")
    if env.is_main:
        n_train = count_trainable(decoder)
        lora_on = cfg.generation.lora_targets_self_attn or cfg.generation.lora_targets_ffn
        print(f"[gen] decoder trainable params (cross-attn"
              f"{' + LoRA' if lora_on else ', LoRA OFF'}): {n_train:,}")
        print(f"[gen] num cross-attention adapters: {len(adapters)}")

    # Gradient checkpointing: recompute layer activations in backward instead of
    # storing them across the deep stack. Essential for the 3B Jamba decoder,
    # whose pure-PyTorch Mamba scan materializes large fp32 [B, d_inner, L,
    # d_state] tensors per layer that otherwise OOM the tile.
    #   - use_reentrant=False is REQUIRED: the backbone is frozen (only adapters/
    #     LoRA train), and reentrant checkpointing demands an input that requires
    #     grad, which a frozen layer input doesn't have.
    #   - Incompatible with the KV/SSM cache, so disable use_cache for training.
    # Applied before the DDP wrap so `decoder` is still the raw model. For the
    # Jamba path the cross-attn adapters are forward hooks inside the layer
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
    # the cross-attn memory) and target projection (for the CVAE posterior).
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
    if args.warm_start_qalign:
        n_ws = warm_start_q_align(adapters, tgt_proj)
        if env.is_main:
            head = "protein_proj" if args.direction == "text2protein" else "text_proj"
            print(f"[gen] warm-started q_align in {n_ws} aligned adapter(s) from {head} "
                  f"(fc1 input layer left freshly initialized)")

    def build_memory(z):
        """Aligned projection -> cross-attn memory. "expanded" lifts z back to
        encoder-hidden space; "aligned" uses z directly, optionally passed through
        the frozen cross-modal map (--memory-map). All frozen; call under no_grad."""
        if not aligned_mem:
            return src_expand(z)
        return retrieval.map_source(z, args.direction) if args.memory_map else z

    # 2b) Generation-side CVAE heads (Feature 1). Trained alongside the adapters;
    # injected as extra cross-attention memory tokens.
    cvae = None
    n_latent = 0
    if args.use_cvae:
        cvae = build_cvae(cfg.generation, mem_dim, cfg.model.embed_dim).to(device)
        n_latent = cvae.cfg.n_latent_tokens
        if env.distributed:
            broadcast_parameters(cvae)            # all ranks start from rank-0 weights
        if env.is_main:
            n_cvae = sum(p.numel() for p in cvae.parameters())
            print(f"[gen] CVAE enabled: d_w={cvae.cfg.d_w} "
                  f"latent_tokens={n_latent} beta_max={cvae.cfg.beta_max} "
                  f"({n_cvae:,} params)")

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

    # Only train needs the target side (CVAE posterior); val conditions on the
    # prior mean, so it never loads the target modality.
    train_ds = GenerationDataset(args.direction, args.cache_dir, pairs,
                                 splits["train"],
                                 cfg.model.protein_hidden, cfg.model.text_hidden,
                                 need_target=args.use_cvae)
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
    rev_prefix_ids = reverse_prefix_ids(decoder, target_tok) if args.direction_augment else None
    if env.is_main and prefix_ids:
        print(f"[gen] decoder target prefix ids: {prefix_ids}"
              + (f" | direction-augment reverse ids: {rev_prefix_ids}" if rev_prefix_ids else ""))

    # Train collate augments direction (if enabled); val stays forward-only so its
    # CE is comparable across runs and reflects the deployment (N-to-C) path.
    train_collate = make_collate(target_tok, cfg.generation.max_target_tokens, prefix_ids,
                                 rev_prefix_ids, direction_augment=args.direction_augment,
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

    # 4) Optim — two LR groups: the adapters/LoRA/CVAE "heads" at --lr, and any
    # unfrozen decoder blocks at --unfreeze-lr (defaults to --lr). The frozen
    # backbone contributes no params. Each group carries its own base_lr; the
    # cosine schedule scales both by the same factor each step (see train loop).
    # With no unfreeze (or --unfreeze-lr unset) this reduces exactly to one LR.
    adapter_param_ids = {id(p) for a in adapters for p in a.parameters()}
    head_params, backbone_params = [], []
    for name, p in core.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in adapter_param_ids or "lora_" in name or "cross_attn" in name:
            head_params.append(p)          # cross-attn adapters + LoRA
        else:
            backbone_params.append(p)      # fully-unfrozen decoder blocks
    if cvae is not None:
        head_params += list(cvae.parameters())

    backbone_lr = args.unfreeze_lr if args.unfreeze_lr is not None else args.lr
    param_groups = [{"params": head_params, "base_lr": args.lr, "lr": args.lr}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "base_lr": backbone_lr, "lr": backbone_lr})
    optim = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=cfg.generation.weight_decay,
    )
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

    global_step = 0
    log = []
    start_epoch = 0

    # Resume: restore the trainable adapter/LoRA weights (+ any unfrozen blocks),
    # the CVAE heads, the optimizer, and the step counter, then continue after the
    # saved epoch. Every rank loads the same file (shared FS) so states stay in
    # sync. adapter_state is a partial state_dict (trainable tensors only), so it
    # loads with strict=False against the full decoder.
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume, ckpt_dir)
        ckpt = torch.load(resume_path, map_location=device)
        # memory_space/memory_map/cross_attn_mode/lora_targets_* default to the
        # pre-flag values ("expanded"/False/"head"/True) for older checkpoints.
        if ckpt.get("cross_attn_every") != args.cross_attn_every or \
           ckpt.get("unfreeze_top") != args.unfreeze_top or \
           ckpt.get("unfreeze_where", "top") != args.unfreeze_where or \
           ckpt.get("warm_start_qalign", False) != args.warm_start_qalign or \
           ckpt.get("memory_space", "expanded") != args.memory_space or \
           ckpt.get("memory_map", False) != args.memory_map or \
           ckpt.get("cross_attn_mode", "head") != args.cross_attn_mode or \
           ckpt.get("lora_targets_self_attn", True) != cfg.generation.lora_targets_self_attn or \
           ckpt.get("lora_targets_ffn", True) != cfg.generation.lora_targets_ffn:
            raise RuntimeError(
                f"--resume checkpoint architecture mismatch: "
                f"cross_attn_every={ckpt.get('cross_attn_every')} unfreeze_top={ckpt.get('unfreeze_top')} "
                f"unfreeze_where={ckpt.get('unfreeze_where', 'top')} "
                f"warm_start_qalign={ckpt.get('warm_start_qalign', False)} "
                f"memory_space={ckpt.get('memory_space', 'expanded')} "
                f"memory_map={ckpt.get('memory_map', False)} "
                f"cross_attn_mode={ckpt.get('cross_attn_mode', 'head')} "
                f"lora_self_attn={ckpt.get('lora_targets_self_attn', True)} "
                f"lora_ffn={ckpt.get('lora_targets_ffn', True)} "
                f"vs args cross_attn_every={args.cross_attn_every} unfreeze_top={args.unfreeze_top} "
                f"unfreeze_where={args.unfreeze_where} warm_start_qalign={args.warm_start_qalign} "
                f"memory_space={args.memory_space} memory_map={args.memory_map} "
                f"cross_attn_mode={args.cross_attn_mode} "
                f"lora_self_attn={cfg.generation.lora_targets_self_attn} "
                f"lora_ffn={cfg.generation.lora_targets_ffn}")
        # `missing` keys are expected (the frozen backbone isn't in adapter_state);
        # only unexpected keys signal a real mismatch.
        _, unexpected = core.load_state_dict(ckpt["adapter_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"--resume unexpected keys in adapter_state: {unexpected}")
        if cvae is not None:
            if "cvae_state" not in ckpt:
                raise RuntimeError(
                    "--resume: --use-cvae set but checkpoint has no cvae_state")
            cvae.load_state_dict(ckpt["cvae_state"])
        elif "cvae_state" in ckpt and env.is_main:
            print("[resume] checkpoint has cvae_state but --use-cvae not set; ignoring it")
        if "optimizer_state" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state"])
        elif env.is_main:
            print("[resume] checkpoint has no optimizer_state; "
                  "optimizer restarts from scratch")
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        if start_epoch >= args.epochs:
            raise RuntimeError(
                f"--resume epoch {ckpt['epoch']} already at/past --epochs {args.epochs}; "
                f"increase --epochs to continue")
        log = ckpt.get("train_log", [])
        if env.is_main:
            print(f"[resume] loaded {resume_path}; resuming at epoch {start_epoch} "
                  f"(global_step={global_step})")

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

            kl = torch.zeros((), device=device)
            if cvae is not None:
                # Posterior sees pooled source + target (both frozen embeddings);
                # sampled latent -> extra cross-attn memory tokens. KL pulls the
                # posterior toward the learned conditional prior p(w|source).
                with torch.no_grad():
                    z_src_pool = masked_mean(z, mask)
                    h_tgt = batch["h_tgt"].to(device).float()
                    m_tgt = batch["m_tgt"].to(device)
                    z_tgt_pool = masked_mean(tgt_proj(h_tgt), m_tgt)
                qmu, qlv = cvae.posterior_params(z_src_pool, z_tgt_pool)
                pmu, plv = cvae.prior_params(z_src_pool)
                w = cvae.reparam(qmu, qlv)
                w_tok = cvae.latent_tokens(w)                # [B, k, mem_dim], carries grad
                mem = torch.cat([mem, w_tok], dim=1)
                k_mask = torch.ones(mem.size(0), n_latent, dtype=torch.bool, device=device)
                mask = torch.cat([mask, k_mask], dim=1)
                kl = cvae.kl(qmu, qlv, pmu, plv)

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

            beta = beta_at(global_step, total_steps, cvae.cfg) if cvae is not None else 0.0
            loss = ce + beta * kl

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
            # DDP syncs the decoder grads; the CVAE heads live outside its forward,
            # so average their grads manually (mirrors the retrieval trainer).
            if cvae is not None:
                average_gradients(cvae)
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
                       f"lr={lr:.2e} ce={ce.item():.4f} ppl={ppl:.2f}")
                if cvae is not None:
                    msg += f" kl={kl.item():.4f} beta={beta:.3f}"
                if gap is not None:
                    msg += f" gap={gap.item():+.4f}"   # train-time content gain (ce_neg-ce_pos)
                print(msg, flush=True)

        dt = time.time() - t0
        if env.is_main:
            print(f"[{args.direction}] epoch={epoch} done in {dt:.1f}s")

        # Val pass + checkpoint on rank 0 only (uses the unwrapped decoder).
        if env.is_main:
            core.eval()
            val_losses = []
            if cvae is not None:
                cvae.eval()
            with torch.no_grad():
                for batch in val_loader:
                    h = batch["h"].to(device).float()
                    mask = batch["m"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attn_mask"].to(device)
                    labels = batch["labels"].to(device)
                    z = src_proj(h)
                    mem = build_memory(z)
                    # Inference-time conditioning: prior mean (no target available).
                    if cvae is not None:
                        pmu, _ = cvae.prior_params(masked_mean(z, mask))
                        mem = torch.cat([mem, cvae.latent_tokens(pmu)], dim=1)
                        k_mask = torch.ones(mem.size(0), n_latent, dtype=torch.bool, device=device)
                        mask = torch.cat([mask, k_mask], dim=1)
                    set_cross_memory(adapters, mem, mask)
                    out = core(input_ids=input_ids, attention_mask=attn_mask)
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
            if cvae is not None:
                cvae.train()
            val_ce = sum(val_losses) / max(len(val_losses), 1)
            print(f"[val] epoch={epoch} ce={val_ce:.4f} ppl={math.exp(val_ce):.2f}")
            log.append({"epoch": epoch, "val_ce": val_ce, "val_ppl": math.exp(val_ce)})

            # Save every trainable tensor (cross-attn + LoRA + any unfrozen
            # blocks), keyed by requires_grad so partial-unfreeze weights persist.
            trainable = {n for n, p in core.named_parameters() if p.requires_grad}
            adapter_state = {k: v for k, v in core.state_dict().items() if k in trainable}
            payload = {"epoch": epoch, "global_step": global_step,
                       "adapter_state": adapter_state,
                       "optimizer_state": optim.state_dict(),
                       "train_log": log,
                       "cross_attn_every": args.cross_attn_every,
                       "unfreeze_top": args.unfreeze_top,
                       "unfreeze_where": args.unfreeze_where,
                       "target_input_dropout": args.target_input_dropout,
                       "dropout_mean_span": args.dropout_mean_span,
                       "contrast_lambda": args.contrast_lambda,
                       "contrast_margin": args.contrast_margin,
                       "unfreeze_lr": args.unfreeze_lr,
                       "warm_start_qalign": args.warm_start_qalign,
                       "memory_space": args.memory_space,
                       "memory_map": args.memory_map,
                       "cross_attn_mode": args.cross_attn_mode,
                       "direction_augment": args.direction_augment,
                       "lora_targets_self_attn": cfg.generation.lora_targets_self_attn,
                       "lora_targets_ffn": cfg.generation.lora_targets_ffn}
            # CVAE heads (Feature 1); absent => downstream loads run without a latent.
            if cvae is not None:
                payload["cvae_state"] = cvae.state_dict()
                payload["cvae_cfg"] = {
                    "d_w": cvae.cfg.d_w, "n_latent_tokens": cvae.cfg.n_latent_tokens,
                    "hidden": cvae.cfg.hidden, "mem_dim": cvae.cfg.mem_dim,
                }
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
