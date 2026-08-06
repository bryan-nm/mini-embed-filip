"""Central configuration for mini-embed-filip.

Retrieval phase trains projection + expansion heads via FILIP-style late
interaction + per-token reconstruction. Generation phase trains a per-direction
decoder cross-attention adapter on top of frozen pretrained decoders.

Encoders and decoders all stay frozen except for the explicitly trainable
adapters listed in each `*Cfg` block.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model and dataset locations
# ---------------------------------------------------------------------------
# THIS FILE OWNS EVERY PATH. Job scripts must not export FILIP_* or pass model /
# CSV locations on the command line — a path that appears in two places
# eventually disagrees with itself, and the .o file then records a path the job
# did not actually read. `python config.py` prints what a run will resolve to;
# every job script banners that output, so the log is the record.
#
# The FILIP_* environment variables remain as an escape hatch for a workstation
# whose models live elsewhere (they are what the local smoke tests use). They are
# deliberately NOT set by any job script. FILIP_MODELS_DIR swaps the base dir for
# all four models at once; the per-model vars override an individual path.
MODELS_DIR = os.environ.get("FILIP_MODELS_DIR", "/flare/NLDesignProtein/bryan/FILIP-dev-space/models")
DATASETS_DIR = os.environ.get(
    "FILIP_DATASETS_DIR", "/flare/NLDesignProtein/bryan/FILIP-dev-space/datasets")

# The corpus: one row per (protein, caption) pair, ONE caption per accession.
# Every entry point reads this one — training, inspection and embedding export
# alike. This build is filtered to proteins of 500 residues or fewer, which is
# what lets max_protein_tokens sit at 512 (see DataCfg).
DATA_CSV = os.environ.get(
    "FILIP_DATA_CSV",
    f"{DATASETS_DIR}/SwissProt_full/fully_annotated_swiss_prot_080326.csv",
)

TEXT_ENCODER_PATH = os.environ.get("FILIP_TEXT_ENCODER", f"{MODELS_DIR}/BioLinkBERT-base")
PROTEIN_ENCODER_PATH = os.environ.get("FILIP_PROTEIN_ENCODER", f"{MODELS_DIR}/SaAMPLIFY_350M")
PROTEIN_DECODER_PATH = os.environ.get("FILIP_PROTEIN_DECODER", f"{MODELS_DIR}/ProtGPT3-112M-dpo")  # Mixtral MoE
TEXT_DECODER_PATH = os.environ.get("FILIP_TEXT_DECODER", f"{MODELS_DIR}/biogpt")


@dataclass
class DataCfg:
    csv_path: str = DATA_CSV
    # Column names in the CSV. The new SwissProt-full file ships with these
    # exact headers; override if a different schema is used.
    csv_id_col: str = "primary_Accession"
    csv_protein_col: str = "protein_sequence"
    csv_text_col: str = "[final]text_caption"
    csv_pfam_col: str = "pfam_label"             # optional; kept for future homology use
    subset_size: int = 0
    seed: int = 0
    splits: tuple = (0.90, 0.05, 0.05)
    max_text_tokens: int = 512
    # The corpus is filtered to proteins <= 500 residues, so 512 (with BOS/EOS)
    # covers every sequence with nothing truncated. Raise this alongside the
    # corpus if an unfiltered SwissProt is ever swapped back in — the long tail
    # runs past 1024.
    max_protein_tokens: int = 512

    # Caption-schema field labels: the UPPERCASE "LABEL:" prefixes the SwissProt-
    # full captions are templated with (all 10 that occur across the corpus). The
    # text tokenizer registers each as a single special token (see
    # encoders.load_text_encoder); cross-modal comparison masks them out. This list
    # is the schema seam — swap it if the caption format changes; a new text
    # embedder just gets these labels registered against its own tokenizer.
    caption_field_labels: tuple = (
        "PROTEIN NAME:", "LINEAGE:", "GENE ONTOLOGY:", "SIMILARITY:", "FUNCTION:",
        "ENZYME CLASS:", "CATALYTIC ACTIVITY:", "PATHWAY:", "DOMAIN:", "PTM:", 
        "SUBCELLULAR LOCATION:",
    )


@dataclass
class ModelCfg:
    text_encoder_path: str = TEXT_ENCODER_PATH
    protein_encoder_path: str = PROTEIN_ENCODER_PATH
    text_hidden: int = 768                    # BioLinkBERT-base
    protein_hidden: int = 960                 # SaAMPLIFY-350M (same 960-d as AMPLIFY-350M)

    proj_d_hidden: int = 512
    proj_d_mid: int = 256
    # Width of the shared retrieval space, and the sweep variable. Nothing else
    # in the repo hard-codes a value: the projection/expansion heads, the aligned
    # cross-attention memory, the round-trip scorer and best-of-N all read it
    # from here, so changing this line is the whole change.
    embed_dim: int = 16
    proj_dropout: float = 0.1

    expand_d_mid: int = 256
    expand_d_hidden: int = 512
    expand_dropout: float = 0.1


@dataclass
class RetrievalCfg:
    """Retrieval (FILIP) training phase."""
    use_cache: bool = True
    cache_dir: str = str(REPO_ROOT / "cache")
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "retrieval")

    device: str = "auto"

    # Cached path can run larger batches because projection+loss is light.
    batch_size: int = 128
    live_batch_size: int = 8
    num_workers: int = 0
    # Batches per worker held in flight. Total shared memory is
    # num_workers * prefetch_factor * batch_bytes per rank, and every rank on a
    # node draws from the same /dev/shm — so this multiplies with num_workers,
    # it does not trade against it. Only bites when num_workers > 0.
    prefetch_factor: int = 2

    phase1_epochs: int = 1
    phase2_epochs: int = 3

    # Loss weights
    phase1_uniformity_weight: float = 0.1
    r2_uniformity_weight: float = 0.01         # token-spread regularizer during R2 (contrastive)
    align_aux_weight: float = 0.1
    recon_weight: float = 0.05

    uniformity_t: float = 2.0

    # Token positions to exclude from FILIP comparisons and uniformity
    # (special tokens have no biological/textual content; they absorb spurious
    # alignment weight otherwise).
    mask_text_special_tokens: bool = True
    mask_protein_special_tokens: bool = True

    # Field-label special tokens (caption prefixes like "FUNCTION:", see
    # DataCfg.caption_field_labels) are emitted as single tokens by the text
    # tokenizer regardless; this flag only controls whether they're masked out of
    # FILIP comparisons + uniformity. They are template artifacts with no content,
    # so mask them for cross-modal comparison — an embedding/export run may keep
    # them (set False). Independent of mask_text_special_tokens (CLS/SEP/PAD).
    mask_text_field_labels: bool = True

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0

    init_temperature: float = 0.07
    max_temperature: float = 100.0

    log_every: int = 50
    eval_every_epoch: bool = True


@dataclass
class HardNegCfg:
    """Hard-decoy negative mining (resume phase on top of a retrieval checkpoint).

    A *decoy* is a caption built from a real caption by replacing exactly one
    templated field (see `DataCfg.caption_field_labels`) with the corresponding
    field taken from a different caption. It shares almost all of its tokens with
    the true caption, so it is a much harder negative than any in-batch caption:
    the model can only rank it below the truth by actually reading the swapped
    field rather than matching the caption's overall topic.

    Two stages, mirroring precompute -> train:
      src/precompute_decoys.py  builds the decoy plan + packed per-token cache
      src/train_hard_negatives.py  resumes a retrieval checkpoint with the R2
                                   objective + a ramped decoy discrimination loss
    """
    # Packed decoy text cache (its own dir; the base cache is left untouched).
    decoy_cache_dir: str = str(REPO_ROOT / "cache" / "decoys")
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "hard_negatives")
    device: str = "auto"

    batch_size: int = 128
    # Nonzero by default, unlike the retrieval phase: this loader reads 1+max_decoys
    # text rows per sample instead of 1, so the per-step random-read count on the
    # shared filesystem triples. With num_workers=0 every one of those reads is
    # serialized into the training step; workers overlap them with compute. The
    # decoy MATH is <1% of the R2 contrastive math, so if a step is slow it is
    # essentially always this.
    # 2, not 4. This is the only phase in the repo that runs loader workers, and
    # so the only one whose batches transit /dev/shm: at the 512-token caps a
    # decoy batch is ~100 MB (protein + text + 2 decoy captions), and
    # num_workers * prefetch_factor * 100 MB * 12 ranks/node is what the node has
    # to hold. At 4x2 that is ~9.6 GB/node steady and ~19 GB across an epoch
    # boundary; a worker that cannot get its shared-memory page dies with SIGBUS,
    # and its peers then fail with oneCCL "RECV entry failed". Raise only
    # alongside the shm figure the job banner prints.
    num_workers: int = 2
    prefetch_factor: int = 2
    epochs: int = 3
    lr: float = 5e-5                          # fine-tune LR: we resume a converged model
    weight_decay: float = 0.01
    warmup_frac: float = 0.02
    grad_clip: float = 1.0
    log_every: int = 50

    # --- decoy construction (precompute side) ---
    decoys_per_row: int = 2                   # decoys generated per caption
    decoy_seed: int = 1234                    # plan RNG; separate from the split seed
    # Restrict which fields may be swapped. Empty = every label in
    # DataCfg.caption_field_labels is swappable.
    swap_fields: tuple = ()
    # Donor rejection: never take the replacement field from a caption of the same
    # protein (its fields may be legitimately true of the anchor) and never from a
    # different train/val/test split (that would leak held-out caption text into
    # training). Both default on; precompute_decoys exposes escape hatches.
    donor_same_protein: bool = False
    donor_cross_split: bool = False

    # --- decoy loss (training side) ---
    max_decoys: int = 2                       # decoys scored per anchor per step
    decoy_weight: float = 0.5                 # final weight after the ramp
    decoy_start_frac: float = 0.05            # fraction of the run before the ramp starts
    decoy_ramp_frac: float = 0.35             # fraction of the run spent ramping 0 -> weight
    # 0 => (K+1)-way softmax discrimination (temperature-scaled, shares logit_scale).
    # >0 => hinge on the raw FILIP-score gap: relu(margin - (s_pos - s_decoy)).
    decoy_margin: float = 0.0


@dataclass
class GenerationCfg:
    """One block per generation direction."""
    direction: str = "text2protein"           # or "protein2text"
    decoder_path: str = PROTEIN_DECODER_PATH  # for text2protein. swap for protein2text.

    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "generation")
    retrieval_ckpt: str = ""                  # path to a trained retrieval checkpoint

    device: str = "auto"

    # Conditioning memory space (what the decoder cross-attends over).
    #   "expanded": expand(project(h)) in the encoder-hidden space (768 text /
    #               960 protein). The original design.
    #   "aligned":  project(h) directly — the shared retrieval space (embed_dim),
    #               no expansion head. Keeps generation in the same space as
    #               retrieval; the expansion cannot add information beyond the
    #               projection, so this drops a provably-redundant lift (and most
    #               of the cross-attn KV width). Requires retraining the adapters
    #               (k/v projections change from encoder-hidden to embed_dim).
    memory_space: str = "expanded"            # or "aligned"

    # Cross-attention scoring space (Strong).
    #   "head":    learned multi-head attention (q_proj/k_proj into the decoder head
    #              space). The original adapter.
    #   "aligned": single-head cosine max-sim in the shared space — queries are
    #              projected to embed_dim and scored directly against the aligned
    #              memory (no k_proj), so the attention weights ARE the FILIP
    #              [residue x source-token] array. Requires memory_space="aligned".
    cross_attn_mode: str = "head"             # or "aligned"

    # Stochastic N-to-C / C-to-N direction augmentation (ProtGPT3 only). Each
    # training target is randomly factorized forward ("1", N-to-C) or reverse
    # ("2", C-to-N, residues reversed). Same protein + same conditioning memory,
    # opposite autoregressive order — an ~free augmentation that also regularizes
    # the conditioning toward direction-invariant (semantic) signal, since local
    # positional shortcuts don't transfer across order. Validation + inference stay
    # forward-only for comparability. Distinct from `direction` (text2protein vs
    # protein2text); this is the within-text2protein sequence order.
    direction_augment: bool = False

    # Cross-attention adapter insertion
    cross_attn_every: int = 2                 # insert at every other decoder block

    # Generation tokenizer caps (max target length)
    max_target_tokens: int = 512

    batch_size: int = 16
    num_workers: int = 0

    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0

    log_every: int = 50


@dataclass
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    hard_neg: HardNegCfg = field(default_factory=HardNegCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)


def default_cfg() -> Cfg:
    return Cfg()


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------
def resolved_paths(cfg: Cfg | None = None) -> dict:
    """Every path a run will actually read or write, already resolved.

    `python config.py` prints this; the job scripts banner it into the .o file so
    a log always says which models and which corpus produced it.
    """
    cfg = cfg or default_cfg()
    return {
        "models_dir": MODELS_DIR,
        "datasets_dir": DATASETS_DIR,
        "text_encoder": cfg.model.text_encoder_path,
        "protein_encoder": cfg.model.protein_encoder_path,
        "protein_decoder": PROTEIN_DECODER_PATH,
        "text_decoder": TEXT_DECODER_PATH,
        "data_csv": cfg.data.csv_path,
        "cache_dir": cfg.retrieval.cache_dir,
        "decoy_cache_dir": cfg.hard_neg.decoy_cache_dir,
    }


def describe_paths(cfg: Cfg | None = None) -> str:
    paths = resolved_paths(cfg)
    width = max(len(k) for k in paths)
    lines = []
    for k, v in paths.items():
        # Flag anything missing: a typo'd path otherwise surfaces minutes later as
        # an opaque HuggingFace or open() error, after the queue slot is spent.
        mark = "" if Path(v).exists() else "   [MISSING]"
        lines.append(f"{k:<{width}} = {v}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_paths())
