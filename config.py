"""Central configuration for mini-embed-filip.

Retrieval phase trains projection + expansion heads via FILIP-style late
interaction + per-token reconstruction. Generation phase trains a per-direction
decoder cross-attention adapter + LoRA on top of frozen pretrained decoders.

Encoders and decoders all stay frozen except for the explicitly trainable
adapters listed in each `*Cfg` block.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Paths default to the local dev (Mac) layout but are environment-overridable, so
# a config.py synced from the dev machine doesn't clobber the cluster's paths.
# On Aurora, set these in the job script, e.g.:
#   export FILIP_MODELS_DIR=/flare/NLDesignProtein/bryan/FILIP-dev-space/models
#   export FILIP_DATA_CSV=/flare/.../SwissProt_full/fully_annotated_...csv
# FILIP_MODELS_DIR swaps the base dir for all four models at once (subdir names
# are assumed identical); the per-model vars override an individual path.
MODELS_DIR = os.environ.get("FILIP_MODELS_DIR", "/Users/bryan/Documents/models")

DATA_CSV = os.environ.get(
    "FILIP_DATA_CSV",
    "/Users/bryan/Documents/datasets/SwissProt_full/fully_annotated_swiss_prot_LEGACY_UPDATE_20260505.csv",
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
    max_protein_tokens: int = 1024            # bumped from 512 — covers long-tail proteins

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
    embed_dim: int = 64
    proj_dropout: float = 0.1

    expand_d_mid: int = 256
    expand_d_hidden: int = 512
    expand_dropout: float = 0.1

    # Cross-modal memory map g: embed_dim -> embed_dim (Medium). A residual MLP,
    # trained in the frozen map phase (train_memory_map), that nudges a source
    # token's aligned projection toward the *target* modality's region of the
    # shared space (text->protein for text2protein). Off unless a decoder run sets
    # --memory-map; this is just the hidden width when it is built.
    map_hidden: int = 128


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
class ReconCfg:
    """Expansion-only reconstruction phase (Feature 2).

    Freezes the projection heads + temperature and trains only the expansion
    heads on reconstruction MSE. Retrieval metrics depend only on the projection,
    so they are unchanged; this sharpens the generation conditioning memory
    expand(project(h)) at zero retrieval cost. Operates on an existing retrieval
    checkpoint and writes the same checkpoint format, so downstream generation /
    inference / round-trip load it transparently.
    """
    cache_dir: str = str(REPO_ROOT / "cache")
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "reconstruction")
    device: str = "auto"
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3                          # recon tolerates a higher LR than joint retrieval
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 50


@dataclass
class MapCfg:
    """Cross-modal memory-map phase (Medium).

    Takes a trained retrieval checkpoint, freezes the projection + expansion heads
    and the temperature, and trains ONLY the two memory maps (text_map,
    protein_map) with a FILIP-contrastive aux computed on *detached* projections:
    map(z_src) must retrieve the true target's aligned tokens. Because the inputs
    are detached, the retrieval geometry (and R@K) is mathematically untouched — a
    map-augmented checkpoint that generation can load and optionally apply. Writes
    the same {"epoch", "model_state"} format as train_retrieval.
    """
    cache_dir: str = str(REPO_ROOT / "cache")
    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "memory_map")
    device: str = "auto"
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3                          # a small head; tolerates a high LR
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 50


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
    #   "aligned":  project(h) directly — the 64-d shared retrieval space, no
    #               expansion head. Keeps generation in the same space as
    #               retrieval; the expansion cannot add information beyond the
    #               64-d projection, so this drops a provably-redundant lift (and
    #               12x the cross-attn KV width). Requires retraining the adapters
    #               (k/v projections change from 768->512 to 64->512).
    memory_space: str = "expanded"            # or "aligned"

    # Apply the retrieval memory map to the source projection before it becomes
    # cross-attn memory (Medium). Requires memory_space="aligned" and a
    # map-augmented retrieval checkpoint (train_memory_map). Off => the raw aligned
    # z is used (the bypass arm of the A/B).
    memory_map: bool = False

    # Cross-attention scoring space (Strong).
    #   "head":    learned multi-head attention (q_proj/k_proj into the decoder head
    #              space). The original adapter.
    #   "aligned": single-head cosine max-sim in the shared 64-d space — queries are
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

    # LoRA on existing self-attn / FFN
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets_self_attn: bool = True
    lora_targets_ffn: bool = True

    # Generation tokenizer caps (max target length)
    max_target_tokens: int = 512

    # Generation-side CVAE (Feature 1). Off by default; --use-cvae enables. The
    # latent w captures the one-to-many residual p(target|source) and is injected
    # as `cvae_n_latent_tokens` extra cross-attention memory tokens.
    use_cvae: bool = False
    cvae_d_w: int = 32                        # latent dim
    cvae_n_latent_tokens: int = 4             # memory tokens decoded from w
    cvae_hidden: int = 256                    # prior/posterior MLP width
    cvae_beta_max: float = 0.1                # KL weight after warmup
    cvae_free_bits: float = 0.5               # per-dim KL floor (anti-collapse)
    cvae_kl_warmup_frac: float = 0.3          # fraction of training to ramp beta 0->max

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
    recon: ReconCfg = field(default_factory=ReconCfg)
    memory_map: MapCfg = field(default_factory=MapCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)


def default_cfg() -> Cfg:
    return Cfg()
