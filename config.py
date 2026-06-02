"""Central configuration for mini-embed-filip.

Retrieval phase trains projection + expansion heads via FILIP-style late
interaction + per-token reconstruction. Generation phase trains a per-direction
decoder cross-attention adapter + LoRA on top of frozen pretrained decoders.

Encoders and decoders all stay frozen except for the explicitly trainable
adapters listed in each `*Cfg` block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

DATA_CSV = "/Users/bryan/Documents/datasets/SwissProt_full/fully_annotated_swiss_prot_LEGACY_UPDATE_20260505.csv"

TEXT_ENCODER_PATH = "/Users/bryan/Documents/models/BioLinkBERT-base"
PROTEIN_ENCODER_PATH = "/Users/bryan/Documents/models/SaAMPLIFY_120M"
PROTEIN_DECODER_PATH = "/Users/bryan/Documents/models/progen2-small"
TEXT_DECODER_PATH = "/Users/bryan/Documents/models/biogpt"


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
    max_text_tokens: int = 1024
    max_protein_tokens: int = 1024            # bumped from 512 — covers long-tail proteins


@dataclass
class ModelCfg:
    text_encoder_path: str = TEXT_ENCODER_PATH
    protein_encoder_path: str = PROTEIN_ENCODER_PATH
    text_hidden: int = 768                    # BioLinkBERT-base
    protein_hidden: int = 640                 # SaAMPLIFY-120M

    proj_d_hidden: int = 512
    proj_d_mid: int = 256
    embed_dim: int = 32
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

    phase1_epochs: int = 1
    phase2_epochs: int = 3

    # Loss weights
    phase1_uniformity_weight: float = 0.1
    align_aux_weight: float = 0.1
    recon_weight: float = 0.05

    uniformity_t: float = 2.0

    # Token positions to exclude from FILIP comparisons and uniformity
    # (special tokens have no biological/textual content; they absorb spurious
    # alignment weight otherwise).
    mask_text_special_tokens: bool = True
    mask_protein_special_tokens: bool = True

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0

    init_temperature: float = 0.07
    max_temperature: float = 100.0

    log_every: int = 50
    eval_every_epoch: bool = True


@dataclass
class GenerationCfg:
    """One block per generation direction."""
    direction: str = "text2protein"           # or "protein2text"
    decoder_path: str = PROTEIN_DECODER_PATH  # for text2protein. swap for protein2text.

    ckpt_dir: str = str(REPO_ROOT / "checkpoints" / "generation")
    retrieval_ckpt: str = ""                  # path to a trained retrieval checkpoint

    device: str = "auto"

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
    generation: GenerationCfg = field(default_factory=GenerationCfg)


def default_cfg() -> Cfg:
    return Cfg()
