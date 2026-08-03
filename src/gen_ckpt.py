"""The architecture header carried by every generation checkpoint.

A generation checkpoint stores its trainable tensors plus the handful of flags
needed to rebuild the decoder that produced them: how many blocks got an adapter,
what space the memory and the attention scores live in, and which blocks were
unfrozen. Five entry points (`train_generation --resume`, `generate`,
`generate_set`, `roundtrip_eval`, `ablate_memory`, `train_rl`) all have to read
those flags back and agree on the defaults for checkpoints written before a flag
existed.

`GenMeta` is that agreement in one place. Adding or retiring a flag is a change
here and nowhere else — previously it was the same five `ckpt.get(k, default)`
blocks edited in parallel, which is exactly how they drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GenMeta:
    """Decoder architecture a generation checkpoint was trained with.

    The defaults are the *pre-flag* values, so a checkpoint written before a
    field existed reads back as the behaviour it actually had.
    """

    cross_attn_every: int = 2
    memory_space: str = "expanded"        # "expanded" | "aligned"
    cross_attn_mode: str = "head"         # "head" | "aligned"
    unfreeze_top: int = 0
    unfreeze_where: str = "top"           # "top" | "bottom"
    warm_start_qalign: bool = False
    direction_augment: bool = False

    @property
    def aligned_memory(self) -> bool:
        return self.memory_space == "aligned"

    def mem_dim(self, cfg, direction: str) -> int:
        """Width of the cross-attention memory for this config + direction.

        "aligned" conditions on the projection itself, so the memory is the
        shared embed_dim. "expanded" lifts it back to the SOURCE modality's
        encoder hidden size — text for text2protein, protein for protein2text.
        """
        if self.aligned_memory:
            return cfg.model.embed_dim
        return (cfg.model.text_hidden if direction == "text2protein"
                else cfg.model.protein_hidden)

    def validate(self) -> None:
        """Reject combinations the adapters cannot express, with the reason."""
        if self.memory_space not in ("expanded", "aligned"):
            raise ValueError(f"memory_space must be expanded|aligned, got {self.memory_space!r}")
        if self.cross_attn_mode not in ("head", "aligned"):
            raise ValueError(f"cross_attn_mode must be head|aligned, got {self.cross_attn_mode!r}")
        if self.unfreeze_where not in ("top", "bottom"):
            raise ValueError(f"unfreeze_where must be top|bottom, got {self.unfreeze_where!r}")
        # Aligned scoring cosine-matches decoder queries against the memory
        # vectors themselves, so the memory has to BE the aligned projection.
        if self.cross_attn_mode == "aligned" and not self.aligned_memory:
            raise ValueError("cross_attn_mode='aligned' requires memory_space='aligned'")
        if self.warm_start_qalign and self.cross_attn_mode != "aligned":
            raise ValueError("warm_start_qalign requires cross_attn_mode='aligned' "
                             "(q_align exists only in aligned mode)")

    @classmethod
    def from_ckpt(cls, ckpt: dict) -> "GenMeta":
        """Read the header out of a loaded checkpoint dict, filling pre-flag defaults."""
        d = cls()
        return cls(**{k: ckpt.get(k, getattr(d, k)) for k in asdict(d)})

    @classmethod
    def from_args(cls, args) -> "GenMeta":
        """Read the header off a parsed argparse namespace (fields share names)."""
        d = cls()
        return cls(**{k: getattr(args, k, getattr(d, k)) for k in asdict(d)})

    def to_payload(self) -> dict:
        """The keys to merge into a checkpoint payload when saving."""
        return asdict(self)

    def describe(self) -> str:
        return " ".join(f"{k}={v}" for k, v in asdict(self).items())

    def assert_matches(self, other: "GenMeta", context: str = "") -> None:
        """Raise if `other` differs, naming every field that disagrees.

        Used by `--resume`: loading adapter weights into a differently-shaped
        decoder either fails on a shape mismatch or, worse, quietly succeeds with
        the adapters in the wrong blocks.
        """
        diffs = [f"{k}: ckpt={v!r} vs args={getattr(other, k)!r}"
                 for k, v in asdict(self).items() if getattr(other, k) != v]
        if diffs:
            raise RuntimeError(
                f"{context or 'generation checkpoint'} architecture mismatch:\n  "
                + "\n  ".join(diffs))


def load_generation_ckpt(path: str, map_location="cpu") -> tuple[dict, GenMeta]:
    """Load a generation checkpoint -> (raw checkpoint dict, its GenMeta)."""
    import torch

    ckpt = torch.load(path, map_location=map_location)
    return ckpt, GenMeta.from_ckpt(ckpt)
