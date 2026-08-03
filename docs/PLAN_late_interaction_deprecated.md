# Late-Interaction Architecture: Retrieval + Generation

Forward-looking plan for the next version of `mini-embed`. Two changes to the
current shipped code, designed so they share a single representation pipeline:

1. **Retrieval** switches from pooled cosine to FILIP-style per-token late
   interaction.
2. **Generation** uses a per-token expansion head feeding decoder cross-
   attention, replacing the facilitator + soft-prefix design from the earlier
   plan.

The two phases share four trained components — per-token projection heads,
per-token expansion heads, a learnable temperature, and an auxiliary
reconstruction objective that ties them together. The 32-d space remains the
universal interlingua.

---

## 1. Architecture overview

```
                ┌──────────────────── shared, trained in Retrieval phase ────────────────────┐

text  ─▶ BioLinkBERT  ─▶ [L_t, 768] ─▶ text_proj (per-token)    ─▶ [L_t, 32] ─┐
        (frozen)                                                              │
                                                                              ├─▶ FILIP retrieval loss
                                                                              │   (token × token max-sim)
protein ▶ SaAMPLIFY   ─▶ [L_p, 640] ─▶ protein_proj (per-token) ─▶ [L_p, 32] ─┘
        (frozen)

                ▲                                                       ▲
                │     reconstruction MSE (per-token, both modalities)   │
                │                                                       │
                └──────────[L_t, 768] ◀── text_expand    ◀──[L_t, 32] ──┘
                └──────────[L_p, 640] ◀── protein_expand ◀──[L_p, 32] ──┘

                ┌─────────────── trained in Generation phase, per direction ────────────────┐

text input ─▶ text_proj ─▶ [L_t, 32] ─▶ text_expand   ─▶ [L_t, 768] ─▶ cross-attn memory
                                                                          │
                                                                          ▼
                                                                  ProGen2 decoder
                                                                  (frozen + cross-attn
                                                                   + LoRA)  ─▶ protein

protein in ─▶ protein_proj ─▶ [L_p, 32] ─▶ protein_expand ─▶ [L_p, 640] ─▶ cross-attn memory
                                                                          │
                                                                          ▼
                                                                   BioGPT decoder
                                                                   (frozen + cross-attn
                                                                   + LoRA)  ─▶ text
```

Three things to notice:

- The **projection + expansion pair functions as an autoencoder** with the
  32-d shared space as its bottleneck code. Retrieval loss aligns the code
  across modalities; reconstruction loss keeps the code informative enough
  to recover per-token encoder state.
- **Same projection head is used per-token in retrieval and as the
  generation-time encoder front-end.** The encoder-side path is identical
  in both phases.
- **Encoders remain frozen everywhere.** The only trainable mass is the
  projection heads, expansion heads, decoder cross-attention, and decoder
  LoRA.

---

## 2. Component definitions

### 2a. Projection heads (per-token)

Slightly deeper than the current shipped heads. Position-wise — no token
mixing inside the projection (that's the encoder's job).

```python
class ProjectionHead(nn.Module):
    """d_in → d_hidden → d_mid → d_out, with LayerNorm + L2-normalized output."""
    def __init__(self, d_in, d_hidden=512, d_mid=256, d_out=32, dropout=0.1):
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_mid)
        self.norm2 = nn.LayerNorm(d_mid)
        self.fc3 = nn.Linear(d_mid, d_out)
        # forward: GELU activations, dropout after fc1, L2-norm on output
```

Param counts (d_out=32, d_hidden=512, d_mid=256):
- `text_proj`     (768 → 512 → 256 → 32): ~530K
- `protein_proj`  (640 → 512 → 256 → 32): ~465K

### 2b. Expansion heads (per-token, symmetric to projections)

Mirror architecture of the projection head, with the dimensions reversed.
Same depth, same widths in reverse order, same activation/norm pattern.

```python
class ExpansionHead(nn.Module):
    """d_in → d_mid → d_hidden → d_out, mirrored layout. No L2-norm on output."""
    def __init__(self, d_in=32, d_mid=256, d_hidden=512, d_out=None, dropout=0.1):
        self.fc1 = nn.Linear(d_in, d_mid)
        self.norm1 = nn.LayerNorm(d_mid)
        self.fc2 = nn.Linear(d_mid, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
```

Output dimension matches the corresponding encoder's hidden dim (768 for
text, 640 for protein), so the expansion has a chance to actually
reconstruct what the encoder produced. Param counts:
- `text_expand`    (32 → 256 → 512 → 768): ~530K
- `protein_expand` (32 → 256 → 512 → 640): ~465K

**Symmetry rationale.** Mirroring the projection makes
`expand(project(h)) ≈ h` a natural objective and gives the autoencoder
loop a coherent shape. Tying weights (literally reusing transposed
matrices) is tempting but too restrictive — it constrains the projection
to be exactly invertible, which doesn't allow the projection to drop
information that's useless for retrieval (which we want it to do). So
**symmetric architecture, separate weights**.

### 2c. Decoder cross-attention adapters

Two decoders, one per generation direction. Use pretrained backbones with
new cross-attention layers inserted into a subset of blocks.

For each block selected for cross-attention insertion, the modified block
becomes:

```
   x ─▶ frozen LayerNorm + SelfAttn ─▶ residual
     ─▶ NEW   LayerNorm + CrossAttn(to encoder memory) ─▶ residual
     ─▶ frozen LayerNorm + FFN ─▶ residual
```

The cross-attention layer is full-rank (Q, K, V, O projections).
**Insertion pattern: every other block** — gets us local correspondence at
every other layer without the full param cost of inserting at every layer.

Per-cross-attention-layer params at hidden=1024: 4 × (1024 × 1024) = 4.2M.

Decoders to use:
- **Protein direction**: ProGen2-small (151M, 12 layers, hidden=1024) →
  6 cross-attention layers ≈ **25M**.
- **Text direction**: BioGPT (347M, 24 layers, hidden=1024) →
  12 cross-attention layers ≈ **50M**.

### 2d. Decoder LoRA (optional, on existing layers)

LoRA-rank-16 on Q/K/V/O of self-attention and on FFN of every decoder
block. Lets the frozen-but-pretrained decoder adjust slightly to the new
cross-attention output it's now reading.

- ProGen2 self-attn LoRA (12 layers × 4 mats × ~32K) ≈ 1.6M
- ProGen2 FFN LoRA       (12 layers × ~160K)         ≈ 2.0M
- BioGPT self-attn LoRA  (24 layers × 4 mats × ~32K) ≈ 3.1M
- BioGPT FFN LoRA        (24 layers × ~160K)         ≈ 3.9M

### 2e. Param budget summary

| component | params | trained when |
|---|---|---|
| `text_proj` + `protein_proj` | ~1.0M | Retrieval phase |
| `text_expand` + `protein_expand` | ~1.0M | Retrieval phase (recon loss) |
| `logit_scale` | 1 | Retrieval phase |
| ProGen2 cross-attn (6 layers) | ~25M | Generation phase (text→protein) |
| ProGen2 self-attn + FFN LoRA | ~3.6M | Generation phase (text→protein) |
| BioGPT cross-attn (12 layers) | ~50M | Generation phase (protein→text) |
| BioGPT self-attn + FFN LoRA | ~7.0M | Generation phase (protein→text) |
| **Total trainable mass** | **~87M** | |

Under the 100M target. Headroom available: bump cross-attention to every
block instead of every-other (→ ~170M, still under 250M), or step up to
ProGen2-medium / BioGPT-large if quality demands it.

---

## 3. Retrieval phase

### 3a. Inputs and per-batch flow

For a batch of B pairs:

1. Tokenize text and protein, build masks excluding `[PAD]`, `[CLS]`,
   `[SEP]`, BOS, EOS positions from the FILIP comparisons (they're
   semantically empty and would absorb spurious alignment weight).
2. Run frozen BioLinkBERT and SaAMPLIFY live (no per-token cache — see
   §3d).
3. Apply per-token projection: `z_p ∈ [B, L_p, 32]`, `z_t ∈ [B, L_t, 32]`,
   L2-normalized along the last dim.
4. Apply per-token expansion: `h_p_hat ∈ [B, L_p, 640]`,
   `h_t_hat ∈ [B, L_t, 768]`. (Used for the reconstruction loss only,
   not retrieval.)

### 3b. FILIP score

For each pair `(p, t)` in the batch:

```
S = z_p[p] @ z_t[t].T          # [L_p, L_t]
Apply masks: invalid positions → -inf
score_p2t = mean_i ( max_j S[i, j] )    # mean over valid protein positions
score_t2p = mean_j ( max_i S[i, j] )    # mean over valid text tokens
filip(p, t) = 0.5 * (score_p2t + score_t2p)
```

Computed for all B² pairs in the batch. With B=64 and L=256, the all-pairs
`[B, B, L_p, L_t]` tensor is ~1 GB at fp32 — fits but tight. Chunked
implementation iterates over the protein axis: ~16 MB at a time.

### 3c. Loss design (mirroring current two-phase structure)

**Phase R1 (warmup, no negatives):**

```
L_R1 = -filip(p, t)                                # maximize positive-pair FILIP score
      + λ_unif * (token_uniformity(z_p) +          # within-modality token spread
                  token_uniformity(z_t)) / 2
      + λ_recon * ( MSE(h_p_hat, h_p) +            # per-token autoencoder loss
                    MSE(h_t_hat, h_t) ) / 2
```

`token_uniformity` is the same Wang & Isola log-mean-exp formula we already
ship, but applied across **all valid tokens in the batch within a modality**
(not just the batch's pooled tokens). With B=64 and L=256, that's ~16K
tokens per modality — plenty of spread signal.

Reconstruction is mean of two MSEs because protein and text have different
encoder hidden dims; we don't want one to dominate.

**Phase R2 (contrastive):**

```
L_R2 = SymInfoNCE( filip_scores * exp(logit_scale) )    # B×B FILIP matrix
      + λ_align * (-filip(p, t))                        # small positive-pair maintenance
      + λ_recon * recon as above
```

The FILIP score matrix replaces the pooled cosine matrix in InfoNCE — same
symmetric cross-entropy, same learnable temperature, same Pfam-masking trick
(when Pfam labels are available) for false-negative removal.

Default weights to start: `λ_unif = 0.1` (matches current default),
`λ_recon = 0.05`, `λ_align = 0.1`. All three are tunable.

### 3d. Caching strategy

The current shipped code caches **pooled** encoder outputs (one vector per
pair per modality). FILIP needs **per-token** outputs, which is much larger
but still feasible on the target hardware (Intel Aurora-class node, ~2 TB
of HBM + DDR5).

**Size estimates (SwissProt scale, 540K pairs, truncation cap 512, bf16):**

| format | protein | text | total |
|---|---|---|---|
| padded | ~330 GB | ~395 GB | **~725 GB** |
| packed (variable-length + offsets) | ~190 GB | ~166 GB | **~360 GB** |

Both fit comfortably; **packed is the recommended default** — the storage
saving is real and the unpacking cost at batch time is a `gather` plus
re-padding, which is negligible against projection-head forward.

**Cache contents** (after one precompute pass through the frozen
encoders):

| file | shape (packed) | dtype |
|---|---|---|
| `cache/protein_h.bin` | flat `[Σ L_p_i × 640]` | bf16 |
| `cache/protein_offsets.pt` | `[N+1]` | int64 |
| `cache/text_h.bin` | flat `[Σ L_t_i × 768]` | bf16 |
| `cache/text_offsets.pt` | `[N+1]` | int64 |
| `cache/pair_ids.json` | list of `N` UniProt IDs | json |
| `cache/splits.json` | as before, with `(n, seed, ratios)` fingerprint | json |

Mask tensors are reconstructed from offsets at load time; no need to store
them separately. We use `np.memmap` on the `.bin` files so we can leave the
data resident on disk for the local pilot and load eagerly into HBM on the
real run.

**Recommended workflow:**

1. **Full-scale training (default).** Precompute the per-token cache once
   on the target node (~4–6 hours on 6 GPUs in bf16). All subsequent
   retrieval training reads from cache; encoder forward is never run
   during the main loop. Projection-head + loss compute is the only cost.
2. **Local pilot.** `--subset-size 10000` builds a small cache (~7 GB
   packed bf16) that fits on a laptop. Same code path, less data. Useful
   for shape-and-loss-decreases sanity before requesting cluster time.
3. **No-cache fallback.** `--no-cache` runs encoders live during training
   for smoke-testing the pipeline on a host that doesn't have the cache
   built yet. Slow (3–5× the cached path) but useful as a one-step
   verification mode.

**Precompute time, full SwissProt on one Aurora node:** 540K pairs at
batch=64, bf16 forward through both encoders, parallelized across 6 GPUs:
roughly 4–6 hours one-time. The cost is dominated by SaAMPLIFY (24 layers,
640-d, no flash attention on Intel GPU — uses PyTorch SDPA path).

**Cache invalidation:** the cache is keyed on (encoder checkpoint hash,
tokenization config, truncation cap). Changing the truncation cap or
swapping the encoder backbone forces a rebuild. Worth fingerprinting in
the cache directory so an outdated cache doesn't silently get reused (the
same trap that hit us with `splits.json` earlier).

### 3e. Diagnostic: token × token similarity matrices

The thing that makes this architecture genuinely interpretable. Provide a
utility module `src/inspect.py` with:

```python
def compute_similarity_matrix(model, protein_seq, text, *, mask_specials=True):
    """Run encoders + projection, return everything needed to view the alignment."""
    # ...
    return {
        "S":          torch.Tensor,         # [L_p, L_t], cos similarity in [-1, 1]
        "mask_p":     torch.BoolTensor,     # [L_p]; True for "real" residue positions
        "mask_t":     torch.BoolTensor,     # [L_t]; True for "real" text tokens
        "tokens_p":   list[str],            # residue characters
        "tokens_t":   list[str],            # text wordpieces
        "z_p":        torch.Tensor,         # [L_p, D] projected, normalized
        "z_t":        torch.Tensor,         # [L_t, D] projected, normalized
    }

def plot_heatmap(out, out_path=None, k_top=None):
    """Heatmap of S with token labels on both axes. Optionally annotate top-k matches."""
```

And a CLI entry point `python -m src.inspect --pair-id P0C9F0` for one-off
visualization on a precomputed cache.

**Dimensionality sweep, as a research direction.** The "easy to retrieve
token × token matrices" requirement is exactly the right setup to study
information loss across embedding dims. Train a small grid:

| run | embed_dim |
|---|---|
| r-16  | 16  |
| r-32  | 32  (default; what we have) |
| r-64  | 64  |
| r-128 | 128 |
| r-256 | 256 |

For a fixed held-out set of pairs, compute `S` from each run. Metrics:

- **Pearson/Spearman correlation** of `S` across dim levels (per pair).
  Plateauing correlation between e.g. r-128 and r-256 suggests adding more
  dimensions stops adding interaction information.
- **Top-k match overlap.** For each protein token, take the top-k text
  tokens by similarity in r-256, see how many are still in the top-k in
  r-32. If overlap is ≥80% for the small-dim runs, you've validated that
  32-d preserves the "important" alignments.
- **Sharpness.** Entropy of the row-wise softmax of `S` — higher dim
  typically gives sharper, more concentrated alignments.
- **Reconstruction MSE on a held-out set** as a clean scalar summary of
  how much information the bottleneck loses at each dim.

These metrics give you principled "is 32 the right size?" evidence; the
heatmap utility gives you per-pair intuition for what specifically is
lost.

---

## 4. Generation phase

### 4a. Per-direction training, independent

Same flow for both directions; differs in which decoder + which encoder
front-end. After retrieval is trained:

1. **Freeze**: BioLinkBERT, SaAMPLIFY, both projection heads, both
   expansion heads.
2. Pass the relevant encoder's per-token output through projection +
   expansion → `h_hat ∈ [L, d_encoder_hidden]`. This is the cross-
   attention memory.
3. The decoder's modified blocks cross-attend to `h_hat`. Self-attention
   and FFN of those blocks remain frozen; the new cross-attention layers
   train from scratch; existing layers get LoRA adapters.
4. Cross-entropy loss on the target sequence, teacher-forced.

### 4b. Loss

```
L_G = L_CE(decoder_output, target_tokens)
```

That's it. The reconstruction loss isn't needed here because expansion
heads are frozen at this stage — they were trained during retrieval and
the decoder learns to work with whatever they produce. The `h_hat` is
fixed (given the input encoder pass) and is just a richer-than-32-d
conditioning memory that's already in the encoder's distribution.

Optional addition if generation quality is low: **unfreeze the expansion
heads and add a small reconstruction term** as joint fine-tuning. Skip
unless needed.

### 4c. Inference

Identical to training for the encoder side:
1. Tokenize input (text for text→protein, protein for protein→text).
2. Frozen encoder → frozen projection → frozen expansion = cross-attention
   memory.
3. Decoder generates autoregressively, attending to that memory at every
   modified block.

No facilitator. No pooled bottleneck used as a single soft prefix. The 32-d
shared space exists in the middle of the pipeline as a diagnostic and a
regularizing autoencoder code, but generation always reads the per-token
expanded representation, never the pooled one.

---

## 5. File-level changes from current `mini-embed`

| file | change |
|---|---|
| `config.py` | new `embed_dim`, `proj_hidden`, `proj_mid` knobs; remove `phase1_uniformity_weight` defaults that no longer apply to pooled (keep token-level version); add `recon_weight`, `expansion_hidden`; per-direction generation config block |
| `src/model.py` | `ProjectionHead` deepened to 3-layer; new `ExpansionHead`; new `MiniEmbed.forward` returns per-token z_p, z_t, h_p_hat, h_t_hat |
| `src/losses.py` | `filip_score(z_p, z_t, mask_p, mask_t)` returning [B, B] matrix; `token_uniformity` (extension of current `uniformity_loss`); rewrite `phase1_loss` / `info_nce_loss` to consume FILIP score matrix |
| `src/encoders.py` | encoders return per-token output + masks (no pooling); pooling helpers removed from this path |
| `src/data.py` | new `PackedPerTokenDataset` with memmap-backed reads from `protein_h.bin` / `text_h.bin` and on-the-fly mask reconstruction from offsets; split fingerprint unchanged; live-mode `RawPairsDataset` kept as a fallback |
| `src/precompute.py` | precomputes packed per-token raw encoder outputs (bf16, with offsets); writes encoder + tokenization fingerprint into the cache directory for staleness detection |
| `src/train.py` | cached-per-token path is the default; per-token mask handling; FILIP-based phase losses; reconstruction term added; `--no-cache` fallback path kept for smoke-testing without a cache |
| `src/evaluate.py` | retrieval R@K uses FILIP score in place of cosine; modality-gap metrics adapted to per-token form (centroid of tokens, not centroid of pools) |
| `src/inspect.py` | **new**: similarity-matrix utilities + CLI |
| `src/generate.py` | **new**: per-direction generation training (one entry point with `--direction text2protein` / `protein2text`); LoRA wiring; cross-attention adapter injection |
| `src/decoder_adapters.py` | **new**: utilities to insert cross-attention layers into a frozen pretrained decoder (ProGen2, BioGPT) and to LoRA-fy self-attn / FFN |

---

## 6. Verification plan

Each step should produce a small, fast smoke test that runs locally before
the full pilot:

1. **Per-token shapes.** Forward pass on 4 pairs, verify
   `z_p`, `z_t`, `h_p_hat`, `h_t_hat` have correct shapes and finite
   values; verify mask-out of special tokens.
2. **FILIP score correctness.** With identical pairs (z_t = z_p, identical
   masks), FILIP should be exactly 1.0; with orthogonal random pairs,
   score should be near zero. Sanity unit tests.
3. **Reconstruction loss sanity.** With expansion = identity-on-encoder-
   hidden-state (initialize with `expand(project(h)) ≈ h` for a random
   batch by training only the autoencoder loop briefly), confirm that
   MSE drops well below the initial random value.
4. **Phase R1 smoke (64 pairs, 1 epoch).** Verify all three loss
   components are decreasing; verify per-token uniformity is going
   negative.
5. **Phase R2 smoke (64 pairs, 1 epoch).** Verify FILIP-InfoNCE accuracy
   above chance; verify gap_l2 (computed over per-token centroids) is
   reasonable.
6. **Token × token heatmap.** On a real (protein, text) pair, generate
   the heatmap. Visually verify it's non-trivial (not uniform, not
   one-hot).
7. **Generation smoke per direction.** With expansion/projection frozen,
   12 cross-attention layers added, LoRA on, run a 32-sample teacher-
   forced epoch. Verify CE loss decreases.
8. **End-to-end inference per direction.** Given a real text prompt,
   generate a protein; verify it's a valid amino-acid sequence and that
   re-encoding it lands near the original z_t in 32-d space (the
   round-trip retrieval check — used as eval only, not training).

---

## 7. Open decisions for the implementation step

These are not blockers for the design, but worth flagging:

- **Batch size for cached-per-token retrieval training.** With the
  encoder forward removed from the loop, projection + FILIP-loss compute
  is light enough to scale batch size significantly. Need to measure
  GPU memory with B=128 vs B=256 vs B=512 at L=512 on a single Intel
  GPU before committing. Larger batches are valuable for contrastive
  learning (more negatives), so push as high as memory allows.
- **Whether to insert cross-attention in every decoder block or every
  other.** Defaulting to every other for the param-budget reason, but
  this can shift after measuring generation quality.
- **Whether to share the `logit_scale` between Phase R2's FILIP-InfoNCE
  and any auxiliary positive-pair term.** Defaulting to "yes" — one
  learnable temperature for all contrastive views.
- **Whether to L2-normalize the input to the expansion head** (i.e.,
  the projection output before expansion). Defaulting to "yes, same
  normalized vectors as retrieval uses" — keeps the pipeline coherent.
- **Special-token handling in the cross-attention memory.** During
  generation, cross-attention sees the expanded per-token output. Should
  `[CLS]` / BOS / EOS positions be masked out of cross-attention (the
  way they're masked out of FILIP)? Probably yes for `[CLS]` and `[SEP]`
  in the text encoder; debatable for protein BOS/EOS. Default: mask
  consistent with what FILIP masks.
