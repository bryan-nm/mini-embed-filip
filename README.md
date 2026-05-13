# mini-embed-filip

FILIP-style multimodal protein/text embedding model with optional cross-modal
generation in both directions. Frozen encoders (BioLinkBERT-base for text,
SaAMPLIFY-120M for protein) feed per-token projection heads into a shared
**32-d L2-normalized** space; per-token expansion heads project back up for
generation-side cross-attention. Only the projection and expansion heads
train during the retrieval phase; cross-attention adapters + LoRA train
during the per-direction generation phase.

Four trained things in total:

1. **Projection heads** (per-token, position-wise): encoder hidden → 32-d.
2. **Expansion heads** (per-token, symmetric to projections): 32-d → encoder hidden.
3. **Decoder cross-attention adapters** (text→protein direction: ProGen2-small;
   protein→text direction: BioGPT). Trained per direction, independently.
4. **Decoder LoRA on existing self-attn/FFN** (small, also per direction).

The 32-d shared space serves as a universal interlingua: retrieval scores
flow through it, generation conditioning flows through it. See
[PLAN_late_interaction.md](PLAN_late_interaction.md) for the architectural
rationale.

---

## Setup

```bash
conda env create -f environment.yml
conda activate mini-embed-filip
```

The env is pip-only inside conda to avoid the macOS `libomp` clash from the
conda/pip duplicate-runtime issue. On a CUDA host, swap the pip `torch>=2.4`
line for the matching pytorch.org wheel index; on Intel XPU, add
`intel-extension-for-pytorch` per Intel's instructions.

You also need four pretrained models on disk. Default paths in `config.py`:

| role | model | path |
|---|---|---|
| text encoder | BioLinkBERT-base | `/Users/bryan/Documents/models/BioLinkBERT-base` |
| protein encoder | SaAMPLIFY-120M | `/Users/bryan/Documents/models/SaAMPLIFY_120M` |
| protein decoder (text→protein) | ProGen2-small | `/Users/bryan/Documents/models/progen2-small` |
| text decoder (protein→text) | BioGPT | `/Users/bryan/Documents/models/biogpt` |

Data source (default in `DataCfg.csv_path`): the SwissProt-full CSV with
columns `primary_Accession`, `protein_sequence`, `[final]text_caption`,
`pfam_label`.

---

## End-to-end workflow

There are four scripts; run them in order. The first is one-shot, the next
two are training, the fourth is inference. Inspection can run at any point
after retrieval training.

```bash
# 1) One-shot per-token cache build (encoder forwards over the dataset).
python -m src.precompute --device cuda --batch-size 64

# 2) Retrieval (FILIP) training.
python -m src.train_retrieval --use-cache --device cuda

# 3a) Generation training: text → protein (ProGen2-small).
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt --device cuda

# 3b) Generation training: protein → text (BioGPT).
python -m src.train_generation --direction protein2text \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt --device cuda

# 4) Inference (either direction).
python -m src.generate --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epochNN.pt \
    --input "PROTEIN NAME: DNA helicase. FUNCTION: Unwinds DNA duplex..."
```

## Local smoke test (no cluster, no cache)

```bash
python -m src.precompute --subset-size 512 --batch-size 16 --device cpu
python -m src.train_retrieval --use-cache --batch-size 32 \
    --phase1-epochs 1 --phase2-epochs 2 --device cpu
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch02.pt \
    --subset-size 64 --batch-size 4 --epochs 1 --device cpu
```

512 pairs precompute in ~100s on Mac CPU. Retrieval epoch ~20s. Generation
epoch ~25s. The full pipeline finishes in under 5 minutes.

---

## CLI flags

### `src/precompute`

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `xpu` |
| `--cache-dir` | `cache/` | output directory for packed bf16 cache |
| `--batch-size` | `32` | encoder batch size |
| `--subset-size` | `0` | `0` = all CSV rows; `>0` = first N |
| `--max-text-tokens` | `1024` | truncation cap for BioLinkBERT |
| `--max-protein-tokens` | `1024` | truncation cap for SaAMPLIFY (incl. BOS/EOS) |
| `--no-mask-text-specials` | off | retain `[CLS]`/`[SEP]`/`[PAD]` in the cache |
| `--no-mask-protein-specials` | off | retain `<bos>`/`<eos>`/`<pad>` in the cache |

Output files: `protein_h.bin`, `protein_offsets.pt`, `protein_mask.bin`,
`text_h.bin`, `text_offsets.pt`, `text_mask.bin`, `pair_ids.json`,
`fingerprint.json`. The fingerprint records the encoder paths, length caps,
and special-token flags; a mismatch on rebuild aborts retrieval training
with a clear error instead of silently training against the wrong cache.

### `src/train_retrieval`

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | |
| `--use-cache` / `--no-cache` | `--use-cache` | live encoders run when `--no-cache` |
| `--cache-dir` | `cache/` | |
| `--ckpt-dir` | `checkpoints/retrieval/` | per-epoch checkpoints + `train_log.json` |
| `--batch-size` | `128` (cached) / `8` (live) | per-step batch |
| `--subset-size` | `0` | live mode only |
| `--phase1-epochs` | `1` | Phase R1 (alignment + uniformity + recon) |
| `--phase2-epochs` | `3` | Phase R2 (FILIP InfoNCE + align aux + recon) |
| `--lr` | `3e-4` | AdamW with cosine schedule + linear warmup |
| `--seed` | `0` | controls the split + init |

Config knobs in `config.py` (`RetrievalCfg`) not on the CLI:

- `phase1_uniformity_weight = 0.1` — Phase R1 within-modality spread.
- `align_aux_weight = 0.1` — Phase R2 positive-pair maintenance.
- `recon_weight = 0.05` — autoencoder loop weight throughout both phases.
- `init_temperature = 0.07` — learnable CLIP temperature, clamped to ≤ 100.

### `src/train_generation`

| flag | default | meaning |
|---|---|---|
| `--direction` | required | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | path to a `train_retrieval` checkpoint |
| `--device` | `auto` | |
| `--cache-dir` | `cache/` | per-token encoder cache used as cross-attn memory source |
| `--ckpt-dir` | `checkpoints/generation/<direction>/` | adapter-only checkpoints |
| `--batch-size` | `16` | tighter than retrieval; decoder forward dominates |
| `--epochs` | `3` | |
| `--lr` | `1e-4` | |
| `--cross-attn-every` | `2` | inject cross-attention into every Nth block |
| `--subset-size` | `0` | |
| `--seed` | `0` | reuses retrieval splits when the dataset size matches |

Config knobs in `GenerationCfg` not on the CLI: `lora_rank=16`,
`lora_alpha=32`, `lora_dropout=0.05`, `lora_targets_self_attn=True`,
`lora_targets_ffn=True`, `max_target_tokens=512`.

### `src/generate`

| flag | default | meaning |
|---|---|---|
| `--direction` | required | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | |
| `--decoder-ckpt` | required | trained adapter checkpoint from generation training |
| `--input` | required | the prompt (text or amino-acid sequence) |
| `--max-new-tokens` | `256` | |
| `--temperature` | `1.0` | sampling temperature |
| `--top-p` | `0.9` | nucleus sampling |
| `--device` | `auto` | |

### `src/inspect`

Returns the token×token similarity matrix for one pair, plus top-k
alignments. Used both as a per-pair interpretability tool and as the
fundamental measurement for dimensionality-sweep studies.

| flag | default | meaning |
|---|---|---|
| `--ckpt` | required | retrieval checkpoint |
| `--cache-dir` | `cache/` | |
| `--pair-id` | none | UniProt accession; looked up in `pair_ids.json` |
| `--pair-idx` | none | direct row index into the cache |
| `--top-k` | `5` | top-k text matches reported per protein position |
| `--plot` | none | path to save heatmap PNG (requires matplotlib) |

`compute_similarity_matrix_live` is also available as a Python API for pairs
not yet in the cache.

---

## Reading the retrieval per-step log

```
[R1-warm] epoch=0 step=1/14 lr=1.50e-04 loss=0.4972 align=0.6182
          unif=-1.8686 recon=1.3177 nce=0.0000 acc@1=0.000
          filip_pos=0.382 tau=0.0700
```

| field | what it is |
|---|---|
| `[R1-warm]` / `[R2-NCE]` | which retrieval phase is active |
| `loss` | total optimized scalar |
| `align` | `1 − FILIP_score(positive_pair)`; ↓ as paired tokens align |
| `unif` | within-modality token-uniformity (averaged across modalities) |
| `recon` | per-token MSE of `expand(project(h))` vs `h` |
| `nce` | FILIP-based symmetric InfoNCE; `0` in Phase R1 |
| `acc@1` | in-batch top-1 retrieval accuracy under InfoNCE; `0` in Phase R1 |
| `filip_pos` | mean FILIP score on positive pairs; ↑ from random (~0.1) toward 1 |
| `tau` | contrastive temperature `1/exp(logit_scale)`; learnable in Phase R2 |

## Reading the retrieval per-epoch val output

```
[val] epoch=2 {'R@1': 0.0962, 'R@5': 0.4231, 'R@10': 0.7692,
               'gap_l2': 0.5994, 'mean_cross_token_cos': 0.3181,
               'uniformity_p_tokens': -0.9216}
```

Full metric dict written to `checkpoints/retrieval/train_log.json`. The
shorter on-screen view shows:

| metric | meaning | what "good" looks like |
|---|---|---|
| `R@1`, `R@5`, `R@10` | symmetric retrieval recall over the val split | climbs from random; R@10 above ~0.5 indicates real alignment |
| `gap_l2` | distance between mean text-token and mean protein-token in 32-d | falls during Phase R1, may rebound in Phase R2 |
| `mean_cross_token_cos` | average cosine between random protein-token and random text-token | should NOT be close to 1 |
| `uniformity_p_tokens` | per-modality Wang-Isola spread; lower (more negative) = better-spread | approaches `-3.7` for fully-spread 32-d on the sphere |

Also in the full dict: `mean_intra_p_token_cos`, `mean_intra_t_token_cos`
(within-modality token cosines; warn of single-modality collapse),
`uniformity_t_tokens`, and per-direction R@K splits.

## Reading the generation per-step log

```
[text2protein] epoch=0 step=1/14 lr=1.00e-04 ce=2.8933 ppl=18.05
```

| field | meaning |
|---|---|
| `ce` | cross-entropy on the target sequence, teacher-forced |
| `ppl` | `exp(ce)`; perplexity. Random baseline = vocab size (32 for ProGen2, 42384 for BioGPT) |

The decoder cross-attention adapters and LoRA layers are zero-initialized,
so the first forward equals the pretrained decoder's unconditional prior.
`ce` at step 0 should already be below random; learning is visible as it
drops further over the first epoch.

---

## Sanity-check patterns

**Retrieval Phase R1 healthy.** `filip_pos` climbs from ~0.1 toward 1.0.
`align` decreases. `unif` becomes more negative. `recon` decreases. R@10
on val starts climbing already in Phase R1 (the model is learning real
alignment, not contrastive separation).

**Retrieval Phase R1 collapsed.** `align` drops too fast (below 0.05 on
epoch 0), `mean_intra_p_token_cos` rises sharply toward 1, `unif` rises
back toward 0. Same failure mode as the pooled `mini-embed`; raise
`phase1_uniformity_weight` from 0.1 toward 0.3.

**Retrieval Phase R2 modality-gap rebound.** `gap_l2` jumps when Phase R2
begins; small bump is expected, monotonic growth past ~1.0 is not. Raise
`align_aux_weight` from 0.1 toward 0.3.

**Generation healthy.** `ce` decreases steadily. With ProGen2 (vocab 32),
ce should reach <2.0 to be meaningfully better than the unconditional
prior on a generic protein. With BioGPT (vocab 42384), the floor is much
higher in absolute terms but the relative drop matters more than the
absolute value.

**Generation decoder ignores conditioning.** `ce` plateaus near the
pretrained model's unconditional perplexity, and outputs from different
`z_t` look interchangeable. Cross-attention is not being used. Try
`--cross-attn-every 1` (more injection points), then check the auxiliary
facilitator-style ideas in the design doc.

---

## Common issues

**`Cache fingerprint mismatch at cache/`.** The cache was built with
different encoder paths, length caps, or special-mask flags than what
`config.py` currently says. Rebuild with `python -m src.precompute`.

**`train_loader has 0 batches`.** Train split is smaller than
`--batch-size`. Either reduce `--batch-size` or grow the dataset
(probably remove a `--subset-size`).

**`Cannot copy out of meta tensor` on first decoder forward.** The custom
ProGen2 code stores `scale_attn` (and possibly others, depending on the
exact ProGen2 variant) as a plain attribute, which transformers ≥5 leaves
on the meta device after load. The fix in `decoder_adapters.py` covers
`scale_attn`; if you swap to a different ProGen2 size, run the
meta-tensor walk:
```python
def walk(mod, prefix=''):
    for name, child in mod.named_children():
        walk(child, prefix + name + '.')
    for k, v in mod.__dict__.items():
        if isinstance(v, torch.Tensor) and v.device.type == 'meta':
            print(f'meta attr: {prefix}{k}')
```
and add re-materialization for whatever it surfaces.

**`Asking to pad but the tokenizer does not have a padding token`.** Only
hits if the decoder tokenizer ships without one. The generation collator
sets `pad_token = eos_token` for ProGen2 already; if you swap decoders,
verify the collator still handles the new tokenizer.

**`You need to install sacremoses to use BioGptTokenizer`.** BioGPT's
tokenizer needs `sacremoses`; it's in `environment.yml` already. If you
hit this anyway, `pip install sacremoses` into the active env.

**`xformers` import errors during AMPLIFY load.** Expected on Mac / CPU /
non-CUDA hosts. `install_xformers_stub_if_missing()` in
`src/encoders.py` patches a weight-compatible stub before the encoder
loads, so xformers is only used when actually available on CUDA.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized.`** macOS conda/pip OpenMP collision. `environment.yml`
ships pip-only inside conda specifically to avoid this; if you've added
conda packages that pull in `llvm-openmp` (numpy, scipy, blas, etc.),
move them to pip or set `KMP_DUPLICATE_LIB_OK=TRUE` as a workaround.

---

## Storage and runtime budgets

Per-token packed bf16 cache, full SwissProt scale (574k pairs, length cap
1024):

| modality | tokens × dim × 2 bytes | size |
|---|---|---|
| protein (avg ~280 valid tokens × 640) | | ~190 GB |
| text (avg ~200 valid tokens × 768) | | ~166 GB |

Local smoke test (512 pairs): ~0.28 GB total.

Precompute time, Mac CPU: ~5 pairs/s; single Aurora GPU: a few hundred
pairs/s. Full SwissProt precompute on a single 6-GPU Aurora node is a
~4–6 hour one-time cost.

Trainable parameter counts (default config):

| stage | params |
|---|---|
| retrieval (projection + expansion heads + temperature) | ~2.0M |
| generation text→protein (ProGen2 cross-attn @ every-2 + LoRA) | ~25M |
| generation protein→text (BioGPT cross-attn @ every-2 + LoRA) | ~48M |
| **total across all three trainings** | **~75M** |

See [PLAN_late_interaction.md](PLAN_late_interaction.md) for the full
design rationale, the rejected alternatives (Q-former, soft prefixes,
unguided alignment warmup), and the open implementation decisions.
