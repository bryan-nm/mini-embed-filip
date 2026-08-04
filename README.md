# mini-embed-filip

FILIP-style multimodal protein/text embedding model with optional cross-modal
generation in both directions. Frozen encoders (BioLinkBERT-base for text,
SaAMPLIFY-350M for protein) feed per-token projection heads into a shared
**L2-normalized** space whose width is `ModelCfg.embed_dim` (currently **16**);
per-token expansion heads project back up for generation-side cross-attention. Only the projection and expansion heads
train during the retrieval phase; cross-attention adapters train during the
per-direction generation phase.

Trained things:

1. **Projection heads** (per-token, position-wise): encoder hidden → `embed_dim`.
2. **Expansion heads** (per-token, symmetric to projections): `embed_dim` → encoder hidden.
   A small auxiliary reconstruction term trains them alongside the contrastive
   objective during retrieval.
3. **Decoder cross-attention adapters** (text→protein direction: ProtGPT3-112M-dpo,
   a Mixtral sparse MoE; protein→text direction: BioGPT). Trained
   per direction, independently. Injection is architecture-dispatched, so a
   direction can be re-pointed at a different decoder by changing its path.
   `--unfreeze-top N` optionally fine-tunes N decoder blocks on top of them.

Items 1–2 train in the retrieval phase; 3 in the per-direction generation phase.

The shared space serves as a universal interlingua: retrieval scores flow
through it, generation conditioning flows through it.

**Sweeping `embed_dim`.** `ModelCfg.embed_dim` in `config.py` is the only place
the width is written down — the heads, the aligned cross-attention memory, the
round-trip scorer and best-of-N all read it from there, and nothing defaults to a
literal. Changing that one line is the whole change; a sweep is one cache (the
encoder outputs do not depend on it) and one retrieval run per value.

Historical design notes live in [docs/](docs/), all suffixed `_deprecated`: they
record the reasoning behind decisions that were made, not the current state of
the code, and several describe features that have since been removed. This
README is the only document that tracks what the code does today.

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

You also need four pretrained models on disk. Default paths in `config.py` are
`$FILIP_MODELS_DIR/<name>`, with `FILIP_MODELS_DIR` defaulting to
`/flare/NLDesignProtein/bryan/FILIP-dev-space/models`:

| role | model | subdir |
|---|---|---|
| text encoder | BioLinkBERT-base | `BioLinkBERT-base` |
| protein encoder | SaAMPLIFY-350M | `SaAMPLIFY_350M` |
| protein decoder (text→protein) | ProtGPT3-112M-dpo (Mixtral MoE) | `ProtGPT3-112M-dpo` |
| text decoder (protein→text) | BioGPT | `biogpt` |

**`config.py` owns every path.** Job scripts do not export `FILIP_*` and do not
pass model or corpus locations on the command line — a path that lives in two
places eventually disagrees with itself, and the job log then records something
the job did not read. To see what a run will resolve to:

```bash
python config.py
```

It prints every model, corpus and cache path and flags anything missing. Every
job script banners that output into its `.o` file.

The `FILIP_*` environment variables remain as an escape hatch for a workstation
whose models live elsewhere (this is what the local smoke tests below use):
`FILIP_MODELS_DIR` swaps all four models at once, `FILIP_DATASETS_DIR` the
dataset root, and `FILIP_DATA_CSV` / the per-model vars override an individual
path.

(If you see `HFValidationError: Repo id must be in the form ...` for a `/Users/...`
path, it's a dev `config.py` on the cluster with no override set.)

One corpus, `DataCfg.csv_path`, is read by every entry point: the SwissProt-full
CSV with columns `primary_Accession`, `protein_sequence`, `[final]text_caption`,
`pfam_label`. The current build is **one caption per accession** and is filtered
to **proteins of 500 residues or fewer**, which is what lets
`DataCfg.max_protein_tokens` sit at 512 with nothing truncated.

The false-negative machinery (accession + caption group ids, group-aware splits,
protein dedup) is kept even though one caption per accession makes the accession
arm inert — it costs nothing, the caption arm still fires on byte-identical
captions across different proteins, and a multi-caption corpus would need it back.

---

## Job scripts

Every `*.pbs` in the repo root follows the same shape:

```bash
MASTER_PORT=29502
source ./pbs_common.sh     # module load, venv, cd, oneCCL/fabric env, MPI_LAUNCH

RETRIEVAL_CKPT=checkpoints/retrieval/epoch50.pt    # checkpoints, written here
BATCH_SIZE=32                                      # hyperparameters, written here
LR=1.5e-4

job_banner "text2protein SFT" RETRIEVAL_CKPT BATCH_SIZE LR   # the run record
require_files RETRIEVAL_CKPT                                 # fail before the queue slot burns

"${MPI_LAUNCH[@]}" python -m src.train_generation --lr "${LR}" ...
finish $?
```

Three conventions, all in service of the log being the record:

- **Checkpoints and hyperparameters are written into the script**, not passed
  with `qsub -v`. Editing the file is the interface; the file is then also what
  the git commit in the banner points at.
- **`job_banner`** prints job id, node/rank topology, git commit (flagged
  `UNCOMMITTED CHANGES` if dirty), every path `config.py` resolved, and the value
  of every named variable — so a `.o` file answers "how was this produced"
  without needing the script as it stood that day.
- **`require_files`** checks each named path exists before any model loads. An
  empty value is skipped, which is how an intentionally-unset optional checkpoint
  (cold-start RL) passes.

`pbs_common.sh` also exports the oneCCL/PMIx settings all the distributed jobs
need and builds `MPI_LAUNCH`, so the fabric configuration lives in one file
rather than eight.

---

## End-to-end workflow

Run the stages in order. Precompute is one-shot, retrieval and generation are
training, generate is inference. Round-trip eval and inspection can run at any
point after retrieval training. On the cluster, run the matching `*.pbs` instead
— they carry the checkpoints and hyperparameters and print the run record.

```bash
# 1) One-shot per-token cache build (encoder forwards over the dataset).
python -m src.precompute --device cuda --batch-size 64

# 2) Retrieval (FILIP) training.
python -m src.train_retrieval --use-cache --device cuda

# 2a) OPTIONAL: hard-negative mining. Builds decoy captions (one templated field
#     swapped in from a different caption), then resumes the retrieval checkpoint
#     with its own R2 objective plus a ramped decoy-discrimination loss. Writes a
#     retrieval-format checkpoint, so 3/4/5 consume it unchanged.
python -m src.precompute_decoys --device cuda --batch-size 64
python -m src.train_hard_negatives \
    --resume checkpoints/retrieval/epochNN.pt --device cuda

# 3a) Generation training: text → protein (ProtGPT3-112M-dpo by default).
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt --device cuda

# 3b) Generation training: protein → text (BioGPT).
python -m src.train_generation --direction protein2text \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt --device cuda

# 3c) OPTIONAL: RL fine-tune on the round-trip reward. --init-ckpt continues an
#      SFT policy; OMIT it to cold-start with no SFT at all (RL then shapes the
#      conditioning from scratch — see src/train_rl below).
python -m src.train_rl --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt \
    --init-ckpt checkpoints/generation/text2protein/epochNN.pt \
    --resume auto --steps 2000 --device cuda

# 4) Inference (either direction). --num-candidates>1 enables best-of-N with
#    contrastive round-trip selection.
python -m src.generate --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epochNN.pt \
    --input "PROTEIN NAME: DNA helicase. FUNCTION: Unwinds DNA duplex..." \
    --num-candidates 8 --selection margin

# 5) Offline round-trip eval: generate -> re-encode -> FILIP-retrieve the source.
#    (Both trainers already run this in-loop each epoch / every --eval-every.)
python -m src.roundtrip_eval --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epochNN.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epochNN.pt \
    --num-samples 1000 --split test --device cuda
```

## Local smoke test (no cluster, no cache)

```bash
python -m src.precompute --subset-size 512 --batch-size 16 --device cpu
python -m src.train_retrieval --use-cache --batch-size 32 \
    --phase1-epochs 1 --phase2-epochs 2 --device cpu

# Generation. The default text2protein decoder (ProtGPT3-112M-dpo) is small
# enough to smoke-test on CPU; FILIP_PROTEIN_DECODER points it elsewhere.
# --subset-size must match the cache so the by-accession split lines up.
# The per-epoch round-trip eval is on by default and loads the target encoder;
# shrink it here (or --no-rt-eval) so a CPU smoke test isn't dominated by decoding.
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch02.pt \
    --subset-size 512 --batch-size 4 --epochs 1 --device cpu \
    --rt-samples 16 --rt-max-new-tokens 64

# Cold-start RL: no SFT checkpoint at all. Two steps is enough to see the loop
# turn over (reward, advantage, KL=0 at step 0).
python -m src.train_rl --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch02.pt \
    --cross-attn-every 4 --cross-attn-mode aligned --warm-start-qalign \
    --steps 2 --prompts-per-rank 2 --group-size 2 --max-new-tokens 16 \
    --eval-samples 8 --eval-max-new-tokens 16 --device cpu

# Best-of-N inference (re-encodes candidates; loads the target encoder too).
FILIP_PROTEIN_DECODER=/path/to/progen2-small \
python -m src.generate --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch02.pt \
    --decoder-ckpt checkpoints/generation/text2protein/epoch00.pt \
    --input "Catalytic protein involved in metabolism." \
    --num-candidates 4 --selection margin --max-new-tokens 24 --device cpu
```

512 pairs precompute in ~100s on Mac CPU. Retrieval epoch ~20s. Generation
epoch with a small decoder is a minute or two on CPU.

---

## CLI flags

### `src/precompute`

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `xpu` |
| `--cache-dir` | `cache/` | output directory for packed bf16 cache |
| `--batch-size` | `32` | encoder batch size |
| `--subset-size` | `0` | `0` = all CSV rows; `>0` = first N |
| `--max-text-tokens` | `DataCfg.max_text_tokens` (512) | truncation cap for BioLinkBERT |
| `--max-protein-tokens` | `DataCfg.max_protein_tokens` (512) | truncation cap for SaAMPLIFY (incl. BOS/EOS) |
| `--no-mask-text-specials` | off | retain `[CLS]`/`[SEP]`/`[PAD]` in the cache |
| `--no-mask-protein-specials` | off | retain `<bos>`/`<eos>`/`<pad>` in the cache |

Output files: `protein_h.bin`, `protein_offsets.pt`, `protein_mask.bin`,
`protein_ids.json`, `text_h.bin`, `text_offsets.pt`, `text_mask.bin`,
`pair_ids.json`, `row_protein_idx.pt`, `fingerprint.json`. The fingerprint
records a format tag, the encoder paths, length caps, and special-token flags;
a mismatch on rebuild aborts retrieval training with a clear error instead of
silently training against the wrong cache.

**Protein dedup.** The protein modality is encoded + stored once per *unique*
protein (so `protein_*` has `N_unique` rows, `text_*` has `N_rows`), and
`row_protein_idx.pt` (`[N_rows]`, CSV row → unique-protein index) joins them at
read time. At one caption per accession the two counts are equal and this saves
nothing; it is kept because it costs one pass and is what makes a multi-caption
corpus affordable — the protein encoder is the precompute bottleneck. Splits are
by accession (no protein straddles train/val/test); the retrieval InfoNCE and
the val/round-trip recall treat every row of the same protein as a positive, so
a multi-caption corpus would not see its own captions as false negatives.

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
- `r2_uniformity_weight = 0.1` — Phase R2 token-spread regularizer (CLI: `--r2-uniformity-weight`). Counters per-token crowding under the contrastive objective, which the InfoNCE negatives alone don't prevent when captions share heavy boilerplate.
- `align_aux_weight = 0.1` — Phase R2 positive-pair maintenance.
- `recon_weight = 0.05` — autoencoder loop weight throughout both phases.
- `init_temperature = 0.07` — learnable CLIP temperature, clamped to ≤ 100.

### `src/precompute_decoys` and `src/train_hard_negatives`

Optional phase after retrieval. A **decoy** is a caption with exactly one
templated field (`DataCfg.caption_field_labels`) replaced by the corresponding
field of a *different* caption — so it shares nearly every token with the truth
and can only be ranked below it by reading the swapped field. In-batch negatives
are separable from overall topic; decoys are not, which is why they keep biting
after R@K looks good.

Captions do **not** all carry the same fields (SwissProt annotation is sparse),
so swap targets and donor pools are both per-label, and rows with no swappable
field simply contribute no decoys.

`src/precompute_decoys` runs in two steps: a single-process **plan** (parse every
caption, index the field spans, assign each row `--decoys-per-row` (field, donor)
swaps, deterministic given the seed) and a **distributed encode + merge** that
writes a packed per-token cache in the same format as `src/precompute`. It is
text-only — the protein side is reused from the base cache.

Donors are rejected when they carry an identical field value (the decoy would
equal the truth), come from the same protein (a sibling caption's field is often
also true of this protein), or come from a different split (held-out caption text
must not reach a training decoy — this needs the retrieval run's `splits.json`).
A decoy whose swapped field lands beyond the text encoder's truncation window
tokenizes identically to the truth; those are detected exactly at encode time and
excluded via `decoy_keep.pt`. Check `decoy_stats.json` (`rows_with_usable_decoy`
vs `rows_total`) before training.

| flag (`precompute_decoys`) | default | meaning |
|---|---|---|
| `--cache-dir` | `cache/` | base cache; supplies row order + fingerprint |
| `--decoy-cache-dir` | `cache/decoys/` | packed decoy cache written here |
| `--decoys-per-row` | `2` | decoys planned per caption |
| `--decoy-seed` | `1234` | plan RNG (separate from the split seed) |
| `--swap-fields` | all labels | restrict which fields may be swapped |
| `--splits` | `<cache-dir>/splits.json` | keeps donors inside the owner's split |
| `--allow-same-protein-donor` | off | permit sibling-caption donors |
| `--allow-cross-split-donors` | off | permit corpus-wide donors |
| `--max-swap-char` | auto | only swap fields starting before this offset |
| `--plan-only` / `--encode-only` / `--merge-only` | | run one phase |

The same-split donor rule needs the retrieval run's `splits.json`. If that file is
stale — most often a small `--subset-size` smoke run left behind in a directory
that was later re-precomputed at full scale — the plan step aborts rather than
guess. `splits.json` is *derived*, not source: it is a deterministic function of
`(pair_ids, text_group_ids, ratios, seed)`, so regenerating reproduces byte-identical
splits. `src/rebuild_splits` does that from the cache alone, no CSV and no trainer:

```bash
python -m src.rebuild_splits --cache-dir cache --check   # report only; exit 1 if stale
python -m src.rebuild_splits --cache-dir cache --seed 0  # rewrite (seed MUST match the retrieval run)
```

A stale split file is also a hint worth chasing: `train_retrieval` rebuilds and
re-saves a stale file at startup, so if it's stale, no retrieval run ever wrote
splits for that cache directory. Confirm the checkpoint you're resuming was
actually trained against that cache before building decoys against it — decoys are
indexed by base-cache row.

`src/train_hard_negatives` then resumes a retrieval checkpoint with **the same R2
objective it finished on** and adds `w(step) * L_decoy`, where `w` ramps linearly
from zero. Pass the same loss weights the retrieval run used (`--align-aux-weight`,
`--recon-weight`, `--r2-uniformity-weight`) — otherwise "resume with its existing
objective" silently isn't. The per-step log adds `w_dec` (current ramp weight),
`d_acc` (anchors already ranked above their hardest decoy), `d_margin` (FILIP-score
lead over it) and `d_rows` (anchors in this batch that had a decoy); the per-epoch
val line adds `val_decoy_acc` / `val_decoy_margin`, and a **baseline** val line is
printed before training so the ramp has a reference point. The goal is `d_acc` up
with `R@K` flat — if R@K sags as the ramp tops out, lower `--decoy-weight` or
switch to the bounded hinge (`--decoy-margin 0.02`).

| flag (`train_hard_negatives`) | default | meaning |
|---|---|---|
| `--resume` | required | retrieval checkpoint (path, or `auto`) |
| `--retrieval-ckpt-dir` | `checkpoints/retrieval/` | searched by `--resume auto` |
| `--decoy-cache-dir` | `cache/decoys/` | |
| `--ckpt-dir` | `checkpoints/hard_negatives/` | retrieval-format checkpoints |
| `--epochs` / `--lr` | `3` / `5e-5` | fresh cosine schedule at fine-tune LR |
| `--load-optimizer` | off | reuse the retrieval run's optimizer state |
| `--decoy-weight` | `0.5` | weight after the ramp |
| `--decoy-start-frac` | `0.05` | fraction of the run before the ramp starts |
| `--decoy-ramp-frac` | `0.35` | fraction spent ramping 0 → weight |
| `--decoy-margin` | `0.0` | `0` = (K+1)-way softmax; `>0` = hinge on the FILIP gap |
| `--max-decoys` | `2` | decoys scored per anchor per step |
| `--group-size` / `--filip-chunk-rows` | `16` / `0` | as in `train_retrieval` |

The decoy term is protein→text only (the reverse would need decoy *proteins*) and
per-anchor: a decoy built from caption *i* says nothing about protein *j*, so it
is scored only against its own protein.

### `src/train_generation`

| flag | default | meaning |
|---|---|---|
| `--direction` | required | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | path to a `train_retrieval` (or `train_hard_negatives`) checkpoint |
| `--device` | `auto` | |
| `--cache-dir` | `cache/` | per-token encoder cache used as cross-attn memory source |
| `--ckpt-dir` | `checkpoints/generation/<direction>/` | adapter-only checkpoints |
| `--batch-size` | `16` | tighter than retrieval; decoder forward dominates |
| `--epochs` | `3` | |
| `--lr` | `1e-4` | |
| `--cross-attn-every` | `2` | inject cross-attention into every Nth block |
| `--unfreeze-top` | `0` | fully fine-tune the top N decoder blocks (on top of the adapters) |
| `--subset-size` | `0` | must match the cache when <full |
| `--seed` | `0` | reuses retrieval splits when the dataset size matches |
| `--rt-eval` / `--no-rt-eval` | on | per-epoch round-trip retrieval eval (below) |
| `--rt-samples` | `256` | val rows in the fixed round-trip pool (0 ⇒ off) |
| `--rt-batch-size` | `8` | per-rank generation batch during the eval |
| `--rt-max-new-tokens` | `--max-target-tokens` | the eval's real cost knob |
| `--rt-temperature` / `--rt-top-p` | `1.0` / `0.9` | eval sampling (0 ⇒ greedy) |
| `--rt-seed` | `1234` | picks the pool; training RNG is restored around each eval |
| `--rt-at-start` / `--no-rt-at-start` | on | epoch=−1 baseline; also smoke-tests the eval |

**Per-epoch round-trip eval.** Val CE is teacher-forced, so it can fall while
free-running generation ignores the conditioning entirely — the confound
`src/train_rl.py` exists to fix. Each epoch the trainer therefore also
generates a *fixed* val pool free-running, re-encodes each output, and scores
accession-grouped R@K/mAP back to the sources, using the same
`score_roundtrip_records` scorer as `python -m src.roundtrip_eval` (so an in-loop
curve and an offline number agree on the same pool). Reported as `[gen][rt]`, with
the mean generated vs true body length, and appended to
`<ckpt-dir>/sft_roundtrip.jsonl` (also stored under `roundtrip` in `train_log`).

Cost is deliberately small: the pool is sharded round-robin across ranks, so it is
one *small* generate per rank, one encoder pass over those rows, and one N×N FILIP
on rank 0 — flat in world size until the pool exceeds `world_size × --rt-batch-size`.
The source features and the true targets (the ceiling row, computed once) come
straight from the packed cache, so the only new resident model is the **target**
encoder — AMPLIFY-350M for `text2protein`, BioLinkBERT for `protein2text` — loaded
up front so a bad config fails in the first minute rather than at the end of epoch 0.

### `src/generate`

| flag | default | meaning |
|---|---|---|
| `--direction` | required | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | |
| `--decoder-ckpt` | required | trained adapter checkpoint from generation training |
| `--input` | required | the prompt (text or amino-acid sequence) |
| `--max-new-tokens` | config cap for the generated modality | never truncates its own output by default |
| `--temperature` | `1.0` | sampling temperature |
| `--top-p` | `0.9` | nucleus sampling |
| `--num-candidates` | `1` | >1 enables best-of-N selection |
| `--selection` | `margin` | `margin` (pos − best panel) or `pos` |
| `--panel-size` | `256` | reference negatives sampled from `--panel-csv` |
| `--panel-csv` | `DATA_CSV` | source of the margin reference panel |
| `--device` | `auto` | |

With `--num-candidates > 1`, N candidates are sampled from the decoder,
re-encoded and ranked by round-trip score (the loaded target encoder closes the
loop). The ranked list is printed and the best candidate is the output.

### `src/train_rl`

GRPO-style RL fine-tuning that optimizes the round-trip FILIP reward directly —
the honest, prefix-free objective. Teacher-forced CE (and the teacher-forced
content-gain metric) can improve while free-running generation ignores the
conditioning entirely, because the true prefix is a confound; this entry point
exists to remove it. Separate from `train_generation`: it shares the frozen
building blocks and no training code.

**With or without SFT.** `--init-ckpt` is optional:

- **Warm start** (`--init-ckpt path/to/sft/epochNN.pt`) continues a supervised
  policy. The architecture comes from that checkpoint's stored header, so the
  `--cross-attn-*` / `--unfreeze-*` flags are ignored — a continued run cannot be
  silently reshaped. KL anchors to the SFT policy.
- **Cold start** (omit `--init-ckpt`) initializes the adapters fresh on the
  frozen decoder, making RL the only thing that ever shapes the conditioning. The
  architecture then comes from the flags, and KL anchors to the decoder's
  unconditional prior. This is legal, not degenerate: `o_proj` is
  zero-initialized, so step 0 *is* that prior and the reward gradient opens the
  adapter from there. Expect a much longer flat stretch before the round-trip
  curve moves. `--warm-start-qalign` matters most here — it is the only thing
  putting the decoder-side query in the right region of the shared space before
  any gradient arrives (text2protein only; it needs
  `dec_hidden == proj_d_hidden`, which BioGPT's 1024 does not satisfy).

Either way the reference copy is initialized from the policy's own starting
weights, so `kl` is exactly 0 at step 0. `--resume auto` continues the latest
checkpoint in `--ckpt-dir` and keeps the KL anchor fixed across restarts, so a
re-queued job after a wall-time kill picks up where it left off.

| flag | default | meaning |
|---|---|---|
| `--direction` | `text2protein` | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | retrieval checkpoint (frozen; supplies memory + reward) |
| `--init-ckpt` | none | SFT checkpoint to continue; omit to cold-start |
| `--resume` | none | `auto` picks the latest `stepNNNNNN.pt` in `--ckpt-dir` |
| `--steps` | `2000` | TOTAL update budget, not per-job |
| `--prompts-per-rank` / `--group-size` | `8` / `8` | B and G; rollout is B×G per rank |
| `--kl-coef` | `0.05` | KL-to-reference penalty (0 skips loading the reference) |
| `--length-lambda` / `--length-tolerance` | `0` / `0.25` | two-sided log-symmetric length band |
| `--reward-contrastive` / `--reward-margin-beta` | off / `1.0` | margin reward instead of the raw positive score |
| `--eval-every` / `--eval-samples` | `200` / `1000` | held-out round-trip metric → `rl_roundtrip.jsonl` |
| `--pppl-passes` | `0` (off) | AMPLIFY pseudo-perplexity of the generations; text2protein only |
| `--cross-attn-every`, `--cross-attn-mode`, `--unfreeze-top`, `--unfreeze-where`, `--warm-start-qalign` | | cold start only |

**Pseudo-perplexity (`--pppl-passes`).** The reward is a FILIP score in the
retrieval space, so a policy can climb it with sequences that embed well without
being proteins — and R@K cannot see that, because it is computed in the same
space the policy is gaming. PPPL is the second axis. AMPLIFY is a masked LM, so
the meaningful number masks each residue before scoring it; scoring without
masking is free (the head's logits are already computed) but nearly useless — a
real protein and a shuffle of its own amino acids come out at 1.40 vs 1.46,
against 1.0–1.2 vs 22–23 for the masked version.

Exact PPPL is one forward per residue. `--pppl-passes N` partitions positions by
`p % N` and masks one class per pass, so N forwards score every residue exactly
once under a real mask. At the default 8 that masks 12.5% at a time — the regime
these models are trained under — and measured within ~3% of exact PPPL at ~1/30
the cost.

Read `pppl_gen` against the `pppl_true` printed beside it (the fixed pool's real
proteins under the identical estimator), never against an absolute scale: AMPLIFY
has memorized well-known proteins and scores those near 1.0. The signature to
watch is `pppl_gen` falling toward `pppl_true` while R@K rises. R@K rising while
`pppl_gen` climbs away from true is reward hacking.

Enabled in `train_t2p_rl.pbs` and off everywhere else, including
`roundtrip_eval` (where the flag exists for spot checks) and the SFT trainers.

### `src/roundtrip_eval`

Generate → re-encode the generation → FILIP-retrieve the source, reporting R@K
against a ceiling (true output re-encoded → source). Distributed like precompute
(one rank per tile generates+encodes a shard; rank 0 merges and scores). With
`--num-candidates > 1` it does best-of-N per source before scoring; the
selection panel is drawn from the **train** split (disjoint from the scored set,
so selection doesn't inflate the reported metric).

| flag | default | meaning |
|---|---|---|
| `--direction` | `text2protein` | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | retrieval checkpoint |
| `--decoder-ckpt` | required | generation checkpoint (architecture flags read from it) |
| `--split` | `test` | `train` / `val` / `test` |
| `--num-samples` | `1000` | sources to evaluate (0 = whole split) |
| `--num-candidates` | `1` | best-of-N per source |
| `--selection` | `margin` | `margin` or `pos` |
| `--panel-size` | `256` | train rows used as the margin reference panel |
| `--temperature` / `--top-p` | `1.0` / `0.9` | sampling |
| `--max-new-tokens` | config cap for the generated modality | never truncates its own output by default |
| `--score-only` | off | re-score existing shards (single process) |
| `--device` | `auto` | |

Outputs (`eval/<direction>/`): `roundtrip_metrics.json` (R@K both directions +
ceiling + config, including `num_candidates`/`selection`), `roundtrip_pairs.tsv`,
and `roundtrip_sequences.fasta` (text2protein only).

### `src/inspect`

Returns the token×token similarity matrix for one pair, plus top-k
alignments. Used both as a per-pair interpretability tool and as the
fundamental measurement for dimensionality-sweep studies. Three input modes:

- **Cache** (`--pair-id` / `--pair-idx`): reads a pair out of the precomputed
  cache. Dedup-aware (maps the CSV row to its unique-protein row).
- **Live, explicit** (`--protein` + `--text`): encode one raw pair fresh.
- **Live, by accession** (`--protein-id`/`--id-file`): look the accession(s) up
  in the corpus (`DataCfg.csv_path`; `--csv PATH` overrides) and encode live.
  Encoders + model load once and are reused across all accessions; the CSV is
  streamed in a single pass. Useful
  with an *old* checkpoint when its cache no longer exists (the live path never
  touches the cache).

| flag | default | meaning |
|---|---|---|
| `--ckpt` | required | retrieval checkpoint |
| `--device` | `cpu` | `cpu` is fine for a handful of proteins |
| `--cache-dir` | `cache/` | cache mode only |
| `--pair-id` | none | accession; looked up in `pair_ids.json` (cache mode) |
| `--pair-idx` | none | direct CSV-row index into the cache |
| `--protein` / `--text` | none | raw sequence + caption for a single live pair |
| `--csv` | none | CSV to resolve `--protein-id` / `--id-file` accessions |
| `--protein-id` | none | one or more accessions to inspect live (needs `--csv`) |
| `--id-file` | none | file with one accession per line (needs `--csv`) |
| `--top-k` | `5` | top-k text matches reported per protein position |
| `--plot` | none | heatmap PNG path (single-pair modes) |
| `--plot-dir` | none | per-accession heatmaps (`<uid>.png`) + index→token TSVs (`<uid>_text_tokens.tsv`, `<uid>_protein_tokens.tsv`) |

`inspect.pbs` wraps the by-accession live mode for an Aurora batch job: it reads
`protein_ids.txt`, pulls sequence+caption from the corpus, and
writes a heatmap per accession. `compute_similarity_matrix_live` /
`load_inspect_encoders` are also available as a Python API.

### `src/export_embeddings`

Dumps the `embed_dim`-wide retrieval embeddings (`z_p`, `z_t`), keyed by primary accession,
for latent-space structure work (clustering, UMAP, modality gap) that training
never emitted. Two sources, with **different parallelism**:

- **Cached** (default): projects the precomputed cache for whole train/val/test
  splits. Just the projection head over cached hidden states — tiny compute,
  I/O-bound — so it runs **single process** (plain `python`, not mpiexec).
  ```bash
  python -m src.export_embeddings --ckpt checkpoints/retrieval/epochNN.pt \
      --splits train,test --device xpu
  ```
- **Live** (`--live`): encodes every row of a corpus (no cache, no splits — all
  sequences). This runs the frozen **encoders**, the expensive part, so it is
  **distributed** — launch under `mpiexec` (`export_embeddings.pbs`) and it
  shards proteins/rows across tiles like precompute (one process falls back
  cleanly). Proteins are deduped (encoded once per unique accession).
  The corpus is `DataCfg.csv_path`; `--csv PATH` overrides it and implies
  `--live`.
  ```bash
  mpiexec -n <12*nodes> -ppn 12 python -m src.export_embeddings --live \
      --ckpt checkpoints/retrieval/epochNN.pt --name swissprot --device xpu
  ```

| flag | default | meaning |
|---|---|---|
| `--ckpt` | required | retrieval checkpoint |
| `--cache-dir` | `cache/` | cached mode: per-token cache to project |
| `--csv` | none | live mode: encode this CSV (no cache/splits); distributed under mpiexec |
| `--out-dir` | `embeddings/` | output directory |
| `--name` | `live` | live mode: output basename (`<name>_pooled.npz`) |
| `--splits` | `train,test` | cached mode: comma list of `train`/`val`/`test`, or `all` |
| `--modalities` | `protein,text` | which to export |
| `--pooling` | `mean` | `mean` = one vector per item; `none` = full per-token packed |
| `--renormalize` | off | L2-normalize pooled vectors |
| `--batch-size` | `512` | rows per batch |
| `--subset-size` | `0` | live mode: first N CSV rows |
| `--device` | `auto` | `cpu`/`xpu`/`cuda` |

`--pooling mean` writes a `_pooled.npz` (`<split>_pooled.npz` cached, or
`<name>_pooled.npz` live) with `z_p [Np,D]`+`acc_p`, `z_t [Nt,D]`+`acc_t`+`row_t`
— proteins deduped (one per unique accession), captions per CSV row. `--pooling
none` streams `<...>_<mod>_z.f32.bin` + `_offsets.npy` + `_accessions.json`
(packed like the cache; text can be tens of GB).

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
| `gap_l2` | distance between mean text-token and mean protein-token in the shared space | falls during Phase R1, may rebound in Phase R2 |
| `mean_cross_token_cos` | average cosine between random protein-token and random text-token (background, across non-matching pairs) | should NOT be close to 1 |
| `mean_pos_token_cos` | average cosine between protein-token and text-token within a *correct* pair | should sit clearly above `mean_cross_token_cos`; the gap between them is the per-token alignment signal |
| `uniformity_p_tokens` | per-modality Wang-Isola spread; lower (more negative) = better-spread | floor depends on `embed_dim`; compare across a sweep, not against a fixed number |

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
| `ppl` | `exp(ce)`; perplexity. Random baseline = vocab size (31 for ProtGPT3's char vocab, ≈32 for ProGen2, 42384 for BioGPT) |

The decoder cross-attention adapters are zero-initialized (`o_proj`), so the
first forward equals the pretrained decoder's unconditional prior. `ce` at step 0
should already be below random; learning is visible as it drops further over the
first epoch.

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

**ProtGPT3 (Mixtral) decoder notes.** `ProtGPT3-112M-dpo` is the default
text→protein decoder: an 8-layer Mixtral sparse MoE (hidden 512, 8 experts,
top-2 routing) with a char-level `WordLevel` vocab of 31. Three quirks the code
handles for you, all in `src/decoder_adapters.py`:

- **Direction marker.** ProtGPT3 was pretrained with a control token immediately
  after BOS — `"1"` for N-to-C, `"2"` for C-to-N. It is an *ordinary* vocab
  token, not a special token, so nothing adds it automatically. `target_prefix_ids`
  supplies it, and every target is built as `[BOS] "1" …residues… [EOS]`; the
  same prompt seeds `generate()` in both `src/generate.py` and
  `src/roundtrip_eval.py`. Omit it and the decoder is off-distribution from
  token 0.
- **Decoding.** The `WordLevel` tokenizer decodes to *space-separated* residues
  (`"1 M K T"`), and `skip_special_tokens` doesn't drop the marker. `decode_target`
  reverses both, so candidates come back as bare sequences that feed straight
  back into the protein encoder for round-trip scoring.
- **Vocab mismatch.** The tokenizer carries three added tokens at ids 31–33
  (`<s>`, `</s>`, `<unk>`) that sit *outside* the model's 31-row embedding table,
  so `tokenizer.unk_token_id` is an out-of-range id. Encoding never emits them
  (the WordLevel fallback is `[UNK]` = 3), but don't call
  `resize_token_embeddings(len(tokenizer))` and don't feed `unk_token_id` to the
  model.

Adapters are injected via forward hooks (the layer returns a bare hidden-state
tensor), which also keeps them inside the layer `__call__` so they recompute
correctly under gradient checkpointing. The MoE feed-forward and router stay
frozen. At `--cross-attn-every 1` on all 8 layers the trainable set is ~11M
params.

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

Per-token packed bf16 cache. Both modalities are stored once per unique item
(proteins by accession, text by CSV row), so at one caption per accession the two
row counts are equal:

| modality | per row | 100k rows |
|---|---|---|
| protein (avg ~250 valid tokens × 960 × 2 B) | ~0.48 MB | ~48 GB |
| text (avg ~200 valid tokens × 768 × 2 B) | ~0.31 MB | ~31 GB |

Multiply by the actual row count — `wc -l` the CSV. **Neither number depends on
`embed_dim`**: the cache holds encoder hidden states, so one cache serves an
entire sweep.

For reference, the previous corpus (574k rows at 8.87 captions per accession,
1024-token protein cap) came to ~285 GB protein + ~166 GB text.

Local smoke test (512 pairs): ~0.28 GB total.

Precompute time, Mac CPU: ~5 pairs/s; single Aurora GPU: a few hundred pairs/s.
The 512-token protein cap makes each protein ~4x cheaper to encode than at 1024
(attention is O(L²)).

**Wall times** in the `*.pbs` files each carry a `WALL TIME BASIS` comment
stating what they were derived from. All of them are ceilings, and the term I
could not compute is the new corpus's row count — read the per-step / per-epoch
timing off the first job of each kind and re-trim.

Trainable parameter counts:

| stage | params |
|---|---|
| retrieval (projection + expansion heads + temperature) | ~2.0M |
| generation text→protein (ProtGPT3 cross-attn @ every-1) | ~11M |
| generation protein→text (BioGPT cross-attn @ every-2) | ~48M |

The text→protein figure is for the shipped default decoder, ProtGPT3-112M-dpo,
with an adapter on every one of its 8 layers; adapter cost scales with the
decoder's hidden size and layer count, so a larger decoder (ProGen2-class ≈25M
@ every-2) moves it.

See [docs/PLAN_late_interaction_deprecated.md](docs/PLAN_late_interaction_deprecated.md)
for the original design rationale and the rejected alternatives (Q-former, soft
prefixes, unguided alignment warmup).
