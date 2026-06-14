# mini-embed-filip

FILIP-style multimodal protein/text embedding model with optional cross-modal
generation in both directions. Frozen encoders (BioLinkBERT-base for text,
SaAMPLIFY-120M for protein) feed per-token projection heads into a shared
**32-d L2-normalized** space; per-token expansion heads project back up for
generation-side cross-attention. Only the projection and expansion heads
train during the retrieval phase; cross-attention adapters + LoRA train
during the per-direction generation phase.

Trained things:

1. **Projection heads** (per-token, position-wise): encoder hidden → 32-d.
2. **Expansion heads** (per-token, symmetric to projections): 32-d → encoder hidden.
   A small auxiliary reconstruction term trains them during retrieval; an optional
   **expansion-only reconstruction phase** then sharpens them further at zero
   retrieval cost (see `src/train_reconstruction`).
3. **Decoder cross-attention adapters** (text→protein direction: Dayhoff-3b-UR90,
   a Jamba hybrid Mamba/attention MoE; protein→text direction: BioGPT). Trained
   per direction, independently. Injection is architecture-dispatched, so a
   direction can be re-pointed at a different decoder by changing its path.
4. **Decoder LoRA on existing self-attn/FFN** (small, also per direction).
5. **CVAE heads** (optional, per direction): a learned conditional prior +
   posterior over a latent `w` that captures the one-to-many residual
   `p(target|source)`; injected as extra cross-attention memory tokens. Trained
   alongside the adapters when `--use-cvae` is set; absent by default.

Items 1–2 train in the retrieval phase; 3–5 in the per-direction generation phase.

The 32-d shared space serves as a universal interlingua: retrieval scores
flow through it, generation conditioning flows through it. See
[PLAN_late_interaction.md](PLAN_late_interaction.md) for the architectural
rationale, and [PLAN_generativity.md](PLAN_generativity.md) for the
reconstruction phase, generation-side CVAE, and best-of-N round-trip selection.

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
| protein decoder (text→protein) | Dayhoff-3b-UR90 (Jamba) | `/Users/bryan/Documents/models/Dayhoff-3b-UR90` |
| text decoder (protein→text) | BioGPT | `/Users/bryan/Documents/models/biogpt` |

These defaults are the dev-machine layout but are **environment-overridable**, so
a `config.py` synced from the dev box doesn't clobber a cluster's paths. Set in
the job script:

```bash
export FILIP_MODELS_DIR=/path/to/models   # swaps the base dir for all four (subdir names assumed equal)
export FILIP_DATA_CSV=/path/to/data.csv   # overrides DATA_CSV
# per-model overrides if a name/location differs:
export FILIP_TEXT_ENCODER=...  FILIP_PROTEIN_ENCODER=...  FILIP_PROTEIN_DECODER=...  FILIP_TEXT_DECODER=...
```

(If you see `HFValidationError: Repo id must be in the form ...` for a `/Users/...`
path, it's this — a dev `config.py` on the cluster with no override set.)

Data source (default in `DataCfg.csv_path`): the SwissProt-full CSV with
columns `primary_Accession`, `protein_sequence`, `[final]text_caption`,
`pfam_label`.

---

## End-to-end workflow

Run the stages in order. Precompute is one-shot, retrieval and generation are
training, generate is inference. The reconstruction phase (2b) is optional but
recommended before generation; round-trip eval and inspection can run at any
point after retrieval training.

```bash
# 1) One-shot per-token cache build (encoder forwards over the dataset).
python -m src.precompute --device cuda --batch-size 64

# 2) Retrieval (FILIP) training.
python -m src.train_retrieval --use-cache --device cuda

# 2b) OPTIONAL: expansion-only reconstruction phase. Freezes the projection
#     heads (R@K unchanged) and trains only the expansion heads harder, so the
#     generation conditioning memory expand(project(h)) is less lossy. Produces a
#     retrieval-format checkpoint that the generation stage consumes in place of
#     the raw retrieval checkpoint.
python -m src.train_reconstruction \
    --ckpt checkpoints/retrieval/epochNN.pt --device cuda

# 3a) Generation training: text → protein (Dayhoff-3b by default).
#     Add --use-cvae to also train the conditional-VAE latent.
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/reconstruction/epochNN.pt --device cuda

# 3b) Generation training: protein → text (BioGPT).
python -m src.train_generation --direction protein2text \
    --retrieval-ckpt checkpoints/reconstruction/epochNN.pt --device cuda

# 4) Inference (either direction). --num-candidates>1 enables best-of-N with
#    contrastive round-trip selection.
python -m src.generate --direction text2protein \
    --retrieval-ckpt checkpoints/reconstruction/epochNN.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epochNN.pt \
    --input "PROTEIN NAME: DNA helicase. FUNCTION: Unwinds DNA duplex..." \
    --num-candidates 8 --selection margin

# 5) Round-trip eval: generate -> re-encode -> FILIP-retrieve the source.
python -m src.roundtrip_eval --direction text2protein \
    --retrieval-ckpt checkpoints/reconstruction/epochNN.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epochNN.pt \
    --num-samples 1000 --split test --device cuda
```

The retrieval checkpoint passed to generation/inference can be either a
`train_retrieval` or a `train_reconstruction` checkpoint — they share the same
format. Use the reconstruction one when you have it.

## Local smoke test (no cluster, no cache)

```bash
python -m src.precompute --subset-size 512 --batch-size 16 --device cpu
python -m src.train_retrieval --use-cache --batch-size 32 \
    --phase1-epochs 1 --phase2-epochs 2 --device cpu

# Optional reconstruction phase (expansion heads only; R@K unchanged).
python -m src.train_reconstruction --ckpt checkpoints/retrieval/epoch02.pt \
    --subset-size 512 --batch-size 32 --epochs 2 --device cpu

# Generation. The default text2protein decoder is Dayhoff-3b (a 3B Jamba MoE,
# impractical on CPU); point the direction at a small decoder for the smoke test.
# --subset-size must match the cache so the by-accession split lines up.
FILIP_PROTEIN_DECODER=/path/to/progen2-small \
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/reconstruction/epoch01.pt \
    --subset-size 512 --batch-size 4 --epochs 1 --device cpu --use-cvae

# Best-of-N inference (re-encodes candidates; loads the target encoder too).
FILIP_PROTEIN_DECODER=/path/to/progen2-small \
python -m src.generate --direction text2protein \
    --retrieval-ckpt checkpoints/reconstruction/epoch01.pt \
    --decoder-ckpt checkpoints/generation/text2protein/epoch00.pt \
    --input "Catalytic protein involved in metabolism." \
    --num-candidates 4 --selection margin --max-new-tokens 24 --device cpu
```

512 pairs precompute in ~100s on Mac CPU. Retrieval epoch ~20s. Reconstruction
epoch is a few seconds (no decoder). Generation epoch with a small decoder is a
minute or two on CPU.

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
`protein_ids.json`, `text_h.bin`, `text_offsets.pt`, `text_mask.bin`,
`pair_ids.json`, `row_protein_idx.pt`, `fingerprint.json`. The fingerprint
records a format tag, the encoder paths, length caps, and special-token flags;
a mismatch on rebuild aborts retrieval training with a clear error instead of
silently training against the wrong cache.

**Protein dedup.** The augmented corpus repeats each protein across ~8.87
caption rows. The protein modality is encoded + stored once per *unique*
protein (so `protein_*` has `N_unique` rows, `text_*` has `N_rows`), and
`row_protein_idx.pt` (`[N_rows]`, CSV row → unique-protein index) joins them at
read time. This avoids ~9× of the protein encoder pass and ~1.5 TB of cache.
Splits are by accession (no protein straddles train/val/test); the retrieval
InfoNCE and the val/round-trip recall mask same-protein siblings so the
multiple captions per protein are not treated as false negatives.

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

### `src/train_reconstruction`

Optional phase between retrieval and generation. Loads a retrieval checkpoint,
**freezes the projection heads + temperature, trains only the expansion heads**
on reconstruction MSE, and writes the same checkpoint format. Because retrieval
scoring depends only on the projection, R@K is mathematically unchanged (the
per-epoch log re-runs the eval to confirm); the gain is a less-lossy
`expand(project(h))` conditioning memory for generation.

| flag | default | meaning |
|---|---|---|
| `--ckpt` | required | trained retrieval checkpoint to refine |
| `--device` | `auto` | |
| `--cache-dir` | `cache/` | per-token cache to reconstruct |
| `--ckpt-dir` | `checkpoints/reconstruction/` | per-epoch checkpoints + `train_log.json` |
| `--batch-size` | `128` | |
| `--epochs` | `5` | |
| `--lr` | `1e-3` | recon tolerates a higher LR than joint retrieval |
| `--subset-size` | `0` | must match the cache when <full |
| `--val-subset` | `1000` | recon MSE + R@K-unchanged check each epoch |

### `src/train_generation`

| flag | default | meaning |
|---|---|---|
| `--direction` | required | `text2protein` or `protein2text` |
| `--retrieval-ckpt` | required | path to a `train_retrieval` *or* `train_reconstruction` checkpoint |
| `--device` | `auto` | |
| `--cache-dir` | `cache/` | per-token encoder cache used as cross-attn memory source |
| `--ckpt-dir` | `checkpoints/generation/<direction>/` | adapter-only checkpoints |
| `--batch-size` | `16` | tighter than retrieval; decoder forward dominates |
| `--epochs` | `3` | |
| `--lr` | `1e-4` | |
| `--cross-attn-every` | `2` | inject cross-attention into every Nth block |
| `--unfreeze-top` | `0` | fully fine-tune the top N decoder blocks (on top of adapters/LoRA) |
| `--subset-size` | `0` | must match the cache when <full |
| `--seed` | `0` | reuses retrieval splits when the dataset size matches |
| `--use-cvae` | off | train the conditional-VAE latent (see below) |
| `--cvae-d-w` | `32` | latent dimension |
| `--cvae-n-latent-tokens` | `4` | extra cross-attn memory tokens decoded from `w` |
| `--cvae-beta-max` | `0.1` | KL weight after warmup |

Config knobs in `GenerationCfg` not on the CLI: `lora_rank=16`,
`lora_alpha=32`, `lora_dropout=0.05`, `lora_targets_self_attn=True`,
`lora_targets_ffn=True`, `max_target_tokens=512`, `cvae_hidden=256`,
`cvae_free_bits=0.5`, `cvae_kl_warmup_frac=0.3`.

**CVAE (`--use-cvae`).** Adds a latent `w` capturing the one-to-many residual
`p(target|source)`. A posterior `q(w|source,target)` (sees both pooled 32-d
embeddings, both frozen) is trained against a learned conditional prior
`p(w|source)`; `w` is decoded to `cvae_n_latent_tokens` extra cross-attention
memory tokens. Loss is `CE + β·KL`, with β warming 0→`cvae_beta_max` and per-dim
free bits guarding against posterior collapse. The CVAE heads are saved under
`cvae_state` in the checkpoint (absent ⇒ downstream runs without a latent). At
inference, `w` is sampled from the prior — sampling N latents yields N distinct
candidates for best-of-N. See [PLAN_generativity.md](PLAN_generativity.md).

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
| `--num-candidates` | `1` | >1 enables best-of-N selection |
| `--selection` | `margin` | `margin` (pos − best panel) or `pos` |
| `--panel-size` | `256` | reference negatives sampled from `--panel-csv` |
| `--panel-csv` | `DATA_CSV` | source of the margin reference panel |
| `--device` | `auto` | |

If the decoder checkpoint carries `cvae_state`, the conditioning memory is
augmented with a latent sampled from the prior `p(w|source)`; with
`--num-candidates > 1`, N independent latent samples give N distinct candidates,
re-encoded and ranked by round-trip score (the loaded target encoder closes the
loop). The ranked list is printed and the best candidate is the output.

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
| `--retrieval-ckpt` | required | retrieval / reconstruction checkpoint |
| `--decoder-ckpt` | required | generation checkpoint (CVAE auto-detected) |
| `--split` | `test` | `train` / `val` / `test` |
| `--num-samples` | `1000` | sources to evaluate (0 = whole split) |
| `--num-candidates` | `1` | best-of-N per source |
| `--selection` | `margin` | `margin` or `pos` |
| `--panel-size` | `256` | train rows used as the margin reference panel |
| `--temperature` / `--top-p` | `1.0` / `0.9` | sampling |
| `--max-new-tokens` | `256` | |
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
- **Live, by accession** (`--protein-id`/`--id-file` + `--csv`): look the
  accession(s) up in a CSV and encode live. Encoders + model load once and are
  reused across all accessions; the CSV is streamed in a single pass. Useful
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
`protein_ids.txt`, pulls sequence+caption from the legacy SwissProt CSV, and
writes a heatmap per accession. `compute_similarity_matrix_live` /
`load_inspect_encoders` are also available as a Python API.

### `src/export_embeddings`

Dumps the 32-d retrieval embeddings (`z_p`, `z_t`), keyed by primary accession,
for latent-space structure work (clustering, UMAP, modality gap) that training
never emitted. Two sources, with **different parallelism**:

- **Cached** (default): projects the precomputed cache for whole train/val/test
  splits. Just the projection head over cached hidden states — tiny compute,
  I/O-bound — so it runs **single process** (plain `python`, not mpiexec).
  ```bash
  python -m src.export_embeddings --ckpt checkpoints/retrieval/epochNN.pt \
      --splits train,test --device xpu
  ```
- **Live** (`--csv PATH`): encodes every row of a CSV (no cache, no splits — all
  sequences). This runs the frozen **encoders**, the expensive part, so it is
  **distributed** — launch under `mpiexec` (`export_embeddings.pbs`) and it
  shards proteins/rows across tiles like precompute (one process falls back
  cleanly). Proteins are deduped (encoded once per unique accession).
  ```bash
  mpiexec -n <12*nodes> -ppn 12 python -m src.export_embeddings \
      --csv data.csv --ckpt checkpoints/retrieval/epochNN.pt --name mydata --device xpu
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
| `gap_l2` | distance between mean text-token and mean protein-token in 32-d | falls during Phase R1, may rebound in Phase R2 |
| `mean_cross_token_cos` | average cosine between random protein-token and random text-token (background, across non-matching pairs) | should NOT be close to 1 |
| `mean_pos_token_cos` | average cosine between protein-token and text-token within a *correct* pair | should sit clearly above `mean_cross_token_cos`; the gap between them is the per-token alignment signal |
| `uniformity_p_tokens` | per-modality Wang-Isola spread; lower (more negative) = better-spread | approaches `-3.7` for fully-spread 32-d on the sphere |

Also in the full dict: `mean_intra_p_token_cos`, `mean_intra_t_token_cos`
(within-modality token cosines; warn of single-modality collapse),
`uniformity_t_tokens`, and per-direction R@K splits.

## Reading the generation per-step log

```
[text2protein] epoch=0 step=1/14 lr=1.00e-04 ce=2.8933 ppl=18.05
[text2protein] epoch=0 step=1/14 lr=1.00e-04 ce=2.8933 ppl=18.05 kl=24.6 beta=0.000   # with --use-cvae
```

| field | meaning |
|---|---|
| `ce` | cross-entropy on the target sequence, teacher-forced |
| `ppl` | `exp(ce)`; perplexity. Random baseline = vocab size (≈32 for ProGen2/Dayhoff char vocab, 42384 for BioGPT) |
| `kl` | (CVAE only) KL(posterior‖prior), floored at `cvae_free_bits × cvae_d_w` |
| `beta` | (CVAE only) current KL weight, warming 0 → `cvae_beta_max` |

The decoder cross-attention adapters and LoRA layers are zero-initialized,
so the first forward equals the pretrained decoder's unconditional prior.
`ce` at step 0 should already be below random; learning is visible as it
drops further over the first epoch. With `--use-cvae`, val CE conditions on the
prior **mean** (no target at inference), so it reflects the inference-time path.

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

**CVAE posterior collapse.** `kl` falls to its free-bits floor early and stays
there, and decoding two latent samples for one source gives identical outputs —
the decoder is ignoring `w`. Lower `cvae_beta_max`, raise `cvae_free_bits`,
inject the latent at more layers (`--cross-attn-every 1`), or add latent tokens
(`--cvae-n-latent-tokens`). Until samples are visibly diverse, best-of-N over
latents has nothing to select from.

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

**Dayhoff-3b / Jamba decoder notes.** It is a hybrid Mamba/attention MoE, so
the text→protein adapters are injected via forward hooks (not module
replacement) to preserve Jamba's per-layer `isinstance` mask routing. It loads
with `use_mamba_kernels=False` (the fused mamba-ssm/causal-conv1d CUDA kernels
aren't available on Mac/XPU — the pure-PyTorch SSM path is used) and
`output_router_logits=False`. Its custom char `ProteinTokenizer` (written for
transformers 4.42) won't load through transformers-5 `AutoTokenizer`, so it is
imported directly from the repo's `tokenizers.py`; the tokenizer adds no
BOS/EOS, so the collator now wraps targets as `[BOS] … [EOS]` explicitly.
LoRA covers the attention layers' self-attn and the dense MLPs; the fused MoE
expert tensors and SSM mixers are left frozen. Generation passes
`use_cache=True` (the model card sets `use_cache=False`, which would disable the
KV/SSM cache and recompute the full sequence every step).

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

Trainable parameter counts:

| stage | params |
|---|---|
| retrieval (projection + expansion heads + temperature) | ~2.0M |
| reconstruction (expansion heads only, refines existing weights) | ~1.0M |
| generation text→protein (ProGen2-class cross-attn @ every-2 + LoRA) | ~25M |
| generation protein→text (BioGPT cross-attn @ every-2 + LoRA) | ~48M |
| CVAE heads, per direction (optional, `--use-cvae`) | ~0.2M |

The text→protein figure is for a ProGen2-class decoder (what training run 1
used); the shipped default decoder is the larger Dayhoff-3b Jamba MoE, whose
cross-attention adapters scale with its hidden size and layer count.
Reconstruction trains the same expansion-head tensors as retrieval (no new
parameters); the CVAE heads are tiny relative to the adapters.

See [PLAN_late_interaction.md](PLAN_late_interaction.md) for the full
design rationale, the rejected alternatives (Q-former, soft prefixes,
unguided alignment warmup), and the open implementation decisions.
