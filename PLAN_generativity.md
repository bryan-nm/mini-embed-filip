# Improving Generativity: Reconstruction Phase, Generation-side CVAE, Best-of-N

Forward-looking plan for the next iteration of the generation pipeline, motivated
by [EVAL_training_run_1.md](EVAL_training_run_1.md): retrieval is excellent, but
**text→protein generation is weak** — generated proteins barely round-trip
(R@1 ≈ 0.01 vs ceiling 0.64). The diagnosis there is twofold:

1. **Train/inference (teacher-forcing vs sampling) gap + one-to-many collapse.**
   A caption maps to many valid proteins; the deterministic decoder produces a
   generic protein "broadly compatible with many captions, specific to none".
2. **Suspected secondary limiter:** generation conditions on the *lossy*
   `expand(project(h))` (recon MSE ~0.12–0.2) whereas retrieval uses the sharp
   32-d vectors — the decoder may read a blurred caption.

Three features, in recommended build order:

| # | Feature | Attacks | Cost |
|---|---|---|---|
| 2 | Expansion-only reconstruction phase | the blur (secondary limiter) | cheap, zero retrieval cost |
| 3 | Best-of-N w/ contrastive round-trip selection | the low-margin / collapse geometry | inference-only |
| 1 | Generation-side CVAE | one-to-many (primary limiter) | training rework |

The unifying implementation fact: generation conditioning is just per-token
**memory tensors set on the adapters** via `set_cross_memory(adapters, mem, mask)`
([decoder_adapters.py](src/decoder_adapters.py)), and the adapter cross-attends to
a *variable-length, masked* memory. So the CVAE latent is injected as **extra
memory tokens** — no decoder internals change.

---

## Feature 2 — Expansion-only reconstruction phase

**Idea.** Retrieval scoring depends only on `project`; the expansion heads play no
role in any retrieval metric. So after retrieval converges we **freeze the
projection heads + temperature and train only the expansion heads** on
reconstruction MSE. Retrieval R@K is mathematically unchanged; the conditioning
memory `expand(project(h))` gets as sharp as the frozen 32-d code permits — a free
lunch up to that ceiling.

**Implementation:** new standalone script `src/train_reconstruction.py`.

- Loads a trained retrieval checkpoint into `MiniEmbedFilip`.
- `model.eval()` (dropout off, so the reconstructed `z` matches what generation
  sees), then `requires_grad_(True)` on `text_expand` / `protein_expand` only.
- Reuses `build_loaders` (imported from `train_retrieval`) over the packed
  per-token cache + by-accession splits.
- Step: `z = project(h)` under `no_grad`; `h_hat = expand(z)`; loss is the
  existing `reconstruction_loss` averaged over both modalities.
- Distributed via the manual `broadcast_parameters` + `average_gradients` path
  (no contrastive coupling → purely local grads; avoids DDP's frozen-param
  bookkeeping).
- Saves `{"epoch", "model_state"}` — **identical format to retrieval**, so
  `generate` / `train_generation` / `roundtrip_eval` load it unchanged.
- Each epoch logs per-modality recon MSE and re-runs `evaluate_split` once to
  *confirm* R@K is unchanged.

**Config:** new `ReconCfg` (epochs=5, lr=1e-3, own `ckpt_dir`).

Run after choosing the converged retrieval epoch:
```bash
python -m src.train_reconstruction \
    --ckpt checkpoints/retrieval/epoch50.pt --device xpu
# then retrain generation off checkpoints/reconstruction/epochNN.pt
```

---

## Feature 3 — Best-of-N with contrastive round-trip selection

**Idea.** Generate N candidates per source, re-encode each, and select the one
with the best contrastive **margin** = `FILIP(cand, src) − max_panel FILIP(cand, panel)`
against a reference panel of *other* sources. Plain pos-score is a poor selector
(bad generations still score high — the documented low-margin failure); the margin
rewards candidates that are *specifically* compatible with their own source.

**New module:** `src/best_of_n.py` — `select_best_of_n(z_cands, mask_cands, z_src,
mask_src, z_panel, mask_panel, mode)` built on the existing
`filip_score_matrix_chunked`.

**Inference (`src/generate.py`):** flags `--num-candidates`, `--selection
{pos,margin}`, `--panel-size`. Loads the **target** encoder + target projection
(currently only the source side is loaded), generates N, re-encodes, selects,
prints the winner.

**Eval (`src/roundtrip_eval.py`):** flags `--num-candidates`, `--selection`. In
`generate_shard`, generate N per item and **select before writing the record**, so
the downstream merge/`_recall` scoring is unchanged. The selection panel is
sampled from the **train** split (disjoint from the scored set) — otherwise
selection optimizes exactly the metric being reported and inflates it.

Works standalone (diversity from sampling temperature); far stronger with the CVAE
(structured diversity). Use it to measure N=1 vs N=8 on the existing decoder
*before* building the CVAE, to confirm the selection signal is exploitable.

---

## Feature 1 — Generation-side CVAE

**Idea.** A conditional VAE injects a sampled latent `w` capturing the one-to-many
residual `p(target | source)` not explained by the deterministic memory. Sampling
`w` yields *structured* diverse candidates, which feed best-of-N. This is the tool
matched to the primary failure (mode collapse).

**Design choices:**

1. **`w` as extra memory tokens.** `to_memory(w) → [B, k, mem_dim]`, concatenated
   onto `mem`, mask extended by `k` True. No decoder-internal changes; the latent
   tokens are *identifiable and ablatable* (preserves interpretability).
2. **Learned conditional prior** `p(w|source)` (not fixed `N(0,I)`) — "which
   protein for this caption" is caption-dependent; at inference we sample from it.
3. **Posterior** `q(w|source,target)` sees both pooled 32-d retrieval embeddings,
   both from **frozen** heads — no new encoder; the target embedding is in the
   cache already.

**New module:** `src/cvae.py` — `CVAEHeads` (prior MLP, posterior MLP, `to_memory`
linear) + `CVAECfg` + a `beta_at` schedule. Closed-form Gaussian KL with **free
bits**; reparameterized sampling.

**Data:** `GenerationDataset` extended to also return the **target-side** cached
hidden states (text2protein: protein cache via `row_protein_idx`; protein2text:
text cache), pooled through the frozen target projection at train time for the
posterior.

**Training (`train_generation.py`, gated on `--use-cvae`):**
```
z_src_pool = mean(project(source));  mem = expand(project(source))
z_tgt_pool = mean(project(target))                      # frozen heads
qmu,qlv = posterior(z_src_pool, z_tgt_pool);  w = reparam(qmu,qlv)
pmu,plv = prior(z_src_pool)
mem_aug = cat([mem, to_memory(w)]);  mask_aug = pad(mask, +k True)
ce = decode(mem_aug);  kl = KL(q||p)
loss = ce + beta(step) * kl
```
The CVAE heads are used *outside* `decoder.forward`, so DDP won't sync them →
their grads are averaged manually with `average_gradients` (mirrors the retrieval
trainer). Saved under `cvae_state` + `cvae_cfg` in the generation checkpoint;
absent ⇒ no latent (**backward compatible**).

**Posterior-collapse controls** (the frozen decoder is a strong prior that wants
to ignore `w`): β warm-up, free bits, and a logged *output-diversity* probe (decode
2–3 `w` samples for one source). If KL→0 or diversity→0, raise `--cross-attn-every
1`, more latent tokens, or lower β.

**Inference / best-of-N:** no target available → sample `w ~ p(w|source)`. Draw N
independent `w` → N structurally distinct candidates → feed straight into
Feature 3's selection.

**Config:** `GenerationCfg.use_cvae` (+ `cvae_d_w`, `cvae_n_latent_tokens`,
`cvae_hidden`, `cvae_beta_max`, `cvae_free_bits`, `cvae_kl_warmup_frac`).

---

## Cross-cutting

- **Checkpoint compatibility:** Feature 2 keeps the retrieval format; Feature 1
  adds an *optional* `cvae_state` key. Existing checkpoints keep loading.
- **`masked_mean`** helper added to `src/losses.py` next to `reconstruction_loss`.
- **Interpretability:** the CVAE latent tokens are appended and separable — ablate
  them to quantify how much generation relies on the latent vs the aligned memory.
- **Eval honesty:** the best-of-N selection panel must be disjoint from the scored
  set in `roundtrip_eval`.

## Suggested validation sequence

1. Feature 2 → confirm R@K unchanged + recon MSE down, retrain generation off the
   sharpened checkpoint, check CE improves.
2. Feature 3 standalone → measure N=1 vs N=8 round-trip on the existing decoder.
3. Feature 1 → train `--use-cvae`, re-run Feature 3 with latent-sampled
   candidates; headline = round-trip R@1 for (CVAE + best-of-N) vs
   (temperature + best-of-N) vs (N=1).
