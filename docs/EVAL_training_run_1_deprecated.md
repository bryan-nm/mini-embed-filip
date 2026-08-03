# Eval — Training Run 1 (SwissProt full, Aurora)

First full-scale end-to-end run of the FILIP retrieval + bidirectional
generation pipeline on Aurora. Dataset: SwissProt-full, 574,627 pairs
(train 517,164 / val 28,731 / test 28,732). All four stages completed:
per-token precompute, FILIP retrieval, and both generation directions.

**TL;DR**
- **Retrieval: excellent.** R@10 = 0.94 on the val subset; the 32-d shared
  space cleanly captures protein/text semantics.
- **protein→text generation: works.** Fluent, specific, sometimes
  verbatim-correct descriptions. Fails by hallucinating a *related* protein,
  not by going generic.
- **text→protein generation: weak.** Generated proteins barely round-trip
  (R@1 ≈ 0.01 vs ceiling 0.64); the decoder produces generic proteins. CE
  improved with more conditioning capacity but round-trip did not — the
  bottleneck is the train/inference gap, not model capacity.

---

## 1. Infrastructure (Aurora notes worth keeping)

The shipped code was single-process; this run added distributed support. The
hard-won launch details (all environment-specific to the `frameworks/2025.3.1`
module) that future runs should reuse:

- **Rank detection via `mpi4py`** (`MPI.COMM_WORLD` + shared-comm split), not
  env vars — Aurora MPICH does not export the `PALS_*`/`PMI_*` names the first
  implementation assumed, so every rank silently saw `world=1`.
- **Distributed backend = `xccl`** (PyTorch-native XPU/oneCCL), not the legacy
  `ccl` (torch-ccl) string, which this torch build doesn't register.
- **`CCL_ZE_IPC_EXCHANGE=pidfd`** (not `drmfd` — rejected by this oneCCL).
- **`export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"`** so tiles enumerate once
  (12/node), not doubled by the OpenCL+L0 default.
- One rank per **tile** (12/node), `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.
- **`trust_remote_code` models (ProGen2) load rank-0-first + barrier** to avoid
  a module-cache write race across ranks.

**Queues:** `capacity` (1–16 nodes, 168 h) is the right queue for all stages —
`prod` (min 256 nodes) is counterproductive for training (8× fewer
steps/epoch). Precompute is merge/startup-bound, so 16 nodes ≈ 256 nodes.

**Parallelism model:** precompute = sharded embarrassingly-parallel +
single-pass merge; retrieval = manual gradient all-reduce (NOT DDP — the
grouped-gather contrastive loss double-marks `logit_scale` under DDP) with
bounded contrastive subgroups (group_size 16 → ~256 negatives); generation =
plain DDP.

---

## 2. Retrieval (FILIP) — **excellent**

`checkpoints/retrieval/epoch50.pt`. 1 R1 (warmup) + 50 R2 (InfoNCE) epochs,
16 nodes, lr 3e-4, batch 16 × group 16. Converged ~epoch 40.

| metric (epoch 50, 1000-pair val) | value |
|---|---|
| R@1 | 0.631 |
| R@5 | 0.893 |
| R@10 | **0.939** |
| gap_l2 | 0.193 (stable) |
| mean_cross_token_cos | 0.017 (healthy, ≉1) |
| uniformity_p_tokens | −2.43 |

**Key fix — R1 collapse.** The default `phase1_uniformity_weight=0.1` collapsed
in R1: `filip_pos` saturated to 0.98 in <50 steps, val R@10 stayed at random
(0.024). Raising it to **0.3** (now a CLI flag, `--phase1-uniformity-weight`)
gave a gradual `filip_pos` climb and R1 R@10 = 0.11, after which R2 trained
cleanly to 0.94. **One R1 epoch is correct — more saturates positives and
hurts R2.**

Caveat: R@K is on a 1000-pair subset (`--val-subset 1000`); the full-test
number will be lower. No standalone full-test eval was run.

---

## 3. Generation — CE summary

| direction | decoder | config | val CE | ppl | uncond. prior | conditioning gain |
|---|---|---|---|---|---|---|
| text→protein | ProGen2-small | every-2, no unfreeze | 1.657 | 5.24 | ~2.0 | 0.34 nats |
| text→protein | ProGen2-small | **every-1, unfreeze-2, lr 5e-5** | 1.512 | 4.54 | ~2.0 | 0.49 nats |
| protein→text | BioGPT | every-1, unfreeze-2, batch 8 | **0.325** | 1.38 | ~1.0 | large (templated) |

All ran 1+30 generation epochs on 16 nodes, converging ~epoch 24–25.
Checkpoints: `checkpoints/generation/{text2protein,protein2text}/epoch29.pt`.

Note: protein→text ppl 1.38 is flattered by **highly templated SwissProt
captions** — low absolute CE here is largely text predictability, not proof of
conditioning. Judge by round-trip, not CE.

---

## 4. Round-trip evaluation (the real test)

Generate → re-encode the generation → FILIP-retrieve the source. `src/roundtrip_eval.py`,
1000 test samples. **Ceiling** = true output re-encoded → source (best a perfect
generator could do with this scorer).

### text→protein (every-1 + unfreeze-2, T=1.0)
| | R@1 | R@5 | R@10 | median rank | pos score |
|---|---|---|---|---|---|
| **generated** protein→text | 0.012 | 0.052 | 0.087 | 199 | 0.824 |
| ceiling (true protein→text) | 0.644 | 0.890 | 0.930 | 1 | 0.985 |

- Essentially **random** (random R@1 over 1000 = 0.001; we get ~10×).
- T=0.5 was *slightly worse* (R@1 0.008) → not a sampling artifact.
- **High pos-score + bad rank = low margin, not low similarity.** True proteins
  (same scorer) discriminate perfectly, so the *space* is fine; the
  *generations* collapse into a generic sub-region — broadly compatible with
  many captions, specific to none. Mode-collapse, amplified by permissive
  max-sim and 32-d crowding.

### protein→text (every-1 + unfreeze-2, T=1.0)
No clean aggregate recorded (the metric is confounded — see below), but
**qualitatively the strongest result of the run**:
- **Verbatim-correct hits** (e.g. `Q6GEV1` MurA, `P60113` ATP synthase c) —
  rank 1, near word-for-word.
- **Common failure = "right neighborhood, wrong member"** (e.g. ribosomal uS15
  generated as uS14) — confident hallucination, *not* generic boilerplate.
- **Metric confound:** SwissProt captions share heavy boilerplate (`LINEAGE`,
  `GENE ONTOLOGY`), so the text-side FILIP round-trip partly scores
  taxonomy/format similarity, not protein identity (a wrong-enzyme caption from
  the same bacterial order ranked #1). Treat protein→text R@K as a soft proxy.
- **Decoding degeneration:** repetition loops in the templated tail at T=1.0.

---

## 5. Key findings

1. **Direction asymmetry is real and large.** protein→text (rich, specific
   input → constrained, low-entropy output) works; text→protein (one-to-many)
   does not. Future effort on text→protein should expect a harder problem.
2. **CE ≠ free-generation faithfulness.** text→protein CE improved 0.34→0.49
   nats with more capacity while round-trip stayed flat. The binding constraint
   is the **train/inference (teacher-forcing vs sampling) gap**, not capacity —
   so *more unfreezing is diminishing returns*.
3. **Retrieval space is not the bottleneck.** Ceiling R@1 = 0.64 proves the
   32-d space carries the semantics; the gap is entirely in the decoder's use
   of the conditioning.
4. **Suspected structural limiter:** generation conditions on
   `expand(project(h))` — a *lossy* reconstruction (recon MSE ~0.12–0.2) —
   whereas retrieval uses the *sharp* 32-d vectors directly. The decoder may be
   reading a blurred caption.

---

## 6. Things to try going forward (prioritized)

**Tier 1 — cheap, directly on the bottleneck**
- [ ] **Best-of-N with *contrastive* selection** (inference-only, no retrain).
  Generate N candidates, keep the one with the highest margin (pos − best
  negative) against a reference panel — directly attacks the train/inference
  gap and the low-margin problem. Highest EV next step for text→protein.
- [ ] **Decoding hygiene for protein→text:** `repetition_penalty≈1.2`,
  `no_repeat_ngram_size=3`, trim `max_new_tokens`. Cleans the looping tails.
- [ ] **Confound-resistant protein→text metric:** round-trip on only the
  identity-bearing fields (`PROTEIN NAME` + `FUNCTION` + `CATALYTIC ACTIVITY`),
  dropping `LINEAGE`/`GENE ONTOLOGY`; and/or a blunt family-match accuracy.

**Tier 2 — structural experiments**
- [ ] **Bypass the generation-side 32-d bottleneck:** condition cross-attention
  on the *raw* encoder hidden states (skip project/expand) and re-measure
  round-trip. If it jumps, the interlingua design needs a richer generation
  path. Most likely fundamental limiter for text→protein.
- [ ] **Recurrent / refinement generation** (draft → re-encode → gap-conditioned
  correction; round-trip distance as the stop signal). Train each step
  supervised (denoise corrupted true target), loop only at inference. Note:
  this *multiplies* a working conditioning mechanism — gated on Tier-1/the base
  actually conditioning.
- [ ] **Round-trip as a training signal (RL/PPO)** with the round-trip score as
  reward. The only objective that operates on free generation. High ceiling,
  high risk (non-differentiable sampling, reward-hacking — keep CE anchor + a
  held-out scorer). Pursue only if best-of-N shows exploitable signal.
- [ ] **Dimensionality sweep** (embed_dim 32 → 64/128) for margin headroom; the
  high-pos/low-margin geometry suggests 32-d crowding may cap discrimination.

**Tier 3 — deprioritized (don't target the bottleneck)**
- Larger frozen models (bigger encoder helps retrieval, already at ceiling;
  bigger decoder strengthens the generic prior).
- TrEMBL corpus (not data-limited; noisier annotations dilute signal).
- Caption cleaning (retrieval already handles current captions).
- Different generator model class (re-solves the same conditioning problem).

---

## 7. Reproducibility (commands)

```bash
# Precompute (16 nodes, sharded + merge)
python -m src.precompute --device xpu --batch-size 64

# Retrieval (the collapse fix is the key flag)
python -m src.train_retrieval --use-cache --device xpu \
    --batch-size 16 --group-size 16 \
    --phase1-epochs 1 --phase2-epochs 50 --lr 3e-4 \
    --phase1-uniformity-weight 0.3

# Generation (both directions; checkpoint stores cross_attn_every/unfreeze_top)
python -m src.train_generation --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch50.pt --device xpu \
    --batch-size 16 --epochs 30 --lr 5e-5 --cross-attn-every 1 --unfreeze-top 2
python -m src.train_generation --direction protein2text \
    --retrieval-ckpt checkpoints/retrieval/epoch50.pt --device xpu \
    --batch-size 8 --epochs 30 --lr 5e-5 --cross-attn-every 1 --unfreeze-top 2

# Round-trip eval (either direction)
python -m src.roundtrip_eval --direction text2protein \
    --retrieval-ckpt checkpoints/retrieval/epoch50.pt \
    --decoder-ckpt   checkpoints/generation/text2protein/epoch29.pt \
    --num-samples 1000 --split test --device xpu
```

All launched via `mpiexec -n <12×nodes> -ppn 12` on the `capacity` queue with
the Aurora env block from §1.
