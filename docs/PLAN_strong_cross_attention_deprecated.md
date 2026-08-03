# Strong: aligned-space max-sim cross-attention

Reference for the `--cross-attn-mode aligned` generation adapter (the "Strong"
plan). It replaces the learned multi-head cross-attention with a **single-head
cosine max-sim** computed in the shared 64-d retrieval space, so the attention
weights *are* the FILIP `[residue × source-token]` alignment array. Everything
here is `text2protein` (ProtGPT3 decoder); the protein→text direction (BioGPT)
uses the same code with the modality roles swapped.

Implemented in `CrossAttentionAdapter` / `CrossAttentionAdapter._forward_aligned`
in [src/decoder_adapters.py](src/decoder_adapters.py). Requires
`--memory-space aligned` (the keys must be the unit-norm retrieval vectors).

---

## 1. The core idea

The frozen protein decoder (ProtGPT3) reasons in its own 512-d hidden space. The
conditioning lives in the 64-d shared space where retrieval aligned text and
protein tokens (a matched text token ≈ its protein token). Strong makes the
decoder **query into that shared space**: a learned `q_align: 512 → 64` projects
each decoder position into the aligned space, where it is scored by cosine
similarity against the source (text) tokens — the exact FILIP max-sim operation,
now between *generated residues* and *caption tokens*.

The single load-bearing new object is **`q_align`** — the map from the decoder
hidden space into the shared memory space. See §4.

---

## 2. Object inventory and dimensions

Fixed sizes (from `config.py` / the model configs):

| symbol | meaning | size |
|---|---|---|
| `D` = `dec_hidden` | ProtGPT3 hidden size | **512** |
| `E` = `embed_dim` = `mem_dim` | shared retrieval space dim | **64** |
| `H_text` | BioLinkBERT hidden (text encoder) | 768 |
| `H_prot` | AMPLIFY hidden (protein encoder) | 960 |
| `n_layers` | ProtGPT3 decoder layers | 8 |
| `n_heads` | ProtGPT3 attention heads | 8 (unused by aligned scoring) |
| `B` | batch size | — |
| `T` | generated/teacher-forced protein length | ≤ `max_target_tokens` (512) |
| `L_mem` | number of source (caption) tokens = memory length | ≤ `max_text_tokens` (512) |

Per-tensor shapes in one aligned cross-attention call:

| tensor | shape | space |
|---|---|---|
| decoder hidden `h_dec` | `[B, T, 512]` | decoder (512-d) |
| conditioning memory `mem` | `[B, L_mem, 64]` | shared aligned (64-d, unit-norm) |
| memory mask | `[B, L_mem]` bool | — |
| query `q = norm(q_align(ln_q(h_dec)))` (q_align = MLP) | `[B, T, 64]` | shared aligned (64-d, unit-norm) |
| key `k = norm(mem)` | `[B, L_mem, 64]` | shared aligned |
| scores `q·kᵀ · temp` | `[B, T, L_mem]` | — (the FILIP array) |
| `attn = softmax(scores)` | `[B, T, L_mem]` | — |
| value `v = v_proj(mem)` | `[B, L_mem, 512]` | decoder (512-d) |
| `attn_out = attn·v` | `[B, T, 512]` | decoder |
| residual `o_proj(attn_out)` | `[B, T, 512]` | decoder |

---

## 3. Building the memory (keys/values source)

For `text2protein`, the source is the caption. The memory is the retrieval
projection of the caption tokens — **not** the expansion (that is the "expanded"
mode). All of this is frozen at generation time.

```
caption text
   → BioLinkBERT (frozen)                     h_text  [B, L_mem, 768]
   → retrieval.text_proj (frozen)             z       [B, L_mem, 64]   (L2-normalized)
   → [optional] retrieval.text_map (frozen)   mem     [B, L_mem, 64]   (Medium; still normalized)
```

- `mem = z` when `--memory-map` is off (the bypass arm), or `mem = text_map(z)`
  when on (the "Medium" cross-modal map, trained in the frozen map phase and
  frozen here). Either way `mem_dim = embed_dim = 64` and `mem` is unit-norm.
- The projection head (`ProjectionHead`) and the memory map (`MemoryMap`) both end
  in `F.normalize`, so the keys are unit vectors — required for cosine max-sim.
- `set_cross_memory(adapters, mem, mask)` stashes `mem` + `mask` on every adapter
  before the decoder forward; `clear_cross_memory` wipes them after.

(If `--use-cvae` were on, `k` extra latent tokens are concatenated →
`[B, L_mem + k, 64]`; they are renormalized as keys. Strong is intended to run
**without** CVAE so the array stays clean.)

---

## 4. The decoder → shared-space mapping (`q_align`) — the crux

This is the object the whole plan hinges on: how a 512-d decoder hidden state
becomes a query in the 64-d shared space.

```
h_dec            [B, T, 512]     decoder hidden state at this layer
  → ln_q         LayerNorm(512)  pre-norm on the decoder side
  → q_align      MLP: 512 → 512 → 256 → 64        ← the decoder→shared map
                 (Linear/GELU/LayerNorm, dropout after block 1)
  → F.normalize(·, dim=-1)       unit vector in the 64-d aligned space
q                [B, T, 64]
```

Properties:

- **`q_align` is a small MLP with the same body as the retrieval `ProjectionHead`
  (§3): `dec_hidden → dec_hidden → dec_hidden//2 → embed_dim`, i.e. `512 → 512 →
  256 → 64` for ProtGPT3, `Linear/GELU/LayerNorm` with dropout after the first
  block; the final `F.normalize` lives in the forward.** One per injected adapter
  layer (not tied) — different depths carry different features.
- **Why an MLP, not a linear map.** The aligned space itself is *nonlinearly*
  constructed — the keys `mem = text_proj(h)` come from a `ProjectionHead` MLP, so
  both sides of the max-sim now reach the shared space through the same *kind* of
  map. The only thing a linear `q_align` would have bought is an analyzable single
  matrix for the decoder↔shared geometry; we don't need that. The **array's
  interpretability is unaffected** — it depends only on cosine-scoring against the
  real keys (§5), not on how the query is computed — while the extra capacity
  gives the map a better chance of landing ProtGPT3's hidden states in the
  AMPLIFY/BioLinkBERT-defined space.
- It is the only place the decoder's native space is bridged to the retrieval
  space. In "head" mode the query stays in 512-d (`q_proj: 512→512`) and the keys
  are lifted to 512-d (`k_proj: 64→512`); Strong instead **pulls the query down to
  64-d and scores against the keys directly**, with no `k_proj`.
- **Initialization:** `q_align` uses default init — it is *not* zeroed. The whole
  adapter is still a no-op at step 0 because `o_proj` is zero-initialized (§6), so
  `q_align` can begin learning as soon as `o_proj` moves off zero.
- **Load-bearing assumption / risk:** `q_align` must learn to land ProtGPT3's
  hidden states in a space that was *defined by AMPLIFY protein tokens and
  BioLinkBERT text tokens* — ProtGPT3 never saw that space. This is learnable but
  is the main bet of the Strong design. (The "Minimal"/aligned-memory path
  sidesteps it by keeping learned `k/v` projections; Strong commits to it in
  exchange for the interpretable array.)

Keys are the memory itself (already aligned), only renormalized defensively:

```
k = F.normalize(mem, dim=-1)     [B, L_mem, 64]     no k_proj — keys ARE the retrieval vectors
```

---

## 5. The cross-attention computation, step by step

From `CrossAttentionAdapter._forward_aligned` (shapes annotated):

```python
h   = ln_q(h_dec)                                   # [B, T, 512]
q   = F.normalize(q_align(h), dim=-1)               # [B, T, 64]   decoder → shared space (q_align = MLP)
k   = F.normalize(mem, dim=-1)                       # [B, L_mem, 64]  aligned keys
temp = logit_scale.clamp(max=log(100)).exp()         # scalar; init exp(log(1/0.07)) ≈ 14.3
scores = (q @ k.transpose(-1,-2)) * temp             # [B, T, L_mem]   cosine max-sim × temperature
scores = scores.masked_fill(~mask[:,None,:], -inf)   # drop padded/invalid source tokens
attn   = softmax(scores, dim=-1)                     # [B, T, L_mem]   ← FILIP alignment array
self.last_attn = attn.detach()                       # kept for interpretability / --dump-attn
v      = v_proj(mem)                                  # [B, L_mem, 512]  aligned → decoder space
attn_out = attn @ v                                   # [B, T, 512]
return h_dec + o_proj(attn_out)                       # residual, o_proj: 512→512 (zero-init)
```

- **Single head.** Scoring is one head in the full 64-d space (not split into
  `n_heads`), so `attn` is a single `[T, L_mem]` distribution per example — that
  is what makes it *the* interpretable alignment array. `n_heads`/`head_dim`/
  `self.scale` from the base class are unused in aligned mode.
- **Learned temperature.** `logit_scale` is a trainable scalar (FILIP-style),
  initialized to `log(1/0.07)` and clamped at use to `log(100)`. It sets the
  sharpness of the softmax over source tokens.
- **Values stay learned and in decoder space.** `v_proj: 64→512` maps the aligned
  memory into a usable 512-d signal. So the *weights* are the alignment while the
  *content* pulled in is a learned function of the memory — interpretability is
  preserved regardless of what `v_proj` learns.
- **Residual + zero-init `o_proj`.** The adapter adds `o_proj(attn_out)` to the
  decoder stream; `o_proj` starts at 0 so the adapter is an exact no-op at step 0.
- **dtype.** The decoder runs bf16; adapter params are fp32. The forward casts the
  hidden state to the adapter dtype and casts the residual back, so a bf16 frozen
  backbone and fp32 trainable adapter coexist under autocast and at inference.

---

## 6. Wiring / injection into ProtGPT3

- ProtGPT3 is a Mixtral sparse-MoE decoder. Adapters are injected on **every
  `cross_attn_every`-th layer** via a `register_forward_hook` on that decoder
  layer (`_mixtral_inject`), so the layer's class is untouched — the hook applies
  `h ← h + adapter(h)` to the layer output. At `cross_attn_every=2` that is layers
  `{0,2,4,6}` → **4 adapters**; at `cross_attn_every=1`, all 8 layers → 8 adapters.
- The adapters are also registered as `model.cross_attn_adapters =
  nn.ModuleList(...)` so their parameters are tracked by the optimizer,
  `state_dict`, `.to(device)`, and DDP. The hook and the ModuleList hold the same
  module objects, so `set_cross_memory()` reaches them.
- The same `mem`/`mask` is set on **all** adapters each step (one shared memory,
  attended independently at each depth).

---

## 7. Trained vs frozen during generation training

**Trained (the only things with `requires_grad=True`):**

| component | params | notes |
|---|---|---|
| per-adapter `q_align` (MLP) | `411,968` each | decoder → shared-space map (512→512→256→64) |
| per-adapter `v_proj` | `64×512` = 32,768 each | shared → decoder values |
| per-adapter `o_proj` | `512×512` = 262,144 each | residual out; **zero-init** |
| per-adapter `ln_q` | `512+512` = 1,024 each | pre-norm |
| per-adapter `logit_scale` | 1 each | learned temperature |
| LoRA on self-attn `q/k/v/o` | `2×512×16` = 16,384 per proj | rank 16, α 32; **all 8 layers** |

**Frozen (everything else):**

- The entire ProtGPT3 backbone: token embeddings, self-attention weights, **MoE
  experts and router**, layernorms, `lm_head`. (LoRA wraps the frozen self-attn
  linears additively; the base weights never update. The MoE experts are fused
  `Parameter` tensors, not `nn.Linear`, so they get no LoRA and stay frozen.)
- The retrieval model: `text_proj`, `protein_proj`, expansion heads, temperature,
  and the `text_map`/`protein_map` (Medium) — the maps are trained in a *separate*
  frozen map phase and are frozen here.
- The text and protein encoders (BioLinkBERT, AMPLIFY).

Loss: next-token cross-entropy on the (teacher-forced) protein sequence. The
gradient reaches the adapters + LoRA only.

---

## 8. Parameter counts (ProtGPT3, `embed_dim=64`)

Per aligned adapter: `1,024 (ln_q) + 411,968 (q_align MLP) + 32,768 (v_proj) +
262,144 (o_proj) + 1 (logit_scale)` = **707,905**. The `q_align` MLP is now the
largest single object (`262,656 + 1,024 + 131,328 + 512 + 16,448`), ahead of
`o_proj`.

| config | adapters | adapter params | LoRA params | total trainable |
|---|---|---|---|---|
| `cross_attn_every=2` | 4 | 2,831,620 | 524,288 | **3,355,908** |
| `cross_attn_every=1` | 8 | 5,663,240 | 524,288 | 6,187,528 |

(LoRA = `4 projections × 8 layers × (512×16×2)` = 524,288, independent of
`cross_attn_every`.) The `3,355,908` figure is exactly the trainable count printed
by a `cross_attn_every=2` aligned build. With `--no-lora` the LoRA term is dropped
(adapters only): `cross_attn_every=2` → 2,831,620 trainable.

---

## 9. Interpretability

Each adapter stores `self.last_attn = attn.detach()` — a `[B, T, L_mem]` tensor
whose `[t, m]` entry is how much generated residue `t` attends to caption token
`m`, in the retrieval geometry. This is the generation-time analogue of the
retrieval FILIP max-sim matrix, and is the reason for the single-head aligned
design. A `--dump-attn` inference path can serialize it into a
`[residue × caption-token]` heatmap.

---

## 10. Requirements and constraints

- **Requires `--memory-space aligned`.** The keys must be the unit-norm 64-d
  retrieval vectors; "expanded" memory (768/960-d, not normalized) is not a valid
  key space for cosine max-sim. `train_generation.py` enforces this.
- **Recommended without `--use-cvae`** (the collapsed latent tokens are not
  aligned-space unit vectors and would muddy the keys/array).
- Composes with **`--memory-map`** (Medium): if on, the keys are `text_map(z)`
  instead of `z`; the query head then matches against the protein-warded memory.
- The `cross_attn_mode` is stored in the generation checkpoint and read back by
  the ablation/inference paths so the adapter is rebuilt in the same scoring space.
