# Model Merging Algorithm Evaluation

This document evaluates the existing merging algorithms in **stmerge**, compares them to widely-used methodologies in the model-merging community (LERP, SLERP, TIES, DARE, Passthrough / Frankenmerging), highlights gaps and weaknesses, and recommends concrete improvements and new algorithms to consider.

---

## Table of Contents

1. [Existing Algorithms](#1-existing-algorithms)
   - [Weighted Sum (LERP)](#11-weighted-sum-lerp)
   - [Consensus Merge (Inverse Distance Weighting)](#12-consensus-merge-inverse-distance-weighting)
2. [Industry-Standard Methodologies](#2-industry-standard-methodologies)
   - [LERP – Linear Interpolation](#21-lerp--linear-interpolation)
   - [SLERP – Spherical Linear Interpolation](#22-slerp--spherical-linear-interpolation)
   - [TIES – Trim, Elect Sign & Merge](#23-ties--trim-elect-sign--merge)
   - [DARE – Drop And REscale](#24-dare--drop-and-rescale)
   - [Passthrough / Frankenmerging](#25-passthrough--frankenmerging)
3. [Comparison Matrix](#3-comparison-matrix)
4. [Suggestions for Existing Algorithms](#4-suggestions-for-existing-algorithms)
   - [Weighted Sum Improvements](#41-weighted-sum-improvements)
   - [Consensus Merge Improvements](#42-consensus-merge-improvements)
5. [Recommended New Algorithms](#5-recommended-new-algorithms)
6. [Implementation Priority](#6-implementation-priority)
7. [Summary](#7-summary)
   - [What stmerge does well](#what-stmerge-does-well)
   - [Critical gaps](#critical-gaps)
   - [Recommended immediate actions](#recommended-immediate-actions)
   - [Hardware Quick Reference](#hardware-quick-reference)
8. [Hardware Considerations](#8-hardware-considerations)
   - [Hardware Target](#81-hardware-target)
   - [Algorithm Hardware Impact Summary](#82-algorithm-hardware-impact-summary)
   - [RAM Requirements by System Configuration](#83-ram-requirements-by-system-configuration)
   - [GPU Acceleration](#84-gpu-acceleration)
   - [CPU vs GPU Decision Guide](#85-cpu-vs-gpu-decision-guide)
   - [Recommendations](#86-recommendations)
   - [RTX 3090 Platform-Specific Analysis](#87-rtx-3090-platform-specific-analysis)
     - [RTX 3090 Laptop (16 GB VRAM)](#871-rtx-3090-laptop-16-gb-vram)
     - [RTX 3090 Desktop (24 GB VRAM)](#872-rtx-3090-desktop-24-gb-vram)

---

## 1. Existing Algorithms

### 1.1 Weighted Sum (LERP)

**Location:** `model_merger/merger.py` → `_weighted_sum_merge()`

#### How It Works

The weighted sum merge is a direct implementation of linear parameter interpolation. Given N models each assigned a scalar weight `w_i`, the merged parameter at every tensor position is:

```
result[k] = Σ (model_i[k] × w_i)   for i in 1..N
```

The implementation uses the **accumulator pattern** to keep memory usage bounded:

```python
# Step 1 – seed the accumulator with the first model's weighted tensors
accumulator[key] = model_0[key].to(torch.float32) * weight_0

# Step 2 – add each subsequent model in-place, then free it immediately
for entry in model_entries[1:]:
    current = load_model(entry.path)
    accumulator[key].add_(current[key].to(torch.float32) * entry.weight)
    del current      # freed before loading the next one
    gc.collect()
```

Only the accumulator and one loaded model live in RAM at any time, making it possible to merge 8+ large SDXL checkpoints (~6.5 GB each) without exceeding typical workstation memory.

#### Strengths

| Strength | Detail |
|---|---|
| **Memory efficiency** | O(2 × model_size) peak RAM regardless of how many models are merged |
| **Speed** | Single pass over the tensors; no per-element statistics needed |
| **Transparency** | The math is trivial to audit and reproduce |
| **Flexibility** | Weights can be arbitrary (negative, > 1, non-summing-to-1) |
| **Predictability** | Output is a deterministic linear combination; easy to reason about |

#### Weaknesses

| Weakness | Detail |
|---|---|
| **No sign-conflict resolution** | When two models push a parameter in opposite directions the arithmetic average can wash out meaningful features from both |
| **No sparsity awareness** | Fine-tuned "task deltas" (LoRA/DreamBooth artifacts) are diluted uniformly; highly fine-tuned parameters are treated identically to ones the fine-tune barely touched |
| **Uniform key treatment** | All tensors (`weight`, `bias`, normalization layers, embeddings) receive the same linear treatment even though their geometric roles differ |
| **No redundancy elimination** | Redundant or noise-level deltas from each model add up and can introduce drift |
| **User must know weights** | The user is responsible for choosing weights that produce coherent results; no guidance or validation is provided |

#### Supported Model Count

**Minimum:** 1 model (trivially copies the single model with its weight applied)  
**Maximum:** Unlimited — the accumulator pattern keeps peak RAM constant at ~2× model size regardless of N  
**Sweet spot:** **8–20+ models** — designed from the ground up for large multi-model merges; this is stmerge's primary strength

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | fp32 accumulator (~13 GB for SDXL) + one fp16 loaded model (~6.5 GB) ≈ **~19.5 GB** peak |
| **N-model scaling** | Constant — accumulator size never grows with N; only 2 models occupy RAM at any point |
| **VRAM usage** | None by default; all computation runs on CPU |
| **GPU acceleration benefit** | ❌ Minimal — the operation is memory-bandwidth bound (load → multiply-add → store). GPU parallelism is rarely the bottleneck |
| **Primary bottleneck** | Disk I/O and RAM bus bandwidth while streaming each model's tensors sequentially |

**Practical notes for 8 × SDXL on a desktop:**

- **16 GB RAM:** Tight — the ~19.5 GB peak exceeds physical capacity by ~3.5 GB. Success depends on OS memory compression (zram/zswap); swap space can bridge the gap but will significantly degrade performance during the disk-I/O-heavy model-loading phases.
- **32 GB RAM:** Comfortable — ~19.5 GB peak leaves ~12 GB headroom.
- **GPU offload:** Adds VRAM pressure with negligible speed benefit. CPU is the recommended execution device for this algorithm; the bottleneck is never the arithmetic unit.

---

### 1.2 Consensus Merge (Inverse Distance Weighting)

**Location:** `model_merger/merger.py` → `_consensus_merge()`, `compute_consensus_weights()`

#### How It Works

For every individual scalar element across all N model tensors, the consensus algorithm:

1. Builds an N×N matrix of pairwise absolute differences.
2. Averages each row to get a scalar "average distance" per model: `d_i = mean(|v_i - v_j|  ∀j≠i)`.
3. Normalizes `d_i` to `[0, 1]`.
4. Inverts so that low-distance (consensus) values score high: `score_i = 1 - d_i_normalized`.
5. Applies an exponent for tunable outlier suppression: `w_i = score_i ^ exponent`.
6. Normalizes weights to a probability distribution and computes the weighted sum.

```python
def compute_consensus_weights(values, exponent=4):
    pairwise_distances = torch.abs(values[:, None] - values)  # N×N
    avg_distances = pairwise_distances.mean(dim=1)            # N
    normalized = (avg_distances - min_d) / (max_d - min_d)
    inverted   = 1.0 - normalized
    powered    = inverted ** exponent
    return powered / powered.sum()
```

The result is that parameter values shared by the majority of models receive high weight, while lone outlier values are suppressed exponentially.

#### Original Design Purpose

The consensus algorithm was specifically designed to solve the **tournament-style merging dilution** problem inherent in tools that merge models two or three at a time.

**The tournament dilution problem:**

In naïve sequential merging, you repeatedly combine pairs — and each earlier model gets geometrically diluted with every additional round:

```
Round 1: A + B → AB         (A: 50%,     B: 50%)
Round 2: AB + C → ABC       (A: 25%,     B: 25%,    C: 50%)
Round 3: ABC + D → ABCD     (A: 12.5%,   B: 12.5%,  C: 25%,   D: 50%)
Round 4: ABCD + E → ABCDE   (A: 6.25%,   B: 6.25%,  C: 12.5%, D: 25%,  E: 50%)
Round 5: … + F              (A: 3.125%,  …                              F: 50%)
Round 6: … + G              (A: 1.5625%, …                              G: 50%)
Round 7: … + H              (A: ~0.78%,  …                              H: 50%)
```

In an 8-model sequential merge, model A's contribution halves every round: `0.5^7 ≈ 0.78%`. With 8 models merged in a binary tournament, the **first model contributes only ~0.8% of its original weight** while the last model contributes 50%. The result is overwhelmingly dominated by whichever model happens to enter last in the merge sequence, regardless of the user's intent.

**The consensus solution:**

Rather than merging round by round, the consensus algorithm ingests **all N models simultaneously** and computes adaptive per-element weights based on pairwise distances. Every model has equal standing from the start:

- No tournament rounds, no ordering bias.
- Each model's contribution to any given parameter is determined by how well that model's value agrees with the rest of the ensemble — not by the sequence in which it was processed.
- User-defined weights (see §4.2.2) can further bias contribution toward preferred models without introducing the geometric dilution of sequential merging.

This makes the consensus algorithm uniquely suited to **merging 8 or more models in a single pass** with equal (or explicitly weighted) contributions from every model — something no sequential pairwise approach can achieve.

#### Strengths

| Strength | Detail |
|---|---|
| **Automatic outlier suppression** | Aberrant parameter values from a single fine-tune are naturally down-weighted without user intervention |
| **No user weight required** | Adaptive per-element weights replace manual guessing |
| **Per-element granularity** | Different layers can have radically different consensus patterns; each element is handled independently |
| **Configurable aggressiveness** | `exponent` from 2 (gentle) to 8 (aggressive) gives fine control |

#### Weaknesses

| Weakness | Detail |
|---|---|
| **Quadratic scaling** | The pairwise distance matrix is O(N²) per element; merging many models with large tensors is very slow (the inner `for elem_idx` Python loop is the bottleneck) |
| **Ignores user weights entirely** | The user cannot bias the consensus toward a preferred model; all models are treated equally as inputs to the distance calculation |
| **No sign awareness** | Like LERP, competing signs are not detected or resolved before averaging |
| **Memory pressure for large tensors** | The `stacked` tensor holding all N model copies of a single tensor layer lives in memory during computation |
| **Consensus ≠ better** | If models intentionally diverge in a layer (style specialisation), forcing them toward consensus destroys the diversity that made each model valuable |
| **Python-level loop** | The `for elem_idx in range(num_elements)` loop is unbatched Python; even modest tensors with millions of parameters are extremely slow |

#### Supported Model Count

**Minimum:** 2 models  
**Maximum:** Unlimited — but RAM scales linearly with N (all models must be in memory simultaneously)  
**Sweet spot:** **8–20 models.** The algorithm's per-element pairwise-consensus approach is purpose-built for large ensembles. With only 2 models it degenerates to a simple average — no better than LERP. The outlier-suppression value emerges at N ≥ 3. Statistical robustness peaks at N ≥ 8, where a single divergent model is reliably outvoted. This is the algorithm most directly targeting stmerge's 8+ model use case — at the cost of higher RAM requirements (all N models must reside in memory simultaneously).

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM (current implementation)** | All N models must be resident simultaneously for the per-layer stacking step: **~N × model size**. For 8 × SDXL ≈ **~52 GB** fp16 — exceeds typical desktop RAM |
| **Peak system RAM (vectorized, §4.2.1)** | Still requires N × model size for the stacked tensors, *plus* an intermediate `(N × N × E)` pairwise distance matrix (a compute optimization, not a memory reduction). For a 512×512 conv layer at N=8: `8 × 8 × 262,144 × 4 B ≈ 536 MB` additional per layer — manageable when processing one layer at a time, but the vectorized form requires *more* total RAM than the naïve Python-loop version. Both require per-layer chunking on systems with less than 64 GB RAM. |
| **N-model scaling** | Linear with N — every additional model adds one full model to the RAM requirement |
| **VRAM usage** | Significant if GPU is used for vectorized ops; per-layer stacks for most SDXL layers fit within 8–16 GB VRAM |
| **GPU acceleration benefit** | ✅ Significant — once vectorized (§4.2.1), the dense `(N × N × E)` distance matrix computation and weighted sum map excellently to GPU SIMD parallelism; expected **10–50×** speedup vs. CPU |
| **Primary bottleneck** | RAM capacity (holding all N models simultaneously) and memory bandwidth for the `(N × N × E)` matrix operations |

**Practical notes for 8 × SDXL on a desktop:**

- **16 GB RAM:** Not feasible without streaming layer-by-layer and immediately discarding model tensors after extracting each slice.
- **32 GB RAM:** Marginal — 8 × 6.5 GB = 52 GB fp16 exceeds capacity. Chunked layer-by-layer processing is mandatory, loading and unloading each model's per-layer slice as needed.
- **64 GB+ RAM:** Comfortable for the vectorized implementation with all models resident.
- **GPU (8–16 GB VRAM):** Provides meaningful acceleration once the Python loop is replaced (§4.2.1); most individual SDXL layers fit within GPU VRAM for per-layer batch processing.
- **This algorithm has the highest RAM requirement of all algorithms in stmerge.** The combination of all-models-in-memory and the large intermediate distance matrix makes it the most memory-intensive option by a significant margin.

---

## 2. Industry-Standard Methodologies

### 2.1 LERP – Linear Interpolation

**What it is:** The simplest possible parameter merge. For two models A and B at blending ratio `t`:

```
result = A × (1 - t) + B × t
```

Extended to N models it becomes the weighted sum already implemented in stmerge.

**Relationship to stmerge:** The `weighted_sum` method *is* LERP. The accumulator pattern is simply an efficient multi-model generalisation.

**Key limitation:** Operates in flat Euclidean space. For unit-normalized weight vectors (common in transformer attention and normalization layers), the interpolated point lies *inside* the hypersphere rather than on its surface, causing magnitude shrinkage.

**Designed model count:** 2 models natively (the formula `A × (1-t) + B × t` is defined for a pair). Extended to N models it is identical to stmerge's Weighted Sum — there is no meaningful algorithmic distinction once N > 2.

---

### 2.2 SLERP – Spherical Linear Interpolation

**What it is:** An interpolation that moves along the geodesic (great-circle arc) of a unit hypersphere rather than through its interior. For two vectors `q₀` and `q₁`:

```
result = sin((1-t)θ)/sin(θ) × q₀  +  sin(tθ)/sin(θ) × q₁

where θ = arccos(q₀ · q₁ / (‖q₀‖ · ‖q₁‖))
```

**Why it matters for SD models:** Weight tensors in attention layers and layer-norm parameters are often compared to points on a high-dimensional sphere. LERP shrinks their magnitude to `cos(θ/2)` of the original; SLERP preserves it. For generation quality this can mean:

- Smoother style transitions between merged models.
- Reduced "muddy average" artefacts at the boundary between two strongly different styles.
- Better preservation of each model's identity at the extremes of the interpolation range.

**Limitation:** SLERP is naturally a two-model operation. Extending it to N models (via iterated pairwise SLERP or barycentric SLERP on the Riemannian manifold) is non-trivial and adds complexity. It is also slower than LERP because it requires the trigonometric `arccos` per tensor.

**Community adoption:** Used in many popular merge GUIs (Supermerger, sd-meh) precisely because it produces cleaner blends for style-heavy model pairs.

**Designed model count:** **2 models only.** SLERP is a pairwise operation by definition — the formula requires exactly two vectors and a single interpolation scalar `t`. Extending to N > 2 via iterated pairwise SLERP is possible, but introduces ordering bias: the result depends on which model is combined first. With each additional iteration the magnitude-preservation guarantee also weakens. This is a fundamentally 2-model technique with no native path to 8+ model merges.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | Both models loaded in fp32 simultaneously: **~26 GB** for two SDXL models (~13 GB each) |
| **N-model scaling** | Pairwise by nature — iterated SLERP for N > 2 requires additional passes, keeping peak RAM at ~2× model size per pass |
| **VRAM usage** | Optional; GPU can hold both fp16 models (~13 GB VRAM combined) for 8+ GB GPU acceleration |
| **GPU acceleration benefit** | ⚠️ Moderate — the `arccos` and `sin` trigonometric operations benefit from GPU parallelism, but memory bandwidth loading both fp32 models dominates runtime |
| **Primary bottleneck** | Loading and holding two full fp32 models simultaneously; trigonometric function overhead is secondary |

**Practical notes for two SDXL models on a desktop:**

- **16 GB RAM:** Tight — two fp32 SDXL models ≈ 26 GB. Requires careful streaming (cast one tensor at a time to fp32, compute SLERP, write output, free) to stay within limits.
- **32 GB RAM:** Comfortable — ~26 GB peak with ~6 GB headroom.
- **GPU (8+ GB VRAM):** Useful for the trigonometric math; both models in fp16 (~13 GB combined) can be held on a 16+ GB GPU. On a 24 GB GPU the full fp16 merge stays resident throughout.

---

### 2.3 TIES – Trim, Elect Sign & Merge

**Paper:** *Resolving Interference When Merging Models* (Yadav et al., 2023).

**What it is:** A task-vector-aware merge that operates on *delta weights* (the fine-tuned model minus its base). Three steps:

1. **Trim:** Zero out delta parameters that are below a threshold τ (by magnitude). This removes noise-level perturbations from each fine-tune, reducing interference.
2. **Elect Sign:** For each parameter position, take a majority vote on the sign of the delta across all models. Use the winning sign only.
3. **Merge:** Average only the models that *agree* with the elected sign at each position.

```
Δθ_i = θ_i_finetuned - θ_base         # task vector for model i

# Trim
Δθ_i_trimmed = Δθ_i × (|Δθ_i| > τ)

# Elect sign
sign_elected[k] = sign( Σ Δθ_i_trimmed[k] )

# Merge (only same-sign contributors)
result = θ_base + mean( Δθ_i[k] for i where sign(Δθ_i[k]) == sign_elected[k] )
```

**Why it matters:** Directly addresses the most common failure mode of LERP: sign conflicts. When model A strongly increases a parameter and model B strongly decreases it, LERP averages them to near-zero — destroying both signals. TIES preserves the dominant direction.

**Limitation:** Requires access to the original base model checkpoint. The deltas must be computed before merging, which adds a preprocessing step and extra disk I/O. Also, the trimming threshold τ is a hyperparameter that requires tuning.

**Designed model count:** N ≥ 2 (any number). The sign-election step aggregates sign votes across all N task vectors simultaneously, making TIES natively suited to large ensembles. The statistical robustness of the sign vote actually *improves* with more models — with 8+ models the majority direction is more reliably determined than with just 2. Well-suited for stmerge's 8+ model target, with a streaming implementation keeping peak RAM constant regardless of N.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM (naïve)** | Base model + all N task vectors resident simultaneously: **(N+2) × model size**. For 8 × SDXL ≈ **~65 GB** |
| **Peak system RAM (streaming)** | Base model (resident throughout) + one delta computed and accumulated at a time: **~3 × model size ≈ ~20 GB** for SDXL |
| **N-model scaling** | Constant peak RAM with streaming regardless of N |
| **VRAM usage** | Optional; delta computation and sign-vote operations are amenable to GPU batching |
| **GPU acceleration benefit** | ⚠️ Moderate — sign-vote weighted averaging and masked accumulation map well to GPU tensor ops; the disk I/O for the extra base-model pass dominates total elapsed time |
| **Primary bottleneck** | Disk I/O: must read the base model once plus all N fine-tuned models; the trim and sign-vote steps are computationally cheap by comparison |

**Practical notes for 8 × SDXL on a desktop:**

- **16 GB RAM:** Feasible only with a streaming implementation (compute and discard each delta immediately after accumulating the sign-vote contribution).
- **32 GB RAM:** Comfortable with streaming (~20 GB peak). The naïve all-deltas-in-memory approach requires 64+ GB.
- The base model checkpoint (~6.5 GB fp16) must remain loaded throughout the entire merge; all fine-tuned models are streamed one at a time on top.

---

### 2.4 DARE – Drop And REscale

**Paper:** *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (Yu et al., 2023).

**What it is:** A sparsification technique applied to task vectors before merging. For each model's delta:

1. **Drop:** Randomly set a fraction `p` of delta parameters to zero (dropout on the delta).
2. **Rescale:** Multiply the remaining deltas by `1/(1-p)` to preserve the expected value.
3. **Merge:** Apply standard LERP/TIES on the rescaled sparse deltas.

```
mask_i[k] ~ Bernoulli(1 - p)          # 1 = keep, 0 = drop
Δθ_i_dare[k] = Δθ_i[k] × mask_i[k] / (1 - p)

result = θ_base + Σ (w_i × Δθ_i_dare)
```

**Why it matters:**

- Dramatically reduces parameter interference because two models are unlikely to have non-zero deltas at the same positions simultaneously. With a typical drop rate of `p=0.9`, each model independently retains 10% of its delta values, so the probability that any two models both have a non-zero delta at the same position is only ~1% — nearly eliminating constructive or destructive interference.
- Enables merging of a larger number of models with less quality degradation.
- Particularly effective when the fine-tunes are from the same base model and cover different but potentially overlapping capabilities.

**Limitation:** Requires the base model. Introduces randomness (though a seed can make it reproducible). The drop probability `p` is another hyperparameter (typically 0.9 for language models).

**Designed model count:** N ≥ 2 (any number). DARE is applied independently to each model's task vector before the final merge, so it is model-count-agnostic. Its interference-reduction effectiveness actually *improves* as N increases: at `p = 0.9`, each model retains only 10% of its delta values, so the probability that any two models share a non-zero delta at the same position is ~1% — and the probability of 3+ models colliding at the same position is ~0.1%. More models means fewer collisions and less interference. Well-suited for stmerge's 8+ model target.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | Identical profile to TIES with streaming: base model resident + one sparsified delta at a time ≈ **~3 × model size ≈ ~20 GB** for SDXL |
| **N-model scaling** | Constant peak RAM with streaming regardless of N |
| **VRAM usage** | Same as TIES; the Bernoulli mask and rescaling add negligible memory on top |
| **GPU acceleration benefit** | ⚠️ Moderate — `torch.bernoulli()` sampling and element-wise masking are GPU-friendly; benefit mirrors TIES |
| **Primary bottleneck** | Same as TIES: base model disk I/O + N model passes; the drop-and-rescale step is nearly free in both compute and memory |

**Practical notes for 8 × SDXL on a desktop:**

- Same RAM profile as TIES: streaming implementation keeps peak at ~20 GB, fitting comfortably in 32 GB.
- The sparsification mask tensor is the same size as one delta (~6.5 GB fp16) and is held only briefly before the delta is accumulated and discarded.
- Setting `dare_seed` ensures bitwise-reproducible results with no change to the memory profile.

---

### 2.5 Passthrough / Frankenmerging

**What it is:** Rather than interpolating parameter values, Frankenmerging transplants entire *layers* (transformer blocks, attention heads, residual stages) from different models. The "recipe" specifies which block in the output comes from which source model.

**Example recipe:**

```
layers 0-11  : model_A   (artistic style)
layers 12-23 : model_B   (photorealistic detail)
layer  24    : model_C   (specific anatomy training)
```

No arithmetic is performed on the weights; layers are just copied verbatim into the target architecture.

**Why it matters:**

- Completely preserves specialised capabilities from each contributor without dilution.
- Allows mixing models from different fine-tune lineages as long as the base architecture is identical.
- Popular for creating "Frankenstein" models with complementary strengths (hence the name).
- Used in community tools like MergeKit's `passthrough` strategy and in creating large Mixtral-style mixture-of-experts structures.

**Limitations:**

- The resulting model can be incoherent at layer boundaries; neighbouring layers that were trained together suddenly must co-operate with alien weights.
- Requires detailed knowledge of which layers encode which capabilities (not always obvious for Stable Diffusion U-Net blocks).
- No smooth blending; it's all-or-nothing per layer.
- Cannot be done in the current stmerge architecture without loading entire models and re-assembling the state dict layer by layer.

**Designed model count:** N ≥ 1 (any number of source models). A layer recipe can draw from any number of distinct source checkpoints; there is no inherent algorithmic limit. In practice, recipes typically reference 2–8 source models, with more sources increasing the likelihood of incoherence at layer boundaries. Unlike parameter-averaging algorithms, each source model contributes a discrete set of layers rather than a fractional blend — so this scales to many sources without dilution concerns.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | ~2 × model size fp16 ≈ **~13 GB** (one source model + the output state dict being assembled). Only models referenced by active recipe steps need to be resident simultaneously |
| **N-model scaling** | Source models can be loaded and unloaded per recipe step; peak RAM is determined by the maximum number of distinct source models needed at the same time (usually 1–2) |
| **VRAM usage** | None required — the operation is a pure memory copy with no arithmetic |
| **GPU acceleration benefit** | ❌ Minimal — layer copying is a memory-to-memory transfer with no floating-point computation; GPU provides no advantage |
| **Primary bottleneck** | Disk I/O to read and slice the relevant layer groups from each source checkpoint |

**Practical notes for 8 × SDXL on a desktop:**

- **16 GB RAM:** Feasible — only one source model (~6.5 GB fp16) plus the output dict (~6.5 GB fp16) need to be in memory at any time.
- **GPU:** Not beneficial for this operation; all work is CPU-side memory management.
- The most RAM-efficient multi-source algorithm when the recipe references at most one or two distinct source models per step.

---

## 3. Comparison Matrix

| Feature | **stmerge Weighted Sum** | **stmerge Consensus** | **LERP** | **SLERP** | **TIES** | **DARE** | **Passthrough** |
|---|---|---|---|---|---|---|---|
| Mathematical basis | Linear combination | Inverse-distance IDW | Linear interpolation | Geodesic interpolation | Task-vector sign election | Stochastic sparsification | Layer transplantation |
| Base model required | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Supported model count** | **N ≥ 1 (designed for 8+)** | **N ≥ 2 (designed for 8+)** | **N = 2 natively; N ≥ 2 via extension** | **⚠️ N = 2 only (pairwise)** | **N ≥ 2 (any; improves with N)** | **N ≥ 2 (any; improves with N)** | **N ≥ 1 (any)** |
| Respects user weights | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A |
| Sign-conflict resolution | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes | Partial | N/A |
| Magnitude preservation | ❌ Shrinks | ❌ Shrinks | ❌ Shrinks | ✅ Yes | Partial | ✅ (rescaled) | ✅ Yes |
| Outlier suppression | ❌ No | ✅ Yes | ❌ No | ❌ No | ✅ Trim step | ✅ Drop step | N/A |
| Memory efficiency | ✅ Excellent | ⚠️ Moderate | ✅ Excellent | ✅ Excellent | ⚠️ Needs deltas | ⚠️ Needs deltas | ⚠️ Moderate |
| Speed | ✅ Fast | ❌ Very slow | ✅ Fast | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Moderate | ✅ Fast |
| Preserves layer identity | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| Deterministic | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ (random) | ✅ Yes |
| Implementation complexity | ⭐ Simple | ⭐⭐⭐ Complex | ⭐ Simple | ⭐⭐ Moderate | ⭐⭐⭐ Complex | ⭐⭐ Moderate | ⭐⭐ Moderate |
| **Peak system RAM** (8 × SDXL) | ~19.5 GB | ~52 GB | ~19.5 GB (2-model fp16, same as Weighted Sum; ~26 GB in fp32) | ~26 GB (2-model fp32) | ~20 GB (streaming) | ~20 GB (streaming) | ~13 GB (fp16) |
| **GPU acceleration benefit** | ❌ Minimal | ✅ Significant | ❌ Minimal | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Moderate | ❌ Minimal |

---

## 4. Suggestions for Existing Algorithms

### 4.1 Weighted Sum Improvements

#### 4.1.1 Add SLERP as an Option Within the Same Pipeline

SLERP is a natural upgrade to LERP for two-model blends and is the most-requested feature in the SD merging community. It can be added as a per-tensor fallback that activates only when the tensors are floating-point and the dot product is not near ±1 (i.e., the vectors are meaningfully non-parallel):

```python
def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    # Flatten to compute dot product
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()
    
    # Compute norms once and reuse to avoid redundant computation
    v0_norm = v0_flat.norm().clamp(min=eps)
    v1_norm = v1_flat.norm().clamp(min=eps)
    dot = torch.dot(v0_flat / v0_norm, v1_flat / v1_norm).clamp(-1.0, 1.0)
    
    # Fall back to LERP when vectors are nearly parallel or anti-parallel
    if dot.abs() > 1.0 - eps:
        return torch.lerp(v0.float(), v1.float(), t).reshape(v0.shape)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    result = (torch.sin((1.0 - t) * theta) / sin_theta) * v0.float() + \
             (torch.sin(t * theta)         / sin_theta) * v1.float()
    return result.reshape(v0.shape)
```

This requires the accumulator pattern to be modified for two-model blends (as SLERP is naturally pairwise), or an iterated approach for N > 2 models.

#### 4.1.2 Weight Normalization Option

Currently stmerge warns if weights do not sum to 1 but allows it. Adding an opt-in normalization flag (`--normalize-weights`) would prevent accidental scale drift, especially for users new to the tool:

```python
if normalize_weights:
    total = sum(e.weight for e in model_entries)
    if abs(total - 1.0) > 1e-3:
        model_entries = [replace(e, weight=e.weight/total) for e in model_entries]
```

#### 4.1.3 Per-Block Weight Support (Block Merging)

The `ROADMAP.md` already lists block merging as a future feature. The weighted sum accumulator is the natural foundation for it. The key change is replacing a single scalar `weight` per model with a dict mapping layer-group names to weights:

```python
# Instead of:  entry.weight (scalar)
# Support:     entry.block_weights = {'input_blocks': 0.7, 'output_blocks': 0.3, ...}
```

This aligns with the popular "MBW" (Merge Block Weighted) technique widely used in the SD community.

#### 4.1.4 Selective Key Merging

Allow users to specify which key prefixes are merged vs. copied from a designated "base" model:

```json
{
  "selective_merge": {
    "merge_prefixes":  ["model.diffusion_model."],
    "copy_from_base":  ["first_stage_model.", "cond_stage_model."]
  }
}
```

This is very useful when you want to merge the U-Net but keep the VAE/text-encoder from a specific model unchanged.

---

### 4.2 Consensus Merge Improvements

#### 4.2.1 Vectorize the Inner Loop (Critical Performance Fix)

The current implementation iterates over every scalar element in Python:

```python
for elem_idx in range(num_elements):          # ← Python loop = slow
    values = stacked_flat[:, elem_idx]
    weights = compute_consensus_weights(values, exponent=exponent)
    merged_flat[elem_idx] = (values * weights).sum()
```

For a single 512×512 convolution weight this is 262,144 Python iterations. This should be replaced with a fully batched PyTorch operation:

```python
# stacked_flat: (num_models, num_elements)
# Compute all pairwise distances at once.
# NOTE: the intermediate (N, N, E) tensor can be large; for N=8 models and E=262,144
# elements this is ~16 GB in float32. Process in element-wise chunks if memory is tight.
diff = stacked_flat.unsqueeze(2) - stacked_flat.unsqueeze(1)   # (N, N, E)
avg_dist = diff.abs().mean(dim=1)                               # (N, E)

min_d = avg_dist.min(dim=0).values
max_d = avg_dist.max(dim=0).values
range_d = (max_d - min_d).clamp(min=1e-10)

normalized = (avg_dist - min_d) / range_d                      # (N, E)
weights    = (1.0 - normalized) ** exponent                    # (N, E)
weights    = weights / weights.sum(dim=0, keepdim=True)        # normalize

merged_flat = (stacked_flat * weights).sum(dim=0)              # (E,)
```

This replaces the Python loop with a single vectorized operation and can achieve **100–1000× speedup** on both CPU and GPU.

#### 4.2.2 Honour User Weights as Priors

Currently the consensus algorithm ignores user-provided weights entirely. A soft prior could blend the user's preference with the data-driven consensus weights:

```python
# user_weight_prior: (N,) tensor of user-specified weights
# consensus_weights: (N, E) data-driven weights
prior = user_weight_prior[:, None].expand_as(consensus_weights)
blended = alpha * prior + (1 - alpha) * consensus_weights
blended = blended / blended.sum(dim=0, keepdim=True)
```

The `alpha` parameter (0 = fully consensus, 1 = fully user-weighted) could be exposed as `--consensus-prior-strength`.

#### 4.2.3 Median as a Faster Consensus Proxy

For use cases where full inverse-distance weighting is too slow, the **element-wise median** is an O(N log N) approximation that also suppresses outliers effectively:

```python
# stacked: (N, *shape) → result: (*shape)
result[key] = stacked.median(dim=0).values
```

The median is outlier-resistant by definition (a single outlier among N models cannot pull the result beyond the median rank) and requires no hyperparameter tuning. It could be offered as `--merge-method median`.

#### 4.2.4 Separate "Consensus" into its Own Config Constant

The consensus merge currently notes `"User-provided weights are ignored in consensus mode"` in a console warning but the behavior is also baked into several places. Moving this to a dedicated manifest field and documenting it clearly in the manifest schema would reduce user confusion.

---

## 5. Recommended New Algorithms

### 5.1 SLERP (Spherical Linear Interpolation) — High Priority

**Value:** The most widely-requested SD merge algorithm, solves magnitude shrinkage inherent in LERP.

**Model count: Exactly 2.** SLERP is a pairwise operation by definition — the formula requires exactly two vectors and one interpolation scalar `t`. The implementation must enforce this constraint. An "iterated SLERP" variant for N > 2 is listed separately (§6, lowest priority); it introduces ordering bias and progressively weakens the magnitude-preservation guarantee with each additional pair combined. **This algorithm addresses a 2-model use case and has no native path to 8+ model merges.**

**Implementation sketch:**

```python
MERGE_METHOD_SLERP = 'slerp'
```

Add `_slerp_merge()` to `merger.py` that:

1. Validates exactly two models are provided (SLERP is pairwise by nature); raise a descriptive error for N > 2 suggesting iterated SLERP or LERP instead.
2. Loads both models.
3. For each tensor key, calls `slerp(t, tensor_a, tensor_b)` where `t` is derived from the relative weights.
4. Falls back to LERP for non-floating-point tensors and for near-parallel vectors.

**CLI flag:** `--merge-method slerp`

**Manifest field:** `"merge_method": "slerp"` (already supported by the manifest schema)

---

### 5.2 TIES Merge — Medium Priority

**Value:** Addresses sign-conflict interference, which is the most significant quality problem with LERP on models fine-tuned from the same base.

**Model count:** N ≥ 2 (any number). The sign-vote and masked-average steps natively aggregate across all N task vectors simultaneously — there is no pairwise constraint. With more models the sign vote becomes more statistically robust, so TIES benefits from having 8+ models rather than just 2. **Well-suited for stmerge's 8+ model target** when a base model is available; the streaming implementation keeps peak RAM constant regardless of N.

**Requirements:** Access to the original base model checkpoint.

**Implementation sketch:**

Add `"base_model"` field to `MergeManifest`:

```python
@dataclass
class MergeManifest:
    ...
    base_model: Optional[str] = None   # path to base checkpoint for TIES/DARE
    ties_trim_ratio: float = 0.2       # fraction of smallest deltas to zero
```

Add `_ties_merge()` to `merger.py`:

```python
def _ties_merge(model_entries, base_path, device, trim_ratio=0.2):
    base, _ = load_model(base_path, device=device)
    
    # Compute task vectors
    deltas = []
    for entry in model_entries:
        model, _ = load_model(entry.path, device=device)
        delta = {k: model[k].float() - base[k].float() for k in base}
        
        # Trim: zero out small deltas by magnitude
        for k, d in delta.items():
            threshold = d.abs().quantile(trim_ratio)
            delta[k] = d * (d.abs() >= threshold)
        
        deltas.append((delta, entry.weight))
        del model
        gc.collect()
    
    # Elect sign by weighted vote
    result = {}
    for key in base:
        stacked = torch.stack([d[key] for d, _ in deltas], dim=0)
        weights = torch.tensor([w for _, w in deltas])
        
        # Broadcast weights over all parameter dimensions: (N,) → (N, 1, 1, ...)
        w_bc = weights.view(-1, *([1] * (stacked.dim() - 1)))
        
        # Weighted sign vote
        elected_sign = torch.sign( (stacked.sign() * w_bc).sum(dim=0) )
        
        # Merge only same-sign contributors
        mask = (stacked.sign() == elected_sign.unsqueeze(0)) | (stacked == 0)
        merged_delta = (stacked * mask * w_bc).sum(dim=0)
        merged_delta /= (mask.float() * w_bc).sum(dim=0).clamp(min=1e-8)
        
        result[key] = base[key].float() + merged_delta
    
    del base, deltas
    gc.collect()
    return result
```

**CLI flag:** `--merge-method ties --base-model path/to/base.safetensors --ties-trim-ratio 0.2`

---

### 5.3 DARE (Drop And REscale) — Medium Priority

**Value:** Reduces parameter interference when merging many fine-tunes from the same base. Pairs naturally with TIES.

**Model count:** N ≥ 2 (any number; effectiveness improves with N). DARE is applied as an independent per-model preprocessing step — each model's task vector is sparsified in isolation, regardless of how many other models exist. This independence is what makes it scale well: at `p = 0.9`, each model retains only 10% of its delta values. Because each model's sparse mask is drawn independently, the probability that any two models share a non-zero delta at the same position is ~1%. With three or more models all colliding at the same position, the probability drops to ~0.1%. More models means fewer collisions and less interference. **The more models you merge, the better DARE's interference reduction works.** Well-suited for stmerge's 8+ model target when a base model is available.

**Requirements:** Base model.

**Implementation sketch:**

Add `_dare_preprocess()` as a pre-merge step that sparsifies task vectors:

```python
def _dare_preprocess(delta: Dict[str, torch.Tensor],
                     drop_rate: float = 0.9,
                     seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Apply DARE sparsification to a task-vector dict."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    
    result = {}
    for key, d in delta.items():
        mask = torch.bernoulli(
            torch.full_like(d, 1.0 - drop_rate), generator=rng
        )
        result[key] = d * mask / (1.0 - drop_rate)   # rescale
    return result
```

DARE can be composed with TIES or plain LERP on task vectors. Expose as:

```
--merge-method dare-lerp  --dare-drop-rate 0.9 --base-model ...
--merge-method dare-ties  --dare-drop-rate 0.9 --ties-trim-ratio 0.2 --base-model ...
```

**Manifest fields:**

```python
dare_drop_rate: float = 0.9
dare_seed: Optional[int] = None
```

---

### 5.4 Passthrough / Frankenmerging — Low-to-Medium Priority

**Value:** Produces models with strongly preserved characteristics from each source by copying whole layers rather than averaging parameter values.

**Model count:** N ≥ 1 (any number of source models). A layer recipe can reference any number of distinct source checkpoints. Unlike parameter-averaging algorithms, each source model contributes a discrete set of layers rather than a fractional blend — there are no dilution concerns and no pairwise constraint. In practice, recipes typically reference 2–8 source models; more sources increase the likelihood of incoherence at layer boundaries.

**Architecture mapping for Stable Diffusion U-Net:**

| Block group | Key prefix |
|---|---|
| Input blocks (encoder) | `model.diffusion_model.input_blocks.{0-11}.*` |
| Middle block | `model.diffusion_model.middle_block.*` |
| Output blocks (decoder) | `model.diffusion_model.output_blocks.{0-11}.*` |
| VAE | `first_stage_model.*` |
| Text encoder (SD 1.x) | `cond_stage_model.*` |
| Text encoder (SDXL) | `conditioner.*` |

**Implementation sketch:**

Add a manifest field `"layer_recipe"` that maps ranges of block indices to source model paths:

```json
{
  "models": [
    {"path": "artistic.safetensors",     "weight": 1.0, "index": 0},
    {"path": "photorealistic.safetensors", "weight": 1.0, "index": 1}
  ],
  "merge_method": "passthrough",
  "layer_recipe": [
    {"model_index": 0, "blocks": "input_blocks.0-5"},
    {"model_index": 1, "blocks": "input_blocks.6-11"},
    {"model_index": 0, "blocks": "middle_block"},
    {"model_index": 1, "blocks": "output_blocks.0-11"}
  ]
}
```

Add `_passthrough_merge()` to `merger.py` that:

1. Parses the `layer_recipe`.
2. Loads each model only as many times as it appears in the recipe (or once with memory mapping).
3. Copies the specified key subsets into the result dict.

---

### 5.5 Weighted Median — Low Priority (Fast Approximation)

**Value:** A fast, hyperparameter-free approximation to consensus merging.

**Model count:** N ≥ 2 (any number; most useful for N ≥ 4). With N = 2, the median is equivalent to the midpoint — identical to LERP at t = 0.5 with no outlier benefit. Outlier-resistance first emerges at N = 3, and becomes practically meaningful at N ≥ 4 (where the median can genuinely exclude an extreme value). **Best suited for 8+ model merges**, where it serves as a faster alternative to Consensus IDW: the sort-based median is O(N log N) per element vs. O(N²) for the full pairwise-distance consensus. The RAM tradeoff is the same as Consensus — all N models must reside in memory simultaneously, so RAM requirements scale linearly with N.

For each parameter position, take the **weighted median** (or unweighted median for equal-weight merges) across all models. The median is inherently outlier-resistant: a single model with an extreme value cannot pull the result beyond the median rank.

```python
def _median_merge(model_entries, device='cpu'):
    # Similar to consensus_merge but replace the weighting loop with:
    merged_flat = stacked_flat.median(dim=0).values
```

This is O(N log N) per element vs O(N²) for consensus, and requires no exponent hyperparameter.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | All N models must be stacked in memory simultaneously: **N × model size**. For 8 × SDXL ≈ **~52 GB** fp16 |
| **N-model scaling** | Linear with N — every additional model adds one full model worth of RAM |
| **VRAM usage** | High if GPU is used for the sort; an 8-model SDXL layer stack can exceed GPU VRAM — process one layer at a time |
| **GPU acceleration benefit** | ✅ Significant — `torch.median()` along the model dimension is highly parallelizable; GPU can accelerate the sort-based median by 10–20× vs. CPU for large tensors |
| **Primary bottleneck** | RAM capacity: holding all N models simultaneously is the dominant constraint; median computation (O(N log N) sort per element) is fast once data is resident |

**Practical notes for 8 × SDXL on a desktop:**

- **16 GB RAM:** Not feasible without chunking — load one tensor key from all N models, compute the median for that key, discard all per-model slices before loading the next key.
- **32 GB RAM:** Possible with careful chunked streaming (process one tensor key group at a time).
- **GPU (16+ GB VRAM):** Can accelerate per-layer median when layers are processed individually; no GPU can hold all 8 SDXL models simultaneously (~52 GB fp16).
- Weighted median (user-defined per-model weights) requires an O(N log N) weighted sort per element — moderately more expensive than unweighted median but still far faster than the full consensus IDW computation.

---

### 5.6 Gradient-Free Task Arithmetic — Future Consideration

**Concept:** *Task arithmetic* (Ilharco et al., 2022) represents each fine-tune as a task vector `τ_i = θ_finetuned_i - θ_base` and merges by adding scaled task vectors back to the base:

```
θ_merged = θ_base + λ₁τ₁ + λ₂τ₂ + ... + λₙτₙ
```

The coefficients `λ_i` act as capability dials — setting `λ_i = 0` removes a capability, setting `λ_i > 1` amplifies it.

This is the theoretical foundation for TIES and DARE. Implementing task arithmetic as a first-class operation (not just a preprocessing step for TIES/DARE) would allow users to arithmetically compose and decompose capabilities.

**Model count:** N ≥ 1 (any number). Each task vector is independently scaled by `λ_i` and accumulated into the result; the operation is additive and scales to any N with constant peak RAM when streaming. **Well-suited for stmerge's 8+ model target** — the accumulator pattern applies directly to task vectors the same way it applies to full model weights.

**Note:** This requires the base model and is most powerful when combining models fine-tuned from the same base on distinct tasks.

#### Hardware Impact

| Metric | Value |
|---|---|
| **Peak system RAM** | Base model resident + one task vector accumulated at a time: **~3 × model size ≈ ~20 GB** for SDXL with streaming |
| **N-model scaling** | Constant peak RAM with streaming regardless of N; each `λ_i × τ_i` is accumulated into the result dict and the source delta is freed |
| **VRAM usage** | Optional; scaled addition of task vectors maps trivially to GPU ops |
| **GPU acceleration benefit** | ❌ Minimal — task arithmetic is scalar-multiply-and-add (structurally identical to weighted sum / LERP); the bottleneck is memory bandwidth, not compute |
| **Primary bottleneck** | Disk I/O: base model read + N fine-tuned model reads to compute and stream each task vector |

**Practical notes for 8 × SDXL on a desktop:**

- Same RAM profile as TIES/DARE with streaming: ~20 GB peak, fits comfortably in 32 GB.
- The `λ_i` coefficient scaling is the same operation as weighted sum; no additional memory pressure beyond the base model and one streaming delta.
- GPU provides no meaningful advantage over CPU for this algorithm.

---

## 6. Implementation Priority

| Priority | Algorithm | Model Count | Effort | Impact | Notes |
|---|---|---|---|---|---|
| 🔴 **High** | Vectorized consensus (performance fix) | N ≥ 2 (designed for 8+) | Low | Very High | Replaces Python loop with batched tensor ops; no API change |
| 🔴 **High** | SLERP | **N = 2 only** ⚠️ | Medium | High | Most-requested in community; adds `slerp` method; pairwise — no path to 8+ models natively |
| 🟡 **Medium** | Weighted median | N ≥ 2 (best at 8+) | Low | Medium | Adds `median` method; no base model required; faster Consensus approximation |
| 🟡 **Medium** | Per-block weights (MBW) | N ≥ 1 (any) | Medium | High | Unlocks the most nuanced community merge workflows |
| 🟡 **Medium** | TIES | N ≥ 2 (any; improves with N) | High | High | Requires base model; solves sign conflicts; scales to 8+ |
| 🟡 **Medium** | DARE + TIES | N ≥ 2 (any; improves with N) | High | High | Pairs with TIES for stronger interference reduction; scales to 8+ |
| 🟢 **Low** | Passthrough / Frankenmerging | N ≥ 1 (any) | Medium | Medium | Niche but powerful; needs manifest schema extension |
| 🟢 **Low** | Task arithmetic | N ≥ 1 (any) | High | Medium | Foundation for TIES/DARE; needs base model; scales to 8+ |
| 🟢 **Low** | Iterated SLERP for N > 2 | N ≥ 3 (order-dependent) ⚠️ | Medium | Low | Useful for quality; pairwise ordering bias weakens guarantees |

---

## 7. Summary

### What stmerge does well

- **Memory-efficient accumulator pattern** is the right architecture. No other community tool handles 8+ large models as gracefully.
- **Consensus merge is a genuinely novel idea** not commonly found in other SD merge tools. The inverse-distance-weighting approach is well-reasoned.
- **Clean separation of concerns** makes it straightforward to add new merge methods in `merger.py` without touching the manifest, loader, or saver modules.

### Algorithm suitability for stmerge's 8+ model use case

| Algorithm | 8+ Model Suitability | Notes |
|---|---|---|
| **Weighted Sum** | ✅ Designed for 8+ | Core strength of stmerge; constant RAM regardless of N |
| **Consensus IDW** | ✅ Designed for 8+ | Best outlier suppression; needs vectorization and 64 GB RAM or chunked streaming |
| **TIES** | ✅ Scales to 8+ | Sign-vote improves with more models; requires base model |
| **DARE** | ✅ Scales to 8+ (improves with N) | Interference reduction improves as N grows; requires base model |
| **Task Arithmetic** | ✅ Scales to 8+ | Additive; same RAM profile as Weighted Sum with streaming |
| **Weighted Median** | ✅ Best at 8+ | Outlier resistance meaningful only at N ≥ 4; same RAM needs as Consensus |
| **Passthrough** | ⚠️ Any N (no dilution) | Layer-level granularity; no parameter blending or dilution |
| **LERP** | ⚠️ N = 2 natively | Identical to Weighted Sum when extended to N > 2 |
| **SLERP** | ❌ N = 2 only | Fundamentally pairwise; no native 8+ model path |

### Critical gaps

1. **No magnitude preservation:** Both current algorithms let tensor magnitudes drift during blending. SLERP would fix this for two-model blends.
2. **No sign-conflict resolution:** Competing fine-tunes can cancel each other out. TIES is the established solution.
3. **Consensus is unusably slow for large models:** The per-element Python loop must be replaced with vectorized tensor operations before consensus can be used on production-size SDXL models.
4. **No base-model workflows:** TIES and DARE — the two algorithms with the strongest theoretical basis for interference reduction — both require a base model. Adding `base_model` to the manifest schema would unlock both.

### Recommended immediate actions

1. **Vectorize `_consensus_merge()`** — replace the `for elem_idx` Python loop with the batched tensor implementation shown in §4.2.1. This is a performance-only change with no API impact.
2. **Add SLERP** — two-model spherical interpolation; add `slerp` as a value for `merge_method` in the manifest schema.
3. **Add `base_model` to `MergeManifest`** — schema addition only; enables TIES and DARE in a follow-up.
4. **Implement TIES** — most impactful algorithmic addition after SLERP; directly addresses the dominant quality failure mode of LERP.

### Hardware Quick Reference

The memory requirements and GPU utility vary significantly across algorithms. Here is a quick reference for typical desktop hardware (16–32 GB RAM, optional GPU with 8–24 GB VRAM):

| System RAM | Feasible Algorithms |
|---|---|
| **16 GB** | Passthrough ✅, Weighted Sum ⚠️ (~19.5 GB peak exceeds limit; needs OS compression/swap — degrades performance) |
| **32 GB** | Weighted Sum ✅, SLERP ✅ (2-model), TIES ✅ (streaming), DARE ✅ (streaming), Task Arithmetic ✅ (streaming) |
| **64 GB+** | All algorithms, including Consensus IDW and Weighted Median without per-layer chunking |

**Key architectural advantage of the accumulator pattern:** The Weighted Sum accumulator keeps peak RAM at ~2× model size regardless of N models merged. Consensus and Weighted Median both require all N models in memory simultaneously (~N× model size), making them impractical without 64 GB+ RAM or chunked streaming.

**On tournament dilution and the consensus algorithm:** The Consensus IDW algorithm is the only algorithm in stmerge designed for true N-model simultaneous merging without tournament-style weight dilution. Sequential pairwise merging dilutes early models exponentially (with 8 models, the first model contributes less than 1% of its original weight if merged in a binary tournament). Consensus avoids this entirely by computing per-element adaptive weights across all N models in a single pass. For memory-constrained systems where Consensus is not feasible, Weighted Sum with equal weights is the next best option — it at least ensures each model contributes the same nominal weight, even if it cannot adaptively suppress outliers.

**GPU acceleration priority for desktop users:**

1. Vectorize Consensus (§4.2.1) and run on GPU — highest impact; 10–50× speedup expected.
2. SLERP on GPU — moderate benefit for two-model blends.
3. TIES/DARE sign-vote and accumulation steps on GPU — moderate benefit.
4. Weighted Sum on GPU — lowest priority; memory-bandwidth bound, <2× benefit.

---

## 8. Hardware Considerations

Stable Diffusion XL checkpoints are large: approximately **6.5 GB on disk in fp16**, expanding to **~13 GB in fp32** during computation. Merging 8+ of them on a consumer desktop or laptop requires careful attention to peak RAM, VRAM availability, and which operations benefit from GPU acceleration.

### 8.1 Hardware Target

stmerge is designed to run on:

- **CPU:** Always available; all algorithms work on CPU-only systems.
- **System RAM:** 16–64 GB typical on modern desktops and workstations. The most important hardware resource for large model merges.
- **GPU (optional):** The primary target platforms are:
  - **RTX 3090 Laptop (Mobile)** — 16 GB GDDR6 VRAM, ~448 GB/s theoretical peak GPU memory bandwidth (~315–380 GB/s effective). Paired with 16–32 GB DDR5 system RAM.
  - **RTX 3090 Desktop** — 24 GB GDDR6X VRAM, ~936 GB/s theoretical peak GPU memory bandwidth (~650–840 GB/s effective). Paired with 32–64 GB DDR4/DDR5 system RAM.

  Other NVIDIA GPUs with 8–24 GB VRAM (RTX 3080, 4080, 4090, A10) follow similar patterns; the two RTX 3090 variants bracket the realistic range for this workload.

> **Key insight:** The bottleneck for most algorithms is **memory bandwidth and capacity**, not floating-point compute. Loading 6.5 GB of model weights from disk or RAM and performing element-wise operations is limited by how fast data can move — not by the speed of the arithmetic units.

---

### 8.2 Algorithm Hardware Impact Summary

The table below sizes each algorithm for **8 × SDXL models** (each 6.5 GB fp16 on disk, ~13 GB fp32 in RAM):

| Algorithm | Peak System RAM | VRAM Usage | GPU Acceleration Benefit | Practical Fit for 8 × SDXL |
|---|---|---|---|---|
| **Weighted Sum (stmerge)** | ~19.5 GB | None | ❌ Minimal | ✅ 32 GB RAM / ⚠️ Tight on 16 GB |
| **Consensus IDW (stmerge)** | ~52 GB (all N loaded) | High if GPU-vectorized | ✅ Significant (once vectorized) | ❌ Requires 64 GB RAM or chunked streaming |
| **LERP** | ~26 GB (2-model fp32) | Optional | ❌ Minimal | ⚠️ N/A for 8-model natively; 32 GB for 2-model |
| **SLERP** | ~26 GB (2-model fp32) | Optional 8+ GB | ⚠️ Moderate | ⚠️ 2-model only; 32 GB recommended |
| **TIES** | ~20 GB (streaming) | Optional | ⚠️ Moderate | ✅ 32 GB with streaming implementation |
| **DARE** | ~20 GB (streaming) | Optional | ⚠️ Moderate | ✅ 32 GB with streaming implementation |
| **Passthrough** | ~13 GB (fp16) | None | ❌ Minimal | ✅ 16 GB RAM |
| **Weighted Median** | ~52 GB (all N loaded) | High if GPU | ✅ Significant | ❌ Requires 64 GB or chunked streaming |
| **Task Arithmetic** | ~20 GB (streaming) | Optional | ❌ Minimal | ✅ 32 GB with streaming implementation |

---

### 8.3 RAM Requirements by System Configuration

#### 16 GB System RAM

| Algorithm | Feasibility | Notes |
|---|---|---|
| Weighted Sum | ⚠️ Tight | ~19.5 GB peak exceeds 16 GB capacity by ~3.5 GB; requires OS memory compression or swap; may succeed with no other heavy processes running, but swap will slow disk-bound model-loading phases significantly |
| Passthrough | ✅ Comfortable | ~13 GB peak (fp16 source + output dict) |
| LERP / SLERP | ⚠️ Marginal | Two fp32 SDXL models ≈ 26 GB; requires per-tensor streaming to stay within limit |
| TIES / DARE / Task Arithmetic | ❌ Requires streaming | Base + streaming delta ≈ 20 GB; exceeds 16 GB without aggressive per-layer chunking |
| Consensus / Weighted Median | ❌ Not feasible | 8 × 6.5 GB = 52 GB minimum — far exceeds capacity even with chunking |

#### 32 GB System RAM

| Algorithm | Feasibility | Notes |
|---|---|---|
| Weighted Sum | ✅ Comfortable | ~19.5 GB peak; ~12 GB headroom |
| Passthrough | ✅ Comfortable | Well within limits |
| LERP / SLERP | ✅ Comfortable (2-model) | ~26 GB peak for two fp32 models |
| TIES / DARE / Task Arithmetic | ✅ With streaming | Streaming implementation keeps peak at ~20 GB |
| Consensus / Weighted Median | ❌ Not feasible | 52 GB minimum exceeds capacity; chunked per-layer streaming is the only path |

#### 64 GB+ System RAM

All algorithms are feasible without special memory management. Consensus IDW and Weighted Median can load all 8 SDXL models simultaneously (~52 GB fp16) with room to spare for the intermediate distance matrix and output state dict.

---

### 8.4 GPU Acceleration

GPU acceleration is most beneficial for **compute-heavy, memory-resident operations** where data can stay on the GPU between steps.

| Algorithm | GPU Benefit Scenario |
|---|---|
| **Consensus IDW (vectorized)** | Dense `(N × N × E)` pairwise matrix computation is highly SIMD-parallel; expected 10–50× CPU speedup |
| **Weighted Median** | `torch.median()` across N stacked tensors is highly parallelizable; 10–20× CPU speedup expected |
| **SLERP** | `arccos` and `sin` per tensor element benefit from GPU parallelism; memory-bound for very large tensors |
| **TIES / DARE** | Sign-vote, masked averaging, and Bernoulli sampling all map well to GPU ops; moderate benefit |
| **Weighted Sum** | Simple multiply-add; memory-bandwidth bound; GPU provides <2× benefit |
| **Passthrough** | Memory copy only — no arithmetic; GPU provides no benefit |

**VRAM guidance for target GPU platforms (8 × SDXL fp16 workload):**

| GPU | VRAM | Weighted Sum | SLERP (2-model) | Consensus IDW (per-layer) | TIES / DARE | Notes |
|---|---|---|---|---|---|---|
| **RTX 3090 Laptop** | 16 GB GDDR6 | ⚠️ CPU preferred | ✅ GPU (fp16 only) | ⚠️ Per-layer GPU | ⚠️ Per-layer GPU | Accumulator (~13 GB fp32) + one model (6.5 GB fp16) = 19.5 GB > 16 GB; SLERP in fp16 (13 GB) fits with ~3 GB headroom |
| **RTX 3090 Desktop** | 24 GB GDDR6X | ✅ GPU | ✅ GPU (fp16) | ⚠️ Per-layer GPU | ✅ GPU | Accumulator + one model (19.5 GB) fits in 24 GB with 4.5 GB headroom; TIES base+delta (13 GB) fits comfortably |

> **Important:** For algorithms that require all N models simultaneously (Consensus, Weighted Median), even a 24 GB GPU cannot hold all 8 SDXL models (~52 GB fp16 combined). These algorithms **must process one layer at a time**, moving each layer's data to GPU, computing the result, and transferring back before loading the next layer.

#### CPU Offloading Analysis

In stmerge's architecture, all model weights reside in system RAM by default; the GPU is used only for the active layer's computation. **"CPU offload"** refers to the percentage of model data that must reside in system RAM rather than VRAM during a merge — a higher percentage means more data must shuttle between RAM and VRAM (or stay on CPU entirely), reducing the GPU's effective acceleration.

| Algorithm | RTX 3090 Laptop (16 GB) CPU Offload | RTX 3090 Desktop (24 GB) CPU Offload | Notes |
|---|---|---|---|
| **Weighted Sum** | ~100% | ~Low | Laptop: accumulator in CPU RAM; layer-by-layer GPU compute bursts only — GPU adds overhead without meaningful speedup. Desktop: accumulator (13 GB fp32) + one model (6.5 GB fp16) fit in VRAM; only subsequent models need RAM→VRAM streaming. |
| **SLERP (fp16, 2 models)** | ~0% | ~0% | Both fp16 models fully VRAM-resident on both platforms (13 GB of 16 GB or 24 GB); no offloading needed during computation. |
| **SLERP (fp32, 2 models)** | ~100% | ~100% | 26 GB > 16 GB and 26 GB > 24 GB; must stream each tensor pair from CPU RAM for both platforms, though per-layer fp32 compute on GPU is feasible on the desktop. |
| **Consensus IDW (per-layer, vectorized)** | ~95% | ~90% | All 8 model state dicts must live in CPU RAM (~52 GB fp16); per-layer slices sent to GPU for compute. Desktop handles larger layer batches more comfortably. |
| **TIES / DARE** | ~90% | ~60% | Laptop: base model + one delta in CPU RAM; layer slices sent to GPU for sign-vote and accumulation. Desktop: base + one delta both fit in VRAM simultaneously (~13 GB), enabling GPU-resident sign-vote and accumulation. |
| **Passthrough** | ~100% | ~100% | Memory copy only; all source weights in CPU RAM; no GPU benefit on either platform. |

---

### 8.5 CPU vs. GPU Decision Guide

For **CPU-only systems**, memory bandwidth is the dominant bottleneck. A high-bandwidth DDR5 system (dual-channel 4800 MT/s ≈ 76 GB/s theoretical peak, realistically 40–50 GB/s effective) can stream a 6.5 GB fp16 model in roughly 0.13–0.16 seconds under ideal sequential-read conditions; actual throughput is lower for state_dict loading due to random-access patterns across thousands of tensor keys. The arithmetic itself is nearly free by comparison.

**Use GPU when:**

- The algorithm does more than one multiply-add per element (e.g., `arccos`/`sin` for SLERP, pairwise distance matrix for Consensus, sort for Weighted Median).
- The working set (stacked tensors, intermediate matrices) fits within VRAM for the layer being processed.
- You have already vectorized the inner loops — GPU acceleration of a Python-level element loop (as in the current Consensus implementation) provides no benefit.

**Use CPU when:**

- The algorithm is purely linear (Weighted Sum, Passthrough, Task Arithmetic).
- The working set exceeds available VRAM (load-to-GPU overhead outweighs compute benefit).
- Streaming from disk is the dominant cost (disk → RAM → CPU is the critical path; adding GPU adds an extra RAM → VRAM copy).

**Practical rule of thumb:** If an algorithm's inner operation is more expensive than a single multiply-add per element, GPU acceleration is likely worthwhile. If it is a multiply-add, stay on CPU.

---

### 8.6 Recommendations

#### For 16 GB RAM systems

1. Use **Passthrough** if layer-level source selection is sufficient for your use case.
2. Use **Weighted Sum** with careful memory discipline (close other applications, disable browser tabs). The ~19.5 GB peak may succeed with OS memory compression; if it fails, add swap space.
3. Avoid Consensus, Weighted Median, and all algorithms requiring both models loaded in fp32 simultaneously.

#### For 32 GB RAM systems

1. **Weighted Sum** for N-model merges — fastest, most memory-predictable, no preprocessing required.
2. **TIES or DARE** (with streaming implementation) when sign-conflict quality issues are observed in the Weighted Sum output.
3. **SLERP** for high-quality two-model blends where magnitude preservation matters.

#### For 64 GB+ RAM systems

1. All algorithms are available without special memory management.
2. **Consensus IDW** (once vectorized per §4.2.1 and run on GPU) for equal-contribution N-model merges with automatic outlier suppression — the algorithm most purpose-built for this use case.
3. **Weighted Median** as a fast approximation to Consensus when the exponent tuning is not needed.

#### GPU acceleration priority (highest impact first)

1. Vectorize Consensus (§4.2.1) → run on GPU — highest impact, unlocks the algorithm that was previously unusably slow.
2. SLERP on GPU — meaningful benefit for two-model blends.
3. TIES / DARE sign-vote and accumulation on GPU — moderate, secondary benefit.
4. Weighted Sum on GPU — lowest priority; bottleneck is memory bandwidth, not compute.

#### For RTX 3090 Laptop users (16 GB VRAM, typically 16–32 GB system RAM)

1. **Weighted Sum** — run on **CPU** (accumulator too large for 16 GB VRAM; GPU adds overhead). With 32 GB system RAM this is comfortable and fast.
2. **SLERP** — run on **GPU with fp16** (both models fit in VRAM). Drop to CPU per-layer streaming for fp32.
3. **Consensus IDW** (once vectorized) — run **per-layer on GPU** after vectorizing (§4.2.1). Requires 64 GB system RAM for all 8 models simultaneously, or a chunked-streaming implementation.
4. **TIES / DARE** — run streaming with per-layer GPU offload. Most effective on 32 GB system RAM.
5. **Passthrough** — CPU always; no VRAM benefit.

#### For RTX 3090 Desktop users (24 GB VRAM, typically 32–64 GB system RAM)

1. **Weighted Sum** — can run **fully on GPU** (accumulator + one model fit in 24 GB VRAM). This is the only consumer GPU class where fully GPU-resident Weighted Sum of SDXL models is feasible.
2. **SLERP** — run on **GPU with fp16** (excellent fit; 11 GB headroom). Comfortable and fast.
3. **TIES / DARE** — run on **GPU** (base + one delta fit in VRAM simultaneously). Lower CPU offload than any other platform.
4. **Consensus IDW** (once vectorized) — run **per-layer on GPU**. With 64 GB system RAM, all 8 models can be held in CPU RAM and served layer-by-layer to the GPU efficiently.
5. **Passthrough** — CPU always.

---

### 8.7 RTX 3090 Platform-Specific Analysis

The two RTX 3090 variants are the primary development and testing targets for stmerge. Their VRAM sizes sit on opposite sides of the critical 19.5 GB threshold (accumulator + one model for Weighted Sum), making them interesting to compare in detail.

#### 8.7.1 RTX 3090 Laptop (16 GB VRAM)

**Typical system:** 16–32 GB DDR5, RTX 3090 Mobile (16 GB GDDR6, ~448 GB/s GPU memory bandwidth), NVMe SSD storage.

| Algorithm | Runs? | VRAM Used | CPU Offload | Recommended Device | Notes |
|---|---|---|---|---|---|
| **Weighted Sum** | ✅ Yes | ~0 MB sustained (layer-by-layer) | ~100% | CPU | Accumulator lives in system RAM; GPU adds overhead without meaningful speedup; run on CPU. May work with 16 GB system RAM if swap is available, though performance will be significantly degraded during model-loading phases. |
| **Consensus IDW (current, Python loop)** | ❌ Unusably slow | ~0 MB | ~100% | CPU (slow) | Python element loop is the bottleneck regardless of GPU; vectorize first (§4.2.1). |
| **Consensus IDW (vectorized)** | ⚠️ Per-layer only | ~470 MB per large layer | ~95% | GPU (per-layer) | Fits easily per-layer; all 8 model state dicts must be in CPU RAM (~52 GB needed — requires 64 GB system RAM or chunked streaming); 10–50× speedup over CPU. |
| **SLERP (2 models, fp16)** | ✅ Yes | ~13 GB (both models) | ~0% | GPU | Both fp16 models fit in 16 GB VRAM with ~3 GB headroom; GPU recommended for trigonometric ops. |
| **SLERP (2 models, fp32)** | ⚠️ Per-layer streaming | ~0 MB sustained | ~100% | CPU | 26 GB > 16 GB VRAM; must cast each tensor to fp32, compute, cast back; run on CPU. |
| **TIES / DARE** | ✅ With streaming | ~0–100 MB per layer | ~90% | CPU or GPU per-layer | Base model + delta computed layer-by-layer; GPU useful for sign-vote and masked accumulation. |
| **Passthrough** | ✅ Yes | ~0 MB | ~100% | CPU | Memory copy only; GPU provides no benefit. |
| **Weighted Median** | ⚠️ Per-layer only | ~50 MB per layer | ~95% | GPU (per-layer) | Same system RAM constraint as Consensus; GPU beneficial for `torch.median()` op. |

##### Important: System RAM is the binding constraint for this platform

With this GPU, VRAM is not the primary limiting factor for most algorithms — system RAM is:

- **16 GB system RAM:** Weighted Sum (~19.5 GB peak) requires swap or memory compression. All other multi-model algorithms are infeasible without per-layer chunking.
- **32 GB system RAM:** Weighted Sum, SLERP (fp16), TIES, and DARE all become comfortable. This is the recommended minimum for productive use.
- **64 GB system RAM:** Consensus IDW and Weighted Median become accessible, with all 8 model state dicts held in RAM simultaneously and per-layer slices streamed to the GPU.

---

#### 8.7.2 RTX 3090 Desktop (24 GB VRAM)

**Typical system:** 32–64 GB DDR4/DDR5, RTX 3090 (24 GB GDDR6X, ~936 GB/s GPU memory bandwidth), NVMe SSD storage.

| Algorithm | Runs? | VRAM Used | CPU Offload | Recommended Device | Notes |
|---|---|---|---|---|---|
| **Weighted Sum** | ✅ Yes | ~19.5 GB (accumulator + model) | ~Low | GPU | Accumulator (13 GB fp32) + one model (6.5 GB fp16) fit in 24 GB VRAM with 4.5 GB headroom; entire weighted sum can run GPU-resident; subsequent models are loaded disk → system RAM → VRAM one at a time and accumulated in-place. |
| **Consensus IDW (current, Python loop)** | ❌ Unusably slow | ~0 MB | ~90% | CPU (slow) | Python loop; vectorize first (§4.2.1). |
| **Consensus IDW (vectorized)** | ⚠️ Per-layer, but comfortable | ~470 MB per large layer | ~90% | GPU (per-layer) | Larger layer batches than laptop; all 8 model state dicts still in CPU RAM; 32 GB system RAM needed. |
| **SLERP (2 models, fp16)** | ✅ Yes | ~13 GB | ~0% | GPU | Both models fully resident in 24 GB VRAM; 11 GB headroom. |
| **SLERP (2 models, fp32)** | ⚠️ Per-layer streaming | ~0 MB sustained | ~100% | CPU or per-layer GPU | 26 GB > 24 GB; just over the limit; per-layer fp32 compute on GPU is feasible since each layer pair is tiny relative to VRAM. |
| **TIES / DARE** | ✅ Yes | ~13 GB (base + one delta) | ~60% | GPU | Base model + one fine-tuned model fit simultaneously in VRAM; sign-vote and accumulation fully in VRAM; remaining models in CPU RAM. |
| **Passthrough** | ✅ Yes | ~0 MB | ~100% | CPU | No GPU benefit. |
| **Weighted Median** | ⚠️ Per-layer only | ~50 MB per layer | ~90% | GPU (per-layer) | Same system RAM constraint; more comfortable layer batches than laptop; 32 GB system RAM strongly recommended. |

##### Key advantage over the laptop

The 24 GB VRAM changes two algorithms fundamentally compared to the 16 GB laptop:

- **Weighted Sum** crosses from CPU-only to fully GPU-resident: the accumulator (13 GB fp32) and one model (6.5 GB fp16) fit simultaneously with 4.5 GB to spare. Subsequent models are loaded disk → system RAM → VRAM one at a time and accumulated in-place, meaning the entire merge arithmetic runs on GPU with no intermediate CPU offloading of results. This is the only mainstream consumer GPU class where this is possible for SDXL.
- **TIES and DARE** benefit significantly: the base model and one fine-tuned model (combined ~13 GB fp16) both fit in VRAM at the same time, enabling the sign-vote, mask computation, and accumulation steps to run entirely in VRAM without shuttling intermediate tensors back to CPU RAM.

---

*This evaluation was written against stmerge v0.5 (`model_merger/merger.py`). All code examples use the PyTorch and safetensors APIs already present in the project dependencies.*
