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

---

## 3. Comparison Matrix

| Feature | **stmerge Weighted Sum** | **stmerge Consensus** | **LERP** | **SLERP** | **TIES** | **DARE** | **Passthrough** |
|---|---|---|---|---|---|---|---|
| Mathematical basis | Linear combination | Inverse-distance IDW | Linear interpolation | Geodesic interpolation | Task-vector sign election | Stochastic sparsification | Layer transplantation |
| Base model required | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| Works on N > 2 models natively | ✅ Yes | ✅ Yes | ⚠️ Requires extension | ⚠️ Pairwise only | ✅ Yes | ✅ Yes | ✅ Yes |
| Respects user weights | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A |
| Sign-conflict resolution | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes | Partial | N/A |
| Magnitude preservation | ❌ Shrinks | ❌ Shrinks | ❌ Shrinks | ✅ Yes | Partial | ✅ (rescaled) | ✅ Yes |
| Outlier suppression | ❌ No | ✅ Yes | ❌ No | ❌ No | ✅ Trim step | ✅ Drop step | N/A |
| Memory efficiency | ✅ Excellent | ⚠️ Moderate | ✅ Excellent | ✅ Excellent | ⚠️ Needs deltas | ⚠️ Needs deltas | ⚠️ Moderate |
| Speed | ✅ Fast | ❌ Very slow | ✅ Fast | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Moderate | ✅ Fast |
| Preserves layer identity | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| Deterministic | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ (random) | ✅ Yes |
| Implementation complexity | ⭐ Simple | ⭐⭐⭐ Complex | ⭐ Simple | ⭐⭐ Moderate | ⭐⭐⭐ Complex | ⭐⭐ Moderate | ⭐⭐ Moderate |

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

For each parameter position, take the **weighted median** (or unweighted median for equal-weight merges) across all models. The median is inherently outlier-resistant: a single model with an extreme value cannot pull the result beyond the median rank.

```python
def _median_merge(model_entries, device='cpu'):
    # Similar to consensus_merge but replace the weighting loop with:
    merged_flat = stacked_flat.median(dim=0).values
```

This is O(N log N) per element vs O(N²) for consensus, and requires no exponent hyperparameter.

---

### 5.6 Gradient-Free Task Arithmetic — Future Consideration

**Concept:** *Task arithmetic* (Ilharco et al., 2022) represents each fine-tune as a task vector `τ_i = θ_finetuned_i - θ_base` and merges by adding scaled task vectors back to the base:

```
θ_merged = θ_base + λ₁τ₁ + λ₂τ₂ + ... + λₙτₙ
```

The coefficients `λ_i` act as capability dials — setting `λ_i = 0` removes a capability, setting `λ_i > 1` amplifies it.

This is the theoretical foundation for TIES and DARE. Implementing task arithmetic as a first-class operation (not just a preprocessing step for TIES/DARE) would allow users to arithmetically compose and decompose capabilities.

**Note:** This requires the base model and is most powerful when combining models fine-tuned from the same base on distinct tasks.

---

## 6. Implementation Priority

| Priority | Algorithm | Effort | Impact | Notes |
|---|---|---|---|---|
| 🔴 **High** | Vectorized consensus (performance fix) | Low | Very High | Replaces Python loop with batched tensor ops; no API change |
| 🔴 **High** | SLERP | Medium | High | Most-requested in community; adds `slerp` method |
| 🟡 **Medium** | Weighted median | Low | Medium | Adds `median` method; no base model required |
| 🟡 **Medium** | Per-block weights (MBW) | Medium | High | Unlocks the most nuanced community merge workflows |
| 🟡 **Medium** | TIES | High | High | Requires base model; solves sign conflicts |
| 🟡 **Medium** | DARE + TIES | High | High | Pairs with TIES for stronger interference reduction |
| 🟢 **Low** | Passthrough / Frankenmerging | Medium | Medium | Niche but powerful; needs manifest schema extension |
| 🟢 **Low** | Task arithmetic | High | Medium | Foundation for TIES/DARE; needs base model |
| 🟢 **Low** | Iterated SLERP for N > 2 | Medium | Low | Useful for quality; not strictly necessary |

---

## 7. Summary

### What stmerge does well

- **Memory-efficient accumulator pattern** is the right architecture. No other community tool handles 8+ large models as gracefully.
- **Consensus merge is a genuinely novel idea** not commonly found in other SD merge tools. The inverse-distance-weighting approach is well-reasoned.
- **Clean separation of concerns** makes it straightforward to add new merge methods in `merger.py` without touching the manifest, loader, or saver modules.

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

---

*This evaluation was written against stmerge v0.5 (`model_merger/merger.py`). All code examples use the PyTorch and safetensors APIs already present in the project dependencies.*
