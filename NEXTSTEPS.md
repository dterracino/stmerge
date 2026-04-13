# Next Steps

Issues identified during algorithm evaluation (see `ALGORITHM_EVALUATION.md`).
Listed in rough priority order within each section.

---

## Bug Fixes

### 1. Consensus merge silently discards user-provided weights

**Location:** `model_merger/merger.py` → `_consensus_merge()`

User-specified weights in the manifest are completely ignored in consensus mode. Every
model is treated with equal standing regardless of what the user requested. The console
prints a note ("User-provided weights are ignored in consensus mode") but offers no way
to honor them.

**Fix:** Blend user weights as a prior against the data-driven consensus weights, exposed
via an `alpha` parameter (`consensus_prior_strength` in the manifest). At `alpha=0.0`
(default) behavior matches today's pure consensus; at `alpha=1.0` it degrades to weighted
sum. Example from §4.2.2 of the evaluation:

```python
# user_weight_prior: (N,) tensor of user-specified weights (normalized)
# consensus_weights: (N, E) data-driven weights computed by IDW
prior = user_weight_prior[:, None].expand_as(consensus_weights)
blended = alpha * prior + (1 - alpha) * consensus_weights
blended = blended / blended.sum(dim=0, keepdim=True)
```

Manifest fields to add: `consensus_prior_strength: float = 0.0` (range 0.0–1.0).

---

### 2. Non-weight/bias tensors are always taken verbatim from model 1

**Location:** `model_merger/merger.py` → `_weighted_sum_merge()` and `_consensus_merge()`

Both merge functions only accumulate/average tensors whose key name contains `"weight"`
or `"bias"`. Any other floating-point tensor (e.g. `running_mean`, `running_var`,
custom embeddings, projection norms) is silently copied from model 1 only — the
equivalent values from models 2–N are discarded without warning.

For SD/SDXL this has low practical impact because GroupNorm layers have no running
stats, but it is semantically incorrect and would produce wrong output for architectures
that do use BatchNorm or custom float tensors.

**Fix:** The filter `if 'weight' in key or 'bias' in key` should be replaced with a
check for floating-point dtype, merging any float tensor and copying integer/bool tensors
(e.g. `position_ids`) from model 1:

```python
if tensor.is_floating_point():
    # merge it (weighted sum or consensus)
else:
    # copy from model 1 (integer indices, etc.)
```

Integer tensors that should never be merged are already handled by `SKIP_MERGE_KEYS`
in `config.py`; the dtype check is the correct generalisation.

---

## Improvements

### 3. Add weight normalization option to weighted sum

**Location:** `model_merger/merger.py` → `_weighted_sum_merge()` / `cli.py` / manifest

Currently stmerge warns if weights do not sum to 1.0 but proceeds anyway, which silently
scales the output tensor magnitudes. Add an opt-in `--normalize-weights` flag (manifest
field: `normalize_weights: bool = false`) that rescales weights to sum to 1.0 before
merging:

```python
if normalize_weights:
    total = sum(e.weight for e in model_entries)
    if abs(total - 1.0) > 1e-3:
        model_entries = [replace(e, weight=e.weight / total) for e in model_entries]
```

This is especially useful for new users who expect a "50/50 blend" without understanding
that `[0.5, 0.5]` and `[1.0, 1.0]` produce different output magnitudes.

---

### 4. Add weighted median as a faster consensus alternative

**Location:** `model_merger/merger.py` (new method `_median_merge()`)

The weighted median is an O(N log N) approximation to the consensus IDW algorithm
with no hyperparameter tuning (no exponent to choose). It is outlier-resistant by
definition and requires no pairwise distance matrix — just `torch.median(dim=0)` over
the stacked model tensors.

```python
stacked = torch.stack([load_tensor(key) for model in models], dim=0)
result[key] = stacked.median(dim=0).values
```

Same RAM profile as consensus (all N models must be stacked); best suited as a
`--merge-method median` option for 64 GB+ systems or with per-layer chunked streaming.
Expose via `merge_method: "median"` in the manifest.

**Note:** RAM requirements match consensus IDW (~N × model size). See §5.5 of the
evaluation for hardware guidance.

---

### 5. Selective key merging (merge U-Net, keep VAE/text-encoder from base)

**Location:** `model_merger/manifest.py`, `model_merger/merger.py`

Allow users to specify which key prefixes are merged vs. copied verbatim from a
designated source model. This is commonly needed when you want to blend the U-Net
across models while keeping the VAE or text encoder from a specific checkpoint:

```json
{
  "selective_merge": {
    "merge_prefixes":  ["model.diffusion_model."],
    "copy_from_base":  ["first_stage_model.", "cond_stage_model."]
  }
}
```

Requires a manifest schema addition and a pre-merge routing step in `merger.py`.

---

*These items were identified during the algorithm evaluation on April 12, 2026.
See also `ALGORITHM_EVALUATION.md` for full context and `ROADMAP.md` for larger
future features (SLERP, TIES, DARE, per-block weights).*
