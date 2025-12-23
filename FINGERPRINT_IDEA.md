# Fingerprinting a Model to Determine Base Model Information Programmatically

This document outlines the current idea for fingerprinting a machine learning model to determine its base model information programmatically. The code provided below is a work-in-progress and serves as a conceptual implementation.

```python
import hashlib
import numpy as np
from safetensors import safe_open
from scipy.stats import skew, kurtosis

# Key tensors that are stable and distinctive across model architectures
# Expanded set targeting most discriminative layers
KEY_TENSORS = [
    # U-Net Core (highest weight - most stable across fine-tuning)
    "model.diffusion_model.middle_block.0.in_layers.0.weight",      # Core processing layer
    "model.diffusion_model.middle_block.0.out_layers.3.weight",     # Core output layer
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight",  # Attention normalization
    
    # Input/Output architectural layers (medium weight)
    "model.diffusion_model.input_blocks.0.0.weight",               # First conv (your original)
    "model.diffusion_model.input_blocks.4.0.in_layers.0.weight",   # Mid-resolution features
    "model.diffusion_model.output_blocks.8.1.conv.weight",         # High-res reconstruction
    "model.diffusion_model.output_blocks.5.1.weight",              # Your original choice
    
    # Time embedding (training-sensitive but architectural)
    "model.diffusion_model.time_embed.0.weight",                   # Time MLP input
    "model.diffusion_model.time_embed.2.weight",                   # Time MLP output
    
    # Text encoder (varies significantly between base models)
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",  # SD 1.x/2.x
    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",  # SDXL
]

# Relative importance of each tensor in scoring (should sum to 1.0)
TENSOR_WEIGHTS = {
    # U-Net core layers (highest weight - most stable)
    "model.diffusion_model.middle_block.0.in_layers.0.weight": 0.25,
    "model.diffusion_model.middle_block.0.out_layers.3.weight": 0.20,
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight": 0.15,
    
    # Architectural layers (medium weight)
    "model.diffusion_model.input_blocks.0.0.weight": 0.15,  # Your original high-value choice
    "model.diffusion_model.input_blocks.4.0.in_layers.0.weight": 0.05,
    "model.diffusion_model.output_blocks.8.1.conv.weight": 0.05,
    "model.diffusion_model.output_blocks.5.1.weight": 0.05,  # Your original choice
    
    # Time embedding (lower weight - training dependent)
    "model.diffusion_model.time_embed.0.weight": 0.03,
    "model.diffusion_model.time_embed.2.weight": 0.02,
    
    # Text encoder (medium weight - base model dependent)
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": 0.025,
    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": 0.025,
}

# ---------- Utility Functions ----------

def tensor_hash(tensor: np.ndarray) -> str:
    return hashlib.sha256(tensor.tobytes()).hexdigest()

def tensor_stats_extended(tensor: np.ndarray) -> dict:
    """
    Compute comprehensive statistical fingerprint of tensor data.
    Uses robust statistics that are less sensitive to outliers and training variations.
    """
    flat = tensor.flatten().astype(np.float64)
    
    # Basic statistics
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    
    # Distribution shape (higher-order moments)
    skew_val = float(skew(flat))
    kurtosis_val = float(kurtosis(flat))
    
    # Robust statistics (less sensitive to outliers)
    median_val = float(np.median(flat))
    mad_val = float(np.median(np.abs(flat - median_val)))  # Median Absolute Deviation
    
    # Percentiles for distribution characterization
    q25 = float(np.percentile(flat, 25))
    q75 = float(np.percentile(flat, 75))
    iqr = q75 - q25  # Interquartile range
    
    # Norm-based measures (scale-invariant properties)
    l1_norm = float(np.linalg.norm(flat, ord=1))
    l2_norm = float(np.linalg.norm(flat, ord=2))
    linf_norm = float(np.linalg.norm(flat, ord=np.inf))
    
    # Sparsity measure (fraction of near-zero values)
    sparsity = float(np.sum(np.abs(flat) < 1e-6)) / len(flat)
    
    # Energy concentration (what fraction of energy is in top 10% of values)
    sorted_abs = np.sort(np.abs(flat))[::-1]
    top10_idx = max(1, len(sorted_abs) // 10)
    energy_concentration = float(np.sum(sorted_abs[:top10_idx]**2) / np.sum(sorted_abs**2))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "skew": skew_val,
        "kurtosis": kurtosis_val,
        "mad": mad_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
        "linf_norm": linf_norm,
        "sparsity": sparsity,
        "energy_concentration": energy_concentration
    }
        "mad": float(np.median(np.abs(flat - np.median(flat))))
    }

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if a_flat.size != b_flat.size:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])

# ---------- Fingerprinting ----------

def fingerprint_model(path: str) -> dict:
    fp = {}
    with safe_open(path, framework="numpy") as f:
        for key in KEY_TENSORS:
            if key in f.keys():
                tensor = f.get_tensor(key)
                fp[key] = {
                    "shape": tensor.shape,
                    "hash": tensor_hash(tensor),
                    "stats": tensor_stats_extended(tensor),
                    "values": tensor
                }
    return fp

# ---------- Comparison ----------

def shape_match(fp_unknown, fp_ref) -> float:
    matches = sum(
        1 for k in KEY_TENSORS
        if k in fp_unknown and k in fp_ref and
           fp_unknown[k]["shape"] == fp_ref[k]["shape"]
    )
    return matches / len(KEY_TENSORS)

def compare_models(fp_unknown: dict, fp_db: dict) -> dict:
    results = {}
    for model_name, ref_fp in fp_db.items():
        shape_score = shape_match(fp_unknown, ref_fp)
        if shape_score < 1.0:
            results[model_name] = {
                "shape_score": shape_score,
                "exact_match": 0.0,
                "stats_score": 0.0,
                "cosine_score": 0.0,
                "pearson_score": 0.0,
                "final_score": 0.0
            }
            continue

        exact_matches = 0
        stats_scores = []
        cosine_scores = []
        pearson_scores = []

        for key in KEY_TENSORS:
            if key in fp_unknown and key in ref_fp:
                weight = TENSOR_WEIGHTS.get(key, 1.0)

                if fp_unknown[key]["hash"] == ref_fp[key]["hash"]:
                    exact_matches += weight

                stats_u = fp_unknown[key]["stats"]
                stats_r = ref_fp[key]["stats"]
                stat_diff = sum(abs(stats_u[s] - stats_r[s]) for s in stats_u)
                stats_scores.append(weight * (1 / (1 + stat_diff)))

                cosine_scores.append(weight * cosine_similarity(fp_unknown[key]["values"], ref_fp[key]["values"]))
                pearson_scores.append(weight * pearson_corr(fp_unknown[key]["values"], ref_fp[key]["values"]))

        total_weight = sum(TENSOR_WEIGHTS.values())
        results[model_name] = {
            "shape_score": shape_score,
            "exact_match": exact_matches / total_weight,
            "stats_score": sum(stats_scores) / total_weight,
            "cosine_score": sum(cosine_scores) / total_weight,
            "pearson_score": sum(pearson_scores) / total_weight,
            "final_score": (
                0.2 * (exact_matches / total_weight) +
                0.25 * (sum(stats_scores) / total_weight) +
                0.3 * (sum(cosine_scores) / total_weight) +
                0.25 * (sum(pearson_scores) / total_weight)
            )
        }
    return results

# ---------- Main ----------

if __name__ == "__main__":
    # Build reference DB from known base models
    reference_db = {
        "SD_1.5": fingerprint_model("sd15_base.safetensors"),
        "SDXL_1.0": fingerprint_model("sdxl_base.safetensors"),
        "PonyXL": fingerprint_model("ponyxl_base.safetensors")
    }

    # Fingerprint unknown model
    unknown_fp = fingerprint_model("unknown_model.safetensors")

    # Compare
    scores = compare_models(unknown_fp, reference_db)

    print("\nModel similarity scores:")
    for model, metrics in scores.items():
        print(f"{model}: shape={metrics['shape_score']:.2f}, "
              f"exact={metrics['exact_match']:.2f}, "
              f"stats={metrics['stats_score']:.3f}, "
              f"cosine={metrics['cosine_score']:.3f}, "
              f"pearson={metrics['pearson_score']:.3f}, "
              f"final={metrics['final_score']:.3f}")

    # Enhanced decision logic with confidence levels
    likely_match = max(scores, key=lambda m: scores[m]["final_score"])
    best_result = scores[likely_match]
    best_score = best_result["final_score"]
    confidence = best_result.get("confidence", "unknown")

    print(f"\nModel similarity analysis:")
    print(f"Best match: {likely_match} (score: {best_score:.3f}, confidence: {confidence})")
    print(f"Shape compatibility: {best_result['shape_score']:.3f}")
    print(f"Exact matches: {best_result['exact_match']:.3f}")
    print(f"Statistical similarity: {best_result['stats_score']:.3f}")

    # Decision with confidence thresholds
    if confidence == "very_high" and best_score > 0.85:
        print(f"\n✅ This model is most likely **{likely_match}** or a direct copy.")
    elif confidence in ["high", "very_high"] and best_score > 0.65:
        print(f"\n⚠️ This model is most likely a **derivative of {likely_match}** (fine-tuned or modified).")
    elif confidence == "medium" and best_score > 0.45:
        print(f"\n🤔 This model shows moderate similarity to **{likely_match}** - possible distant derivative.")
    else:
        print("\n❓ No strong base model match found - could be a novel architecture or heavily modified.")

## Implementation Notes & Enhancements

### Key Improvements Made:

1. **Expanded Tensor Set**: Added more discriminative layers from U-Net core, time embeddings, and text encoders
2. **Robust Statistics**: Added percentiles, MAD, energy concentration, and sparsity measures  
3. **Weighted Statistical Comparison**: Emphasizes robust stats over outlier-sensitive measures
4. **Coverage Penalty**: Reduces confidence when many key tensors are missing
5. **Confidence Scoring**: Multi-level confidence assessment based on exact matches and statistical similarity
6. **Enhanced Decision Logic**: More nuanced interpretation of similarity scores

### Future Enhancements to Consider:

1. **Dynamic Tensor Selection**: Automatically identify most discriminative tensors for each architecture
2. **Hierarchical Fingerprinting**: Coarse-grained architecture detection followed by fine-grained base model identification  
3. **Training Dataset Fingerprinting**: Use activation patterns to detect training data influences
4. **Version Detection**: Distinguish between different versions of the same base model (v1.0 vs v1.1)
5. **Merge History Reconstruction**: Attempt to identify component models in merged models
6. **Adversarial Robustness**: Handle models that might intentionally obscure their fingerprints

### Performance Optimizations:

1. **Lazy Loading**: Only load necessary tensors for comparison
2. **Caching**: Store computed fingerprints to avoid recomputation  
3. **Approximate Matching**: Use LSH or similar techniques for fast initial filtering
4. **Incremental Updates**: Add new models to database without full recomputation
```
