# Usage Guide

Complete guide to using Model Merger for all common workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Converting Legacy Models](#converting-legacy-models)
- [Verifying Conversions](#verifying-conversions)
- [Scanning Models](#scanning-models)
- [Understanding the Manifest](#understanding-the-manifest)
- [Merging Models](#merging-models)
- [Advanced Workflows](#advanced-workflows)

## Quick Start

The basic workflow is three simple steps:

```bash
# 1. Scan a folder of models
python run.py scan ./my_models

# 2. (Optional) Edit the generated manifest file
# Review weights, architecture tags, output filename, etc.

# 3. Merge!
python run.py merge --manifest ./my_models/merge_manifest.json
```

Done! Your merged model will be saved in the location specified in the manifest.

## Converting Legacy Models

Got old `.ckpt` or `.pt` files? Convert them safely to safetensors with advanced robustness features.

### Basic Conversion

```bash
# Basic conversion (auto-prunes, creates model.safetensors)
python run.py convert old_model.ckpt

# With custom output name
python run.py convert model.pt --output new_name.safetensors

# Skip pruning (keep all training artifacts)
python run.py convert model.ckpt --no-prune

# Overwrite existing files
python run.py convert model.pt --output existing.safetensors --overwrite
```

### Desktop Notifications

For long conversions that take more than 30 seconds:

```bash
# Get Windows toast notification when done (v0.5.1!)
python run.py convert huge_model.ckpt --notify
```

Perfect for converting files while working on other tasks!

### Unrecognized File Extensions

The tool validates file extensions and only accepts:

- `.ckpt`
- `.pt`
- `.pth`
- `.bin`

For files with unusual extensions:

```bash
# Will prompt for confirmation
python run.py convert unusual_file.model

# Skip confirmation (for automation/scripts)
python run.py convert unusual_file.model --force
```

### What Makes Our Converter Special

- ✅ **Smart Format Detection** - Auto-detects SD models, VAEs, LoRAs, embeddings, and upscalers
- ✅ **Extension Validation** - Warns about unrecognized file types
- ✅ **Adaptive Pruning** - Different strategies for different file types
- ✅ **Desktop Notifications** - Toast notifications for operations >30 seconds
- ✅ **DataParallel Support** - Automatically removes `module.` prefixes from multi-GPU trained models
- ✅ **Shared Tensor Detection** - Clones tensors to prevent memory-sharing errors
- ✅ **Output Verification** - Validates the converted file for quality
- ✅ **Numpy Fallback** - If standard save fails, uses numpy roundtrip as fallback
- ✅ **Security** - Pickle files can contain malicious code, we only use safe loading
- ✅ **Speed** - Safetensors loads 10x faster than pickles
- ✅ **Reliability** - Can't get corrupted as easily
- ✅ **Compatibility** - Works everywhere (ComfyUI, A1111, merging tools)

## Verifying Conversions

Want to be 100% sure your conversion is perfect? Deep verify compares the original and converted files tensor-by-tensor.

### Basic Verification

```bash
# Convert a model
python run.py convert old_model.ckpt

# Verify it matches the original perfectly
python run.py verify old_model.ckpt old_model.safetensors
```

### Verbose Verification

See every tensor comparison:

```bash
python run.py verify original.pt converted.safetensors --verbose
```

### What the Verifier Checks

- ✅ **Key sets match** - All tensor names are present in both files
- ✅ **Shapes match** - Every tensor has the same dimensions
- ✅ **Values match** - Numerical values are identical (within floating-point tolerance)
- ✅ **Module prefix handling** - Correctly handles `module.` prefix removal

The verifier uses `torch.allclose()` with tolerances (rtol=1e-5, atol=1e-8) to account for normal floating-point precision differences. This is the same standard used in PyTorch's own testing!

## Scanning Models

The `scan` command finds all `.safetensors` files in a folder and generates a merge manifest.

### Basic Scanning

```bash
# Scan without VAE (just merge models)
python run.py scan ./models

# Scan with VAE (bake VAE into merged model)
python run.py scan ./models --vae vae.safetensors

# Custom output location
python run.py scan ./models --output my_merge.json
```

### Scan Options

```bash
# Compute SHA-256 hashes (slow but useful for tracking)
python run.py scan ./models --compute-hashes

# Don't auto-calculate equal weights (you'll edit manually)
python run.py scan ./models --no-equal-weights

# Skip files that can't be loaded (useful during copying)
python run.py scan ./models --skip-errors
```

### What Happens During Scan

1. Finds all `.safetensors` in the folder
2. Detects architecture from filenames (Pony, SDXL, Illustrious, etc.)
3. Detects precision (fp16/fp32) for each model
4. Optionally computes SHA-256 hashes (with `--compute-hashes`)
5. Assigns equal weights (1/N for each model)
6. Generates a smart output filename
7. Saves a JSON manifest you can review/edit

**Note on hashing:** Computing hashes for 8x 7GB models takes time (several minutes). The output model hash is ALWAYS computed after merging - that's quick since it's just one file!

## Understanding the Manifest

The generated manifest is a JSON file that describes your merge:

```json
{
  "models": [
    {
      "path": "/path/to/model1.safetensors",
      "weight": 0.25,
      "architecture": "Pony",
      "precision_detected": "fp16",
      "sha256": "abc123..."
    },
    {
      "path": "/path/to/model2.safetensors",
      "weight": 0.25,
      "architecture": "Pony",
      "precision_detected": "fp16"
    }
  ],
  "vae": {
    "path": "/path/to/vae.safetensors",
    "sha256": "def456...",
    "precision_detected": "fp16"
  },
  "output": {
    "path": "Pony_Model1_Model2_merged.safetensors",
    "sha256": "789xyz...",
    "precision_written": "fp16"
  },
  "output_precision": "match",
  "device": "cpu",
  "prune": true,
  "overwrite": false
}
```

### Key Fields

- **models[].weight** - How much each model contributes (can be any value, don't need to sum to 1.0)
- **vae** - Structured object with path, hash, and precision (or `null` if no VAE)
- **output** - Gets populated with hash and precision after merge
- **output_precision** - `"match"` uses first model's precision, or specify `"fp16"` / `"fp32"`
- **device** - `"cpu"` (default, safest) or `"cuda"` (if you have VRAM to spare)
- **prune** - Remove training artifacts (recommended, saves space)
- **overwrite** - Whether to overwrite existing output file

**Edit this file** to adjust weights, change architectures, tweak the output name, etc.

## Merging Models

Once you're happy with the manifest, merge your models:

```bash
# Basic merge
python run.py merge --manifest my_merge.json

# Use GPU acceleration
python run.py merge --manifest my_merge.json --device cuda

# Disable pruning
python run.py merge --manifest my_merge.json --no-prune

# Overwrite existing file
python run.py merge --manifest my_merge.json --overwrite
```

### CLI Overrides

Command-line flags override manifest settings:

- `--overwrite` - Overwrite output file
- `--device cpu|cuda` - Force specific device
- `--no-prune` - Disable pruning

**Important:** CLI overrides update the manifest file! The manifest always reflects what actually happened in the merge, making it a true "build record."

### What Happens During Merge

1. Loads and validates the manifest
2. Checks CUDA availability (auto-falls back to CPU if needed)
3. Displays device information
4. Checks model compatibility (matching shapes)
5. Merges models using weighted accumulation
6. Bakes VAE (if specified)
7. Converts precision (if needed)
8. Prunes unnecessary keys
9. Saves the result
10. Computes and stores output hash

### Memory-Efficient Accumulator Pattern

The merge uses an accumulator pattern that only keeps 2 models in memory at once:

```text
result = model1 * weight1
result += model2 * weight2  (load, add, free)
result += model3 * weight3  (load, add, free)
...
```

This means you can merge 8+ models without needing 56GB of RAM!

## Advanced Workflows

### Unequal Weights

Want to emphasize certain models? Edit the weights in the manifest:

```json
"models": [
  {"path": "base.safetensors", "weight": 0.4},
  {"path": "style_a.safetensors", "weight": 0.3},
  {"path": "style_b.safetensors", "weight": 0.2},
  {"path": "style_c.safetensors", "weight": 0.1}
]
```

Weights don't need to sum to 1.0 (though they usually should). Going outside 0-1 can create "spicy" results!

### GPU Acceleration

If you have VRAM to spare, merging on GPU is ~50x faster:

```json
"device": "cuda"
```

Or use the CLI flag:

```bash
python run.py merge --manifest config.json --device cuda
```

**Warning:** An SDXL model is ~7GB. Merging 8 models on GPU needs ~14GB VRAM (accumulator + current model). If you run out of VRAM, stick with CPU!

### Precision Control

**Match first model (default):**

```json
"output_precision": "match"
```

**Force specific precision:**

```json
"output_precision": "fp16"  // or "fp32"
```

Why fp16? It halves file size with minimal quality loss. Most modern models are trained in fp16 anyway.

### VAE Baking

Why bake a VAE?

- Different VAEs affect colors, contrast, detail
- Some VAEs are optimized for specific styles
- Baking is permanent - no need to load VAE separately during generation

When to skip VAE baking:

- You want flexibility to swap VAEs later
- Your generation tool loads VAEs separately anyway

### Architecture Detection

The scanner tries to guess model architecture from filenames:

- `pony_realistic_v2.safetensors` → "Pony"
- `illustrious_anime.safetensors` → "Illustrious"
- `my_cool_model_sdxl.safetensors` → "SDXL"

If it guesses wrong, just edit the manifest! This is mainly for naming/organization.

See [Customization Guide](customization.md) to add your own architecture patterns.

### Tournament vs True Blend

**Tournament style** (manual merging):

```text
Round 1: (A+B)/2 → AB, (C+D)/2 → CD
Round 2: (AB+CD)/2 → ABCD
```

Problem: A and B get diluted more than C and D!

**This tool's approach** (accumulator):

```text
result = A*0.25 + B*0.25 + C*0.25 + D*0.25
```

All models contribute equally. Much better! 🎯

### File Size Expectations

A typical SDXL model:

- Raw merged: ~6.5GB (fp32) or ~3.3GB (fp16)
- After pruning: ~6.2GB (fp32) or ~3.1GB (fp16)

Pruning removes ~200-400MB of training artifacts. Always recommended!

## Next Steps

- [Customization Guide](customization.md) - Configure architecture patterns
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Installation Guide](installation.md) - GPU setup and requirements
- [FAQ](FAQ.md) - Frequently asked questions
