# Model Merger v0.3.0

A clean, simple tool for merging multiple Stable Diffusion models with optional VAE baking, plus robust conversion of legacy checkpoint formats with advanced safety features.

## Features

- ✨ **Multi-model merging** - Combine 2+ models with configurable weights
- 🎯 **Accumulator pattern** - Memory-efficient streaming (only 2 models in RAM at once)
- 🎨 **VAE baking** - Inject custom VAEs into merged models
- 🔄 **Robust format converter** - Safely convert .ckpt/.pt/.pth/.bin to safetensors
  - 🧹 **DataParallel prefix removal** - Auto-removes `module.` prefixes from multi-GPU trained models
  - 🔗 **Shared tensor handling** - Detects and fixes memory-sharing issues
  - ✅ **Output verification** - Validates converted files for quality
- 📝 **Manifest workflow** - Scan → Review → Merge with JSON configs
- 🔒 **Security first** - Safe loading of legacy formats (no code execution!)
- 🧹 **Auto-pruning** - Strips training artifacts to keep files lean
- 🔍 **Architecture detection** - Guesses model types from filenames (Pony, SDXL, etc.)
- ⏱️ **Performance tracking** - Shows elapsed time and file hashes for verification
- 🎨 **Beautiful CLI** - Rich progress bars and formatted output

## Installation

```bash
# Clone or download this repo, then:
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- PyTorch 2.0+
- safetensors
- rich (for beautiful terminal output)
- packaging (dependency of safetensors)

## Quick Start

### Converting Legacy Models (Enhanced in v0.3!)

Got old `.ckpt` or `.pt` files? Convert them safely to safetensors with advanced robustness features:

```bash
# Basic conversion (auto-prunes, creates model.safetensors)
python run.py convert old_model.ckpt

# With options
python run.py convert model.pt --output new_name.safetensors --no-prune --overwrite
```

**What makes v0.3.0's converter special:**

- ✅ **DataParallel Support** - Automatically removes `module.` prefixes from multi-GPU trained models
- ✅ **Shared Tensor Detection** - Clones tensors to prevent memory-sharing errors
- ✅ **Output Verification** - Validates the converted file for quality
- ✅ **Numpy Fallback** - If standard save fails, uses numpy roundtrip as fallback
- ✅ **Security** - Pickle files can contain malicious code, we only use safe loading
- ✅ **Speed** - Safetensors loads 10x faster than pickles
- ✅ **Reliability** - Can't get corrupted as easily
- ✅ **Compatibility** - Works everywhere (ComfyUI, A1111, merging tools)

### The Simple Merge Flow

```bash
# 1. Scan a folder of models
python run.py scan ./my_models --vae my_vae.safetensors

# 2. Edit the generated manifest file (optional)
# Review weights, architecture tags, output filename, etc.

# 3. Merge!
python run.py merge --manifest ./my_models/merge_manifest.json
```

Done! Your merged model will be saved in the location specified in the manifest.

## Detailed Usage

### Scanning Models

The `scan` command finds all `.safetensors` files in a folder and generates a merge manifest:

```bash
python run.py scan ./models --vae vae.safetensors --output my_merge.json
```

**Options:**

- `--vae PATH` - VAE file to bake into the merged model
- `--output PATH` - Where to save the manifest (default: `folder/merge_manifest.json`)
- `--compute-hashes` - Calculate SHA-256 hashes for input models/VAE (slow but useful)
- `--no-equal-weights` - Don't auto-calculate equal weights (you'll edit them manually)
- `--skip-errors` - Skip files that can't be loaded (useful when files are still copying)

**What happens:**

1. Finds all `.safetensors` in the folder
2. Detects architecture from filenames (Pony, SDXL, Illustrious, etc.)
3. Detects precision (fp16/fp32) for each model
4. Optionally computes SHA-256 hashes (with `--compute-hashes`)
5. Assigns equal weights (1/N for each model)
6. Generates a smart output filename
7. Saves a JSON manifest you can review/edit

**Note on hashing:** Computing hashes for 8x 7GB models takes time (several minutes).
The output model hash is ALWAYS computed after merging - that's quick since it's just one file!

### The Manifest File

The generated manifest looks like this:

```json
{
  "models": [
    {
      "path": "/path/to/model1.safetensors",
      "weight": 0.25,
      "architecture": "Pony",
      "precision_detected": "fp16",
      "sha256": "abc123..." // Optional, if --compute-hashes used
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
    "sha256": "789xyz...", // Filled in after merge
    "precision_written": "fp16" // Filled in after merge
  },
  "output_precision": "match",
  "device": "cpu",
  "prune": true,
  "overwrite": false
}
```

**Key fields:**

- `models[].weight` - How much each model contributes (can be any value, don't need to sum to 1.0)
- `vae` - Structured object with path, hash, and precision (or `null` if no VAE)
- `output` - Structured object that gets populated with hash and precision after merge
- `output_precision` - `"match"` uses first model's precision, or specify `"fp16"` / `"fp32"`
- `device` - `"cpu"` (default, safest) or `"cuda"` (if you have VRAM to spare)
- `prune` - Remove training artifacts (recommended, saves space)
- `overwrite` - Whether to overwrite existing output file

**Edit this file** to adjust weights, change architectures, tweak the output name, etc.

### Merging Models

Once you're happy with the manifest, merge:

```bash
python run.py merge --manifest my_merge.json
```

**Options:**

- `--overwrite` - Overwrite output file (overrides manifest setting)
- `--device cpu|cuda` - Force specific device (overrides manifest setting)
- `--no-prune` - Disable pruning (overrides manifest setting)

**CLI overrides:** When you use these flags, they update the manifest file! So the manifest always reflects what actually happened in the merge. This makes it a true "build record" of your merge.

**What happens:**

1. Loads and validates the manifest
2. Checks model compatibility (matching shapes)
3. Merges models using weighted accumulation
4. Bakes VAE (if specified)
5. Converts precision (if needed)
6. Prunes unnecessary keys
7. Saves the result

The merge uses an **accumulator pattern** that only keeps 2 models in memory at once:

```text
result = model1 * weight1
result += model2 * weight2  (load, add, free)
result += model3 * weight3  (load, add, free)
...
```

This means you can merge 8+ models without needing 56GB of RAM!

## Advanced Usage

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

### CUDA Acceleration

If you have VRAM to spare, merging on GPU is ~50x faster:

```json
"device": "cuda"
```

**Warning:** An SDXL model is ~7GB. Merging 8 models on GPU needs ~14GB VRAM (accumulator + current model). If you run out of VRAM, stick with CPU!

### Precision Control

**Match first model** (default):

```json
"output_precision": "match"
```

**Force specific precision:**

```json
"output_precision": "fp16"  // or "fp32"
```

Why fp16? It halves file size with minimal quality loss. Most modern models are trained in fp16 anyway.

### Architecture Detection

The scanner tries to guess model architecture from filenames:

- `pony_realistic_v2.safetensors` → "Pony"
- `illustrious_anime.safetensors` → "Illustrious"
- `my_cool_model_sdxl.safetensors` → "SDXL"

If it guesses wrong, just edit the manifest! This is mainly for naming/organization.

## Tips & Tricks

### Tournament vs True Blend

**Tournament style** (what you were doing manually):

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

### When Models Won't Merge

If you get "models are incompatible" errors, it means:

- They have different architectures (Pony vs Illustrious)
- They have different tensor shapes (can't merge!)

Solution: Only merge models from the same family. Use the architecture detection to check!

### VAE Baking

Why bake a VAE?

- Different VAEs affect colors, contrast, detail
- Some VAEs are optimized for specific styles
- Baking is permanent - no need to load VAE separately during generation

When to skip VAE baking:

- You want flexibility to swap VAEs later
- Your generation tool loads VAEs separately anyway

### File Size

A typical SDXL model:

- Raw merged: ~6.5GB (fp32) or ~3.3GB (fp16)
- After pruning: ~6.2GB (fp32) or ~3.1GB (fp16)

Pruning removes ~200-400MB of training artifacts. Always recommended!

## Project Structure

```text
model_merger/
├── __init__.py       # Package exports
├── config.py         # Constants, patterns, defaults
├── loader.py         # Load models/VAEs, compute hashes
├── manifest.py       # Scan folders, generate/validate manifests
├── merger.py         # Core accumulator merge logic
├── vae.py            # VAE baking
├── saver.py          # Save merged models
├── converter.py      # Convert legacy formats to safetensors
├── console.py        # Rich UI formatting and progress bars
└── cli.py            # Command-line interface

run.py                # Entry point
```

Each module has a single, clear responsibility. Easy to test, easy to extend!

## Roadmap

**v0.3.0 - ✅ Complete!**

- [x] DataParallel `module.` prefix removal
- [x] Shared tensor detection and cloning
- [x] Output verification for converted files
- [x] Numpy fallback for stubborn shared tensors
- [x] Enhanced converter robustness

**v0.2.0 - ✅ Complete!**

- [x] Convert .ckpt/.pt files to safetensors
- [x] Progress bars with elapsed time tracking
- [x] Beautiful Rich-powered CLI output
- [x] CLI override persistence to manifest
- [x] Skip-errors mode for scanning

**Future ideas (v0.4.0+):**

- [ ] Batch conversion (convert entire folders at once)
- [ ] Block-weighted merging (different weights per layer)
- [ ] Extract VAE from merged model
- [ ] Interactive TUI for editing manifests
- [ ] API lookup for model hashes (CivitAI, HuggingFace)
- [ ] Validate model compatibility before loading (peek at metadata)

## Troubleshooting

### "Model file not found"

- Check paths in manifest are absolute or relative to where you run the script
- Manifest might have been generated on different machine with different paths

### "Models are incompatible"

- Only merge models from same architecture family
- Check that models aren't corrupted (try loading in your SD UI first)

### "Out of memory"

- Use `"device": "cpu"` instead of CUDA
- Close other applications
- Merge fewer models at once

### "Output file exists"

- Use `--overwrite` flag or change output filename in manifest

## Contributing

This is a clean, focused tool. If you want to add features:

1. Keep separation of concerns (one module = one job)
2. Follow existing code style
3. Add docstrings!
4. Test with real models before submitting

## License

Do whatever you want with this. No warranty, use at your own risk, etc.

Made with ❤️ and too much coffee by someone who was tired of clicking through Supermerger's UI 8 times per merge session.
