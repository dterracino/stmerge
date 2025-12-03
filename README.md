# Model Merger

Memory-efficient merging of multiple Stable Diffusion models using weighted accumulation. Built for merging 8+ Pony/SDXL/Illustrious models without needing 56GB of RAM.

## Why This Tool?

Supermerger and similar tools make you merge models in "tournament style" - pairing them up round by round. This causes unequal blending where early models get diluted more than later ones. **Model Merger uses a proper accumulator pattern** where all models contribute equally based on their weights.

**Key Features:**

- ✅ **Memory Efficient** - Merge 8+ models without massive RAM requirements
- ✅ **Shape-Only Validation** - Ultra-fast compatibility checks without loading full models  
- ✅ **True Weighted Blending** - All models contribute equally (no tournament dilution!)
- ✅ **Smart Conversion** - Legacy `.ckpt`/`.pt` to safetensors with adaptive pruning
- ✅ **Deep Verification** - Tensor-by-tensor comparison of conversions
- ✅ **VAE Baking** - Optional VAE integration into merged models
- ✅ **GPU Acceleration** - CUDA support with auto-fallback to CPU
- ✅ **Desktop Notifications** - Toast notifications for long operations (Windows)
- ✅ **Customizable** - JSON-based architecture detection patterns
- ✅ **Safe & Fast** - Uses safetensors format (10x faster loading, can't execute malicious code)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Convert old models to safetensors
python run.py convert old_model.ckpt

# Merge models in three steps
python run.py scan ./my_models                          # 1. Find models, generate manifest
# (optionally edit the manifest to adjust weights)
python run.py merge --manifest ./my_models/merge_manifest.json  # 2. Merge!
```

Done! Your merged model is ready to use.

## Simple Example

```bash
# Scan a folder with 4 Pony models
python run.py scan ~/models/pony --vae ~/models/pony_vae.safetensors

# This generates merge_manifest.json with equal weights (0.25 each)
# Edit if you want to emphasize specific models

# Merge with GPU acceleration
python run.py merge --manifest ~/models/pony/merge_manifest.json --device cuda

# Result: Pony_Model1_Model2_Model3_Model4_merged.safetensors
```

## The Accumulator Difference

**Tournament style** (what you were doing manually):

```text
Round 1: (A+B)/2 → AB, (C+D)/2 → CD
Round 2: (AB+CD)/2 → ABCD
```

Problem: A and B get diluted more than C and D!

**Accumulator style** (this tool):

```text
result = A*0.25 + B*0.25 + C*0.25 + D*0.25
```

All models contribute equally. Much better! 🎯

The accumulator pattern only keeps 2 models in memory at once:

```text
result = model1 * weight1
result += model2 * weight2  (load, add, free)
result += model3 * weight3  (load, add, free)
...
```

This means you can merge 8+ models without needing 56GB of RAM!

## Documentation

- **[Installation Guide](docs/installation.md)** - Setup, requirements, GPU acceleration
- **[Usage Guide](docs/usage.md)** - Converting, verifying, scanning, merging
- **[Customization Guide](docs/customization.md)** - Architecture patterns, manifest editing
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[FAQ](docs/FAQ.md)** - Frequently asked questions
- **[CHANGELOG](CHANGELOG.md)** - Version history and release notes
- **[ROADMAP](ROADMAP.md)** - Future plans and development ideas

## Features Deep Dive

### Smart Format Detection & Conversion

Convert legacy `.ckpt`, `.pt`, `.pth`, `.bin` files to safetensors:

```bash
python run.py convert old_model.ckpt
python run.py convert model.pt --output new_name.safetensors --notify
```

**Why our converter is better:**

- Auto-detects file type (SD model, VAE, LoRA, embedding, upscaler)
- Adaptive pruning strategies for each type
- DataParallel prefix removal (`module.*`)
- Shared tensor detection and cloning
- Extension validation with `--force` bypass
- Desktop notifications for long operations
- Output verification
- Numpy fallback for problematic saves

### Deep Verification

Verify conversions are pixel-perfect:

```bash
python run.py convert old_model.ckpt
python run.py verify old_model.ckpt old_model.safetensors --verbose
```

Checks:

- Key sets match
- Tensor shapes match  
- Values match (within floating-point tolerance)
- Module prefix handling

Uses PyTorch's own testing standards (rtol=1e-5, atol=1e-8).

### Flexible Merging

**Equal weights:**

```json
{"models": [
  {"path": "model1.safetensors", "weight": 0.25},
  {"path": "model2.safetensors", "weight": 0.25},
  {"path": "model3.safetensors", "weight": 0.25},
  {"path": "model4.safetensors", "weight": 0.25}
]}
```

**Emphasize specific models:**

```json
{"models": [
  {"path": "base.safetensors", "weight": 0.5},
  {"path": "style_a.safetensors", "weight": 0.3},
  {"path": "detail.safetensors", "weight": 0.2}
]}
```

**Experimental "spicy" weights:**

```json
{"models": [
  {"path": "model1.safetensors", "weight": 1.5},
  {"path": "model2.safetensors", "weight": -0.2}
]}
```

Weights don't need to sum to 1.0!

### GPU Acceleration

50x faster merging with CUDA:

```bash
python run.py merge --manifest config.json --device cuda
```

Auto-detects CUDA availability and falls back to CPU with helpful installation instructions.

**Requirements:**

- NVIDIA GPU with CUDA support
- CUDA-enabled PyTorch (see [Installation Guide](docs/installation.md))
- ~14GB VRAM for merging 8 SDXL models

### Customizable Architecture Detection

Create `~/.model_merger/architecture_patterns.json` to add custom architectures:

```json
{
  "patterns": {
    "Pony": ["pony", "ponyxl", "ponydiffusion"],
    "Flux": ["flux", "flux-dev", "flux-schnell"],
    "MyCustomArch": ["mycustom", "custom-model"]
  },
  "default": "SDXL"
}
```

No code changes needed! See [Customization Guide](docs/customization.md).

## Project Structure

```text
model_merger/
├── __init__.py                    # Package exports
├── config.py                      # Constants, patterns, defaults
├── architecture_patterns.json     # Default architecture detection patterns
├── loader.py                      # Load models/VAEs, compute hashes
├── manifest.py                    # Scan folders, generate/validate manifests
├── merger.py                      # Core accumulator merge logic
├── vae.py                         # VAE baking
├── saver.py                       # Save merged models
├── converter.py                   # Convert legacy formats to safetensors
├── verifier.py                    # Deep verification of conversions
├── pruner.py                      # Smart format detection and pruning
├── notifier.py                    # Desktop notifications for long operations
├── console.py                     # Rich UI formatting and progress bars
└── cli.py                         # Command-line interface

run.py                              # Entry point
docs/                               # Documentation
```

Each module has a single, clear responsibility. Easy to test, easy to extend!

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CPU or CUDA)
- safetensors 0.4.0+
- rich 13.0+ (beautiful terminal output)
- numpy 1.24+
- packaging 21.0+
- win10toast 0.9+ (optional, Windows only)

See [Installation Guide](docs/installation.md) for detailed setup.

## Contributing

This is a clean, focused tool. If you want to add features:

1. Keep separation of concerns (one module = one job)
2. Follow existing code style
3. Add docstrings!
4. Test with real models before submitting

## License

Do whatever you want with this. No warranty, use at your own risk, etc.

Made with ❤️ and too much coffee by someone who was tired of clicking through Supermerger's UI 8 times per merge session.
