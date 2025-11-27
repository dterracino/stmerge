# Customization Guide

Learn how to customize Model Merger for your specific workflow and naming conventions.

## Table of Contents

- [Architecture Pattern Detection](#architecture-pattern-detection)
- [Editing Merge Manifests](#editing-merge-manifests)
- [Custom Merge Strategies](#custom-merge-strategies)

## Architecture Pattern Detection

The tool detects model architectures (Pony, SDXL, Illustrious, etc.) from filenames. You can customize these patterns without editing code!

### Why Customize?

- You use custom/niche architectures (Flux, AnimagineXL, etc.)
- Your file naming convention differs from defaults
- You work with specialized models that need specific tags
- You want better organization in your manifest files

### Creating Custom Patterns

#### Step 1: Create config directory

```bash
mkdir -p ~/.model_merger
```

#### Step 2: Copy default patterns

```bash
cp model_merger/architecture_patterns.json ~/.model_merger/
```

#### Step 3: Edit the patterns

```bash
# Use your favorite editor
nano ~/.model_merger/architecture_patterns.json
# or
code ~/.model_merger/architecture_patterns.json
```

### Pattern File Format

```json
{
  "_comment": "Architecture detection patterns for model filenames",
  "_usage": "Copy this file to ~/.model_merger/architecture_patterns.json to customize",
  "patterns": {
    "Pony": ["pony", "ponyxl", "ponydiffusion", "pony-diffusion", "ponyv6"],
    "Illustrious": ["illustrious", "illus", "ill", "illustrious-xl"],
    "SDXL": ["sdxl", "xl"],
    "SD1.5": ["sd15", "sd1.5", "v1-5", "v1_5"],
    "SD2.1": ["sd21", "sd2.1", "v2-1", "v2_1"],
    "Noobai": ["noobai", "noob"]
  },
  "default": "SDXL",
  "_note": "Patterns are case-insensitive. Add your own architectures or patterns as needed."
}
```

### Adding Custom Architectures

Add your own architecture types:

```json
{
  "patterns": {
    "Pony": ["pony", "ponyxl", "ponydiffusion"],
    "Illustrious": ["illustrious", "illus"],
    "SDXL": ["sdxl", "xl"],
    "Flux": ["flux", "flux-dev", "flux-schnell"],
    "AnimagineXL": ["animagine", "animaginexl"],
    "MyCustomArch": ["mycustom", "custom-model", "cm"]
  },
  "default": "SDXL"
}
```

### Adding Pattern Aliases

Add more patterns for existing architectures:

```json
{
  "patterns": {
    "Pony": [
      "pony",
      "ponyxl",
      "ponydiffusion",
      "pony-diffusion",
      "ponyv6",
      "pdxl",              // Your custom shorthand
      "pony_realism",      // Your specific variant
      "autismmix"          // Community model based on Pony
    ]
  }
}
```

### How Pattern Matching Works

1. **Case-insensitive**: `"pony"` matches `"Pony"`, `"PONY"`, `"PoNy"`
2. **Substring matching**: `"pony"` matches `"my_pony_model_v2.safetensors"`
3. **First match wins**: Patterns are checked in order (top to bottom)
4. **User overrides defaults**: Your patterns take precedence over built-in ones

### Examples

**Filename:** `realistic_pony_v6_fp16.safetensors`

- Matches: `"pony"` pattern → Architecture: `"Pony"`

**Filename:** `illustrious_anime_style.safetensors`

- Matches: `"illustrious"` pattern → Architecture: `"Illustrious"`

**Filename:** `my_cool_flux_model.safetensors`

- Matches: `"flux"` pattern (if added) → Architecture: `"Flux"`
- Without custom pattern → Fallback: `"SDXL"` (default)

**Filename:** `random_model_name.safetensors`

- No matches → Fallback: `"SDXL"` (default)

### Changing the Default

Change which architecture is used when no patterns match:

```json
{
  "patterns": { ... },
  "default": "Flux"  // Changed from "SDXL"
}
```

### Override Behavior

The tool loads patterns in this priority order:

1. **Default patterns** (built into the package)
2. **User patterns** (from `~/.model_merger/architecture_patterns.json`)
3. **Merge** (user patterns override defaults)
4. **Fallback** (if JSON missing/corrupt, use hardcoded defaults)

**Example:**

Default patterns:

```json
{
  "patterns": {
    "Pony": ["pony", "ponyxl"]
  }
}
```

Your user patterns:

```json
{
  "patterns": {
    "Pony": ["pony", "ponyxl", "pdxl", "autismmix"]
  }
}
```

Result:

```json
{
  "patterns": {
    "Pony": ["pony", "ponyxl", "pdxl", "autismmix"]  // User version wins
  }
}
```

### Benefits of External Patterns

- ✅ **No code changes needed** - Just edit JSON
- ✅ **No PRs required** - Handle your naming conventions locally
- ✅ **Shareable** - Share pattern files with team/community
- ✅ **Future-proof** - Add new architectures as they're released
- ✅ **Works with frozen apps** - Even compiled/packaged versions can use custom patterns

### Testing Your Patterns

Scan a folder to see if patterns match correctly:

```bash
python run.py scan ./models --output test_manifest.json
```

Check the generated manifest - each model should have the correct `architecture` tag.

## Editing Merge Manifests

The manifest file gives you complete control over the merge process.

### Basic Manifest Structure

```json
{
  "models": [...],      // Models to merge
  "vae": {...},         // Optional VAE to bake
  "output": {...},      // Output configuration
  "output_precision": "match",
  "device": "cpu",
  "prune": true,
  "overwrite": false
}
```

### Adjusting Model Weights

**Equal weights (default):**

```json
"models": [
  {"path": "model1.safetensors", "weight": 0.25},
  {"path": "model2.safetensors", "weight": 0.25},
  {"path": "model3.safetensors", "weight": 0.25},
  {"path": "model4.safetensors", "weight": 0.25}
]
```

**Emphasizing specific models:**

```json
"models": [
  {"path": "base.safetensors", "weight": 0.5},       // 50% base
  {"path": "style_a.safetensors", "weight": 0.3},    // 30% style A
  {"path": "detail.safetensors", "weight": 0.2}      // 20% detail
]
```

**Experimental "spicy" weights:**

```json
"models": [
  {"path": "model1.safetensors", "weight": 1.5},     // Amplified
  {"path": "model2.safetensors", "weight": -0.2}     // Negative (reduces influence)
]
```

Weights don't need to sum to 1.0! Going outside typical ranges can create interesting (or broken) results.

### Changing Output Filename

```json
"output": {
  "path": "MyCustom_MergedModel_v2.safetensors"
}
```

Use absolute paths or relative paths (relative to where you run the script).

### Removing Models from Merge

Just delete the model entry from the manifest:

```json
"models": [
  {"path": "model1.safetensors", "weight": 0.33},
  {"path": "model2.safetensors", "weight": 0.33},
  {"path": "model3.safetensors", "weight": 0.33}
  // model4 removed - will not be included in merge
]
```

Remember to adjust weights if you want them to sum to 1.0!

### Adding Models to Merge

Add a new entry:

```json
"models": [
  {"path": "model1.safetensors", "weight": 0.25, "architecture": "Pony", "precision_detected": "fp16"},
  {"path": "model2.safetensors", "weight": 0.25, "architecture": "Pony", "precision_detected": "fp16"},
  {"path": "model3.safetensors", "weight": 0.25, "architecture": "Pony", "precision_detected": "fp16"},
  {"path": "/new/path/model4.safetensors", "weight": 0.25, "architecture": "Pony", "precision_detected": "fp16"}
]
```

### Removing VAE

Set to `null`:

```json
"vae": null
```

### Adding/Changing VAE

```json
"vae": {
  "path": "/path/to/my_vae.safetensors",
  "precision_detected": "fp16"
}
```

The `sha256` field is optional (only generated with `--compute-hashes`).

### Correcting Architecture Tags

If scanner guessed wrong:

```json
"models": [
  {
    "path": "weird_name.safetensors",
    "weight": 0.5,
    "architecture": "Illustrious",  // Change from "SDXL" to "Illustrious"
    "precision_detected": "fp16"
  }
]
```

## Custom Merge Strategies

### Strategy: Base + Style Mix

Use one strong base model, then blend multiple style models:

```json
"models": [
  {"path": "base_realistic.safetensors", "weight": 0.6},
  {"path": "style_anime.safetensors", "weight": 0.15},
  {"path": "style_painterly.safetensors", "weight": 0.15},
  {"path": "style_detailed.safetensors", "weight": 0.1}
]
```

### Strategy: Iterative Refinement

Merge → test → adjust weights → re-merge:

1. Start with equal weights
2. Generate test images
3. Identify which models contribute best qualities
4. Adjust weights to emphasize those models
5. Re-merge and test again

### Strategy: Negative Weights (Experimental!)

Subtract unwanted characteristics:

```json
"models": [
  {"path": "good_model.safetensors", "weight": 1.2},
  {"path": "has_unwanted_style.safetensors", "weight": -0.2}
]
```

⚠️ **Warning:** This is experimental and may produce unstable results!

### Strategy: Three-Way Balance

Balance between three distinct characteristics:

```json
"models": [
  {"path": "realism.safetensors", "weight": 0.4},
  {"path": "creativity.safetensors", "weight": 0.4},
  {"path": "detail.safetensors", "weight": 0.2}
]
```

### Strategy: Precision Mixing

Mix fp32 and fp16 models (output precision determined by `output_precision`):

```json
"models": [
  {"path": "fp32_model.safetensors", "weight": 0.5, "precision_detected": "fp32"},
  {"path": "fp16_model.safetensors", "weight": 0.5, "precision_detected": "fp16"}
],
"output_precision": "fp16"  // Convert final result to fp16
```

## Next Steps

- [Usage Guide](usage.md) - Learn core workflows
- [Troubleshooting](troubleshooting.md) - Solve common issues
- [Installation Guide](installation.md) - GPU setup and requirements
- [FAQ](FAQ.md) - Frequently asked questions
