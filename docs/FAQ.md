# Frequently Asked Questions (FAQ)

Quick answers to common questions about Model Merger.

## Table of Contents

- [General Questions](#general-questions)
- [Conversion Questions](#conversion-questions)
- [Merging Questions](#merging-questions)
- [Performance Questions](#performance-questions)
- [Compatibility Questions](#compatibility-questions)

## General Questions

### What is Model Merger?

Model Merger is a tool for combining multiple Stable Diffusion models using weighted accumulation. Unlike "tournament style" merging (where you merge pairs repeatedly), this tool merges all models at once with proper weights, avoiding unequal dilution.

### Why would I merge models?

Merging models lets you combine the strengths of multiple models:

- Blend different art styles
- Mix realism with creativity
- Combine character knowledge from multiple finetunes
- Create unique model combinations not available elsewhere

### Is this safe to use?

Yes! The tool:

- Uses safetensors format (can't execute malicious code)
- Safe loads all files with `weights_only=True`
- Verifies outputs after conversion
- Open source - you can review the code

### What models can I merge?

You can merge any Stable Diffusion models of the **same architecture**:

- ✅ Pony with Pony
- ✅ SDXL with SDXL
- ✅ Illustrious with Illustrious
- ✅ SD1.5 with SD1.5
- ❌ Can't mix different architectures (Pony + SDXL = error)

### Do I need GPU to use this?

No! CPU mode works fine, just slower:

- **CPU merging:** ~5-10 minutes for 8 SDXL models
- **GPU merging:** ~10-20 seconds for 8 SDXL models

GPU is recommended but not required.

### How much RAM do I need?

The accumulator pattern is memory-efficient:

- **8 SDXL models:** ~16-20GB RAM (CPU) or ~14GB VRAM (GPU)
- **4 SDXL models:** ~10-12GB RAM (CPU) or ~10GB VRAM (GPU)

Much less than loading all models at once!

## Conversion Questions

### Why convert to safetensors?

Safetensors format is:

- **10x faster** to load than pickle (.ckpt/.pt)
- **Safer** - can't execute malicious code
- **More reliable** - can't get corrupted as easily
- **Universal** - works everywhere (ComfyUI, A1111, merging tools)

### Will conversion change my model?

No! Conversion preserves the model exactly. We verify this with tensor-by-tensor comparison using the `verify` command.

### What does pruning do?

Pruning removes training artifacts (optimizer states, EMA weights, etc.) that aren't needed for inference. This saves 200-400MB per model with zero quality loss.

### Should I always prune?

Yes, unless:

- You plan to continue training the model
- You specifically need the training artifacts
- You want to keep the original file size for some reason

For inference (generation), always prune!

### Can I convert files from untrusted sources?

**Be careful!** Pickle files (.ckpt, .pt) can contain malicious code. The tool uses safe loading, but if a file can't be loaded safely, DON'T force it unless you created the file yourself.

See [Troubleshooting: Safe Loading Issues](troubleshooting.md#safe-loading-issues).

### What's the "format detection" feature?

The converter auto-detects:

- **SD models** - Full Stable Diffusion checkpoints
- **VAEs** - Variational autoencoders
- **LoRAs** - Low-rank adaptations
- **Embeddings** - Textual inversions
- **Upscalers** - Super-resolution models

Each type gets appropriate pruning rules!

## Merging Questions

### How do weights work?

Weights control how much each model contributes:

```json
"models": [
  {"path": "model1.safetensors", "weight": 0.5},   // 50%
  {"path": "model2.safetensors", "weight": 0.3},   // 30%
  {"path": "model3.safetensors", "weight": 0.2}    // 20%
]
```

The final model = `model1 * 0.5 + model2 * 0.3 + model3 * 0.2`

### Do weights need to sum to 1.0?

No! Common patterns:

- **Sum to 1.0:** Typical blend (0.25, 0.25, 0.25, 0.25)
- **Sum > 1.0:** Amplified result (1.2, 0.5, 0.3)
- **Negative weights:** Subtract qualities (-0.2 to remove style)

Going outside typical ranges is experimental but fun!

### What's "tournament style" and why is it bad?

**Tournament style:**

```text
Round 1: (A+B)/2 → AB, (C+D)/2 → CD
Round 2: (AB+CD)/2 → ABCD
```

Problem: A and B each contribute 25%, but C and D each contribute 50%!

**Accumulator style (this tool):**

```text
result = A*0.25 + B*0.25 + C*0.25 + D*0.25
```

All models contribute equally!

### Should I bake a VAE?

**Bake when:**

- You always use the same VAE with this model
- You want a self-contained model
- Your generation tool doesn't handle external VAEs well

**Don't bake when:**

- You like swapping VAEs for different looks
- Your tool loads VAEs separately anyway
- You want flexibility

### What's the difference between fp16 and fp32?

- **fp32:** Full precision, ~6.5GB per SDXL model
- **fp16:** Half precision, ~3.3GB per SDXL model

fp16 is standard for modern models - minimal quality difference, half the file size!

### Can I merge more than 8 models?

Yes! The tool has no limit. The accumulator pattern processes one at a time:

- 8 models? No problem.
- 16 models? Sure!
- 100 models? Technically possible, but why? 😄

### How long does merging take?

**CPU mode:**

- 4 models: ~3-5 minutes
- 8 models: ~5-10 minutes
- 16 models: ~10-20 minutes

**GPU mode (CUDA):**

- 4 models: ~5-10 seconds
- 8 models: ~10-20 seconds
- 16 models: ~20-40 seconds

### Can I undo a merge?

No - merging is permanent! That's why the tool:

- Doesn't overwrite inputs (unless you force it)
- Creates a new output file
- Lets you adjust weights and re-merge

Always keep your source models!

## Performance Questions

### GPU or CPU for merging?

**Use GPU when:**

- ✅ You have 14GB+ VRAM (for 8 SDXL models)
- ✅ You're merging frequently
- ✅ Speed matters

**Use CPU when:**

- ✅ Limited VRAM (< 14GB)
- ✅ Merging occasionally (speed doesn't matter)
- ✅ GPU is busy with other tasks

The tool auto-falls back to CPU if GPU unavailable!

### Why is my GPU not being used?

Common reasons:

1. **PyTorch is CPU-only** - Default pip install doesn't include CUDA
2. **CUDA not installed** - Need NVIDIA drivers + CUDA toolkit
3. **Wrong CUDA version** - PyTorch and system CUDA must match

See [Troubleshooting: CUDA Issues](troubleshooting.md#cuda-and-gpu-issues).

### Can I use multiple GPUs?

The tool uses one GPU at a time. For merging, this is actually fine since:

- Merging is sequential (accumulator pattern)
- Network overhead would slow down multi-GPU
- Single GPU is already 50x faster than CPU

### How can I speed up conversion?

Conversion is I/O-bound (disk speed), not CPU/GPU bound:

- ✅ Use SSD instead of HDD
- ✅ Use `--no-prune` to skip pruning (if you don't need it)
- ❌ GPU won't help (it's all disk reads/writes)

### Does the tool support batch processing?

Not directly, but you can script it:

```bash
#!/bin/bash
for file in *.ckpt; do
  python run.py convert "$file" --notify
done
```

Or merge multiple sets:

```bash
python run.py scan ./set1 --output set1.json
python run.py scan ./set2 --output set2.json
python run.py merge --manifest set1.json --notify
python run.py merge --manifest set2.json --notify
```

## Compatibility Questions

### What Stable Diffusion versions are supported?

- ✅ SD 1.5
- ✅ SD 2.1
- ✅ SDXL
- ✅ Pony
- ✅ Illustrious
- ✅ Noobai
- ✅ Any other SDXL-based architectures

As long as they're the same architecture, they'll merge!

### Will merged models work in ComfyUI?

Yes! Safetensors format works everywhere:

- ✅ ComfyUI
- ✅ Automatic1111
- ✅ InvokeAI
- ✅ Fooocus
- ✅ Any tool that supports safetensors

### Will merged models work in Forge?

Yes! Same as ComfyUI - safetensors is universal.

### Can I merge LoRAs?

Not with this tool - it's designed for full models. LoRAs have different merge math (they're deltas, not full weights).

For LoRA merging, use specialized tools.

### Can I merge embeddings/textual inversions?

Not with this tool. Embeddings are tiny and use different merge strategies.

### What about merging with different precisions?

You can merge fp16 and fp32 models together! Use `output_precision` to control the result:

```json
"models": [
  {"path": "fp32_model.safetensors", "precision_detected": "fp32"},
  {"path": "fp16_model.safetensors", "precision_detected": "fp16"}
],
"output_precision": "fp16"  // or "fp32" or "match"
```

### Can I use models from CivitAI?

Yes! Download safetensors versions when available. If only .ckpt available:

1. Download the .ckpt
2. Convert with `python run.py convert model.ckpt`
3. Merge the converted .safetensors

### Why can't I merge Pony and SDXL?

They have different architectures (tensor shapes don't match). It's like trying to merge a cat and a dog - the DNA is incompatible! 🐱🐶

Stick to merging models from the same family.

## Still Have Questions?

- Check the [Usage Guide](usage.md) for detailed workflows
- See [Troubleshooting](troubleshooting.md) for problem solving
- Review [Customization](customization.md) for advanced options
- Read the [Installation Guide](installation.md) for setup help
