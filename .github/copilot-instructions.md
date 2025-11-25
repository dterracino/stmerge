# GitHub Copilot Instructions for Model Merger

## Project Overview

This is a **Stable Diffusion model merging tool** that combines multiple `.safetensors` models using weighted accumulation, with support for VAE baking and legacy format conversion. The tool emphasizes memory efficiency, security, and clean architecture.

## Core Architecture Principles

### 1. **Separation of Concerns**
- Each module in `model_merger/` has ONE responsibility
- No mixing of I/O, computation, and presentation logic
- Follow the existing pattern: `loader.py` loads, `merger.py` merges, `saver.py` saves

### 2. **Memory-Efficient Design**
- Use the **accumulator pattern** - only 2 models in RAM at once
- Never load all models simultaneously
- Always clean up after processing each model

### 3. **Security First**
- Use `safe_torch_load()` from `loader.py` for ALL pickle-based formats
- Never use `torch.load()` with `weights_only=False` on user files
- Prefer safetensors format; conversion is a safety measure

## Code Style Guidelines

### Python Standards
- Python 3.8+ compatible (no walrus operators, use `Optional` from typing)
- Type hints on all function signatures
- Docstrings in Google style for all public functions
- Use pathlib `Path` objects, not string paths

### Naming Conventions
```python
# Good
def merge_models(model_entries: List[ModelEntry], device: str) -> Dict[str, torch.Tensor]:
    """Merge multiple models using weighted accumulation."""

# Bad
def merge(models, dev):  # No types, unclear name
```

### Error Handling
- Use Rich console for user-facing messages (via `console.py`)
- Raise descriptive exceptions with context
- Validate inputs early, fail fast

```python
# Good
if not input_path.exists():
    raise FileNotFoundError(f"Model file not found: {input_path}")

# Bad  
assert input_path.exists()  # Silent failure
```

## Module-Specific Guidelines

### `loader.py` - Loading Models/VAEs
- **Always** compute SHA-256 hashes after loading for verification
- Detect precision (fp16/fp32) from the first tensor found
- Use `safe_torch_load()` for .ckpt/.pt/.pth files
- Handle corrupted files gracefully (skip or warn, don't crash)

### `merger.py` - Core Merge Logic
- Validate tensor shapes BEFORE merging (save RAM on failures)
- Accumulator pattern: `result = model1 * w1; result += model2 * w2; ...`
- Skip keys in `config.SKIP_MERGE_KEYS` (they cause issues)
- Support both CPU and CUDA devices

### `manifest.py` - Manifest Generation/Validation
- Use dataclasses with `@dataclass` decorator for manifest structures
- Always validate file paths exist before merging
- Generate smart output filenames from input model names
- Architecture detection is a guess - user can override

### `vae.py` - VAE Baking
- VAE keys need `first_stage_model.` prefix when baking
- Verify VAE has expected keys before baking
- Warn if VAE already exists in target model (overwriting)

### `saver.py` - Saving Models
- Include metadata: merge weights, architectures, timestamps
- Never overwrite unless explicitly requested
- Always compute output hash after saving
- Use `safetensors.torch.save_file()` for saving

### `converter.py` - Format Conversion
- Only convert FROM (.ckpt/.pt/.pth/.bin) TO (.safetensors)
- Auto-prune by default (strip training artifacts)
- Warn about risks of loading pickle formats
- Show progress with Rich progress bars

### `console.py` - Rich UI/UX
- Use Rich for ALL user-facing output (no raw `print()`)
- Progress bars for long operations (loading, merging, saving)
- Color coding: cyan=info, yellow=warning, red=error, green=success
- Show elapsed time for operations

### `config.py` - Configuration
- ALL magic strings/numbers go here
- Architecture patterns for detection
- Keys to prune, skip, keep
- Default settings and filenames

### `cli.py` - Command Line Interface
- Three subcommands: `scan`, `merge`, `convert`
- CLI flags override manifest settings
- Update manifest with actual used settings after merge
- Validate before executing, fail gracefully

## Common Patterns

### Loading a Model
```python
from model_merger import loader

# Load with hash computation
state_dict, precision, file_hash = loader.load_model(
    model_path,
    device='cpu',
    compute_hash=True
)
```

### Merging Models (Accumulator Pattern)
```python
result = None
for i, entry in enumerate(models):
    model_dict, _, _ = loader.load_model(entry.path, device)
    
    if result is None:
        result = {k: v * entry.weight for k, v in model_dict.items()}
    else:
        for key in result.keys():
            result[key] += model_dict[key] * entry.weight
    
    del model_dict  # Free memory immediately
```

### Using Rich Console
```python
from model_merger.console import console, print_success, print_error

with console.status("[cyan]Loading model..."):
    model = load_model(path)

print_success(f"Model loaded: {path}")
```

## Testing Strategy

### What to Test
- Manifest validation (missing files, invalid weights)
- Tensor shape compatibility checking
- Precision conversion (fp32 ↔ fp16)
- Hash computation (if file changes, hash must change)
- VAE baking (verify keys are correct)

### How to Test
- Use small dummy models (save test time)
- Test error cases explicitly (missing files, incompatible shapes)
- Verify memory cleanup (check that old models are freed)

## Domain Knowledge

### Stable Diffusion Model Structure
- **Main model**: `model.diffusion_model.*` keys (U-Net)
- **VAE**: `first_stage_model.*` keys (image encoder/decoder)
- **Text encoder**: `cond_stage_model.*` (SD 1.x/2.x) or `conditioner.*` (SDXL)

### Model Architectures
- **SDXL**: Base model, ~6.5GB fp32
- **Pony Diffusion**: SDXL-based, trained on specific styles
- **Illustrious**: SDXL variant optimized for anime/illustration
- All are fundamentally SDXL architecture with different training

### Merging Theory
- **Linear interpolation**: `result = model_a * 0.5 + model_b * 0.5`
- **Weights don't need to sum to 1.0** (but usually should)
- **Block merging** (future feature): Different weights per layer

### VAE Behavior
- Different VAEs affect color, contrast, detail rendering
- Common VAEs: MSE, EMA, custom trained variants
- Baking permanently embeds VAE (can't swap later)

## Feature Requests / New Features

### Before Implementing
1. Does it fit the "clean, focused tool" philosophy?
2. Can it be added without breaking existing code?
3. Does it maintain separation of concerns?

### Architecture for New Features
- New functionality → new module or function
- Update `config.py` for new constants
- Add CLI argument if user-facing
- Update manifest schema if needed
- Document in README.md

### Likely Future Features
- **Block-weighted merging**: Different weights per layer (complex, needs research)
- **Batch conversion**: Convert entire folders (easy, just loop)
- **VAE extraction**: Pull VAE from model (inverse of baking)
- **Interactive TUI**: Terminal UI for editing manifests (use Rich's TUI features)
- **Metadata lookup**: Query CivitAI/HuggingFace for model info by hash

## Common Gotchas

### Don't Do This
❌ Load all models into memory at once (OOM on large merges)  
❌ Use `torch.load()` without `weights_only=True` on user files (security risk)  
❌ Assume all models have same keys (validate compatibility first)  
❌ Skip hash computation on output (users need verification)  
❌ Mix string paths and Path objects (stick to Path)  
❌ Nest modules (keep flat structure in `model_merger/`)

### Do This Instead
✅ Use accumulator pattern (only 2 models in RAM)  
✅ Use `safe_torch_load()` wrapper for pickle formats  
✅ Check tensor shapes before merging  
✅ Always compute output hash for verification  
✅ Use `Path` everywhere for cross-platform compatibility  
✅ Keep modules independent and focused

## Questions to Ask

When suggesting code changes, consider:
1. Does this maintain the memory-efficient accumulator pattern?
2. Is the security model preserved (no unsafe pickle loading)?
3. Are errors handled gracefully with Rich console output?
4. Does this fit into the existing module structure?
5. Will this work on both CPU and CUDA devices?
6. Are type hints and docstrings included?

## Resources

- **Safetensors**: https://github.com/huggingface/safetensors
- **Rich**: https://rich.readthedocs.io/
- **PyTorch**: https://pytorch.org/docs/stable/
- **SD Model Structure**: Research papers on Stable Diffusion architecture

## Tone and Style

- Code comments: Brief and practical
- Error messages: Helpful and specific (tell user HOW to fix it)
- Documentation: Concise with examples
- Variable names: Clear and descriptive (no cryptic abbreviations)
- Functions: Small and focused (if >50 lines, consider splitting)

---

**Remember**: This tool was built by someone "tired of clicking through Supermerger's UI 8 times per merge session." Keep it clean, keep it simple, keep it focused. No feature creep! 🎨
