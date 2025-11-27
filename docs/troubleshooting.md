# Troubleshooting Guide

Solutions to common issues when using Model Merger.

## Table of Contents

- [File and Path Issues](#file-and-path-issues)
- [CUDA and GPU Issues](#cuda-and-gpu-issues)
- [Memory Issues](#memory-issues)
- [Conversion Issues](#conversion-issues)
- [Merge Compatibility Issues](#merge-compatibility-issues)
- [Safe Loading Issues](#safe-loading-issues)

## File and Path Issues

### "Model file not found"

**Symptoms:**

```text
Error: Model file not found: /path/to/model.safetensors
```

**Causes:**

- File doesn't exist at specified path
- Manifest was generated on a different machine
- Paths are relative but script is run from wrong directory
- File was moved/renamed after manifest generation

**Solutions:**

1. **Check file exists:**

   ```bash
   ls -la /path/to/model.safetensors
   ```

2. **Use absolute paths in manifest:**

   ```json
   "models": [
     {"path": "/full/absolute/path/to/model.safetensors", "weight": 0.5}
   ]
   ```

3. **Run script from correct directory:**

   ```bash
   cd /directory/where/manifest/was/generated
   python run.py merge --manifest merge_manifest.json
   ```

4. **Re-scan if files moved:**

   ```bash
   python run.py scan ./new_location --output new_manifest.json
   ```

### "Output file exists"

**Symptoms:**

```text
Error: Output file already exists: output.safetensors
```

**Solutions:**

1. **Use --overwrite flag:**

   ```bash
   python run.py merge --manifest config.json --overwrite
   ```

2. **Change output filename in manifest:**

   ```json
   "output": {
     "path": "my_merge_v2.safetensors"
   }
   ```

3. **Delete existing file:**

   ```bash
   rm output.safetensors
   ```

### "Permission denied"

**Symptoms:**

```text
PermissionError: [Errno 13] Permission denied: 'output.safetensors'
```

**Solutions:**

1. **Check directory permissions:**

   ```bash
   ls -ld /output/directory
   ```

2. **Use directory you have write access to:**

   ```json
   "output": {
     "path": "~/models/my_merge.safetensors"
   }
   ```

3. **On Linux/Mac, check ownership:**

   ```bash
   sudo chown $USER:$USER /output/directory
   ```

## CUDA and GPU Issues

### "CUDA requested but not available"

**Symptoms:**

```text
⚠️  WARNING: CUDA requested but not available (PyTorch not built with CUDA support)
Falling back to CPU. To fix this:
...
```

**Diagnosis Steps:**

1. **Check if you have an NVIDIA GPU:**

   ```bash
   nvidia-smi
   ```

   If this fails, you don't have NVIDIA drivers installed or no compatible GPU.

2. **Check CUDA version:**

   ```bash
   nvidia-smi
   ```

   Look for "CUDA Version: X.X" in top-right corner.

3. **Check PyTorch CUDA support:**

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
   ```

#### Solution 1: Install CUDA-enabled PyTorch

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# For CUDA 13.0 (latest)
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

#### Solution 2: Force reinstall existing PyTorch

```bash
# Replace cu130 with your CUDA version
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu130
```

#### Solution 3: Verify installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Device name: {torch.cuda.get_device_name(0)}')"
```

Should output:

```text
CUDA available: True
Device count: 1
Device name: NVIDIA GeForce RTX 4090
```

#### Solution 4: Just use CPU

If you can't get CUDA working, CPU mode works fine (just slower):

```bash
python run.py merge --manifest config.json --device cpu
```

### "CUDA out of memory"

**Symptoms:**

```text
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Causes:**

- Models are too large for your GPU VRAM
- Merging too many models simultaneously
- Other applications using GPU

**Solutions:**

1. **Use CPU instead:**

   ```bash
   python run.py merge --manifest config.json --device cpu
   ```

2. **Close other GPU applications:**
   - Close games, video editing software
   - Close other AI tools (Stable Diffusion UIs, etc.)

3. **Check GPU memory usage:**

   ```bash
   nvidia-smi
   ```

4. **Reduce batch size (if applicable)**

5. **Upgrade GPU or use cloud GPU:**
   - Consider Google Colab, RunPod, Vast.ai

**Memory requirements:**

- SDXL model: ~7GB
- Merging 8 models on GPU: ~14GB VRAM needed (accumulator + current model)
- CPU mode: Works with less RAM due to swapping

## Memory Issues

### "Out of memory" (System RAM)

**Symptoms:**

```text
MemoryError: Unable to allocate X.XX GiB for an array
```

**Solutions:**

1. **Close other applications**

2. **Merge fewer models at once:**
   - Split 8 models into two 4-model merges
   - Merge the results together

3. **Increase swap space (Linux):**

   ```bash
   sudo fallocate -l 32G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Use CPU mode (if using CUDA):**
   CUDA mode requires both VRAM and RAM. CPU mode is more memory-efficient.

### Process killed / "Killed"

**Symptoms:**

```text
Killed
```

(Process terminates with no other error)

**Cause:**

- Linux OOM (Out Of Memory) killer terminated the process
- System ran out of RAM

**Solutions:**

See ["Out of memory" solutions above](#out-of-memory-system-ram)

## Conversion Issues

### "Safe loading failed"

**Symptoms:**

```text
Error: Cannot load checkpoint.ckpt safely (possibly contains custom Python code)
```

**Cause:**
The checkpoint uses old pickle format with custom Python code that our safety checks (`weights_only=True`) reject. This is INTENTIONAL - the file could contain malicious code!

**Solution for trusted files:**

If you created this file yourself or completely trust the source, use the manual unsafe converter:

```python
#!/usr/bin/env python3
"""
Manual unsafe checkpoint converter - USE AT YOUR OWN RISK!
Only use this for files YOU created or completely trust.
"""
import torch
from safetensors.torch import save_file
from pathlib import Path

# Your checkpoint file
input_file = Path("my_checkpoint.ckpt")
output_file = input_file.with_suffix(".safetensors")

print(f"⚠️  WARNING: Loading {input_file} WITHOUT safety checks!")
print("This file could contain malicious code. Proceed? [y/N]: ", end="")
if input().lower() != 'y':
    print("Cancelled.")
    exit()

# Load checkpoint (UNSAFE - can execute code!)
checkpoint = torch.load(input_file, map_location='cpu', weights_only=False)

# Extract state dict from various formats
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'string_to_param' in checkpoint:  # Textual inversion
        state_dict = checkpoint['string_to_param']
    else:
        state_dict = checkpoint  # Assume it's already a state dict
else:
    state_dict = {input_file.stem: checkpoint}  # Single tensor

# Remove 'module.' prefixes if present
cleaned = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        cleaned[key[7:]] = value
    else:
        cleaned[key] = value

# Make tensors contiguous and independent
for key in cleaned:
    if isinstance(cleaned[key], torch.Tensor):
        cleaned[key] = cleaned[key].contiguous().clone()

# Save as safetensors
print(f"Saving to {output_file}...")
save_file(cleaned, str(output_file))
print(f"✓ Converted! Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
```

Save this script, modify the `input_file` path, and run it. The interactive prompt forces you to acknowledge the risk.

**⚠️ NEVER use this script on files from untrusted sources!** Malicious pickle files can delete your data, steal credentials, or install malware.

### "Conversion verification failed"

**Symptoms:**

```text
Warning: Output verification failed - file may be corrupted
```

**Causes:**

- Source file was corrupt
- Disk error during write
- Interrupted conversion

**Solutions:**

1. **Try conversion again:**

   ```bash
   python run.py convert model.ckpt --overwrite
   ```

2. **Use deep verification:**

   ```bash
   python run.py verify model.ckpt model.safetensors --verbose
   ```

3. **Check source file:**
   - Try loading in your SD UI first
   - Verify file size matches expected size

4. **Check disk space:**

   ```bash
   df -h
   ```

### "Unrecognized file extension"

**Symptoms:**

```text
Warning: Unrecognized extension '.model' (not in .ckpt, .pt, .pth, .bin)
...
Attempt conversion anyway? [y/N]:
```

**Solutions:**

1. **Confirm it's a valid PyTorch file** and answer `y`

2. **Use --force flag to skip prompt:**

   ```bash
   python run.py convert unusual_file.model --force
   ```

3. **Rename file to standard extension:**

   ```bash
   mv file.model file.ckpt
   python run.py convert file.ckpt
   ```

## Merge Compatibility Issues

### "Models are incompatible"

**Symptoms:**

```text
Error: Models have incompatible tensor shapes
Model 1: tensor 'key' has shape [X, Y, Z]
Model 2: tensor 'key' has shape [A, B, C]
```

**Causes:**

- Different architectures (Pony vs Illustrious)
- Different model types (SD1.5 vs SDXL)
- One model is corrupted

**Solutions:**

1. **Check architectures in manifest:**

   ```json
   "models": [
     {"path": "model1.safetensors", "architecture": "Pony"},  // All should match!
     {"path": "model2.safetensors", "architecture": "Pony"},
     {"path": "model3.safetensors", "architecture": "SDXL"}   // ← Wrong!
   ]
   ```

2. **Only merge same-architecture models:**
   - Pony with Pony
   - SDXL with SDXL
   - Illustrious with Illustrious

3. **Test models individually:**
   - Try loading each model in your SD UI
   - Verify none are corrupted

4. **Re-scan to detect architectures:**

   ```bash
   python run.py scan ./models --output new_manifest.json
   ```

### "Tensor shape mismatch"

**Symptoms:**
Similar to "Models are incompatible" but for specific tensors.

**Solutions:**

1. **Check model files aren't corrupted:**

   ```bash
   # Should load without error
   python -c "from safetensors.torch import load_file; load_file('model.safetensors')"
   ```

2. **Check precision matches:**
   - Don't merge fp16 with fp32 directly
   - Or use `"output_precision": "fp16"` to force conversion

3. **Verify models are from same family:**
   - Some Pony variants have modified architectures
   - Check model descriptions/documentation

## Safe Loading Issues

### "ImportError: No module named 'torch'"

**Symptoms:**

```text
ImportError: No module named 'torch'
```

**Solutions:**

```bash
pip install torch
```

Or for CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### "ImportError: DLL load failed" (Windows)

**Symptoms:**

```text
ImportError: DLL load failed while importing _C
```

**Cause:**
Missing Visual C++ Redistributable

**Solution:**

Download and install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

Then restart your terminal.

## Getting More Help

If you're still stuck:

1. **Check your manifest file** - Most issues come from incorrect paths or settings
2. **Try with CPU mode** - Rules out GPU-specific issues
3. **Test with smaller models** - Rules out size/memory issues
4. **Verify individual files** - Make sure source files work in your SD UI
5. **Check the CHANGELOG** - Your issue might be a known bug that's fixed in newer version

## Next Steps

- [Usage Guide](usage.md) - Learn core workflows
- [Customization Guide](customization.md) - Configure for your needs
- [Installation Guide](installation.md) - GPU setup and requirements
- [FAQ](FAQ.md) - Frequently asked questions
