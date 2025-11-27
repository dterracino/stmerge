# Installation Guide

## Basic Installation

```bash
# Clone or download this repo
git clone https://github.com/yourusername/model-merger.git
cd model-merger

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+**
- **safetensors 0.4.0+**
- **rich 13.0+** (for beautiful terminal output)
- **numpy 1.24+**
- **packaging 21.0+** (dependency of safetensors)
- **win10toast 0.9+** (optional, Windows only - for desktop notifications)

## GPU Acceleration (Optional)

The default installation uses **CPU-only PyTorch**. For GPU acceleration with CUDA:

### Step 1: Check Your CUDA Version

```bash
nvidia-smi
```

Look for the CUDA version in the top-right of the output (e.g., "CUDA Version: 13.0").

### Step 2: Install CUDA-Enabled PyTorch

Choose the command matching your CUDA version:

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

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

### Already Have PyTorch Installed?

Force reinstall with CUDA support:

```bash
# Replace cu130 with your CUDA version
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu130
```

### Using GPU Acceleration

Once installed, use the `--device cuda` flag for merging operations:

```bash
python run.py merge --manifest config.json --device cuda
```

The tool will automatically detect if CUDA is unavailable and fall back to CPU with a helpful warning.

**Note:** GPU acceleration is most beneficial for merging large models. Conversions are I/O-bound and won't benefit much from GPU.

## Troubleshooting Installation

### "No module named 'torch'"

You haven't installed PyTorch yet. Run:

```bash
pip install torch
```

### "CUDA requested but not available"

See the [Troubleshooting Guide](troubleshooting.md#cuda-requested-but-not-available) for detailed solutions.

### "ImportError: DLL load failed"

On Windows, you may need to install Visual C++ Redistributable:

- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Install and restart your terminal

### Permission Errors

On Linux/Mac, you may need to use `pip install --user`:

```bash
pip install --user -r requirements.txt
```

## Next Steps

- [Usage Guide](usage.md) - Learn how to use the tool
- [Customization](customization.md) - Configure architecture patterns
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [FAQ](FAQ.md) - Frequently asked questions
