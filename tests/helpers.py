"""
Test helper utilities.

Provides functions for creating dummy models and cleaning up test files.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
from safetensors.torch import save_file


def create_dummy_model(
    name: str,
    precision: str = "fp32",
    size: int = 10,
    temp_dir: Path = Path("tests/temp"),
    seed: Optional[int] = None,
    include_vae: bool = False,
    custom_keys: Optional[Dict[str, torch.Tensor]] = None
) -> Path:
    """
    Create a tiny dummy model for testing.
    
    Args:
        name: Filename for the model
        precision: 'fp16', 'fp32', or 'bf16'
        size: Size of tensor dimensions
        temp_dir: Directory to save the model
        seed: Random seed for reproducibility
        include_vae: Whether to include VAE keys
        custom_keys: Custom tensors to include
        
    Returns:
        Path to the created model file
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float32)
    
    # Create minimal model structure
    state_dict = {
        "model.diffusion_model.layer1.weight": torch.randn(size, size, dtype=dtype),
        "model.diffusion_model.layer1.bias": torch.randn(size, dtype=dtype),
        "model.diffusion_model.layer2.weight": torch.randn(size, size, dtype=dtype),
        "model.diffusion_model.layer2.bias": torch.randn(size, dtype=dtype),
        "cond_stage_model.transformer.text_model.embeddings.position_ids": torch.arange(77),
    }
    
    if include_vae:
        state_dict.update({
            "first_stage_model.encoder.conv_in.weight": torch.randn(size, size, 3, 3, dtype=dtype),
            "first_stage_model.encoder.conv_in.bias": torch.randn(size, dtype=dtype),
            "first_stage_model.decoder.conv_out.weight": torch.randn(3, size, 3, 3, dtype=dtype),
            "first_stage_model.decoder.conv_out.bias": torch.randn(3, dtype=dtype),
        })
    
    if custom_keys:
        state_dict.update(custom_keys)
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    save_file(state_dict, str(path))
    return path


def create_dummy_vae(
    name: str,
    precision: str = "fp32",
    size: int = 10,
    temp_dir: Path = Path("tests/temp"),
    seed: Optional[int] = None
) -> Path:
    """
    Create a tiny dummy VAE for testing.
    
    Args:
        name: Filename for the VAE
        precision: 'fp16', 'fp32', or 'bf16'
        size: Size of tensor dimensions
        temp_dir: Directory to save the VAE
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created VAE file
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float32)
    
    # Create minimal VAE structure (without first_stage_model prefix)
    state_dict = {
        "encoder.conv_in.weight": torch.randn(size, 3, 3, 3, dtype=dtype),
        "encoder.conv_in.bias": torch.randn(size, dtype=dtype),
        "encoder.conv_out.weight": torch.randn(size, size, 3, 3, dtype=dtype),
        "encoder.conv_out.bias": torch.randn(size, dtype=dtype),
        "decoder.conv_in.weight": torch.randn(size, size, 3, 3, dtype=dtype),
        "decoder.conv_in.bias": torch.randn(size, dtype=dtype),
        "decoder.conv_out.weight": torch.randn(3, size, 3, 3, dtype=dtype),
        "decoder.conv_out.bias": torch.randn(3, dtype=dtype),
    }
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    save_file(state_dict, str(path))
    return path


def create_incompatible_model(
    name: str,
    precision: str = "fp32",
    size: int = 20,
    temp_dir: Path = Path("tests/temp")
) -> Path:
    """
    Create a model with different tensor shapes for incompatibility testing.
    
    Args:
        name: Filename for the model
        precision: 'fp16' or 'fp32'
        size: Different size for tensors (default 20 vs 10 for normal models)
        temp_dir: Directory to save the model
        
    Returns:
        Path to the created model file
    """
    dtype = torch.float16 if precision == "fp16" else torch.float32
    
    state_dict = {
        "model.diffusion_model.layer1.weight": torch.randn(size, size, dtype=dtype),
        "model.diffusion_model.layer1.bias": torch.randn(size, dtype=dtype),
        "model.diffusion_model.layer2.weight": torch.randn(size, size, dtype=dtype),
        "model.diffusion_model.layer2.bias": torch.randn(size, dtype=dtype),
        "cond_stage_model.transformer.text_model.embeddings.position_ids": torch.arange(77),
    }
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    save_file(state_dict, str(path))
    return path


def create_dummy_checkpoint(
    name: str,
    temp_dir: Path = Path("tests/temp"),
    format_type: str = "wrapped",
    size: int = 10
) -> Path:
    """
    Create a dummy pickle checkpoint for conversion testing.
    
    Args:
        name: Filename for the checkpoint (should end in .ckpt, .pt, or .pth)
        temp_dir: Directory to save the checkpoint
        format_type: 'bare', 'wrapped', or 'nested'
        size: Size of tensor dimensions
        
    Returns:
        Path to the created checkpoint file
    """
    state_dict = {
        "model.diffusion_model.layer1.weight": torch.randn(size, size),
        "model.diffusion_model.layer1.bias": torch.randn(size),
        "first_stage_model.encoder.weight": torch.randn(size, size),
    }
    
    if format_type == "bare":
        checkpoint = state_dict
    elif format_type == "wrapped":
        checkpoint = {
            "state_dict": state_dict,
            "optimizer_state": {"lr": 0.001},
            "epoch": 10,
        }
    elif format_type == "nested":
        checkpoint = {
            "model": state_dict,
            "epoch": 10,
        }
    else:
        checkpoint = state_dict
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    torch.save(checkpoint, path)
    return path


def cleanup_test_files(files: List[Path]) -> None:
    """
    Clean up test files.
    
    Args:
        files: List of file paths to delete
    """
    for file_path in files:
        if file_path.exists():
            file_path.unlink()


def get_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors file and return its state dict.
    
    Args:
        path: Path to the safetensors file
        
    Returns:
        State dict
    """
    from safetensors.torch import load_file
    return load_file(str(path))
