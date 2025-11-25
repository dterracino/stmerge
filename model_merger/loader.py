"""
Model and VAE loading utilities.

Handles reading safetensors files, detecting their precision (fp16/fp32),
and computing SHA-256 hashes for verification/lookup.
"""

import hashlib
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from safetensors.torch import load_file
from tqdm import tqdm

from . import config


def compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Reads the file in chunks to avoid loading huge models into memory.
    This is useful for verification and for looking up models in databases
    like CivitAI later.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the SHA-256 hash
    """
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    
    return sha256.hexdigest()


def detect_precision(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Detect the precision of a model by examining its tensors.
    
    We look at the first tensor with 'weight' in its name and check its dtype.
    Models are typically all fp16 or all fp32, so sampling one is enough.
    
    Args:
        state_dict: The model's state dictionary
        
    Returns:
        'fp16', 'fp32', or 'unknown'
    """
    # Find first weight tensor
    for key, tensor in state_dict.items():
        if 'weight' in key:
            if tensor.dtype == torch.float16:
                return 'fp16'
            elif tensor.dtype == torch.float32:
                return 'fp32'
            elif tensor.dtype == torch.bfloat16:
                return 'bf16'
            else:
                return 'unknown'
    
    return 'unknown'


def load_model(
    filepath: Path,
    device: str = 'cpu',
    compute_hash: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a model from a safetensors file.
    
    Args:
        filepath: Path to the .safetensors file
        device: Device to load tensors to ('cpu' or 'cuda')
        compute_hash: Whether to compute SHA-256 hash (slow for large files)
        
    Returns:
        Tuple of (state_dict, metadata)
        - state_dict: Dictionary of tensor names to tensors
        - metadata: Dict with 'precision', 'hash' (if computed), 'filename'
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file isn't a .safetensors file
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if filepath.suffix not in config.SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. "
                        f"Supported formats: {config.SUPPORTED_MODEL_EXTENSIONS}")
    
    print(f"Loading model: {filepath.name}")
    
    # Load the state dict
    state_dict = load_file(str(filepath), device=device)
    
    # Detect precision
    precision = detect_precision(state_dict)
    
    # Build metadata
    metadata = {
        'filename': filepath.name,
        'precision': precision,
    }
    
    # Optionally compute hash (slow!)
    if compute_hash:
        print(f"  Computing hash for {filepath.name}...")
        metadata['hash'] = compute_file_hash(filepath)
    
    print(f"  Loaded {len(state_dict)} keys, precision: {precision}")
    
    return state_dict, metadata


def load_vae(
    filepath: Path,
    device: str = 'cpu',
    compute_hash: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a VAE from a safetensors file.
    
    VAEs are just smaller models, so this is basically the same as load_model,
    but we're being explicit about the intent and could add VAE-specific
    validation later if needed.
    
    Args:
        filepath: Path to the VAE .safetensors file
        device: Device to load tensors to ('cpu' or 'cuda')
        compute_hash: Whether to compute SHA-256 hash
        
    Returns:
        Tuple of (state_dict, metadata)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file isn't a .safetensors file
    """
    if not filepath.exists():
        raise FileNotFoundError(f"VAE file not found: {filepath}")
    
    if filepath.suffix not in config.SUPPORTED_VAE_EXTENSIONS:
        raise ValueError(f"Unsupported VAE format: {filepath.suffix}")
    
    print(f"Loading VAE: {filepath.name}")
    
    # Load the state dict
    state_dict = load_file(str(filepath), device=device)
    
    # Detect precision
    precision = detect_precision(state_dict)
    
    # Build metadata
    metadata = {
        'filename': filepath.name,
        'precision': precision,
    }
    
    # Optionally compute hash
    if compute_hash:
        print(f"  Computing hash for {filepath.name}...")
        metadata['hash'] = compute_file_hash(filepath)
    
    print(f"  Loaded VAE with {len(state_dict)} keys, precision: {precision}")
    
    return state_dict, metadata


def validate_models_compatible(
    model_a_dict: Dict[str, torch.Tensor],
    model_b_dict: Dict[str, torch.Tensor],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> Tuple[bool, Optional[str]]:
    """
    Check if two models are compatible for merging.
    
    Models are compatible if:
    1. They have the same keys (or at least overlapping keys)
    2. Corresponding tensors have the same shapes
    
    This prevents us from trying to merge a Pony model with an Illustrious
    model, which would explode spectacularly.
    
    Args:
        model_a_dict: First model's state dict
        model_b_dict: Second model's state dict
        model_a_name: Name for error messages
        model_b_name: Name for error messages
        
    Returns:
        Tuple of (is_compatible, error_message)
        - is_compatible: True if models can be merged
        - error_message: Description of incompatibility, or None if compatible
    """
    # Get keys that exist in both models
    keys_a = set(model_a_dict.keys())
    keys_b = set(model_b_dict.keys())
    common_keys = keys_a & keys_b
    
    # Check if there's significant overlap
    if len(common_keys) < 0.8 * min(len(keys_a), len(keys_b)):
        missing_in_b = len(keys_a - keys_b)
        missing_in_a = len(keys_b - keys_a)
        return False, (
            f"Models have insufficient key overlap. "
            f"{model_a_name} has {missing_in_b} keys not in {model_b_name}. "
            f"{model_b_name} has {missing_in_a} keys not in {model_a_name}. "
            f"These models likely have different architectures."
        )
    
    # Check shapes for common keys
    for key in common_keys:
        if config.should_skip_merge_key(key):
            continue
            
        shape_a = model_a_dict[key].shape
        shape_b = model_b_dict[key].shape
        
        if shape_a != shape_b:
            return False, (
                f"Shape mismatch at key '{key}': "
                f"{model_a_name} has shape {shape_a}, "
                f"{model_b_name} has shape {shape_b}. "
                f"These models are not compatible for merging."
            )
    
    return True, None