"""
Model and VAE loading utilities.

Handles reading safetensors files, detecting their precision (fp16/fp32),
and computing SHA-256 hashes for verification/lookup.
"""

import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from safetensors.torch import load_file

from . import config
from .console import console, print_info
from .hasher import compute_file_hash


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
    
    console.print(f"[cyan]Loading model:[/cyan] {filepath.name}")
    
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
        print_info(f"Computing hash for {filepath.name}...")
        metadata['hash'] = compute_file_hash(filepath)
    
    console.print(f"  [dim]Loaded {len(state_dict)} keys, precision: {precision}[/dim]")
    
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
    
    console.print(f"[cyan]Loading VAE:[/cyan] {filepath.name}")
    
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
        print_info(f"Computing hash for {filepath.name}...")
        metadata['hash'] = compute_file_hash(filepath)
    
    console.print(f"  [dim]Loaded VAE with {len(state_dict)} keys, precision: {precision}[/dim]")
    
    return state_dict, metadata


def validate_models_compatible(
    reference_shapes: Dict[str, torch.Size],
    reference_keys: set,
    current_model: Dict[str, torch.Tensor],
    reference_name: str,
    current_name: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate that two models have compatible structures for merging.
    
    Args:
        reference_shapes: Tensor shapes from reference model
        reference_keys: Set of keys from reference model
        current_model: Current model state dict to validate
        reference_name: Name of reference model (for error messages)
        current_name: Name of current model (for error messages)
        
    Returns:
        (is_compatible, error_message) tuple
    """
    current_keys = set(current_model.keys())
    
    # Check if key sets match
    if reference_keys != current_keys:
        missing_in_current = reference_keys - current_keys
        extra_in_current = current_keys - reference_keys
        
        error_parts = []
        if missing_in_current:
            error_parts.append(f"Missing {len(missing_in_current)} keys in {current_name}")
        if extra_in_current:
            error_parts.append(f"Extra {len(extra_in_current)} keys in {current_name}")
        
        return False, " | ".join(error_parts)
    
    # Check if shapes match for all keys
    for key in reference_keys:
        if key not in current_model:
            continue  # Already caught above
            
        if isinstance(current_model[key], torch.Tensor):
            reference_shape = reference_shapes.get(key)
            current_shape = current_model[key].shape
            
            if reference_shape != current_shape:
                return False, (
                    f"Shape mismatch for key '{key}': "
                    f"{reference_name} has {reference_shape}, "
                    f"{current_name} has {current_shape}"
                )
    
    return True, None