"""
Multi-model merging with accumulator pattern.

This module implements the core weighted merge algorithm using an accumulator
pattern to avoid loading all models into memory at once.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from . import config
from .loader import load_model, validate_models_compatible
from .manifest import ModelEntry


def merge_models(
    model_entries: List[ModelEntry],
    device: str = 'cpu',
    validate_compatibility: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple models using weighted accumulation.
    
    This is the heart of the whole operation! We use an accumulator pattern:
    1. Load first model, multiply by its weight -> this is our accumulator
    2. For each subsequent model:
       - Load it
       - Multiply by its weight
       - Add to accumulator
       - Free it from memory
    3. Return the accumulated result
    
    This way we only ever have 2 models in memory at once (accumulator + current),
    instead of loading all 8+ models simultaneously and exploding your RAM.
    
    The math works because addition is associative:
        (A*0.125 + B*0.125) + C*0.125 == A*0.125 + B*0.125 + C*0.125
    
    Args:
        model_entries: List of models with their weights
        device: Device to compute on ('cpu' or 'cuda')
        validate_compatibility: Check models are compatible before merging
        
    Returns:
        The merged model state dict
        
    Raises:
        ValueError: If models are incompatible or no models provided
    """
    if not model_entries:
        raise ValueError("No models provided for merging")
    
    if len(model_entries) == 1:
        print("Warning: Only one model provided, returning it unmodified")
        state_dict, _ = load_model(Path(model_entries[0].path), device=device)
        return state_dict
    
    print(f"\nMerging {len(model_entries)} models...")
    print("=" * 60)
    
    # Load first model - this becomes our accumulator
    first_entry = model_entries[0]
    print(f"\n[1/{len(model_entries)}] Loading base model: {Path(first_entry.path).name}")
    print(f"  Weight: {first_entry.weight}")
    
    accumulator, first_metadata = load_model(
        Path(first_entry.path),
        device=device,
        compute_hash=False
    )
    
    # Multiply first model by its weight
    print(f"  Applying weight {first_entry.weight}...")
    for key in tqdm(accumulator.keys(), desc="  Weighting base model"):
        if config.should_skip_merge_key(key):
            continue
        if 'weight' in key or 'bias' in key:
            accumulator[key] = accumulator[key].to(torch.float32) * first_entry.weight
    
    # Store reference model for compatibility checking
    reference_dict = {k: v for k, v in accumulator.items()}
    
    # Process remaining models
    for idx, entry in enumerate(model_entries[1:], start=2):
        model_path = Path(entry.path)
        print(f"\n[{idx}/{len(model_entries)}] Loading: {model_path.name}")
        print(f"  Weight: {entry.weight}")
        
        # Load model
        current_model, current_metadata = load_model(
            model_path,
            device=device,
            compute_hash=False
        )
        
        # Validate compatibility if requested
        if validate_compatibility:
            compatible, error_msg = validate_models_compatible(
                reference_dict,
                current_model,
                Path(first_entry.path).name,
                model_path.name
            )
            if not compatible:
                raise ValueError(f"Models are incompatible: {error_msg}")
        
        # Accumulate this model
        print(f"  Merging with weight {entry.weight}...")
        merged_keys = 0
        skipped_keys = 0
        
        for key in tqdm(current_model.keys(), desc="  Accumulating"):
            if config.should_skip_merge_key(key):
                skipped_keys += 1
                continue
            
            # Only merge keys that exist in both models
            if key not in accumulator:
                skipped_keys += 1
                continue
            
            if 'weight' in key or 'bias' in key:
                # Add weighted tensor to accumulator
                accumulator[key] = (
                    accumulator[key] + 
                    current_model[key].to(torch.float32) * entry.weight
                )
                merged_keys += 1
        
        print(f"  Merged {merged_keys} tensors, skipped {skipped_keys}")
        
        # Free memory
        del current_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean up reference dict
    del reference_dict
    
    print("\n" + "=" * 60)
    print(f"✓ Merge complete! Combined {len(model_entries)} models.")
    print(f"  Total tensors in result: {len(accumulator)}")
    
    return accumulator


def convert_precision(
    state_dict: Dict[str, torch.Tensor],
    target_precision: str
) -> Dict[str, torch.Tensor]:
    """
    Convert model precision to target dtype.
    
    After accumulation in fp32 (for numerical stability), we might want
    to convert back to fp16 to save space. Or vice versa!
    
    Args:
        state_dict: Model state dictionary
        target_precision: Target precision ('fp16', 'fp32', 'bf16')
        
    Returns:
        State dict with converted tensors
    """
    if target_precision not in ['fp16', 'fp32', 'bf16']:
        raise ValueError(f"Unknown precision: {target_precision}")
    
    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    
    target_dtype = dtype_map[target_precision]
    
    print(f"\nConverting model to {target_precision}...")
    
    converted = {}
    for key, tensor in tqdm(state_dict.items(), desc="Converting precision"):
        # Only convert floating point tensors
        if tensor.dtype in {torch.float32, torch.float16, torch.bfloat16, torch.float64}:
            converted[key] = tensor.to(target_dtype)
        else:
            # Keep integer tensors, etc. as-is
            converted[key] = tensor
    
    return converted


def prune_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove unnecessary keys from the model.
    
    This strips out training artifacts, optimizer states, EMA weights,
    and other cruft that bloats the file without affecting generation.
    
    We keep only:
    - model.diffusion_model.* (the actual SD model)
    - first_stage_model.* (the VAE)
    - cond_stage_model.* / conditioner.* (text encoders)
    
    Everything else gets the axe! This can save hundreds of MB.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Pruned state dict with only essential keys
    """
    print("\nPruning unnecessary keys...")
    
    original_count = len(state_dict)
    pruned = {}
    
    for key, tensor in tqdm(state_dict.items(), desc="Pruning"):
        if not config.should_prune_key(key):
            pruned[key] = tensor
    
    removed_count = original_count - len(pruned)
    print(f"  Removed {removed_count} keys, kept {len(pruned)} keys")
    
    return pruned