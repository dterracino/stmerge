"""
Multi-model merging with accumulator pattern.

This module implements the core weighted merge algorithm using an accumulator
pattern to avoid loading all models into memory at once.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .loader import load_model, validate_models_compatible
from .manifest import ModelEntry
from .console import console, create_progress, print_step, print_success, print_section


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
        console.print("[yellow]Warning:[/yellow] Only one model provided, returning it unmodified")
        state_dict, _ = load_model(Path(model_entries[0].path), device=device)
        return state_dict
    
    print_section(f"Merging {len(model_entries)} Models")
    
    # Load first model - this becomes our accumulator
    first_entry = model_entries[0]
    print_step(1, len(model_entries), f"Loading base model: [bold]{Path(first_entry.path).name}[/bold]")
    console.print(f"  [dim]Weight: {first_entry.weight}[/dim]")
    
    accumulator, first_metadata = load_model(
        Path(first_entry.path),
        device=device,
        compute_hash=False
    )
    
    # Multiply first model by its weight
    console.print(f"  [cyan]Applying weight {first_entry.weight}...[/cyan]")
    with create_progress() as progress:
        task = progress.add_task("Weighting base model", total=len(accumulator))
        for key in accumulator.keys():
            if config.should_skip_merge_key(key):
                continue
            if 'weight' in key or 'bias' in key:
                accumulator[key] = accumulator[key].to(torch.float32) * first_entry.weight
            progress.advance(task)
    
    # Store reference model for compatibility checking
    reference_dict = {k: v for k, v in accumulator.items()}
    
    # Process remaining models
    for idx, entry in enumerate(model_entries[1:], start=2):
        model_path = Path(entry.path)
        print_step(idx, len(model_entries), f"Loading: [bold]{model_path.name}[/bold]")
        console.print(f"  [dim]Weight: {entry.weight}[/dim]")
        
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
        console.print(f"  [cyan]Merging with weight {entry.weight}...[/cyan]")
        merged_keys = 0
        skipped_keys = 0
        
        with create_progress() as progress:
            task = progress.add_task("Accumulating", total=len(current_model))
            for key in current_model.keys():
                if config.should_skip_merge_key(key):
                    skipped_keys += 1
                    progress.advance(task)
                    continue
                
                # Only merge keys that exist in both models
                if key not in accumulator:
                    skipped_keys += 1
                    progress.advance(task)
                    continue
                
                if 'weight' in key or 'bias' in key:
                    # Add weighted tensor to accumulator
                    accumulator[key] = (
                        accumulator[key] + 
                        current_model[key].to(torch.float32) * entry.weight
                    )
                    merged_keys += 1
                
                progress.advance(task)
        
        console.print(f"  [dim]Merged {merged_keys} tensors, skipped {skipped_keys}[/dim]")
        
        # Free memory
        del current_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean up reference dict
    del reference_dict
    
    print_success(f"Merge complete! Combined {len(model_entries)} models.")
    console.print(f"  [dim]Total tensors in result: {len(accumulator)}[/dim]\n")
    
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
    
    console.print(f"\n[cyan]Converting model to {target_precision}...[/cyan]")
    
    converted = {}
    with create_progress() as progress:
        task = progress.add_task("Converting precision", total=len(state_dict))
        for key, tensor in state_dict.items():
            # Only convert floating point tensors
            if tensor.dtype in {torch.float32, torch.float16, torch.bfloat16, torch.float64}:
                converted[key] = tensor.to(target_dtype)
            else:
                # Keep integer tensors, etc. as-is
                converted[key] = tensor
            progress.advance(task)
    
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
    console.print("\n[cyan]Pruning unnecessary keys...[/cyan]")
    
    original_count = len(state_dict)
    pruned = {}
    
    with create_progress() as progress:
        task = progress.add_task("Pruning", total=len(state_dict))
        for key, tensor in state_dict.items():
            if not config.should_prune_key(key):
                pruned[key] = tensor
            progress.advance(task)
    
    removed_count = original_count - len(pruned)
    console.print(f"  [dim]Removed {removed_count} keys, kept {len(pruned)} keys[/dim]")
    
    return pruned