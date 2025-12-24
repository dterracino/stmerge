"""
Multi-model merging with accumulator pattern.

This module implements the core weighted merge algorithm using an accumulator
pattern to avoid loading all models into memory at once.
"""

import gc
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import config
from .loader import load_model, validate_models_compatible
from .manifest import ModelEntry
from .console import console, create_progress, print_step, print_success, print_section


def compute_consensus_weights(
    values: Union[List[float], torch.Tensor],
    exponent: int = 4
) -> torch.Tensor:
    """
    Compute consensus-based weights for merging using inverse distance weighting.
    
    This algorithm suppresses outliers by giving higher weights to values that are
    closer to the consensus (average position) of all values. The algorithm:
    
    1. Compute average pairwise distance for each value to all others
    2. Normalize distances (min=0, max=1)
    3. Invert (so consensus values get high scores: 1 - normalized)
    4. Apply power to exponentially suppress outliers
    5. Normalize to sum=1 (create probability distribution)
    
    Args:
        values: List or tensor of values from different models (one per model)
        exponent: Power to apply for outlier suppression (higher = more aggressive)
                  Typical range: 2-8, default 4
        
    Returns:
        Tensor of normalized weights summing to 1.0
        
    Example:
        >>> values = [1.0, 1.1, 1.05, 5.0]  # Last value is outlier
        >>> weights = compute_consensus_weights(values, exponent=4)
        >>> # weights ≈ [0.33, 0.34, 0.33, 0.00] - outlier suppressed
    """
    if isinstance(values, list):
        values = torch.tensor(values, dtype=torch.float32)
    else:
        values = values.float()
    
    n = len(values)
    if n == 1:
        return torch.ones(1)
    
    # Step 1: Compute pairwise distances (average distance from each value to all others)
    # Broadcasting: values[:, None] is shape (n, 1), values is shape (n,)
    # Result: (n, n) matrix of all pairwise distances
    pairwise_distances = torch.abs(values[:, None] - values)
    avg_distances = pairwise_distances.mean(dim=1)  # Average distance for each value
    
    # Step 2: Normalize to [0, 1] range
    min_dist = avg_distances.min()
    max_dist = avg_distances.max()
    
    if max_dist - min_dist < 1e-10:  # All values identical
        return torch.ones(n) / n
    
    normalized = (avg_distances - min_dist) / (max_dist - min_dist)
    
    # Step 3: Invert (consensus values now have scores near 1, outliers near 0)
    inverted = 1.0 - normalized
    
    # Step 4: Apply power (exponentially suppress outliers)
    powered = inverted ** exponent
    
    # Step 5: Normalize to probability distribution
    weights = powered / powered.sum()
    
    return weights


def merge_models(
    model_entries: List[ModelEntry],
    device: str = 'cpu',
    validate_compatibility: bool = True,
    merge_method: str = config.MERGE_METHOD_WEIGHTED_SUM,
    consensus_exponent: int = config.DEFAULT_CONSENSUS_EXPONENT
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple models using the specified merging strategy.
    
    Supports two merge methods:
    - 'weighted_sum': Traditional weighted linear interpolation using accumulator pattern
    - 'consensus': Outlier-resistant merging using inverse distance weighting
    
    Weighted Sum (default):
        Uses an accumulator pattern to avoid loading all models at once.
        Only 2 models in memory at a time. Fast and memory-efficient.
        Math: result = model_a * weight_a + model_b * weight_b + ...
    
    Consensus:
        Uses per-element consensus weighting to suppress outliers.
        Loads one tensor at a time from all models using memory-mapped files.
        Slower but produces better results when merging many diverse models.
        For each weight position, computes adaptive weights based on inter-model agreement.
    
    Args:
        model_entries: List of models with their weights
        device: Device to compute on ('cpu' or 'cuda')
        validate_compatibility: Check models are compatible before merging
        merge_method: 'weighted_sum' or 'consensus'
        consensus_exponent: Power for consensus outlier suppression (2-8, default 4)
        
    Returns:
        The merged model state dict
        
    Raises:
        ValueError: If models are incompatible, no models provided, or invalid merge_method
    """
    if not model_entries:
        raise ValueError("No models provided for merging")
    
    # Route to appropriate merge strategy
    if merge_method == config.MERGE_METHOD_CONSENSUS:
        return _consensus_merge(
            model_entries,
            device=device,
            validate_compatibility=validate_compatibility,
            exponent=consensus_exponent
        )
    elif merge_method == config.MERGE_METHOD_WEIGHTED_SUM:
        return _weighted_sum_merge(
            model_entries,
            device=device,
            validate_compatibility=validate_compatibility
        )
    else:
        raise ValueError(
            f"Unknown merge method: {merge_method}. "
            f"Must be '{config.MERGE_METHOD_WEIGHTED_SUM}' or '{config.MERGE_METHOD_CONSENSUS}'"
        )


def _weighted_sum_merge(
    model_entries: List[ModelEntry],
    device: str = 'cpu',
    validate_compatibility: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Internal: Merge models using weighted sum with accumulator pattern.
    
    This is the traditional merge algorithm. See merge_models() docstring for details.
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
    
    # Store only shapes for compatibility checking (memory efficient!)
    reference_shapes = {k: v.shape for k, v in accumulator.items()}
    reference_keys = set(accumulator.keys())

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
        compatible = True
        error_msg = None
        if validate_compatibility:
            compatible, error_msg = validate_models_compatible(
                reference_shapes,  # Changed from reference_dict
                reference_keys,    # Added
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
                    # In-place addition to avoid creating intermediate tensors
                    weighted_tensor = current_model[key].to(torch.float32) * entry.weight
                    accumulator[key].add_(weighted_tensor)  # In-place!
                    del weighted_tensor  # Free immediately
                    merged_keys += 1
                
                progress.advance(task)
        
        console.print(f"  [dim]Merged {merged_keys} tensors, skipped {skipped_keys}[/dim]")
        
        # AGGRESSIVE MEMORY CLEANUP
        # Move tensors to CPU before deletion (frees GPU memory if on CUDA)
        if device != 'cpu':
            current_model = {k: v.cpu() for k, v in current_model.items()}
        
        # Delete the model
        del current_model
        
        # Force Python's garbage collector to run NOW
        # This is critical - Python's GC is lazy and we need immediate cleanup
        gc.collect()
        
        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA ops to finish
    
    # Clean up reference data
    del reference_shapes
    del reference_keys

    print_success(f"Merge complete! Combined {len(model_entries)} models.")
    console.print(f"  [dim]Total tensors in result: {len(accumulator)}[/dim]\n")
    
    return accumulator


def _consensus_merge(
    model_entries: List[ModelEntry],
    device: str = 'cpu',
    validate_compatibility: bool = True,
    exponent: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Internal: Merge models using consensus-based outlier-resistant weighting.
    
    This algorithm computes adaptive weights for each individual parameter position
    based on inter-model agreement. Values that are outliers get suppressed.
    
    Uses memory-mapped file loading to only load one tensor at a time from all models,
    avoiding the need to load all models into memory simultaneously.
    
    Args:
        model_entries: List of models (user-provided weights are ignored in consensus mode)
        device: Device to compute on ('cpu' or 'cuda')
        validate_compatibility: Check models are compatible before merging
        exponent: Power for outlier suppression (higher = more aggressive)
    
    Returns:
        The consensus-merged model state dict
    """
    import safetensors.torch
    
    if not model_entries:
        raise ValueError("No models provided for merging")
    
    if len(model_entries) == 1:
        console.print("[yellow]Warning:[/yellow] Only one model provided, returning it unmodified")
        state_dict, _ = load_model(Path(model_entries[0].path), device=device)
        return state_dict
    
    print_section(f"Consensus Merging {len(model_entries)} Models")
    console.print(f"  [dim]Using exponent={exponent} for outlier suppression[/dim]")
    console.print(f"  [yellow]Note:[/yellow] User-provided weights are ignored in consensus mode\n")
    
    # Extract paths for easier handling
    model_paths = [Path(entry.path) for entry in model_entries]
    
    # Memory-map all model files (cheap operation, just file handles)
    console.print("[cyan]Opening model files (memory-mapped)...[/cyan]")
    try:
        mapped_files = [safetensors.torch.load_file(str(path), device=device) for path in model_paths]
    except Exception as e:
        raise ValueError(f"Failed to open model files: {e}")
    
    # Get tensor keys from first model
    reference_keys = set(mapped_files[0].keys())
    console.print(f"  [dim]Found {len(reference_keys)} tensors to merge[/dim]\n")
    
    # Validate compatibility if requested
    if validate_compatibility:
        console.print("[cyan]Validating model compatibility...[/cyan]")
        reference_shapes = {k: mapped_files[0][k].shape for k in reference_keys}
        
        for idx, mmap in enumerate(mapped_files[1:], start=2):
            model_name = model_paths[idx-1].name
            current_keys = set(mmap.keys())
            
            # Use existing validation function
            compatible, error_msg = validate_models_compatible(
                reference_shapes,
                reference_keys,
                {k: mmap[k] for k in current_keys},  # Create dict view for validation
                model_paths[0].name,
                model_name
            )
            
            if not compatible:
                # Clean up file handles
                del mapped_files
                raise ValueError(f"Models are incompatible: {error_msg}")
        
        console.print("  [green]✓[/green] All models compatible\n")
    
    # Process each tensor with consensus weighting
    result = {}
    keys_to_merge = [k for k in reference_keys if not config.should_skip_merge_key(k)]
    
    console.print(f"[cyan]Computing consensus merge for {len(keys_to_merge)} tensors...[/cyan]")
    
    with create_progress() as progress:
        task = progress.add_task("Consensus merging", total=len(keys_to_merge))
        
        for key in keys_to_merge:
            # Load this tensor from all models
            tensors = [mmap[key].to(device) for mmap in mapped_files]
            
            # Check if this is a weight/bias tensor that should be merged
            if 'weight' in key or 'bias' in key:
                # Ensure fp32 for numerical stability
                tensors = [t.to(torch.float32) for t in tensors]
                
                # Apply consensus weighting element-wise
                # Stack tensors: shape (num_models, *tensor_shape)
                stacked = torch.stack(tensors, dim=0)
                
                # Flatten to (num_models, num_elements)
                original_shape = stacked.shape[1:]
                stacked_flat = stacked.reshape(stacked.shape[0], -1)
                
                # Compute consensus weights for each element position
                # This is the expensive operation but it's what makes consensus work
                num_elements = stacked_flat.shape[1]
                merged_flat = torch.zeros(num_elements, dtype=torch.float32, device=device)
                
                for elem_idx in range(num_elements):
                    # Get all model values at this position
                    values = stacked_flat[:, elem_idx]
                    
                    # Compute consensus weights
                    weights = compute_consensus_weights(values, exponent=exponent)
                    
                    # Weighted sum using consensus weights
                    merged_flat[elem_idx] = (values * weights).sum()
                
                # Reshape back to original tensor shape
                result[key] = merged_flat.reshape(*original_shape)
            else:
                # Non-weight tensors: just copy from first model
                result[key] = tensors[0]
            
            # Free memory immediately
            del tensors
            progress.advance(task)
    
    # Clean up memory-mapped files
    del mapped_files
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_success(f"Consensus merge complete! Combined {len(model_entries)} models.")
    console.print(f"  [dim]Total tensors in result: {len(result)}[/dim]\n")
    
    return result


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
    
    Uses smart pruning that adapts to the file format:
    - Full SD checkpoints: Aggressive pruning (keep only model weights)
    - Standalone files: Conservative pruning (remove only training artifacts)
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Pruned state dict with only essential keys
    """
    from . import pruner as pruner_module
    
    console.print("\n[cyan]Pruning unnecessary keys...[/cyan]")
    
    original_count = len(state_dict)
    
    # Use pruner module for smart format-aware pruning
    pruned, removed_count, format_type = pruner_module.prune_state_dict(state_dict)
    
    console.print(f"  [dim]Detected format: {pruner_module.get_format_description(format_type)}[/dim]")
    console.print(f"  [dim]Removed {removed_count} keys, kept {len(pruned)} keys[/dim]")
    
    return pruned