"""
Checkpoint format converter.

This module handles converting legacy checkpoint formats (.ckpt, .pt, .pth, .bin)
to the modern safetensors format. It uses safe loading (weights_only=True) to
prevent arbitrary code execution from pickle files.

Key safety features:
- Only loads with weights_only=True (refuses to execute code)
- Validates that we actually got tensors
- Prunes unnecessary keys by default
- Clear error messages for suspicious files
"""

import gc
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from . import config
from .loader import compute_file_hash
from .saver import save_model
from .console import console, print_warning, print_error, print_success, print_info

# Supported input formats
LEGACY_FORMATS = {'.ckpt', '.pt', '.pth', '.bin'}


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Remove 'module.' prefixes from state dict keys.
    
    When models are trained with PyTorch's DataParallel (multi-GPU training),
    it wraps the model and adds 'module.' prefixes to all keys. This makes
    the checkpoint incompatible with single-GPU loading.
    
    Example:
        'module.encoder.weight' -> 'encoder.weight'
        'module.decoder.bias' -> 'decoder.bias'
    
    Args:
        state_dict: State dictionary potentially with module prefixes
        
    Returns:
        Tuple of (cleaned_state_dict, count_of_prefixes_removed)
    """
    cleaned = {}
    prefix_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('module.'):
            # Remove the 'module.' prefix (7 characters)
            cleaned_key = key[7:]
            cleaned[cleaned_key] = value
            prefix_count += 1
        else:
            cleaned[key] = value
    
    return cleaned, prefix_count


def detect_checkpoint_format(checkpoint: Dict[str, Any]) -> str:
    """
    Detect the format/structure of a loaded checkpoint.
    
    Checkpoints can be structured in different ways:
    - Bare state dict: {'model.layer.weight': tensor, ...}
    - Wrapped: {'state_dict': {...}, 'optimizer_state': {...}, ...}
    - Training checkpoint: {'model': {...}, 'epoch': 42, ...}
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        
    Returns:
        Format identifier: 'bare', 'wrapped', or 'nested'
    """
    # Check if it looks like a bare state dict
    # (all keys are strings with dots, values are tensors)
    if all(isinstance(k, str) and '.' in k for k in list(checkpoint.keys())[:10]):
        # Sample a few values to see if they're tensors
        sample_values = [v for k, v in list(checkpoint.items())[:5]]
        if all(isinstance(v, torch.Tensor) for v in sample_values):
            return 'bare'
    
    # Check for wrapped format (has 'state_dict' key)
    if 'state_dict' in checkpoint:
        return 'wrapped'
    
    # Check for nested format (has 'model' key)
    if 'model' in checkpoint:
        return 'nested'
    
    # Unknown format
    return 'unknown'


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract the state dictionary from a checkpoint.
    
    Navigates different checkpoint structures to find the actual model weights.
    Removes training state, optimizer state, and other metadata.
    
    Handles:
    - Standard model checkpoints (state_dict, model, weights, model_state_dict keys)
    - Textual inversion embeddings (string_to_param key)
    - Bare state dicts (already just tensors)
    - Single tensor files
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Clean state dict with just model tensors
        
    Raises:
        ValueError: If we can't find valid tensors in the checkpoint
    """
    format_type = detect_checkpoint_format(checkpoint)
    
    if format_type == 'bare':
        # Already a state dict, just return it
        console.print("[dim]  Format: Bare state dict[/dim]")
        return checkpoint
    
    elif format_type == 'wrapped':
        # Extract from 'state_dict' key
        console.print("[dim]  Format: Wrapped (has 'state_dict' key)[/dim]")
        state_dict = checkpoint['state_dict']
        
        # Validate it's actually tensors
        if not isinstance(state_dict, dict):
            raise ValueError("'state_dict' key doesn't contain a dictionary!")
        
        return state_dict
    
    elif format_type == 'nested':
        # Extract from 'model' key
        console.print("[dim]  Format: Nested (has 'model' key)[/dim]")
        model_data = checkpoint['model']
        
        # Sometimes 'model' contains another layer of nesting
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            return model_data['state_dict']
        elif isinstance(model_data, dict):
            return model_data
        else:
            raise ValueError("'model' key doesn't contain a valid state dict!")
    
    else:
        # Unknown format - try some heuristics
        
        # Check for textual inversion embedding format
        # These have keys like: string_to_token, string_to_param, name, step, etc.
        if 'string_to_param' in checkpoint:
            console.print("[dim]  Format: Textual Inversion Embedding[/dim]")
            # The actual embedding tensor is in string_to_param
            # It's a dict mapping token strings to parameter tensors
            return checkpoint['string_to_param']
        
        # Check for other known patterns
        if 'weights' in checkpoint:
            console.print("[dim]  Format: Has 'weights' key[/dim]")
            return checkpoint['weights']
        
        if 'model_state_dict' in checkpoint:
            console.print("[dim]  Format: Has 'model_state_dict' key[/dim]")
            return checkpoint['model_state_dict']
        
        # Last resort: treat the whole thing as a state dict
        # This works for some exotic checkpoint formats where the tensors
        # are at the top level mixed with metadata
        console.print("[dim]  Format: Unknown, treating as bare state dict[/dim]")
        console.print("[dim]  (This might include non-tensor metadata)[/dim]")
        
        # Filter to only keep tensors
        filtered = {}
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                filtered[key] = value
        
        if not filtered:
            # No tensors found at all!
            keys = list(checkpoint.keys())
            raise ValueError(
                f"Cannot find tensors in checkpoint!\n"
                f"Top-level keys: {keys[:10]}\n"
                f"Expected one of: 'state_dict', 'model', 'weights', 'string_to_param', or bare tensors"
            )
        
        console.print(f"  [dim]Extracted {len(filtered)} tensors from checkpoint[/dim]")
        return filtered


def load_checkpoint(
    filepath: Path,
    device: str = 'cpu'
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a checkpoint file safely (weights only, no code execution).
    
    Supports .ckpt, .pt, .pth, and .bin formats. Uses PyTorch's weights_only=True
    to prevent arbitrary code execution from pickle files.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load tensors to (default: 'cpu')
        
    Returns:
        Tuple of (state_dict, metadata) where metadata contains file info
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or contains suspicious content
        RuntimeError: If safe loading fails
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    if filepath.suffix.lower() not in LEGACY_FORMATS:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}\n"
            f"Supported formats: {', '.join(LEGACY_FORMATS)}"
        )
    
    console.print(f"[cyan]Loading checkpoint:[/cyan] {filepath.name}")
    console.print(f"[yellow]⚠ Warning:[/yellow] Loading pickle file (legacy format)")
    
    try:
        # CRITICAL: Load with weights_only=True to prevent code execution!
        # This is the safety feature that prevents malicious pickles from running code
        console.print("[dim]  Using safe mode (weights_only=True)...[/dim]")
        
        checkpoint = torch.load(
            filepath,
            map_location=device,
            weights_only=True  # ← THE CRITICAL SAFETY LINE
        )
        
    except Exception as e:
        # If safe loading fails, provide helpful error with PyTorch's context
        from rich.panel import Panel
        
        # Format the PyTorch error nicely
        pytorch_error = str(e)
        error_panel = Panel(
            f"[yellow]{pytorch_error}[/yellow]",
            title="[red]PyTorch Loading Error Details[/red]",
            border_style="red",
            padding=(1, 2)
        )
        
        console.print()
        print_error(f"Cannot load {filepath.name} safely!")
        console.print("\n[yellow]This file cannot be loaded with safety checks enabled.[/yellow]")
        console.print("Possible reasons:")
        console.print("  [dim]• Contains custom Python classes or code[/dim]")
        console.print("  [dim]• Uses an incompatible/old pickle format[/dim]")
        console.print("  [dim]• File corruption[/dim]")
        
        console.print()
        console.print(error_panel)
        
        console.print()
        console.print("[bold cyan]If you created this file yourself:[/bold cyan]")
        console.print("  See the [bold]Troubleshooting[/bold] section in README.md for manual")
        console.print("  conversion instructions (search for 'Safe loading failed')")
        console.print()
        console.print("[bold red]⚠ For your safety, this tool will NOT attempt unsafe loading.[/bold red]")
        
        # Raise a simple exception (details already printed above)
        raise RuntimeError(f"Safe loading failed for {filepath.name}")
    
    # Extract the actual state dict
    try:
        state_dict = extract_state_dict(checkpoint)
    except ValueError as e:
        raise ValueError(f"Failed to extract model weights: {e}")
    
    # Validate we got tensors
    if not state_dict:
        raise ValueError("Checkpoint contains no model weights!")
    
    # Sample a few values to make sure they're actually tensors
    sample_values = [v for k, v in list(state_dict.items())[:5]]
    if not all(isinstance(v, torch.Tensor) for v in sample_values):
        raise ValueError("Checkpoint contains non-tensor data in state dict!")
    
    # Build metadata
    metadata = {
        'filename': filepath.name,
        'format': filepath.suffix,
        'num_keys': len(state_dict),
    }
    
    console.print(f"  [dim]Loaded {len(state_dict)} tensors[/dim]")
    
    return state_dict, metadata


def convert_to_safetensors(
    input_path: Path,
    output_path: Optional[Path] = None,
    prune: bool = True,
    compute_hash: bool = False,
    overwrite: bool = False
) -> str:
    """
    Convert a checkpoint file to safetensors format.
    
    Main conversion function that orchestrates the entire process:
    1. Load checkpoint safely (weights_only mode)
    2. Extract state dict
    3. Remove 'module.' prefixes (from DataParallel training)
    4. Optionally prune unnecessary keys
    5. Prepare tensors (contiguous + clone to break shared memory)
    6. Save as safetensors
    7. Verify output
    8. Optionally compute output hash
    
    Args:
        input_path: Path to input checkpoint (.ckpt/.pt/.pth/.bin)
        output_path: Output path (default: same name with .safetensors extension)
        prune: Whether to prune non-model keys (default: True)
        compute_hash: Whether to compute SHA-256 hash (default: False, it's slow!)
        overwrite: Whether to overwrite existing output file
        
    Returns:
        SHA-256 hash of the output file (for verification)
        
    Raises:
        Various exceptions if loading/conversion fails
    """
    # Determine output path
    if output_path is None:
        output_path = input_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)
    
    # Check if output exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite to replace it."
        )
    
    # Load the checkpoint
    state_dict, metadata = load_checkpoint(input_path, device='cpu')
    
    # Remove 'module.' prefixes from DataParallel models
    console.print("\n[cyan]Cleaning state dict...[/cyan]")
    state_dict, prefix_count = remove_module_prefix(state_dict)
    if prefix_count > 0:
        console.print(f"  [dim]Removed 'module.' prefix from {prefix_count} keys[/dim]")
    else:
        console.print(f"  [dim]No 'module.' prefixes found (good!)[/dim]")
    
    # Show info about the model
    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
    console.print(f"\n[cyan]Model info:[/cyan]")
    console.print(f"  [dim]State dict contains {len(state_dict)} keys[/dim]")
    console.print(f"  [dim]Total parameters: {total_params:,}[/dim]")
    
    # Show sample keys
    sample_keys = list(state_dict.keys())[:3]
    if sample_keys:
        console.print(f"  [dim]Sample keys:[/dim]")
        for key in sample_keys:
            if hasattr(state_dict[key], 'shape'):
                console.print(f"    [dim]{key}: {state_dict[key].shape}[/dim]")
    
    # Optionally prune unnecessary keys using smart pruner
    if prune:
        from . import pruner as pruner_module
        
        # Detect format and decide pruning strategy
        format_type = pruner_module.detect_format(state_dict)
        format_desc = pruner_module.get_format_description(format_type)
        
        if pruner_module.should_skip_pruning(format_type):
            console.print(f"\n[yellow]⚠ Detected {format_desc} (non-prunable format)[/yellow]")
            console.print("  [dim]Skipping pruning - file contains only essential data[/dim]")
        else:
            console.print(f"\n[cyan]Pruning unnecessary keys...[/cyan]")
            console.print(f"  [dim]Detected format: {format_desc}[/dim]")
            original_count = len(state_dict)
            
            state_dict, removed_count, _ = pruner_module.prune_state_dict(state_dict, format_type)
            
            console.print(f"  [dim]Removed {removed_count} keys, kept {len(state_dict)} keys[/dim]")
    else:
        console.print("\n[dim]Pruning disabled (--no-prune flag used)[/dim]")
    
    # Prepare tensors (contiguous + clone)
    # This is handled by save_model, but we import it for the numpy fallback
    from .saver import prepare_tensors
    state_dict = prepare_tensors(state_dict)
    
    # Save as safetensors
    console.print(f"\n[cyan]Saving as safetensors:[/cyan] {output_path.name}")
    
    # Build metadata for the safetensors file
    save_metadata = {
        'converted_from': metadata['filename'],
        'original_format': metadata['format'],
        'pruned': 'true' if prune else 'false',
    }
    
    # Try standard save
    try:
        # Use our existing saver module (which now has prepare_tensors built in)
        from .saver import save_model
        output_hash = save_model(
            state_dict=state_dict,
            output_path=output_path,
            overwrite=overwrite,
            metadata=save_metadata
        )
    except RuntimeError as e:
        if "Some tensors share memory" in str(e):
            # This shouldn't happen after prepare_tensors, but just in case...
            console.print("[yellow]⚠ Shared tensor error detected! Trying numpy fallback...[/yellow]")
            
            # Force complete independence via numpy roundtrip
            independent_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    # Go through numpy to break ALL connections
                    independent_state_dict[key] = torch.tensor(
                        value.cpu().numpy(),
                        dtype=value.dtype,
                        device=value.device
                    )
                else:
                    independent_state_dict[key] = value
            
            # Try saving again with independent tensors
            output_hash = save_model(
                state_dict=independent_state_dict,
                output_path=output_path,
                overwrite=overwrite,
                metadata=save_metadata
            )
            console.print(f"  [green]✓ Saved using numpy fallback approach![/green]")
        else:
            raise
    
    # Verify the output file
    console.print("\n[cyan]Verifying output...[/cyan]")
    verify_conversion(output_path)
    
    # Aggressive cleanup
    del state_dict
    gc.collect()
    
    return output_hash


def verify_conversion(output_path: Path) -> None:
    """
    Verify the converted safetensors file.
    
    Checks:
    1. File exists and has reasonable size
    2. Can be loaded back successfully
    3. No remaining 'module.' prefixes (these cause loading issues)
    
    Args:
        output_path: Path to the converted safetensors file
    """
    from safetensors.torch import load_file
    
    if not output_path.exists():
        raise FileNotFoundError("Output file was not created!")
    
    # Check file size
    file_size = output_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    console.print(f"  [dim]File size: {size_mb:.2f} MB[/dim]")
    
    # Try loading it back
    try:
        verify_data = load_file(str(output_path))
        console.print(f"  [green]✓ File loads successfully ({len(verify_data)} keys)[/green]")
        
        # Check for remaining 'module.' prefixes
        module_keys = [k for k in verify_data.keys() if k.startswith('module.')]
        if module_keys:
            print_warning(f"Found {len(module_keys)} keys with 'module.' prefix still present!")
            print_warning("This may cause loading issues in some applications.")
        else:
            console.print(f"  [green]✓ No 'module.' prefixes found[/green]")
            
    except Exception as e:
        print_error(f"Failed to verify output file: {e}")