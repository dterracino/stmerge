"""
Model verification utilities.

This module provides tools to verify that converted models are identical to their
originals. It performs deep comparison of tensors to ensure no data loss or
corruption occurred during format conversion.
"""

import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Set
from safetensors.torch import load_file

from .console import console, print_success, print_error, print_warning, print_info, create_progress


def load_for_verification(file_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a model file for verification purposes.
    
    Supports both legacy formats (.ckpt, .pt, .pth, .bin) and safetensors.
    Uses the SAME extraction logic as the converter to ensure we're comparing
    apples-to-apples (extracted state dicts, not raw checkpoint structures).
    
    Args:
        file_path: Path to model file
        
    Returns:
        State dictionary with tensors (extracted using same logic as converter)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported or file is corrupted
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    console.print(f"[cyan]Loading:[/cyan] {file_path.name}")
    
    try:
        if ext == '.safetensors':
            # Load safetensors (already extracted format)
            data = load_file(str(file_path))
            console.print(f"  [dim]Loaded as safetensors ({len(data)} keys)[/dim]")
            return data
            
        elif ext in ['.ckpt', '.pt', '.pth', '.bin']:
            # Load legacy format - use the CONVERTER'S extraction logic!
            # This ensures we compare the same thing the converter saved
            from . import converter as converter_module
            
            try:
                # Use converter's load_checkpoint (safe mode)
                checkpoint, metadata = converter_module.load_checkpoint(file_path, device='cpu')
            except Exception as e:
                # If safe load fails, try unsafe for old files
                console.print("  [yellow]Safe load failed, trying unsafe mode for verification[/yellow]")
                checkpoint = torch.load(file_path, map_location='cpu')
                # Extract using converter's logic
                checkpoint = converter_module.extract_state_dict(checkpoint)
            
            console.print(f"  [dim]Loaded as {ext} ({len(checkpoint)} keys)[/dim]")
            return checkpoint
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        raise ValueError(f"Failed to load {file_path.name}: {e}")


def compare_key_sets(
    original_keys: Set[str],
    converted_keys: Set[str]
) -> Tuple[bool, Set[str], Set[str]]:
    """
    Compare key sets between original and converted models.
    
    Args:
        original_keys: Set of keys from original model
        converted_keys: Set of keys from converted model
        
    Returns:
        Tuple of (keys_match, missing_keys, extra_keys)
    """
    missing = original_keys - converted_keys  # In original but not converted
    extra = converted_keys - original_keys    # In converted but not original
    
    return (len(missing) == 0 and len(extra) == 0), missing, extra


def compare_tensors(
    key: str,
    original_tensor: torch.Tensor,
    converted_tensor: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Tuple[bool, Optional[str]]:
    """
    Compare two tensors for equivalence.
    
    Checks:
    1. Both are actually tensors
    2. Shapes match
    3. Values match within floating point tolerance
    
    Args:
        key: Tensor name (for error reporting)
        original_tensor: Tensor from original model
        converted_tensor: Tensor from converted model
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison
        
    Returns:
        Tuple of (is_match, error_message)
    """
    # Check both are tensors
    if not isinstance(original_tensor, torch.Tensor):
        return False, f"Original '{key}' is not a tensor (type: {type(original_tensor)})"
    if not isinstance(converted_tensor, torch.Tensor):
        return False, f"Converted '{key}' is not a tensor (type: {type(converted_tensor)})"
    
    # Check shapes match
    if original_tensor.shape != converted_tensor.shape:
        return False, f"Shape mismatch: original={original_tensor.shape}, converted={converted_tensor.shape}"
    
    # Check values match (within floating point tolerance)
    # Move to CPU for comparison to avoid device mismatch issues
    try:
        if not torch.allclose(original_tensor.cpu(), converted_tensor.cpu(), rtol=rtol, atol=atol):
            # Calculate max difference for debugging
            diff = torch.abs(original_tensor.cpu() - converted_tensor.cpu())
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            return False, f"Values differ: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    except Exception as e:
        return False, f"Comparison failed: {e}"
    
    return True, None


def verify_conversion(
    original_path: Path,
    converted_path: Path,
    verbose: bool = False
) -> bool:
    """
    Verify that a converted model matches the original.
    
    Performs comprehensive comparison:
    1. Loads both files
    2. Compares key sets
    3. Compares tensor shapes
    4. Compares tensor values (with floating point tolerance)
    
    Args:
        original_path: Path to original checkpoint
        converted_path: Path to converted safetensors
        verbose: If True, show details for every tensor checked
        
    Returns:
        True if verification passes, False otherwise
    """
    console.print("\n[bold cyan]Starting Verification[/bold cyan]")
    console.print(f"[cyan]Original:[/cyan] {original_path.name}")
    console.print(f"[cyan]Converted:[/cyan] {converted_path.name}\n")
    
    # Load both files
    try:
        original_sd = load_for_verification(original_path)
        converted_sd = load_for_verification(converted_path)
    except Exception as e:
        print_error(f"Failed to load files: {e}")
        return False
    
    # Compare key sets
    console.print("\n[cyan]Checking key sets...[/cyan]")
    original_keys = set(original_sd.keys())
    converted_keys = set(converted_sd.keys())
    
    keys_match, missing, extra = compare_key_sets(original_keys, converted_keys)
    
    if not keys_match:
        if missing:
            print_error(f"{len(missing)} keys from original are missing in converted file")
            print_info(f"Sample missing keys: {list(missing)[:5]}")
        if extra:
            print_warning(f"{len(extra)} extra keys in converted file (not in original)")
            print_info(f"Sample extra keys: {list(extra)[:5]}")
        
        # This might be okay if the converter intentionally removed prefixes
        console.print("\n[yellow]Note:[/yellow] Key mismatch might be expected if 'module.' prefixes were removed")
        
        # Check if the mismatch is just module prefixes
        # Convert module. keys to non-prefixed versions for comparison
        original_cleaned = {k.replace('module.', '', 1) if k.startswith('module.') else k 
                          for k in original_keys}
        converted_cleaned = {k.replace('module.', '', 1) if k.startswith('module.') else k 
                           for k in converted_keys}
        
        if original_cleaned == converted_cleaned:
            print_success("Key sets match after removing 'module.' prefixes - this is expected!")
            # Use cleaned keys for comparison
            # Map cleaned keys back to actual keys for tensor comparison
            keys_to_compare = converted_cleaned
        else:
            return False
    else:
        print_success(f"Key sets match! Both files have {len(original_keys)} keys")
        keys_to_compare = original_keys
    
    # Compare tensors
    console.print("\n[cyan]Comparing tensors...[/cyan]")
    mismatch_count = 0
    checked_count = 0
    
    with create_progress() as progress:
        task = progress.add_task("Verifying tensors", total=len(keys_to_compare))
        
        for key in keys_to_compare:
            # Handle potential module prefix mismatch
            orig_key = key if key in original_sd else f'module.{key}'
            conv_key = key if key in converted_sd else f'module.{key}'
            
            # Skip if key doesn't exist in either (shouldn't happen but be safe)
            if orig_key not in original_sd or conv_key not in converted_sd:
                progress.advance(task)
                continue
            
            original_tensor = original_sd[orig_key]
            converted_tensor = converted_sd[conv_key]
            
            # Compare tensors
            is_match, error_msg = compare_tensors(key, original_tensor, converted_tensor)
            
            if not is_match:
                if verbose or mismatch_count < 5:  # Show first 5 errors always
                    print_error(f"Tensor '{key}': {error_msg}")
                mismatch_count += 1
            elif verbose:
                print_success(f"Tensor '{key}': OK")
            
            checked_count += 1
            progress.advance(task)
    
    # Report results
    console.print()
    if mismatch_count == 0:
        print_success(f"✨ VERIFICATION PASSED! ✨")
        console.print(f"  [dim]All {checked_count} tensors match in shape and values[/dim]")
        console.print(f"  [dim]Comparison tolerance: rtol=1e-5, atol=1e-8[/dim]")
        return True
    else:
        print_error(f"❌ VERIFICATION FAILED!")
        console.print(f"  [dim]{mismatch_count}/{checked_count} tensors failed comparison[/dim]")
        if mismatch_count > 5:
            console.print(f"  [dim](Showing first 5 errors, {mismatch_count - 5} more not shown)[/dim]")
        return False