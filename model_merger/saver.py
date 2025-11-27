"""
Model saving utilities.

Handles writing merged models to disk in safetensors format, with
proper handling of overwrite protection and tensor contiguity.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
from safetensors.torch import save_file

from . import config
from .loader import compute_file_hash
from .console import console, create_progress, print_info


def prepare_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Prepare tensors for safetensors saving.
    
    Safetensors requires:
    1. Tensors must be contiguous in memory
    2. Tensors must NOT share memory with each other
    
    The shared memory issue occurs when multiple keys point to the same
    underlying tensor data. Example:
        tied_weights = torch.randn(256, 256)
        state_dict = {
            'encoder.weight': tied_weights,
            'decoder.weight': tied_weights  # Same memory!
        }
    
    This function:
    1. Makes all tensors contiguous (safetensors requirement)
    2. Clones all shared tensors to ensure independence (breaks memory sharing)
    
    The clone() operation creates new memory, so if tensors were sharing
    before, they won't be after this. It's brute-force but foolproof!
    
    Args:
        state_dict: State dictionary with tensors
        
    Returns:
        Prepared state dictionary with contiguous, independent tensors
    """
    console.print("\n[cyan]Preparing tensors for safetensors...[/cyan]")
    
    prepared = {}
    contiguous_count = 0
    cloned_count = 0
    total_tensors = 0
    
    # Track tensor data pointers to detect sharing
    seen_data_ptrs = {}
    
    with create_progress() as progress:
        task = progress.add_task("Preparing tensors", total=len(state_dict))
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                total_tensors += 1
                
                # Step 1: Make contiguous if needed
                if not value.is_contiguous():
                    value = value.contiguous()
                    contiguous_count += 1
                
                # Step 2: Check if this tensor shares memory with another
                data_ptr = value.data_ptr()
                
                if data_ptr in seen_data_ptrs:
                    # Shared memory detected! Clone to make independent
                    prepared[key] = value.clone()
                    cloned_count += 1
                else:
                    # Not shared, use as-is (no clone needed!)
                    prepared[key] = value
                    seen_data_ptrs[data_ptr] = key
            else:
                # Keep non-tensor values as-is
                prepared[key] = value
            
            progress.advance(task)
    
    console.print(f"  [dim]Made {contiguous_count} tensors contiguous[/dim]")
    console.print(f"  [dim]Cloned {cloned_count} shared tensors (out of {total_tensors} total)[/dim]")
    
    return prepared


def save_model(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    overwrite: bool = False,
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """
    Save a model to a safetensors file.
    
    This is the final step! We write the merged (and optionally VAE-baked)
    model to disk. Safetensors is great because:
    - It's fast to load (memory-mapped)
    - It's secure (no pickle exploits)
    - It's portable (works across frameworks)
    - It supports metadata
    
    Args:
        state_dict: The model state dictionary to save
        output_path: Where to save the file
        overwrite: If False, error if file exists
        metadata: Optional metadata to embed in the file
        
    Returns:
        SHA-256 hash of the saved file (for verification/tracking)
        
    Raises:
        FileExistsError: If file exists and overwrite=False
        OSError: If unable to write file
    """
    output_path = Path(output_path)
    
    # Check if file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite flag to overwrite existing files."
        )
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[cyan]Saving merged model to:[/cyan] {output_path}")
    
    # Prepare tensors (make contiguous + clone to break memory sharing)
    state_dict = prepare_tensors(state_dict)
    
    # Prepare metadata
    # Safetensors metadata must be strings
    save_metadata = metadata or {}
    
    # Convert all metadata values to strings
    str_metadata = {k: str(v) for k, v in save_metadata.items()}
    
    console.print(f"  [cyan]Writing {len(state_dict)} tensors to disk...[/cyan]")
    
    try:
        # Save as safetensors
        save_file(state_dict, str(output_path), metadata=str_metadata)
        
        # Get file size for reporting
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        # Compute hash of output file
        print_info("Computing SHA-256 hash of output file...")
        output_hash = compute_file_hash(output_path)
        
        # We'll let the CLI handle the fancy completion message
        # Just return the data it needs
        return output_hash
        
    except Exception as e:
        raise OSError(f"Failed to save model: {e}")


def save_manifest_metadata(
    manifest,
    merged_precision: str
) -> Dict[str, str]:
    """
    Generate metadata for the merged model from the manifest.
    
    This creates a record of how the model was created - which source
    models, their weights, etc. Useful for reproducibility and tracking.
    
    Args:
        manifest: The MergeManifest used for this merge
        merged_precision: The final precision of the merged model
        
    Returns:
        Dictionary of metadata (all string values)
    """
    metadata = {
        'merge_tool': 'model_merger',
        'merge_precision': merged_precision,
        'num_models': str(len(manifest.models)),
        'output_precision_setting': manifest.output_precision,
    }
    
    # Add model info
    for idx, model in enumerate(manifest.models, start=1):
        prefix = f'model_{idx}_'
        metadata[prefix + 'path'] = model.path
        metadata[prefix + 'weight'] = str(model.weight)
        metadata[prefix + 'architecture'] = model.architecture
        if model.sha256:
            metadata[prefix + 'sha256'] = model.sha256
    
    # Add VAE info if present
    if manifest.vae:
        metadata['vae_path'] = manifest.vae.path
        if manifest.vae.sha256:
            metadata['vae_sha256'] = manifest.vae.sha256
        if manifest.vae.precision_detected:
            metadata['vae_precision'] = manifest.vae.precision_detected
    
    return metadata