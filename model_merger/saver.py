"""
Model saving utilities.

Handles writing merged models to disk in safetensors format, with
proper handling of overwrite protection and tensor contiguity.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
from safetensors.torch import save_file
from tqdm import tqdm

from . import config
from .loader import compute_file_hash


def ensure_contiguous(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Ensure all tensors are contiguous in memory.
    
    Safetensors requires tensors to be contiguous (stored in a single
    block of memory without gaps). After all our merging operations,
    some tensors might be non-contiguous, which would cause save_file()
    to error out.
    
    This is a quirk of how PyTorch handles tensor operations - sometimes
    operations like transpose() or slice() create "views" that reference
    the original tensor's memory in a non-contiguous way. Calling
    .contiguous() forces PyTorch to make a new contiguous copy.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        State dict with all tensors contiguous
    """
    print("\nEnsuring tensors are contiguous...")
    
    contiguous_dict = {}
    for key, tensor in tqdm(state_dict.items(), desc="Checking contiguity"):
        if not tensor.is_contiguous():
            contiguous_dict[key] = tensor.contiguous()
        else:
            contiguous_dict[key] = tensor
    
    return contiguous_dict


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
    
    print(f"\nSaving merged model to: {output_path}")
    print("=" * 60)
    
    # Ensure tensors are contiguous (safetensors requirement)
    state_dict = ensure_contiguous(state_dict)
    
    # Prepare metadata
    # Safetensors metadata must be strings
    save_metadata = metadata or {}
    
    # Convert all metadata values to strings
    str_metadata = {k: str(v) for k, v in save_metadata.items()}
    
    print(f"Writing {len(state_dict)} tensors to disk...")
    
    try:
        # Save as safetensors
        save_file(state_dict, str(output_path), metadata=str_metadata)
        
        # Get file size for reporting
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        # Compute hash of output file
        print("\nComputing SHA-256 hash of output file...")
        output_hash = compute_file_hash(output_path)
        
        print(f"\n✓ Model saved successfully!")
        print(f"  Location: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Tensors: {len(state_dict)}")
        print(f"  SHA-256: {output_hash}")
        
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
        metadata['vae_path'] = manifest.vae
        if manifest.vae_sha256:
            metadata['vae_sha256'] = manifest.vae_sha256
    
    return metadata