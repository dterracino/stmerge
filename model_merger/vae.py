"""
VAE baking utilities.

Handles injecting VAE tensors into a merged model. This is done after
merging is complete, as the final step before saving.
"""

import torch
from pathlib import Path
from typing import Dict

from . import config
from .loader import load_vae
from .console import console, create_progress, print_section, print_success


def bake_vae(
    model_state_dict: Dict[str, torch.Tensor],
    vae_path: Path,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Bake a VAE into a merged model.
    
    This replaces the VAE tensors in the merged model with ones from
    a separate VAE file. The magic is pretty simple:
    
    1. Load the VAE (which has keys like 'encoder.conv_in.weight')
    2. Add 'first_stage_model.' prefix to each key
    3. Replace those keys in the model state dict
    
    The reason this works is that SD models store their VAE under the
    'first_stage_model.' namespace. By swapping those tensors, we're
    effectively changing which VAE the model will use during generation.
    
    Why would you want this? Different VAEs can affect color saturation,
    contrast, and detail levels. Some VAEs are optimized for specific
    styles (anime vs realistic, etc). Baking in a good VAE can make a
    noticeable difference in output quality!
    
    Args:
        model_state_dict: The merged model's state dict (modified in-place)
        vae_path: Path to the VAE .safetensors file
        device: Device to load VAE to
        
    Returns:
        The model state dict with VAE baked in (same object, for chaining)
        
    Note:
        The model_state_dict is modified in-place for memory efficiency,
        but we also return it to allow method chaining.
    """
    print_section(f"Baking VAE: {vae_path.name}")
    
    # Load the VAE
    vae_state_dict, vae_metadata = load_vae(vae_path, device=device)
    
    console.print(f"  [dim]VAE loaded with {len(vae_state_dict)} tensors[/dim]")
    
    # Inject VAE tensors into model
    replaced_count = 0
    added_count = 0
    
    with create_progress() as progress:
        task = progress.add_task("Baking VAE", total=len(vae_state_dict))
        for vae_key, vae_tensor in vae_state_dict.items():
            # Add the first_stage_model prefix
            model_key = config.VAE_KEY_PREFIX + vae_key
            
            if model_key in model_state_dict:
                # Replace existing VAE tensor
                model_state_dict[model_key] = vae_tensor
                replaced_count += 1
            else:
                # Add new VAE tensor (in case merged model was missing some)
                model_state_dict[model_key] = vae_tensor
                added_count += 1
            
            progress.advance(task)
    
    print_success("VAE baked successfully!")
    console.print(f"  [dim]Replaced {replaced_count} tensors[/dim]")
    if added_count > 0:
        console.print(f"  [dim]Added {added_count} new tensors[/dim]")
    
    # Free VAE memory
    del vae_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model_state_dict


def extract_vae(
    model_state_dict: Dict[str, torch.Tensor],
    output_path: Path
) -> Dict[str, torch.Tensor]:
    """
    Extract VAE from a model and save it as a separate file.
    
    This is the inverse operation of bake_vae. Useful if you want to
    extract a VAE from a merged model to use it with other models.
    
    Note: This is a v2 feature - not implementing the save logic yet,
    just extracting the tensors.
    
    Args:
        model_state_dict: Model state dict containing VAE
        output_path: Where to save the extracted VAE
        
    Returns:
        Dictionary of VAE tensors (without the 'first_stage_model.' prefix)
    """
    print(f"\nExtracting VAE from model...")
    
    vae_dict = {}
    
    for key, tensor in model_state_dict.items():
        if key.startswith(config.VAE_KEY_PREFIX):
            # Strip the prefix
            vae_key = key[len(config.VAE_KEY_PREFIX):]
            vae_dict[vae_key] = tensor
    
    print(f"Extracted {len(vae_dict)} VAE tensors")
    
    # TODO: Implement saving in v2
    # For now just return the dict
    return vae_dict