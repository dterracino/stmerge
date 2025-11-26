"""
Pruning logic for model checkpoints.

This module handles detection and removal of unnecessary keys from checkpoints,
such as training artifacts, optimizer states, and other metadata that bloats
file sizes without adding value.

Different file types require different pruning strategies:
- Full SD checkpoints: Aggressive pruning (keep only model weights)
- Standalone files (VAE, LoRA, embeddings): Conservative pruning (skip or light touch)
"""

from typing import Dict, Tuple, Set, Optional
import torch


# Patterns that indicate training artifacts (always safe to remove)
TRAINING_ARTIFACT_PATTERNS = [
    'optimizer',
    'optimizer_state',
    'lr_scheduler',
    'ema',
    'global_step',
    'epoch',
    'training_state',
    'loss',
]

# Keys for full SD model components (keep these when pruning full checkpoints)
SD_MODEL_PREFIXES = [
    'model.diffusion_model.',     # The actual SD model
    'first_stage_model.',          # VAE (when baked in)
    'cond_stage_model.',           # Text encoder (SD 1.x/2.x)
    'conditioner.',                # Text encoder (SDXL)
]


def detect_format(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Detect what type of checkpoint this is.
    
    Checks key patterns to identify:
    - Full SD checkpoints (has diffusion_model)
    - Standalone VAEs (has encoder/decoder without first_stage_model prefix)
    - LoRAs (has lora in key names)
    - Textual inversion embeddings (very few keys, short names)
    - Upscalers (has body.X.body.Y pattern, ESRGAN/Real-ESRGAN)
    - Unknown (anything else)
    
    Args:
        state_dict: The checkpoint's state dictionary
        
    Returns:
        Format identifier: 'sd_checkpoint', 'vae', 'lora', 'embedding', 'upscaler', or 'unknown'
    """
    if not state_dict:
        return 'unknown'
    
    # Sample keys for analysis (check first 20 for speed)
    sample_keys = list(state_dict.keys())[:20]
    all_keys = state_dict.keys()
    
    # Check for full SD checkpoint (has diffusion model)
    if any('diffusion_model' in key for key in sample_keys):
        return 'sd_checkpoint'
    
    # Check for LoRA (has lora in key names)
    # LoRA keys look like: lora_unet_..., lora_te_text_model_...
    if any('lora' in key.lower() for key in sample_keys):
        return 'lora'
    
    # Check for textual inversion embedding
    # - Very few keys (typically 1-5)
    # - Short key names (embedding name or "*")
    if len(state_dict) <= 5 and all(len(k) < 50 for k in all_keys):
        return 'embedding'
    
    # Check for upscaler (ESRGAN, Real-ESRGAN, etc.)
    # Upscalers have characteristic nested body blocks: body.0.body.0.body.0.*
    # They also tend to be relatively small (10-50M parameters)
    upscaler_patterns = ['body.', 'conv_first', 'conv_body', 'conv_up', 'conv_last', 'upconv']
    if any(any(pattern in key for pattern in upscaler_patterns) for key in sample_keys):
        # Also check for the characteristic nested structure
        if any('body.' in key and key.count('body.') >= 2 for key in sample_keys):
            return 'upscaler'
    
    # Check for standalone VAE (has encoder/decoder WITHOUT first_stage_model prefix)
    vae_patterns = ['encoder.', 'decoder.', 'quant_conv', 'post_quant_conv']
    has_vae_keys = any(
        any(pattern in key for pattern in vae_patterns)
        for key in sample_keys
    )
    # Make sure it's NOT a baked VAE (which would have first_stage_model prefix)
    has_baked_vae = any('first_stage_model' in key for key in sample_keys)
    if has_vae_keys and not has_baked_vae:
        return 'vae'
    
    # Unknown format - be conservative
    return 'unknown'


def should_prune_key(key: str, format_type: str) -> bool:
    """
    Determine if a key should be pruned based on file format.
    
    Strategy depends on format:
    - sd_checkpoint: Keep only SD model prefixes (aggressive)
    - vae/lora/embedding/unknown: Only remove obvious training artifacts (conservative)
    
    Args:
        key: The tensor key name
        format_type: Format from detect_format()
        
    Returns:
        True if key should be removed, False if it should be kept
    """
    # For full SD checkpoints, use aggressive pruning
    if format_type == 'sd_checkpoint':
        # Keep only keys that match SD model prefixes
        for prefix in SD_MODEL_PREFIXES:
            if key.startswith(prefix):
                return False  # Keep this key
        return True  # Prune everything else
    
    # For standalone files (VAE, LoRA, embedding, unknown), be conservative
    # Only remove obvious training artifacts
    key_lower = key.lower()
    for pattern in TRAINING_ARTIFACT_PATTERNS:
        if pattern in key_lower:
            return True  # This is training junk, remove it
    
    # Keep everything else for standalone files
    return False


def prune_state_dict(
    state_dict: Dict[str, torch.Tensor],
    format_type: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor], int, str]:
    """
    Prune unnecessary keys from a state dictionary.
    
    Auto-detects format if not provided, then applies appropriate pruning strategy.
    
    Args:
        state_dict: The state dictionary to prune
        format_type: Optional format hint (auto-detected if None)
        
    Returns:
        Tuple of (pruned_dict, removed_count, detected_format)
    """
    # Auto-detect format if not provided
    if format_type is None:
        format_type = detect_format(state_dict)
    
    # Prune based on format
    pruned = {}
    for key, tensor in state_dict.items():
        if not should_prune_key(key, format_type):
            pruned[key] = tensor
    
    removed_count = len(state_dict) - len(pruned)
    
    return pruned, removed_count, format_type


def should_skip_pruning(format_type: str) -> bool:
    """
    Check if pruning should be skipped entirely for this format.
    
    Some formats are already minimal and contain only essential data,
    so pruning provides no benefit.
    
    Args:
        format_type: Format from detect_format()
        
    Returns:
        True if pruning should be skipped
    """
    # These formats are already minimal - don't bother pruning
    return format_type in ['embedding', 'vae', 'lora', 'upscaler', 'unknown']


def get_format_description(format_type: str) -> str:
    """
    Get a human-readable description of the detected format.
    
    Args:
        format_type: Format from detect_format()
        
    Returns:
        Human-readable description string
    """
    descriptions = {
        'sd_checkpoint': 'Stable Diffusion checkpoint',
        'vae': 'Standalone VAE',
        'lora': 'LoRA (Low-Rank Adaptation)',
        'embedding': 'Textual inversion embedding',
        'upscaler': 'Upscaler model (ESRGAN/Real-ESRGAN)',
        'unknown': 'Unknown format',
    }
    return descriptions.get(format_type, 'Unknown format')