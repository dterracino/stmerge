"""
Configuration and constants for the model merger.

This module contains all the magic numbers, patterns, and defaults
so we don't have random strings scattered everywhere in the codebase.

Architecture patterns are loaded from architecture_patterns.json.
Users can customize by creating ~/.model_merger/architecture_patterns.json
"""

import json
import os
from typing import List, Set, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supported file formats
SUPPORTED_MODEL_EXTENSIONS = {'.safetensors'}
SUPPORTED_VAE_EXTENSIONS = {'.safetensors'}
SUPPORTED_CONVERT_EXTENSIONS = {'.ckpt', '.pt', '.pth', '.bin'}


def load_architecture_patterns() -> tuple[Dict[str, List[str]], str]:
    """
    Load architecture patterns from JSON files.
    
    Loads in order of priority:
    1. Default patterns (shipped with tool)
    2. User patterns from ~/.model_merger/architecture_patterns.json (if exists)
    
    User patterns override/extend default patterns.
    
    Returns:
        Tuple of (patterns_dict, default_architecture)
    """
    # Load default patterns from package
    default_file = Path(__file__).parent / 'architecture_patterns.json'
    
    try:
        with open(default_file, 'r', encoding='utf-8') as f:
            default_data = json.load(f)
            patterns = default_data.get('patterns', {})
            default_arch = default_data.get('default', 'SDXL')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Fallback to hardcoded if JSON is missing/corrupt
        patterns = {
            'Pony': ['pony', 'ponyxl', 'ponydiffusion'],
            'Illustrious': ['illustrious', 'illus', 'ill'],
            'SDXL': ['sdxl', 'xl'],
            'SD1.5': ['sd15', 'sd1.5', 'v1-5'],
            'SD2.1': ['sd21', 'sd2.1', 'v2-1'],
            'Noobai': ['noobai', 'noob'],
        }
        default_arch = 'SDXL'
    
    # Check for user overrides
    user_file = Path.home() / '.model_merger' / 'architecture_patterns.json'
    
    if user_file.exists():
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                user_patterns = user_data.get('patterns', {})
                
                # Merge user patterns (user patterns override defaults)
                for arch, pattern_list in user_patterns.items():
                    patterns[arch] = pattern_list
                
                # Use user's default if specified
                if 'default' in user_data:
                    default_arch = user_data['default']
                    
        except (json.JSONDecodeError, IOError):
            # If user file is corrupt, just use defaults
            pass
    
    return patterns, default_arch


# Load architecture patterns from JSON
ARCHITECTURE_PATTERNS, DEFAULT_ARCHITECTURE = load_architecture_patterns()

# Keys that should be pruned (removed) from models
# These are training artifacts and other cruft we don't need
PRUNE_KEYS_PATTERNS = [
    'optimizer',
    'lr_scheduler',
    'ema',
    'global_step',
    'epoch',
]

# Keys that are part of the core model and should be kept
# Anything not matching these prefixes gets pruned
KEEP_KEY_PREFIXES = [
    'model.diffusion_model.',     # The actual SD model
    'first_stage_model.',          # VAE
    'cond_stage_model.',           # Text encoder (SD 1.x/2.x)
    'conditioner.',                # Text encoder (SDXL)
]

# VAE key prefix - VAE tensors need this prepended when baking
VAE_KEY_PREFIX = 'first_stage_model.'

# Keys to skip during merge (these cause issues or aren't actual weights)
SKIP_MERGE_KEYS = [
    'cond_stage_model.transformer.text_model.embeddings.position_ids',
]

# Merge settings defaults
DEFAULT_MERGE_SETTINGS = {
    'output_precision': 'match',  # 'match', 'fp16', or 'fp32'
    'device': 'cpu',              # Keep on CPU to avoid VRAM issues
    'prune': True,                # Remove unnecessary keys
    'overwrite': False,           # Don't overwrite existing files by default
}

# Manifest defaults
DEFAULT_MANIFEST_FILENAME = 'merge_manifest.json'
DEFAULT_OUTPUT_FILENAME = 'merged_output.safetensors'
DEFAULT_OUTPUT_DIR = 'output'

# Filename generation settings
FILENAME_PREFIX_LENGTH = 8  # How many chars to take from each model name
FILENAME_STRIP_CHARS = r'[_\-\.\s]+'  # Regex pattern for chars to strip

# CivitAI API configuration
CIVITAI_API_KEY_ENV_VAR = 'CIVITAI_API_KEY'
CIVITAI_API_BASE_URL = 'https://civitai.com/api/v1'


def detect_architecture_from_filename(filename: str) -> str:
    """
    Attempt to detect model architecture from filename.
    
    Searches the filename (case-insensitive) for known architecture patterns.
    Returns the first match found, or DEFAULT_ARCHITECTURE if none match.
    
    Args:
        filename: The model filename to analyze
        
    Returns:
        Architecture name (e.g., 'Pony', 'SDXL', 'Illustrious')
        
    Example:
        >>> detect_architecture_from_filename('pony_realistic_v2.safetensors')
        'Pony'
        >>> detect_architecture_from_filename('my_cool_model.safetensors')
        'SDXL'
    """
    filename_lower = filename.lower()
    
    for arch_name, patterns in ARCHITECTURE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return arch_name
    
    return DEFAULT_ARCHITECTURE


def should_prune_key(key: str) -> bool:
    """
    Determine if a key should be pruned from the model.
    
    Keys are kept if they match one of the KEEP_KEY_PREFIXES.
    Everything else gets the axe! This removes training artifacts,
    optimizer states, and other cruft that bloats the file.
    
    Args:
        key: Tensor key name
        
    Returns:
        True if key should be removed, False if it should be kept
    """
    # Check if it's a key we want to keep
    for prefix in KEEP_KEY_PREFIXES:
        if key.startswith(prefix):
            return False
    
    # If it doesn't match any keep prefix, prune it!
    return True


def should_skip_merge_key(key: str) -> bool:
    """
    Check if a key should be skipped during merging.
    
    Some keys (like position_ids) cause issues or aren't actual weights,
    so we skip them entirely during the merge process.
    
    Args:
        key: Tensor key name
        
    Returns:
        True if key should be skipped during merge
    """
    return key in SKIP_MERGE_KEYS


def get_civitai_api_key() -> Optional[str]:
    """
    Get the CivitAI API key from environment variables.
    
    Looks for the API key in the CIVITAI_API_KEY environment variable.
    If using a .env file, python-dotenv will load it automatically.
    
    Returns:
        API key string if found, None otherwise
        
    Example:
        >>> key = get_civitai_api_key()
        >>> if key:
        ...     # Use the API key
        ... else:
        ...     print("No API key configured")
    """
    return os.environ.get(CIVITAI_API_KEY_ENV_VAR)


def has_civitai_api_key() -> bool:
    """
    Check if a CivitAI API key is configured.
    
    Returns:
        True if API key is available, False otherwise
    """
    return get_civitai_api_key() is not None