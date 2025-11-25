"""
Model Merger - A tool for merging Stable Diffusion models.

This package provides utilities for:
- Scanning folders of models and generating merge manifests
- Merging multiple models with configurable weights
- Baking VAEs into merged models
- Saving results in safetensors format

Main workflow:
    1. Scan folder → generate manifest
    2. Edit manifest to adjust weights
    3. Process manifest → merge models
    4. Save result with optional VAE baking
"""

__version__ = '0.1.0'

# Expose main classes and functions at package level
from .config import (
    detect_architecture_from_filename,
    should_prune_key,
    should_skip_merge_key,
)

from .loader import (
    load_model,
    load_vae,
    compute_file_hash,
    validate_models_compatible,
)

from .manifest import (
    MergeManifest,
    ModelEntry,
    scan_folder,
    validate_manifest,
)

from .merger import (
    merge_models,
    convert_precision,
    prune_model,
)

from .vae import (
    bake_vae,
    extract_vae,
)

from .saver import (
    save_model,
    save_manifest_metadata,
)

__all__ = [
    # Config
    'detect_architecture_from_filename',
    'should_prune_key',
    'should_skip_merge_key',
    
    # Loader
    'load_model',
    'load_vae',
    'compute_file_hash',
    'validate_models_compatible',
    
    # Manifest
    'MergeManifest',
    'ModelEntry',
    'scan_folder',
    'validate_manifest',
    
    # Merger
    'merge_models',
    'convert_precision',
    'prune_model',
    
    # VAE
    'bake_vae',
    'extract_vae',
    
    # Saver
    'save_model',
    'save_manifest_metadata',
]