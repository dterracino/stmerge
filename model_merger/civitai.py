"""
CivitAI API integration for model metadata lookup.

This module provides functions to query the CivitAI API for model information
using file hashes. This allows automatic detection of model metadata, architecture,
and other details without manual input.

Key features:
- Look up model versions by hash (AutoV1, AutoV2, SHA256, CRC32, Blake3)
- Extract architecture information from model metadata
- Graceful fallback when API key not configured or model not found
"""

from typing import Optional, Dict, Any
import requests
from pathlib import Path

from .config import get_civitai_api_key, CIVITAI_API_BASE_URL, detect_architecture_from_filename, DEFAULT_ARCHITECTURE
from .console import console, print_warning, print_error


def get_model_version_by_hash(file_hash: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch model version information from CivitAI using a file hash.
    
    Queries the CivitAI API to retrieve detailed information about a model version
    based on its file hash. Supports multiple hash algorithms: AutoV1, AutoV2, 
    SHA256, CRC32, and Blake3.
    
    Args:
        file_hash: The hash of the model file (any supported algorithm)
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        Dictionary containing model version data if found, None otherwise
        
    Response includes:
        - id: Model version ID
        - name: Version name
        - description: Version description/changelog
        - model: Parent model info (name, type, nsfw, etc.)
        - modelId: Parent model ID
        - trainedWords: Trigger words for the model
        - baseModel: Base model (SD 1.5, SDXL, etc.)
        - files: File information (size, format, fp, etc.)
        - images: Preview images
        - stats: Download/rating stats
        
    Example:
        >>> file_hash = "ABC123..."
        >>> info = get_model_version_by_hash(file_hash)
        >>> if info:
        ...     print(f"Model: {info['model']['name']}")
        ...     print(f"Type: {info['model']['type']}")
    """
    api_key = get_civitai_api_key()
    
    # Build URL
    url = f"{CIVITAI_API_BASE_URL}/model-versions/by-hash/{file_hash}"
    
    # Prepare headers and params
    headers = {"Content-Type": "application/json"}
    params = {}
    
    # Add API key if available (some models require authentication)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        
        # Handle not found (model not in CivitAI database yet)
        if response.status_code == 404:
            return None
        
        # Handle other errors
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.Timeout:
        print_warning(f"CivitAI API timeout after {timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        print_error(f"CivitAI API error: {e}")
        return None


def detect_architecture_from_civitai(file_hash: str, fallback_filename: Optional[str] = None) -> Optional[str]:
    """
    Detect model architecture from CivitAI metadata.
    
    Queries CivitAI for model information and extracts the architecture type.
    Falls back to filename detection if CivitAI lookup fails or returns no data.
    
    The architecture is determined by:
    1. CivitAI 'baseModel' field (most reliable)
    2. Model name/tags (secondary)
    3. Fallback to filename detection (last resort)
    
    Args:
        file_hash: The hash of the model file
        fallback_filename: Optional filename to use if CivitAI lookup fails
        
    Returns:
        Architecture string ('Pony', 'SDXL', 'Illustrious', etc.) or None
        
    Example:
        >>> file_hash = "ABC123..."
        >>> arch = detect_architecture_from_civitai(file_hash, "pony_model.safetensors")
        >>> print(arch)
        'Pony'
    """
    # Try CivitAI first
    model_info = get_model_version_by_hash(file_hash)
    
    if model_info:
        # Try to extract architecture from baseModel field
        base_model = model_info.get('baseModel')
        if base_model:
            arch = _map_base_model_to_architecture(base_model)
            if arch:
                return arch
        
        # Try model name and tags
        model_data = model_info.get('model', {})
        model_name = model_data.get('name', '')
        model_tags = model_data.get('tags', [])
        
        # Check model name for architecture hints
        # Only accept if it's not the default (meaning a pattern was found)
        arch = detect_architecture_from_filename(model_name)
        if arch and arch != DEFAULT_ARCHITECTURE:
            return arch
        
        # Check tags for architecture hints
        # Only accept if it's not the default
        for tag in model_tags:
            arch = detect_architecture_from_filename(tag)
            if arch and arch != DEFAULT_ARCHITECTURE:
                return arch
        
        # If we only found defaults, return the default
        if model_name or model_tags:
            return DEFAULT_ARCHITECTURE
    
    # Fall back to filename detection if provided
    if fallback_filename:
        return detect_architecture_from_filename(fallback_filename)
    
    return None


def _map_base_model_to_architecture(base_model: str) -> Optional[str]:
    """
    Map CivitAI baseModel field to our architecture names.
    
    CivitAI uses different naming conventions than we do internally.
    This function translates their baseModel field to our architecture names.
    
    Args:
        base_model: The baseModel string from CivitAI (e.g., "SDXL 1.0", "Pony", "Illustrious")
        
    Returns:
        Our internal architecture name or None if unknown
        
    CivitAI baseModel values include:
        - SD 1.4, SD 1.5, SD 2.0, SD 2.1, SDXL 0.9, SDXL 1.0, Pony, Illustrious, etc.
    """
    base_model_lower = base_model.lower()
    
    # Check specific architectures first (before generic SDXL)
    # Pony detection
    if 'pony' in base_model_lower:
        return 'Pony'
    
    # Illustrious detection (SDXL-based but distinct)
    if 'illustrious' in base_model_lower or 'illus' in base_model_lower:
        return 'Illustrious'
    
    # Noobai detection (SDXL-based but distinct)
    if 'noobai' in base_model_lower or 'noob' in base_model_lower:
        return 'Noobai'
    
    # Generic SDXL detection (after specific variants)
    if 'sdxl' in base_model_lower or 'xl' in base_model_lower:
        return 'SDXL'
    
    # SD 1.x detection
    if 'sd 1' in base_model_lower or 'sd1' in base_model_lower:
        return 'SD1.5'
    
    # SD 2.x detection
    if 'sd 2' in base_model_lower or 'sd2' in base_model_lower:
        return 'SD2.1'
    
    # Unknown base model
    return None


def get_model_metadata_summary(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get a simplified summary of model metadata from CivitAI.
    
    Returns only the most commonly needed fields in a clean format.
    Useful for display or logging purposes.
    
    Args:
        file_hash: The hash of the model file
        
    Returns:
        Dictionary with summary fields or None if not found
        
    Summary fields:
        - model_name: Name of the model
        - version_name: Name of this version
        - architecture: Detected architecture (Pony, SDXL, etc.)
        - base_model: Base model from CivitAI
        - type: Model type (Checkpoint, LORA, etc.)
        - nsfw: Whether model is NSFW
        - trained_words: Trigger words
        - download_url: URL to download this version
        
    Example:
        >>> summary = get_model_metadata_summary(file_hash)
        >>> if summary:
        ...     print(f"{summary['model_name']} - {summary['architecture']}")
    """
    model_info = get_model_version_by_hash(file_hash)
    
    if not model_info:
        return None
    
    model_data = model_info.get('model', {})
    
    return {
        'model_name': model_data.get('name'),
        'version_name': model_info.get('name'),
        'architecture': detect_architecture_from_civitai(file_hash),
        'base_model': model_info.get('baseModel'),
        'type': model_data.get('type'),
        'nsfw': model_data.get('nsfw', False),
        'trained_words': model_info.get('trainedWords', []),
        'download_url': model_info.get('downloadUrl'),
    }
