"""
Manifest generation and validation.

This module handles scanning model folders, generating merge manifests,
and validating manifest files before processing.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from . import config
from .loader import compute_file_hash, load_model
from .console import console, print_info, print_success


@dataclass
class ModelEntry:
    """Represents a single model in the merge manifest."""
    path: str
    weight: float
    architecture: str
    sha256: Optional[str] = None
    precision_detected: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MergeManifest:
    """Complete merge configuration."""
    models: List[ModelEntry]
    vae: Optional[str] = None
    vae_sha256: Optional[str] = None
    output: str = config.DEFAULT_OUTPUT_FILENAME
    output_precision: str = 'match'
    device: str = 'cpu'
    prune: bool = True
    overwrite: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'models': [m.to_dict() for m in self.models],
            'vae': self.vae,
            'vae_sha256': self.vae_sha256,
            'output': self.output,
            'output_precision': self.output_precision,
            'device': self.device,
            'prune': self.prune,
            'overwrite': self.overwrite,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MergeManifest':
        """Create manifest from dictionary (loaded from JSON)."""
        models = [ModelEntry(**m) for m in data['models']]
        return cls(
            models=models,
            vae=data.get('vae'),
            vae_sha256=data.get('vae_sha256'),
            output=data.get('output', config.DEFAULT_OUTPUT_FILENAME),
            output_precision=data.get('output_precision', 'match'),
            device=data.get('device', 'cpu'),
            prune=data.get('prune', True),
            overwrite=data.get('overwrite', False),
        )
    
    def save(self, filepath: Path) -> None:
        """Save manifest to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print_success(f"Manifest saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'MergeManifest':
        """Load manifest from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_output_filename(model_files: List[Path], architecture: str) -> str:
    """
    Generate a sensible output filename from input models.
    
    Takes the first N characters from each model filename (stripping
    punctuation), concatenates them, and adds architecture prefix.
    
    Example:
        ['pony_realistic.safetensors', 'pony_furry_v2.safetensors']
        -> 'Pony_PonyReal_PonyFurr_merged.safetensors'
    
    Args:
        model_files: List of model file paths
        architecture: Detected architecture name
        
    Returns:
        Generated filename
    """
    # Extract prefixes from each filename
    prefixes = []
    for model_file in model_files:
        # Get filename without extension
        name = model_file.stem
        
        # Strip punctuation and get first N chars
        clean_name = re.sub(config.FILENAME_STRIP_CHARS, '', name)
        prefix = clean_name[:config.FILENAME_PREFIX_LENGTH]
        
        if prefix:
            prefixes.append(prefix)
    
    # Limit to reasonable number of prefixes (avoid super long names)
    if len(prefixes) > 5:
        prefixes = prefixes[:5]
    
    # Build filename: Architecture_Prefix1_Prefix2_..._merged.safetensors
    filename_parts = [architecture] + prefixes + ['merged']
    filename = '_'.join(filename_parts) + '.safetensors'
    
    return filename


def scan_folder(
    folder: Path,
    vae_file: Optional[Path] = None,
    compute_hashes: bool = False,
    equal_weights: bool = True
) -> MergeManifest:
    """
    Scan a folder for model files and generate a merge manifest.
    
    This is the main entry point for the "scan and generate manifest" workflow.
    It finds all .safetensors files in the folder, detects their architecture,
    and creates a manifest with sensible defaults.
    
    Args:
        folder: Directory containing model files
        vae_file: Optional VAE file to bake in
        compute_hashes: Whether to compute SHA-256 hashes (slow!)
        equal_weights: If True, give all models equal weights (1/N each)
        
    Returns:
        Generated MergeManifest
        
    Raises:
        ValueError: If no model files found in folder
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    
    # Find all model files
    model_files = sorted(folder.glob('*.safetensors'))
    
    if not model_files:
        raise ValueError(f"No .safetensors files found in {folder}")
    
    console.print(f"[cyan]Found {len(model_files)} model files in {folder}[/cyan]")
    
    # Detect architecture from first file (assume all same architecture)
    first_file = model_files[0]
    detected_arch = config.detect_architecture_from_filename(first_file.name)
    print_info(f"Detected architecture: [bold]{detected_arch}[/bold] (from {first_file.name})")
    
    # Calculate equal weights if requested
    if equal_weights:
        weight = 1.0 / len(model_files)
    else:
        weight = 1.0  # User will need to edit these
    
    # Build model entries
    model_entries = []
    for model_file in model_files:
        # Detect architecture for this specific file
        arch = config.detect_architecture_from_filename(model_file.name)
        
        # Optionally compute hash
        file_hash = None
        if compute_hashes:
            print_info(f"Computing hash for {model_file.name}...")
            file_hash = compute_file_hash(model_file)
        
        # Load model briefly to detect precision
        console.print(f"[cyan]Detecting precision for {model_file.name}...[/cyan]")
        state_dict, metadata = load_model(model_file, device='cpu', compute_hash=False)
        precision = metadata['precision']
        del state_dict  # Free memory immediately
        
        entry = ModelEntry(
            path=str(model_file),
            weight=weight,
            architecture=arch,
            sha256=file_hash,
            precision_detected=precision
        )
        model_entries.append(entry)
    
    # Handle VAE if provided
    vae_path = None
    vae_hash = None
    if vae_file:
        if not vae_file.exists():
            raise FileNotFoundError(f"VAE file not found: {vae_file}")
        vae_path = str(vae_file)
        if compute_hashes:
            print_info(f"Computing hash for VAE: {vae_file.name}...")
            vae_hash = compute_file_hash(vae_file)
            console.print(f"  [dim]VAE SHA-256: {vae_hash}[/dim]")
    
    # Generate output filename
    output_filename = generate_output_filename(model_files, detected_arch)
    
    # Create manifest
    manifest = MergeManifest(
        models=model_entries,
        vae=vae_path,
        vae_sha256=vae_hash,
        output=output_filename,
        output_precision='match',
        device='cpu',
        prune=True,
        overwrite=False,
    )
    
    return manifest


def validate_manifest(manifest: MergeManifest) -> List[str]:
    """
    Validate a manifest for common issues.
    
    Checks:
    - All model files exist
    - VAE file exists (if specified)
    - Weights are reasonable (warn if they don't sum to ~1.0)
    - Output directory is writable
    
    Args:
        manifest: The manifest to validate
        
    Returns:
        List of warning/error messages (empty if all good)
    """
    issues = []
    
    # Check model files exist
    for model in manifest.models:
        model_path = Path(model.path)
        if not model_path.exists():
            issues.append(f"Model file not found: {model.path}")
    
    # Check VAE file exists
    if manifest.vae:
        vae_path = Path(manifest.vae)
        if not vae_path.exists():
            issues.append(f"VAE file not found: {manifest.vae}")
    
    # Check weights sum (warn, don't error)
    total_weight = sum(m.weight for m in manifest.models)
    if not (0.9 <= total_weight <= 1.1):
        issues.append(
            f"Warning: Weights sum to {total_weight:.3f}, not 1.0. "
            f"This is fine if intentional, but may produce unexpected results."
        )
    
    # Check output directory exists/is writable
    output_path = Path(manifest.output)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
    if not output_dir.exists():
        issues.append(f"Output directory does not exist: {output_dir}")
    
    return issues