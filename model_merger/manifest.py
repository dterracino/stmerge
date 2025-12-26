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
from .loader import compute_file_hash, load_model, load_vae
from .console import console, print_info, print_success


@dataclass
class ModelEntry:
    """Represents a single model in the merge manifest."""
    path: str
    weight: float
    architecture: str
    index: int  # Required: position in merge sequence (0-based, contiguous)
    sha256: Optional[str] = None
    crc32: Optional[str] = None  # 8-character hex CRC32 (for legacy A1111 hash)
    precision_detected: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VAEEntry:
    """Represents the VAE in the merge manifest."""
    path: str
    sha256: Optional[str] = None
    crc32: Optional[str] = None  # 8-character hex CRC32 (for legacy A1111 hash)
    precision_detected: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class OutputEntry:
    """Represents the output model from the merge."""
    path: str
    sha256: Optional[str] = None
    precision_written: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MergeManifest:
    """Complete merge configuration."""
    models: List[ModelEntry]
    vae: Optional[VAEEntry] = None
    output: Optional[OutputEntry] = None  # Mark as optional since it can be None initially
    output_precision: str = 'match'  # Keep for pre-merge config
    device: str = 'cpu'
    prune: bool = True
    overwrite: bool = False
    merge_method: str = config.MERGE_METHOD_WEIGHTED_SUM
    consensus_exponent: int = config.DEFAULT_CONSENSUS_EXPONENT
    
    def __post_init__(self):
        """Ensure output is an OutputEntry, even if initialized with just a string."""
        if isinstance(self.output, str):
            # Convert string path to OutputEntry for backwards compat
            self.output = OutputEntry(path=self.output)
        elif self.output is None:
            # Default output
            self.output = OutputEntry(path=config.DEFAULT_OUTPUT_FILENAME)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'models': [m.to_dict() for m in self.models],
            'vae': self.vae.to_dict() if self.vae else None,
            'output': self.output.to_dict() if self.output else None,
            'output_precision': self.output_precision,
            'device': self.device,
            'prune': self.prune,
            'overwrite': self.overwrite,
            'merge_method': self.merge_method,
            'consensus_exponent': self.consensus_exponent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MergeManifest':
        """Create manifest from dictionary (loaded from JSON)."""
        models = [ModelEntry(**m) for m in data['models']]
        
        # Validate model indices
        indices = [m.index for m in models]
        expected = list(range(len(models)))
        
        # Check for duplicates
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Duplicate model indices found: {indices}\n"
                f"Each model must have a unique index."
            )
        
        # Check for contiguous 0-based sequence
        if sorted(indices) != expected:
            raise ValueError(
                f"Model indices must be contiguous starting from 0.\n"
                f"Expected: {expected}\n"
                f"Got: {sorted(indices)}\n"
                f"Please renumber your models in the manifest."
            )
        
        # Sort models by index (so JSON array order doesn't matter)
        models.sort(key=lambda m: m.index)
        
        # Handle VAE - could be old format (string) or new format (dict)
        vae_data = data.get('vae')
        if vae_data:
            if isinstance(vae_data, dict):
                # New format
                vae = VAEEntry(**vae_data)
            else:
                # Old format (just a path string) - convert for backwards compat
                vae = VAEEntry(path=vae_data, sha256=data.get('vae_sha256'))
        else:
            vae = None
        
        # Handle output - could be old format (string) or new format (dict)
        output_data = data.get('output', config.DEFAULT_OUTPUT_FILENAME)
        if isinstance(output_data, dict):
            # New format - OutputEntry
            output = OutputEntry(**output_data)
        else:
            # Old format (just a path string) - convert for backwards compat
            # Check if there's an old-style output_sha256 field
            output = OutputEntry(
                path=output_data,
                sha256=data.get('output_sha256')
            )
        
        return cls(
            models=models,
            vae=vae,
            output=output,
            output_precision=data.get('output_precision', 'match'),
            device=data.get('device', 'cpu'),
            prune=data.get('prune', True),
            overwrite=data.get('overwrite', False),
            merge_method=data.get('merge_method', config.MERGE_METHOD_WEIGHTED_SUM),
            consensus_exponent=data.get('consensus_exponent', config.DEFAULT_CONSENSUS_EXPONENT),
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
    equal_weights: bool = True,
    skip_errors: bool = False
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
        skip_errors: If True, skip files that can't be loaded and continue
        
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
    
    # Build model entries
    model_entries = []
    skipped_files = []
    
    for idx, model_file in enumerate(model_files):
        try:
            # Detect architecture for this specific file
            arch = config.detect_architecture_from_filename(model_file.name)
            
            # Load model briefly to detect precision (and optionally compute hashes)
            console.print(f"[cyan]Processing {model_file.name}...[/cyan]")
            state_dict, metadata = load_model(model_file, device='cpu', compute_hash=compute_hashes)
            precision = metadata['precision']
            file_hash = metadata.get('sha256')
            file_crc32 = metadata.get('crc32')
            del state_dict  # Free memory immediately
            
            # Don't set weight yet - we'll do that after we know how many loaded successfully
            entry = ModelEntry(
                path=str(model_file),
                weight=0.0,  # Placeholder, will be set below
                architecture=arch,
                index=idx,  # Explicit sequential index
                sha256=file_hash,
                crc32=file_crc32,
                precision_detected=precision
            )
            model_entries.append(entry)
            
        except Exception as e:
            if skip_errors:
                from .console import print_warning
                print_warning(f"Skipping {model_file.name}: {e}")
                skipped_files.append(model_file.name)
                continue
            else:
                raise
    
    # Check if we have any valid entries
    if not model_entries:
        if skipped_files:
            raise ValueError(
                f"No valid models found. All {len(skipped_files)} files were skipped due to errors. "
                f"Files: {', '.join(skipped_files)}"
            )
        else:
            raise ValueError("No model entries created (this shouldn't happen!)")
    
    # Report skipped files at the end
    if skipped_files:
        console.print(f"\n[yellow]⚠ Skipped {len(skipped_files)} file(s) due to errors:[/yellow]")
        for skipped in skipped_files:
            console.print(f"  [dim]- {skipped}[/dim]")
        
        # Renumber indices to be contiguous after skipping files
        print_info("Renumbering model indices to be contiguous...")
        for new_idx, entry in enumerate(model_entries):
            entry.index = new_idx
    
    # NOW calculate weights based on actual successful entries
    if equal_weights:
        weight = 1.0 / len(model_entries)
        print_info(f"Setting equal weights: {weight:.4f} for each of {len(model_entries)} models")
        for entry in model_entries:
            entry.weight = weight
    else:
        # User will edit these manually
        for entry in model_entries:
            entry.weight = 1.0
    
    # Handle VAE if provided
    vae_entry = None
    if vae_file:
        if not vae_file.exists():
            raise FileNotFoundError(f"VAE file not found: {vae_file}")
        
        # Load VAE and optionally compute hashes
        console.print(f"[cyan]Processing VAE: {vae_file.name}...[/cyan]")
        vae_state_dict, vae_metadata = load_vae(vae_file, device='cpu', compute_hash=compute_hashes)
        vae_precision = vae_metadata['precision']
        vae_hash = vae_metadata.get('sha256')
        vae_crc32 = vae_metadata.get('crc32')
        del vae_state_dict  # Free memory
        
        vae_entry = VAEEntry(
            path=str(vae_file),
            sha256=vae_hash,
            crc32=vae_crc32,
            precision_detected=vae_precision
        )
    
    # Generate output filename
    output_filename = generate_output_filename(model_files, detected_arch)
    
    # Create manifest
    manifest = MergeManifest(
        models=model_entries,
        vae=vae_entry,
        output=OutputEntry(path=output_filename),  # Create OutputEntry with path
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
        vae_path = Path(manifest.vae.path)  # Extract path string from VAEEntry
        if not vae_path.exists():
            issues.append(f"VAE file not found: {manifest.vae.path}")
    
    # Check weights sum (warn, don't error)
    total_weight = sum(m.weight for m in manifest.models)
    if not (0.9 <= total_weight <= 1.1):
        issues.append(
            f"Warning: Weights sum to {total_weight:.3f}, not 1.0. "
            f"This is fine if intentional, but may produce unexpected results."
        )
    
    # Check output directory exists/is writable
    # Note: output is never None after __post_init__, but Pylance doesn't know that
    assert manifest.output is not None, "Output should always be set"
    output_path = Path(manifest.output.path)  # Extract path from OutputEntry
    output_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
    if not output_dir.exists():
        issues.append(f"Output directory does not exist: {output_dir}")
    
    return issues