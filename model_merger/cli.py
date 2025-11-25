"""
Command-line interface for the model merger.

Handles argument parsing and orchestrates the merge workflow.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

from . import config
from . import manifest as manifest_module
from . import merger as merger_module
from . import vae as vae_module
from . import saver as saver_module
from . import loader as loader_module
from .console import (
    console, print_header, print_section, print_success, print_error, 
    print_warning, print_info, print_manifest_summary, print_completion,
    print_validation_issues
)


def cmd_scan(args):
    """
    Handle the 'scan' subcommand.
    
    Scans a folder for models and generates a manifest file.
    """
    folder = Path(args.folder)
    
    # Print header
    print_header("🎨 Model Merger - Scan Mode 🎨")
    
    # Handle VAE if provided
    vae_file = Path(args.vae) if args.vae else None
    if vae_file and not vae_file.exists():
        print_error(f"VAE file not found: {vae_file}")
        return 1
    
    # Generate manifest
    try:
        manifest = manifest_module.scan_folder(
            folder=folder,
            vae_file=vae_file,
            compute_hashes=args.compute_hashes,
            equal_weights=not args.no_equal_weights,
            skip_errors=args.skip_errors
        )
    except Exception as e:
        print_error(f"Error scanning folder: {e}")
        return 1
    
    # Determine output path for manifest
    if args.output:
        manifest_path = Path(args.output)
    else:
        # Save in the scanned folder by default
        manifest_path = folder / config.DEFAULT_MANIFEST_FILENAME
    
    # Save manifest
    try:
        manifest.save(manifest_path)
        
        console.print()
        print_info(f"Review and edit: [bold]{manifest_path}[/bold]")
        print_info(f"Then run: [bold cyan]python run.py merge --manifest {manifest_path}[/bold cyan]")
        
    except Exception as e:
        print_error(f"Error saving manifest: {e}")
        return 1
    
    return 0


def cmd_merge(args):
    """
    Handle the 'merge' subcommand.
    
    Loads a manifest and performs the merge operation.
    """
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print_error(f"Manifest file not found: {manifest_path}")
        return 1
    
    # Print header
    print_header("🎨 Model Merger v0.1.0 🎨")
    
    # Load manifest
    try:
        manifest = manifest_module.MergeManifest.load(manifest_path)
    except Exception as e:
        print_error(f"Error loading manifest: {e}")
        return 1
    
    # Apply CLI overrides
    if args.overwrite:
        manifest.overwrite = True
    if args.device:
        manifest.device = args.device
    
    # Display manifest summary
    print_manifest_summary(manifest)
    
    # Validate manifest
    console.print("\n[cyan]Validating manifest...[/cyan]")
    issues = manifest_module.validate_manifest(manifest)
    if issues:
        print_validation_issues(issues)
        
        # Check if any are errors (not warnings)
        errors = [i for i in issues if not i.startswith("Warning:")]
        if errors:
            print_error("Cannot proceed with errors. Please fix the manifest.")
            return 1
        else:
            # Just warnings, ask to continue
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                console.print("[yellow]Aborted.[/yellow]")
                return 1
    else:
        print_success("Manifest validation passed!")
    
    # Start timing the merge process
    start_time = time.time()
    
    # Step 1: Merge models
    try:
        merged_dict = merger_module.merge_models(
            model_entries=manifest.models,
            device=manifest.device,
            validate_compatibility=True
        )
    except Exception as e:
        print_error(f"Error during merge: {e}")
        return 1
    
    # Step 2: Bake VAE if specified
    if manifest.vae:
        try:
            vae_path = Path(manifest.vae.path)  # Extract path from VAEEntry
            merged_dict = vae_module.bake_vae(
                model_state_dict=merged_dict,
                vae_path=vae_path,
                device=manifest.device
            )
        except Exception as e:
            print_error(f"Error baking VAE: {e}")
            return 1
    
    # Step 3: Handle precision conversion
    # Determine target precision
    if manifest.output_precision == 'match':
        # Use first model's precision
        target_precision = manifest.models[0].precision_detected or 'fp32'
        console.print(f"\n[cyan]Matching first model's precision: {target_precision}[/cyan]")
    else:
        target_precision = manifest.output_precision
        console.print(f"\n[cyan]Converting to specified precision: {target_precision}[/cyan]")
    
    # Convert if needed
    try:
        merged_dict = merger_module.convert_precision(
            state_dict=merged_dict,
            target_precision=target_precision
        )
    except Exception as e:
        print_error(f"Error converting precision: {e}")
        return 1
    
    # Step 4: Prune if requested
    if manifest.prune:
        try:
            merged_dict = merger_module.prune_model(merged_dict)
        except Exception as e:
            print_error(f"Error pruning model: {e}")
            return 1
    
    # Step 5: Save the result
    try:
        # Generate metadata
        metadata = saver_module.save_manifest_metadata(
            manifest=manifest,
            merged_precision=target_precision
        )
        
        # Save model
        # Note: output is never None after __post_init__, but Pylance doesn't know that
        assert manifest.output is not None, "Output should always be set by __post_init__"
        output_path = Path(manifest.output.path)  # Extract path from OutputEntry
        output_hash = saver_module.save_model(
            state_dict=merged_dict,
            output_path=output_path,
            overwrite=manifest.overwrite,
            metadata=metadata
        )
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        # Update the OutputEntry with hash and precision info
        manifest.output.sha256 = output_hash
        manifest.output.precision_written = target_precision
        
        # Write the updated manifest back
        manifest.save(manifest_path)
        print_success(f"Updated manifest with output info: {manifest_path}")
        
        # Calculate total elapsed time
        elapsed_seconds = time.time() - start_time
        
        # Print beautiful completion message with timing
        print_completion(str(output_path), size_mb, output_hash, elapsed_seconds)
        
    except Exception as e:
        print_error(f"Error saving model: {e}")
        return 1
    
    return 0


def main():
    """
    Main entry point for the CLI.
    
    Sets up argument parsing with subcommands and dispatches to handlers.
    """
    parser = argparse.ArgumentParser(
        description='Model Merger - Merge multiple Stable Diffusion models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a folder and generate a manifest
  python run.py scan ./my_models --vae my_vae.safetensors
  
  # Edit the generated manifest file, then merge
  python run.py merge --manifest ./my_models/merge_manifest.json
  
  # Override settings from command line
  python run.py merge --manifest config.json --overwrite --device cuda
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Scan subcommand
    scan_parser = subparsers.add_parser(
        'scan',
        help='Scan a folder for models and generate a manifest'
    )
    scan_parser.add_argument(
        'folder',
        type=str,
        help='Folder containing model files (.safetensors)'
    )
    scan_parser.add_argument(
        '--vae',
        type=str,
        help='Path to VAE file to bake into merged model'
    )
    scan_parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output path for manifest file (default: folder/merge_manifest.json)'
    )
    scan_parser.add_argument(
        '--compute-hashes',
        action='store_true',
        help='Compute SHA-256 hashes (slow, but useful for verification)'
    )
    scan_parser.add_argument(
        '--no-equal-weights',
        action='store_true',
        help='Don\'t auto-calculate equal weights (weights will be 1.0, user must edit)'
    )
    scan_parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Skip files that can\'t be loaded (useful when files are still copying)'
    )
    
    # Merge subcommand
    merge_parser = subparsers.add_parser(
        'merge',
        help='Merge models using a manifest file'
    )
    merge_parser.add_argument(
        '--manifest',
        '-m',
        type=str,
        required=True,
        help='Path to merge manifest JSON file'
    )
    merge_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    merge_parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to use for merging (overrides manifest setting)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Dispatch to appropriate handler
    if args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())