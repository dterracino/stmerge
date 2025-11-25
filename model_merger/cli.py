"""
Command-line interface for the model merger.

Handles argument parsing and orchestrates the merge workflow.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from . import config
from . import manifest as manifest_module
from . import merger as merger_module
from . import vae as vae_module
from . import saver as saver_module
from . import loader as loader_module


def cmd_scan(args):
    """
    Handle the 'scan' subcommand.
    
    Scans a folder for models and generates a manifest file.
    """
    folder = Path(args.folder)
    
    # Handle VAE if provided
    vae_file = Path(args.vae) if args.vae else None
    if vae_file and not vae_file.exists():
        print(f"Error: VAE file not found: {vae_file}")
        return 1
    
    # Generate manifest
    try:
        manifest = manifest_module.scan_folder(
            folder=folder,
            vae_file=vae_file,
            compute_hashes=args.compute_hashes,
            equal_weights=not args.no_equal_weights
        )
    except Exception as e:
        print(f"Error scanning folder: {e}")
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
        
        print("\n" + "=" * 60)
        print("Manifest generated successfully!")
        print(f"Review and edit: {manifest_path}")
        print("Then run: python run.py merge --manifest", manifest_path)
        
    except Exception as e:
        print(f"Error saving manifest: {e}")
        return 1
    
    return 0


def cmd_merge(args):
    """
    Handle the 'merge' subcommand.
    
    Loads a manifest and performs the merge operation.
    """
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}")
        return 1
    
    # Load manifest
    try:
        manifest = manifest_module.MergeManifest.load(manifest_path)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return 1
    
    # Apply CLI overrides
    if args.overwrite:
        manifest.overwrite = True
    if args.device:
        manifest.device = args.device
    
    # Validate manifest
    print("Validating manifest...")
    issues = manifest_module.validate_manifest(manifest)
    if issues:
        print("\nManifest validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        # Check if any are errors (not warnings)
        errors = [i for i in issues if not i.startswith("Warning:")]
        if errors:
            print("\nCannot proceed with errors. Please fix the manifest.")
            return 1
        else:
            # Just warnings, ask to continue
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1
    
    print("\n" + "=" * 60)
    print("Starting merge process...")
    print("=" * 60)
    
    # Step 1: Merge models
    try:
        merged_dict = merger_module.merge_models(
            model_entries=manifest.models,
            device=manifest.device,
            validate_compatibility=True
        )
    except Exception as e:
        print(f"\nError during merge: {e}")
        return 1
    
    # Step 2: Bake VAE if specified
    if manifest.vae:
        try:
            vae_path = Path(manifest.vae)
            merged_dict = vae_module.bake_vae(
                model_state_dict=merged_dict,
                vae_path=vae_path,
                device=manifest.device
            )
        except Exception as e:
            print(f"\nError baking VAE: {e}")
            return 1
    
    # Step 3: Handle precision conversion
    # Determine target precision
    if manifest.output_precision == 'match':
        # Use first model's precision
        target_precision = manifest.models[0].precision_detected or 'fp32'
        print(f"\nMatching first model's precision: {target_precision}")
    else:
        target_precision = manifest.output_precision
        print(f"\nConverting to specified precision: {target_precision}")
    
    # Convert if needed
    try:
        merged_dict = merger_module.convert_precision(
            state_dict=merged_dict,
            target_precision=target_precision
        )
    except Exception as e:
        print(f"\nError converting precision: {e}")
        return 1
    
    # Step 4: Prune if requested
    if manifest.prune:
        try:
            merged_dict = merger_module.prune_model(merged_dict)
        except Exception as e:
            print(f"\nError pruning model: {e}")
            return 1
    
    # Step 5: Save the result
    try:
        # Generate metadata
        metadata = saver_module.save_manifest_metadata(
            manifest=manifest,
            merged_precision=target_precision
        )
        
        # Save model
        output_path = Path(manifest.output)
        saver_module.save_model(
            state_dict=merged_dict,
            output_path=output_path,
            overwrite=manifest.overwrite,
            metadata=metadata
        )
        
        print("\n" + "=" * 60)
        print("🎉 MERGE COMPLETE! 🎉")
        print("=" * 60)
        print(f"Your merged model is ready at: {output_path}")
        
    except Exception as e:
        print(f"\nError saving model: {e}")
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