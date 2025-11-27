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
from . import converter as converter_module
from . import verifier as verifier_module
from .__init__ import __version__
from .console import (
    console, print_header, print_section, print_success, print_error, 
    print_warning, print_info, print_manifest_summary, print_completion,
    print_validation_issues
)


def check_cuda_availability(requested_device: str) -> str:
    """
    Check if CUDA is available when requested, warn if not.
    
    Args:
        requested_device: The device the user requested ('cuda' or 'cpu')
        
    Returns:
        The actual device to use (may fall back to 'cpu' if CUDA unavailable)
    """
    if requested_device != 'cuda':
        return requested_device
    
    import torch
    if not torch.cuda.is_available():
        console.print()
        console.print("[yellow]⚠ Warning: CUDA requested but not available![/yellow]")
        console.print("[yellow]  Possible reasons:[/yellow]")
        console.print("[yellow]  • PyTorch CPU-only version installed[/yellow]")
        console.print("[yellow]  • No NVIDIA GPU detected[/yellow]")
        console.print("[yellow]  • CUDA drivers not installed[/yellow]")
        console.print()
        console.print("[cyan]To enable GPU acceleration:[/cyan]")
        console.print("[dim]  1. Check CUDA version: nvidia-smi[/dim]")
        console.print("[dim]  2. Install CUDA PyTorch (match your CUDA version):[/dim]")
        console.print("[dim]     pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8[/dim]")
        console.print("[dim]     pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1[/dim]")
        console.print("[dim]     pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8[/dim]")
        console.print("[dim]     pip install torch --index-url https://download.pytorch.org/whl/cu130  # CUDA 13.0[/dim]")
        console.print()
        console.print("[yellow]Falling back to CPU...[/yellow]")
        console.print()
        return 'cpu'
    
    return 'cuda'


def print_device_info(selected_device: str):
    """
    Display device information (CUDA availability and selected device).
    
    Args:
        selected_device: The device that will be used ('cuda' or 'cpu')
    """
    import torch
    
    cuda_available = torch.cuda.is_available()
    
    console.print()
    console.print("[cyan]Device Information:[/cyan]")
    console.print(f"  CUDA Available: {'[green]Yes[/green]' if cuda_available else '[yellow]No[/yellow]'}")
    
    if cuda_available:
        cuda_device_name = torch.cuda.get_device_name(0)
        console.print(f"  GPU: [dim]{cuda_device_name}[/dim]")
    
    console.print(f"  Selected Device: [bold]{'[green]CUDA[/green]' if selected_device == 'cuda' else '[blue]CPU[/blue]'}[/bold]")
    console.print()


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


def cmd_convert(args) -> int:
    """Handle the convert subcommand."""
    from . import notifier as notifier_module
    
    input_path = Path(args.input)
    
    # Print header
    print_header("🔄 Model Merger - Convert Mode 🔄")
    
    # Check input exists
    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        return 1
    
    # Validate file extension
    if input_path.suffix.lower() not in config.SUPPORTED_CONVERT_EXTENSIONS:
        if not args.force:
            console.print()
            console.print(f"[yellow]⚠ Warning: Unrecognized file extension: {input_path.suffix}[/yellow]")
            console.print(f"[yellow]  Supported formats: {', '.join(sorted(config.SUPPORTED_CONVERT_EXTENSIONS))}[/yellow]")
            console.print()
            console.print("[cyan]This might be:[/cyan]")
            console.print("  [dim]• A typo in the filename[/dim]")
            console.print("  [dim]• A renamed file with wrong extension[/dim]")
            console.print("  [dim]• An unusual format that might work anyway[/dim]")
            console.print()
            
            response = input("Attempt conversion anyway? [y/N]: ")
            if response.lower() != 'y':
                console.print("[yellow]Conversion cancelled.[/yellow]")
                return 1
            
            console.print("[dim]Proceeding with conversion attempt...[/dim]")
            console.print()
        else:
            # --force flag specified, just show brief warning
            console.print(f"[yellow]Note: Unrecognized extension {input_path.suffix} (--force specified)[/yellow]")
            console.print()
    
    # Determine output path
    output_path = Path(args.output) if args.output else None
    
    # Convert!
    try:
        # Start timing
        start_time = time.time()
        
        output_hash = converter_module.convert_to_safetensors(
            input_path=input_path,
            output_path=output_path,
            prune=not args.no_prune,  # Default is True, --no-prune makes it False
            compute_hash=args.compute_hash,
            overwrite=args.overwrite
        )
        
        # Calculate elapsed time
        elapsed_seconds = time.time() - start_time
        
        # Get the actual output path that was used
        if output_path is None:
            output_path = input_path.with_suffix('.safetensors')
        
        # Get file size
        size_bytes = output_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Print completion message
        print_completion(str(output_path), size_mb, output_hash, elapsed_seconds)
        
        # Send notification if requested and operation took long enough
        if notifier_module.should_notify(args.notify, elapsed_seconds):
            notifier_module.notify_conversion_success(
                output_path.name,
                size_bytes,
                elapsed_seconds
            )
        
        # Helpful tip
        console.print()
        print_info("You can now use this model in ComfyUI, A1111, or merge it with other models!")
        print_info(f"Consider deleting the original: [dim]{input_path}[/dim]")
        
    except Exception as e:
        error_msg = str(e)
        print_error(f"Conversion failed: {error_msg}")
        
        # Send failure notification if requested
        if args.notify and notifier_module.is_available():
            notifier_module.notify_conversion_failure(input_path.name, error_msg)
        
        return 1
    
    return 0


def cmd_verify(args) -> int:
    """Handle the verify subcommand."""
    original_path = Path(args.original)
    converted_path = Path(args.converted)
    
    # Print header
    print_header("🔍 Model Merger - Verify Mode 🔍")
    
    # Check files exist
    if not original_path.exists():
        print_error(f"Original file not found: {original_path}")
        return 1
    
    if not converted_path.exists():
        print_error(f"Converted file not found: {converted_path}")
        return 1
    
    # Run verification
    try:
        passed = verifier_module.verify_conversion(
            original_path=original_path,
            converted_path=converted_path,
            verbose=args.verbose
        )
        
        if passed:
            console.print()
            print_success("Conversion verified successfully! Files are identical.")
            return 0
        else:
            console.print()
            print_error("Verification failed! Files do not match.")
            return 1
            
    except Exception as e:
        print_error(f"Verification error: {e}")
        return 1


def cmd_merge(args):
    """
    Handle the 'merge' subcommand.
    
    Loads a manifest and performs the merge operation.
    """
    from . import notifier as notifier_module
    from . import __version__
    
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print_error(f"Manifest file not found: {manifest_path}")
        return 1
    
    # Print header
    print_header(f"🎨 Model Merger v{__version__} 🎨")
    
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
    if args.no_prune:
        manifest.prune = False  # CLI says don't prune, update manifest
    
    # Check CUDA availability and warn if needed
    manifest.device = check_cuda_availability(manifest.device)
    
    # Display device information
    print_device_info(manifest.device)
    
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
        error_msg = str(e)
        print_error(f"Error during merge: {error_msg}")
        
        # Send failure notification if requested
        if args.notify and notifier_module.is_available():
            notifier_module.notify_merge_failure(error_msg)
        
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
            error_msg = str(e)
            print_error(f"Error baking VAE: {error_msg}")
            
            # Send failure notification if requested
            if args.notify and notifier_module.is_available():
                notifier_module.notify_merge_failure(f"VAE baking failed: {error_msg}")
            
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
        size_bytes = output_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Update the OutputEntry with hash and precision info
        manifest.output.sha256 = output_hash
        manifest.output.precision_written = target_precision
        
        # Write the updated manifest back
        # This includes any CLI overrides (--no-prune, --device, etc.) so the manifest
        # reflects what ACTUALLY happened in this merge
        manifest.save(manifest_path)
        print_success(f"Manifest updated with merge results: {manifest_path}")
        
        # Calculate total elapsed time
        elapsed_seconds = time.time() - start_time
        
        # Print beautiful completion message with timing
        print_completion(str(output_path), size_mb, output_hash, elapsed_seconds)
        
        # Send notification if requested and operation took long enough
        if notifier_module.should_notify(args.notify, elapsed_seconds):
            notifier_module.notify_merge_success(
                output_path.name,
                size_bytes,
                elapsed_seconds
            )
        
    except Exception as e:
        error_msg = str(e)
        print_error(f"Error saving model: {error_msg}")
        
        # Send failure notification if requested
        if args.notify and notifier_module.is_available():
            notifier_module.notify_merge_failure(error_msg)
        
        return 1
    
    return 0


def main():
    """
    Main entry point for the CLI.
    
    Sets up argument parsing with subcommands and dispatches to handlers.
    """
    parser = argparse.ArgumentParser(
        description='Model Merger ' + __version__ + ' - Merge multiple Stable Diffusion models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a folder and generate a manifest
  python run.py scan ./my_models --vae my_vae.safetensors
  
  # Edit the generated manifest file, then merge
  python run.py merge --manifest ./my_models/merge_manifest.json
  
  # Override settings from command line
  python run.py merge --manifest config.json --overwrite --device cuda
  
  # Convert legacy checkpoint
  python run.py convert old_model.ckpt
  
  # Verify conversion quality
  python run.py verify old_model.ckpt new_model.safetensors
        """
    )
    
    # Add version argument
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'Model Merger v{__version__}'
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
    merge_parser.add_argument(
        '--no-prune',
        action='store_true',
        help='Don\'t prune unnecessary keys (overrides manifest setting)'
    )
    merge_parser.add_argument(
        '--notify',
        action='store_true',
        help='Send desktop notification when complete (long operations only)'
    )
    
    # Convert subcommand
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert legacy checkpoint formats (.ckpt/.pt/.pth/.bin) to safetensors'
    )
    convert_parser.add_argument(
        'input',
        type=str,
        help='Input checkpoint file to convert'
    )
    convert_parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output path (default: same name with .safetensors extension)'
    )
    convert_parser.add_argument(
        '--no-prune',
        action='store_true',
        help='Don\'t prune unnecessary keys (keep training state, optimizer, etc.)'
    )
    convert_parser.add_argument(
        '--compute-hash',
        action='store_true',
        help='Compute SHA-256 hash of input file (slow but provides verification)'
    )
    convert_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    convert_parser.add_argument(
        '--notify',
        action='store_true',
        help='Send desktop notification when complete (long operations only)'
    )
    convert_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts (for unrecognized file extensions)'
    )
    
    # Verify subcommand
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify that a converted model matches the original'
    )
    verify_parser.add_argument(
        'original',
        type=str,
        help='Path to original model file (.ckpt/.pt/.pth/.bin)'
    )
    verify_parser.add_argument(
        'converted',
        type=str,
        help='Path to converted safetensors file'
    )
    verify_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed comparison for every tensor'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Dispatch to appropriate handler
    if args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    elif args.command == 'convert':
        return cmd_convert(args)
    elif args.command == 'verify':
        return cmd_verify(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())