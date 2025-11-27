"""
Example of using CivitAI API integration.

This shows how to:
1. Check if an API key is configured
2. Hash a model file
3. Look up model metadata from CivitAI by hash
4. Display model information
"""

from pathlib import Path
from model_merger.config import has_civitai_api_key
from model_merger.console import console, print_warning, print_success, print_error
from model_merger.loader import compute_file_hash
from model_merger.civitai import (
    get_model_version_by_hash,
    detect_architecture_from_civitai,
    get_model_metadata_summary
)

# Example model path - update this to an actual model on your system
#EXAMPLE_MODEL_PATH = Path("models/miaomiaoHarem_v175.safetensors")
EXAMPLE_MODEL_PATH = Path("models/cyberrealisticPony_catalystV40DMD2.safetensors")


def example_civitai_integration():
    """Example of checking for and using the CivitAI API key.
    
    Returns:
        File hash if successful, None otherwise
    """
    
    # Check if API key is configured
    if not has_civitai_api_key():
        print_error("No CivitAI API key configured")
        console.print("\n[yellow]To configure your API key:[/yellow]")
        console.print("1. Copy .env.example to .env")
        console.print("2. Get your API key from https://civitai.com/user/account")
        console.print("3. Add it to .env: CIVITAI_API_KEY=your_key_here")
        return None
    
    print_success("CivitAI API key is configured!")
    
    # Check if example model exists
    if not EXAMPLE_MODEL_PATH.exists():
        print_error(f"Example model not found: {EXAMPLE_MODEL_PATH}")
        console.print("\n[yellow]Update EXAMPLE_MODEL_PATH in this file to point to an actual model.[/yellow]")
        return None
    
    console.print(f"\n[cyan]Analyzing model:[/cyan] {EXAMPLE_MODEL_PATH.name}")
    
    # Step 1: Compute file hash
    console.print("\n[cyan]Step 1: Computing file hash...[/cyan]")
    with console.status("[cyan]Hashing file..."):
        file_hash = compute_file_hash(EXAMPLE_MODEL_PATH)
    print_success(f"Hash: {file_hash}")
    
    # Step 2: Look up model info from CivitAI
    console.print("\n[cyan]Step 2: Querying CivitAI API...[/cyan]")
    with console.status("[cyan]Fetching metadata..."):
        model_info = get_model_version_by_hash(file_hash)
    
    if not model_info:
        print_warning("Model not found in CivitAI database")
        console.print("This could mean:")
        console.print("  - Model is not hosted on CivitAI")
        console.print("  - Hash hasn't been indexed yet (new model)")
        console.print("  - Hash doesn't match any known versions")
        return None
    
    print_success("Model found in CivitAI!")
    
    # Step 3: Display model metadata
    console.print("\n[bold cyan]═══ Model Metadata ═══[/bold cyan]")
    
    # Get summary for easy display
    summary = get_model_metadata_summary(file_hash)
    if summary:
        console.print(f"\n[bold]Model Name:[/bold] {summary['model_name']}")
        console.print(f"[bold]Version:[/bold] {summary['version_name']}")
        console.print(f"[bold]Type:[/bold] {summary['type']}")
        console.print(f"[bold]Base Model:[/bold] {summary['base_model']}")
        console.print(f"[bold]NSFW:[/bold] {summary['nsfw']}")
        
        # Show detected architecture
        architecture = summary['architecture']
        if architecture:
            console.print(f"[bold]Detected Architecture:[/bold] [green]{architecture}[/green]")
        else:
            console.print(f"[bold]Detected Architecture:[/bold] [yellow]Unknown[/yellow]")
        
        # Show trigger words if any
        trained_words = summary['trained_words']
        if trained_words:
            console.print(f"\n[bold]Trigger Words:[/bold]")
            for word in trained_words[:5]:  # Show first 5
                console.print(f"  • {word}")
            if len(trained_words) > 5:
                console.print(f"  ... and {len(trained_words) - 5} more")
        
        console.print(f"\n[bold]Download URL:[/bold] {summary['download_url']}")
    
    # Show raw data structure
    console.print("\n[dim]Full API response available in model_info variable[/dim]")
    console.print(f"[dim]Keys: {', '.join(model_info.keys())}[/dim]")
    
    # Return the hash for use by other examples
    return file_hash


def example_architecture_detection(file_hash: str):
    """Example of automatic architecture detection."""
    
    console.print("\n[bold cyan]═══ Architecture Detection Example ═══[/bold cyan]")
    
    # Method 1: Detect from CivitAI with filename fallback
    arch = detect_architecture_from_civitai(file_hash, EXAMPLE_MODEL_PATH.name)
    
    if arch:
        console.print(f"\n[green]✓[/green] Detected architecture: [bold]{arch}[/bold]")
        console.print(f"This model can be merged with other {arch} models.")
    else:
        console.print("\n[yellow]?[/yellow] Could not detect architecture")
        console.print("You may need to specify architecture manually.")


if __name__ == '__main__':
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   CivitAI API Integration Example   [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    
    # Run main example and get the file hash
    file_hash = example_civitai_integration()
    
    # Only run architecture detection if we got a hash
    if file_hash:
        example_architecture_detection(file_hash)
    
    console.print("\n[bold green]✓[/bold green] Example complete!")

