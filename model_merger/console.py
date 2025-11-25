"""
Beautiful console output using Rich.

This module wraps Rich's formatting capabilities so we have consistent,
gorgeous output throughout the tool. No more boring print statements!
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import List, Dict, Any

# Create a single console instance we'll use everywhere
console = Console()


def print_header(title: str) -> None:
    """
    Print a fancy header banner.
    
    Example:
        print_header("Model Merger v0.1.0")
        
    Outputs a nice panel with the title centered.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            box=box.DOUBLE
        )
    )


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
    console.print(f"[bold yellow]{title}[/bold yellow]")
    console.print(f"[bold yellow]{'=' * 60}[/bold yellow]\n")


def print_success(message: str) -> None:
    """Print a success message in green with a checkmark."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message in red with an X."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow with a warning symbol."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message in cyan with an info symbol."""
    console.print(f"[cyan]ℹ[/cyan] {message}")


def print_step(current: int, total: int, message: str) -> None:
    """
    Print a step indicator.
    
    Example: [1/8] Loading base model...
    """
    console.print(f"[bold cyan][{current}/{total}][/bold cyan] {message}")


def print_models_table(models: List[Any]) -> None:
    """
    Display models in a beautiful table.
    
    Args:
        models: List of ModelEntry objects from manifest
        
    Shows model names, weights, and architectures in a formatted table.
    """
    table = Table(
        title="[bold]Models to Merge[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    table.add_column("Model", style="white", no_wrap=False, width=40)
    table.add_column("Weight", justify="right", style="yellow")
    table.add_column("Arch", style="green")
    table.add_column("Precision", style="magenta")
    
    for model in models:
        # Extract just the filename from the path
        from pathlib import Path
        filename = Path(model.path).name  # Use attribute, not subscript!
        
        # Truncate if too long
        if len(filename) > 37:
            filename = filename[:34] + "..."
        
        weight = f"{model.weight:.3f}"
        arch = model.architecture if hasattr(model, 'architecture') else '?'
        precision = model.precision_detected if hasattr(model, 'precision_detected') else '?'
        
        table.add_row(filename, weight, arch, precision)
    
    console.print(table)


def print_manifest_summary(manifest) -> None:
    """
    Print a summary of the merge manifest.
    
    Shows all the key info: models, VAE, output settings, etc.
    """
    print_section("Merge Configuration")
    
    # Print models table
    print_models_table(manifest.models)
    
    console.print()
    
    # Print other settings
    if manifest.vae:
        from pathlib import Path
        vae_name = Path(manifest.vae.path).name  # Extract path from VAEEntry
        console.print(f"[cyan]VAE:[/cyan] {vae_name}")
        if manifest.vae.precision_detected:
            console.print(f"[cyan]VAE Precision:[/cyan] {manifest.vae.precision_detected}")
    else:
        console.print(f"[dim]VAE: None[/dim]")
    
    # Output is never None after __post_init__, but help Pylance understand that
    assert manifest.output is not None, "Output should always be set"
    console.print(f"[cyan]Output:[/cyan] {manifest.output.path}")
    if manifest.output.sha256:
        console.print(f"[cyan]Output Hash:[/cyan] [dim]{manifest.output.sha256}[/dim]")
    if manifest.output.precision_written:
        console.print(f"[cyan]Output Precision:[/cyan] {manifest.output.precision_written}")
    console.print(f"[cyan]Precision Setting:[/cyan] {manifest.output_precision}")
    console.print(f"[cyan]Device:[/cyan] {manifest.device}")
    console.print(f"[cyan]Prune:[/cyan] {'Yes' if manifest.prune else 'No'}")


def create_progress() -> Progress:
    """
    Create a Rich Progress instance with our preferred styling.
    
    This replaces tqdm with something way prettier!
    
    Returns:
        Configured Progress object. Use with context manager:
        
        with create_progress() as progress:
            task = progress.add_task("Processing...", total=100)
            for i in range(100):
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )


def print_completion(output_path: str, size_mb: float, hash_value: str) -> None:
    """
    Print the final success message with all the details.
    
    Makes a beautiful panel with all the info about the merged model.
    """
    console.print()
    
    message = (
        f"[bold green]✨ MERGE COMPLETE! ✨[/bold green]\n\n"
        f"[cyan]Location:[/cyan] {output_path}\n"
        f"[cyan]Size:[/cyan] {size_mb:.2f} MB\n"
        f"[cyan]SHA-256:[/cyan] [dim]{hash_value}[/dim]\n\n"
        f"[dim]💡 Tip: Save this hash for verification or model lookup![/dim]"
    )
    
    console.print(
        Panel(
            message,
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2)
        )
    )


def print_validation_issues(issues: List[str]) -> None:
    """
    Print validation issues in a formatted way.
    
    Separates warnings from errors and uses appropriate colors.
    """
    console.print("\n[bold yellow]Manifest Validation Issues:[/bold yellow]")
    
    for issue in issues:
        if issue.startswith("Warning:"):
            print_warning(issue[9:])  # Strip "Warning: " prefix
        else:
            print_error(issue)