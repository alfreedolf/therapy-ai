"""
main.py — top-level CLI launcher

Usage:
  python main.py ui          → launch Streamlit app
  python main.py info        → print hardware backend info
  python main.py check       → validate config + model availability
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app     = typer.Typer(help="Therapy AI — local privacy-preserving psychotherapy assistant")
console = Console()


@app.command()
def ui(
    port:   int  = typer.Option(8501, help="Streamlit port"),
    config: str  = typer.Option("config/default_config.yaml"),
) -> None:
    """Launch the Streamlit UI."""
    console.print(Panel.fit(
        "🌿 [bold green]Therapy AI[/bold green] — starting local UI\n"
        f"[dim]http://localhost:{port}[/dim]",
        border_style="green",
    ))
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "therapy_ai/ui/app.py",
        f"--server.port={port}",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",   # privacy: no Streamlit telemetry
    ]
    subprocess.run(cmd, check=True)


@app.command()
def info() -> None:
    """Print detected hardware backend information."""
    from therapy_ai.core.backend import BackendFactory
    backend_info = BackendFactory.detect()

    table = Table(title="Hardware Backend")
    table.add_column("Property",  style="cyan")
    table.add_column("Value",     style="green")
    table.add_row("Backend",      backend_info.backend.name)
    table.add_row("Device",       backend_info.device_name)
    table.add_row("Memory (GB)",  f"{backend_info.available_memory_gb:.1f}")
    table.add_row("Fine-tuning",  "✅" if backend_info.supports_finetuning else "❌")
    console.print(table)


@app.command()
def check(
    config: str = typer.Option("config/default_config.yaml"),
) -> None:
    """Validate config and verify model is accessible on HuggingFace Hub."""
    from therapy_ai.core.model_manager import ModelConfig
    from huggingface_hub import model_info

    cfg = ModelConfig.from_yaml(config)
    console.print(f"[cyan]Model ID:[/cyan]       {cfg.model_id}")
    console.print(f"[cyan]Quantisation:[/cyan]   {cfg.quantization}")
    console.print(f"[cyan]Max seq len:[/cyan]    {cfg.max_seq_length}")

    with console.status("Checking HuggingFace Hub ..."):
        try:
            info = model_info(cfg.model_id)
            console.print(f"[green]✅ Model found:[/green] {info.id} ({info.likes} likes)")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not verify on Hub (offline mode?):[/yellow] {e}")


if __name__ == "__main__":
    app()
