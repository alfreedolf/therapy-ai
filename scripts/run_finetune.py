"""
scripts/run_finetune.py

CLI for all fine-tuning operations.
Usage examples:
  python scripts/run_finetune.py export
  python scripts/run_finetune.py export --min-rating 4 --val-split 0.2
  python scripts/run_finetune.py train
  python scripts/run_finetune.py train --method grpo
  python scripts/run_finetune.py preview --session-id <uuid>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Therapy AI — fine-tuning CLI")


@app.command()
def export(
    passphrase: str = typer.Option(..., prompt=True, hide_input=True,
                                    help="Your encryption passphrase"),
    min_rating:  int   = typer.Option(None,  help="Minimum session rating to include (1–5)"),
    val_split:   float = typer.Option(0.15,  help="Fraction of data for validation"),
    output_dir:  str   = typer.Option("data/finetune", help="Output directory"),
    skip_flagged:bool  = typer.Option(True, help="Skip crisis-flagged sessions"),
) -> None:
    """Export encrypted sessions → JSONL for fine-tuning."""
    from therapy_ai.memory.session_store import SessionStore
    from therapy_ai.memory.exporter import FinetuneExporter

    store    = SessionStore(passphrase=passphrase)
    exporter = FinetuneExporter(store)

    with console.status("Exporting sessions ..."):
        counts = exporter.export(
            output_dir=output_dir,
            val_split=val_split,
            min_rating=min_rating,
            skip_flagged=skip_flagged,
        )

    table = Table(title="Export Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_row("train", str(counts["train"]))
    table.add_row("val",   str(counts["val"]))
    console.print(table)


@app.command()
def train(
    method:            str  = typer.Option("sft",  help="sft | grpo | dpo"),
    model_config:      str  = typer.Option("config/default_config.yaml"),
    training_config:   str  = typer.Option("config/training_config.yaml"),
    train_data:        str  = typer.Option("data/finetune/train.jsonl"),
    val_data:          str  = typer.Option("data/finetune/val.jsonl"),
) -> None:
    """Run LoRA / GRPO fine-tuning locally."""
    from therapy_ai.training.trainer import LocalTrainer, TrainingConfig
    import yaml

    # Patch method from CLI into config
    with open(training_config) as f:
        cfg_data = yaml.safe_load(f)
    cfg_data["training"]["method"] = method
    patched_path = Path(training_config).with_name("_patched_training_config.yaml")
    with open(patched_path, "w") as f:
        yaml.dump(cfg_data, f)

    trainer = LocalTrainer.from_config(
        model_config_path=model_config,
        train_config_path=patched_path,
    )

    console.print(f"[bold green]Starting {method.upper()} training ...[/bold green]")
    adapter_path = trainer.train(train_data=train_data, val_data=val_data)
    console.print(f"[bold green]✅ Adapter saved → {adapter_path}[/bold green]")
    patched_path.unlink(missing_ok=True)


@app.command()
def preview(
    passphrase:  str = typer.Option(..., prompt=True, hide_input=True),
    session_id:  str = typer.Argument(..., help="Session UUID to preview"),
) -> None:
    """Preview how a session will look as training data."""
    from therapy_ai.memory.session_store import SessionStore
    from therapy_ai.memory.exporter import FinetuneExporter

    store    = SessionStore(passphrase=passphrase)
    exporter = FinetuneExporter(store)
    console.print(exporter.preview(session_id))


if __name__ == "__main__":
    app()
