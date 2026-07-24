"""
Fine-tuning orchestrator — prepare → train → test → export.

Everything runs locally on Apple Silicon via MLX: no cloud costs, and
my writing data never leaves the machine. The base model downloads once
(~4.5GB), trains in ~10 min, and produces a small LoRA adapter.

USAGE:
  python -m src.finetune --all    # full pipeline
  (or --prepare / --train / --test / --export individually)
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from configs.config import PROJECT_ROOT, finetune_config

console = Console()


# ──────────────────────────────────────────────────────────────
# Step 1: Data Preparation
# ──────────────────────────────────────────────────────────────

def prepare_data() -> bool:
    """Split training data into train/valid/test (thin wrapper around data_splitter)."""
    console.print(Panel(
        "[bold]Step 1: Data Preparation[/bold]\n\n"
        "Splitting writing_style.jsonl into train/valid/test sets.\n"
        "MLX-LM expects these three files in its data directory.",
        border_style="blue",
    ))

    try:
        from src.data_splitter import split_data
        counts = split_data()
        console.print(f"\n[green]Data preparation complete![/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Data preparation failed: {e}[/red]")
        return False


# ──────────────────────────────────────────────────────────────
# Step 2: LoRA Training
# ──────────────────────────────────────────────────────────────

def train_model() -> bool:
    """
    Run MLX LoRA fine-tuning on the prepared data.

    This is where the magic happens. We use mlx_lm's high-level run()
    function which handles:
    - Downloading the base model from HuggingFace (first run only)
    - Freezing base weights and adding LoRA adapters
    - Training with our hyperparameters
    - Saving adapter checkpoints
    - Evaluating on validation data

    THE KEY INSIGHT:
    ────────────────
    We're NOT retraining the whole model. LoRA adds tiny trainable
    matrices (rank 8) to a few layers (8 of 32) while keeping the
    original 8 billion parameters frozen. This means:
    - Training is fast (~5-15 min on Apple Silicon)
    - The adapter is small (~50-200MB vs 4.5GB for the model)
    - The base model's knowledge is preserved
    - We only learn my writing STYLE, not new facts
    """
    console.print(Panel(
        "[bold]Step 2: LoRA Fine-Tuning[/bold]\n\n"
        f"Base model:    [cyan]{finetune_config.base_model}[/cyan]\n"
        f"LoRA rank:     [cyan]{finetune_config.lora_rank}[/cyan]\n"
        f"Num layers:    [cyan]{finetune_config.num_layers}[/cyan]\n"
        f"Dropout:       [cyan]{finetune_config.lora_dropout}[/cyan]\n"
        f"Learning rate: [cyan]{finetune_config.learning_rate}[/cyan]\n"
        f"Iterations:    [cyan]{finetune_config.iters}[/cyan]\n"
        f"Batch size:    [cyan]{finetune_config.batch_size}[/cyan]\n"
        f"Mask prompts:  [cyan]{finetune_config.mask_prompt}[/cyan]\n"
        f"Adapter path:  [cyan]{finetune_config.adapter_path}[/cyan]",
        border_style="green",
    ))

    prepared_dir = PROJECT_ROOT / finetune_config.prepared_dir
    train_file = prepared_dir / "train.jsonl"
    valid_file = prepared_dir / "valid.jsonl"

    if not train_file.exists() or not valid_file.exists():
        console.print("[yellow]Prepared data not found. Running data preparation first...[/yellow]\n")
        if not prepare_data():
            return False

    adapter_path = PROJECT_ROOT / finetune_config.adapter_path
    adapter_path.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_lm.lora import run, CONFIG_DEFAULTS
    except ImportError:
        console.print("[red]mlx-lm not installed![/red]")
        console.print("Run: [cyan]pip install 'mlx-lm>=0.30.0'[/cyan]")
        return False

    # mlx_lm's CONFIG_DEFAULTS as the base, our config on top
    args = SimpleNamespace(**{
        **CONFIG_DEFAULTS,
        "model": finetune_config.base_model,
        "data": str(prepared_dir),
        "train": True,
        "test": False,
        "fine_tune_type": "lora",
        "iters": finetune_config.iters,
        "batch_size": finetune_config.batch_size,
        "learning_rate": finetune_config.learning_rate,
        "num_layers": finetune_config.num_layers,
        "adapter_path": str(adapter_path),
        "save_every": finetune_config.steps_per_save,
        "steps_per_report": 10,
        "steps_per_eval": finetune_config.steps_per_eval,
        "mask_prompt": finetune_config.mask_prompt,
        "lora_parameters": {
            "rank": finetune_config.lora_rank,
            "dropout": finetune_config.lora_dropout,
            "scale": 20.0,  # default LoRA scaling factor
        },
        "max_seq_length": 1024,  # capped so it doesn't learn ultra-long generation patterns
        "grad_checkpoint": False,
        "seed": 42,
    })

    console.print("\n[bold]Starting training...[/bold]")
    console.print("[dim]First run will download the base model (~4.5GB). Subsequent runs are instant.[/dim]\n")

    start_time = time.time()

    try:
        run(args)
        elapsed = time.time() - start_time

        console.print(f"\n[bold green]Training complete![/bold green]")
        console.print(f"[dim]Time: {elapsed / 60:.1f} minutes[/dim]")
        console.print(f"[dim]Adapter saved to: {adapter_path}[/dim]")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        console.print(f"\n[red]Training failed after {elapsed / 60:.1f} minutes: {e}[/red]")

        error_str = str(e).lower()
        if "out of memory" in error_str or "mps" in error_str:
            console.print("\n[yellow]Memory issue detected. Try these fixes:[/yellow]")
            console.print("  1. Close other apps to free memory")
            console.print("  2. Reduce batch_size in configs/config.py (currently: {})".format(
                finetune_config.batch_size
            ))
            console.print("  3. Reduce num_layers (currently: {})".format(
                finetune_config.num_layers
            ))
        elif "not found" in error_str or "no such file" in error_str:
            console.print("\n[yellow]Model or data file not found.[/yellow]")
            console.print("  Make sure to run --prepare first")
            console.print(f"  Expected data at: {prepared_dir}")

        return False


# ──────────────────────────────────────────────────────────────
# Step 3: A/B Testing
# ──────────────────────────────────────────────────────────────

def test_model() -> bool:
    """
    Side-by-side comparison between base and fine-tuned model. Loss curves
    say IF it learned; reading the output is how you know WHAT it learned.
    """
    console.print(Panel(
        "[bold]Step 3: A/B Model Comparison[/bold]\n\n"
        "Generating responses from both the base model and fine-tuned\n"
        "model for the same prompts. Read them side-by-side to see\n"
        "if the fine-tuned one sounds more like you.",
        border_style="yellow",
    ))

    try:
        from src.model_tester import run_comparison
        results = run_comparison()
        return True
    except FileNotFoundError as e:
        console.print(f"\n[red]{e}[/red]")
        console.print("[dim]Run training first: python -m src.finetune --train[/dim]")
        return False
    except Exception as e:
        console.print(f"\n[red]Testing failed: {e}[/red]")
        return False


# ──────────────────────────────────────────────────────────────
# Step 4: Ollama Export
# ──────────────────────────────────────────────────────────────

def export_model() -> bool:
    """Export to Ollama as 'llama3.1-assistant' — usable anywhere Ollama models work."""
    console.print(Panel(
        "[bold]Step 4: Ollama Export[/bold]\n\n"
        f"Creating Ollama model: [cyan]{finetune_config.ollama_model_name}[/cyan]\n"
        "This will try direct adapter import first, then fall back\n"
        "to GGUF conversion if needed.",
        border_style="magenta",
    ))

    try:
        from src.ollama_exporter import export_to_ollama
        success = export_to_ollama()
        return success
    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        return False


# ──────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────

def run_all() -> bool:
    """
    Run the complete fine-tuning pipeline from start to finish.

    Steps:
      1. Prepare data (split into train/valid/test)
      2. Train (MLX LoRA fine-tuning)
      3. Test (A/B comparison)
      4. Export (create Ollama model)

    Each step depends on the previous one. If any step fails,
    we stop and report what happened.
    """
    console.print(Panel(
        "[bold]Full Fine-Tuning Pipeline[/bold]\n\n"
        "Running all steps: prepare → train → test → export\n\n"
        f"Model:    [cyan]{finetune_config.base_model}[/cyan]\n"
        f"Data:     [cyan]{finetune_config.data_dir}/writing_style.jsonl[/cyan]\n"
        f"Output:   [cyan]{finetune_config.ollama_model_name}[/cyan]",
        border_style="bold green",
    ))

    pipeline_start = time.time()

    steps = [
        ("Prepare Data", prepare_data),
        ("Train Model", train_model),
        ("Test Model", test_model),
        ("Export to Ollama", export_model),
    ]

    results = []

    for step_name, step_fn in steps:
        console.print(f"\n{'=' * 60}")
        step_start = time.time()
        success = step_fn()
        step_time = time.time() - step_start

        results.append({
            "step": step_name,
            "success": success,
            "time_seconds": round(step_time, 1),
        })

        if not success:
            console.print(f"\n[red]Pipeline stopped at: {step_name}[/red]")
            break

    # Summary
    total_time = time.time() - pipeline_start

    console.print(f"\n{'=' * 60}")
    table = Table(title="Pipeline Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right", style="dim")

    for r in results:
        status = "[green]\u2713 Pass[/green]" if r["success"] else "[red]\u2717 Fail[/red]"
        time_str = f"{r['time_seconds']:.1f}s"
        if r["time_seconds"] > 60:
            time_str = f"{r['time_seconds'] / 60:.1f}m"
        table.add_row(r["step"], status, time_str)

    console.print(table)
    console.print(f"\n[dim]Total time: {total_time / 60:.1f} minutes[/dim]")

    all_passed = all(r["success"] for r in results)

    if all_passed:
        console.print(Panel(
            "[bold green]Pipeline complete![/bold green]\n\n"
            f"Your fine-tuned model is ready: [cyan]{finetune_config.ollama_model_name}[/cyan]\n\n"
            "Try it out:\n"
            f"  [cyan]ollama run {finetune_config.ollama_model_name}[/cyan]\n"
            f"  [cyan]python -m src.chat --model {finetune_config.ollama_model_name.split(':')[0]}[/cyan]",
            border_style="green",
        ))
    else:
        console.print("\n[yellow]Pipeline did not complete. Check the errors above.[/yellow]")

    return all_passed


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MLX LoRA Fine-Tuning Pipeline for Writing Style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.finetune --all          # Full pipeline (recommended first run)
  python -m src.finetune --prepare      # Just split the data
  python -m src.finetune --train        # Just run training
  python -m src.finetune --test         # Just run A/B comparison
  python -m src.finetune --export       # Just create Ollama model

Step-by-step workflow:
  1. python -m src.finetune --prepare   # ~1 second
  2. python -m src.finetune --train     # ~5-15 min on Apple Silicon
  3. python -m src.finetune --test      # ~2-5 min
  4. python -m src.finetune --export    # ~5-10 min
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prepare", action="store_true",
        help="Split training data into train/valid/test sets",
    )
    group.add_argument(
        "--train", action="store_true",
        help="Run MLX LoRA fine-tuning",
    )
    group.add_argument(
        "--test", action="store_true",
        help="Run A/B comparison (base vs fine-tuned)",
    )
    group.add_argument(
        "--export", action="store_true",
        help="Export fine-tuned model to Ollama",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run full pipeline: prepare → train → test → export",
    )

    args = parser.parse_args()

    if args.prepare:
        success = prepare_data()
    elif args.train:
        success = train_model()
    elif args.test:
        success = test_model()
    elif args.export:
        success = export_model()
    elif args.all:
        success = run_all()
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
