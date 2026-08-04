"""
Data splitter — 80/10/10 into the train/valid/test files MLX-LM expects.

Shuffles with a fixed seed so the split is identical every run —
differences between training runs come from hyperparams, not the data.
"""

import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from configs.config import PROJECT_ROOT, finetune_config

console = Console()


def _fold_system_into_user(example: dict) -> dict:
    """
    Merge a leading system message into the first user turn.

    Mistral's template has no system role and straight-up rejects one,
    so the training data has to bake it in.
    """
    messages = example.get("messages", [])
    if len(messages) < 2 or messages[0].get("role") != "system":
        return example

    system_content = messages[0]["content"]
    rest = [dict(m) for m in messages[1:]]
    if rest[0].get("role") == "user":
        rest[0]["content"] = f"{system_content}\n\n{rest[0]['content']}"
    return {**example, "messages": rest}


def split_data(
    input_path: Path = None,
    output_dir: Path = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """
    Split a JSONL dataset into train/valid/test files.

    Args:
        input_path:  Path to the source JSONL (default: writing_style.jsonl)
        output_dir:  Where to write the split files (default: data/training/prepared/)
        train_ratio: Fraction of data for training (default: 0.8)
        valid_ratio: Fraction of data for validation (default: 0.1)
        seed:        Random seed for reproducibility (default: 42)

    Returns:
        Dict with counts: {"train": N, "valid": N, "test": N}
    """
    if input_path is None:
        input_path = PROJECT_ROOT / finetune_config.data_dir / "writing_style.jsonl"
    if output_dir is None:
        output_dir = PROJECT_ROOT / finetune_config.prepared_dir

    if not input_path.exists():
        console.print(f"[red]Training data not found: {input_path}[/red]")
        console.print("[dim]Run the training data parser first to generate writing_style.jsonl[/dim]")
        raise FileNotFoundError(f"Missing training data: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if not finetune_config.supports_system_role:
        examples = [_fold_system_into_user(e) for e in examples]
        console.print(
            f"[dim]{finetune_config.model_family}: no system role — "
            "folded system prompts into user turns[/dim]"
        )

    total = len(examples)
    if total == 0:
        console.print("[red]Training data file is empty![/red]")
        raise ValueError("No training examples found")

    console.print(f"[bold blue]Splitting {total} examples...[/bold blue]\n")

    random.seed(seed)
    random.shuffle(examples)

    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_data = examples[:train_end]
    valid_data = examples[train_end:valid_end]
    test_data = examples[valid_end:]

    # Every set needs at least 1 example
    if len(valid_data) == 0 and len(train_data) > 2:
        valid_data = [train_data.pop()]
    if len(test_data) == 0 and len(train_data) > 2:
        test_data = [train_data.pop()]

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }

    counts = {}
    for name, data in splits.items():
        output_path = output_dir / f"{name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        counts[name] = len(data)
        console.print(f"  [green]\u2713[/green] {output_path.name}: {len(data)} examples")

    console.print()
    table = Table(title="Data Split Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="dim")

    for name, count in counts.items():
        pct = f"{count / total * 100:.0f}%"
        table.add_row(name, str(count), pct)
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "[bold]100%[/bold]")

    console.print(table)
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]")
    console.print(f"[dim]Random seed: {seed} (deterministic — same split every time)[/dim]")

    return counts


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    split_data()
