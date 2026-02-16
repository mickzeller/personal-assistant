"""
Model Tester — A/B comparison between base model and fine-tuned adapter.

THE GOAL:
─────────
Numbers (loss curves) tell you IF the model learned.
Side-by-side comparisons tell you WHAT it learned.

This script generates responses from both the base Llama 3.1 8B
(no adapter) and the fine-tuned version (with LoRA adapter), using
the same prompts across different writing contexts. You read both
and decide: "Does the fine-tuned one sound more like me?"

The comparison covers all the contexts we trained on:
  - Casual friend texts
  - Work Slack messages
  - Church/ministry
  - Online reviews
  - Technical explanations
  - Personal messages

Results are displayed side-by-side in the terminal AND saved to
a JSON file so you can review them later.
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table

from configs.config import PROJECT_ROOT, finetune_config
from src.training_data_parser import SYSTEM_PROMPT

console = Console()


def generate_response_mlx(
    prompt: str,
    context: str,
    adapter_path: str = None,
    max_tokens: int = 512,
    temp: float = 0.7,
) -> str:
    """
    Generate a response using MLX-LM directly (no Ollama).

    WHY MLX DIRECTLY:
    ─────────────────
    The LoRA adapter lives as MLX weights on disk. We can load the
    base model + adapter together using mlx_lm.load(), which is the
    fastest way to test without exporting to Ollama first.

    Args:
        prompt:       The writing task
        context:      Writing context (e.g., "casual text to a friend")
        adapter_path: Path to LoRA adapter (None = base model only)
        max_tokens:   Maximum response length
        temp:         Sampling temperature (0.7 = creative but coherent)
    """
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        console.print("[red]mlx-lm not installed. Run: pip install 'mlx-lm>=0.30.0'[/red]")
        raise

    # Build the chat-formatted prompt
    user_content = f"Context: {context}\n\nTask: {prompt}" if context else prompt

    # Load model (with or without adapter)
    model_path = finetune_config.base_model
    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    # Build chat messages using the tokenizer's chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template to get the formatted prompt string
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: manual formatting
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # mlx_lm 0.30+ uses a sampler function instead of a temp kwarg.
    # make_sampler(temp=0.0) = greedy, temp=0.5 = balanced, temp=1.0 = creative
    sampler = make_sampler(temp=temp)

    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    return response.strip()


def run_comparison(
    adapter_path: str = None,
    output_path: Path = None,
    prompts: list[dict] = None,
) -> list[dict]:
    """
    Run A/B comparison between base model and fine-tuned model.

    Generates responses for each test prompt from both models,
    displays them side-by-side, and saves results to JSON.

    Args:
        adapter_path: Path to the LoRA adapter (default: from config)
        output_path:  Where to save comparison JSON (default: data/training/)
        prompts:      Test prompts to use (default: from config)

    Returns:
        List of comparison result dicts
    """
    if adapter_path is None:
        adapter_path = str(PROJECT_ROOT / finetune_config.adapter_path)
    if output_path is None:
        output_path = PROJECT_ROOT / finetune_config.data_dir / "comparison_results.json"
    if prompts is None:
        prompts = finetune_config.test_prompts

    # Verify adapter exists
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        console.print(f"[red]Adapter not found: {adapter_path}[/red]")
        console.print("[dim]Run training first: python -m src.finetune --train[/dim]")
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")

    console.print(Panel(
        f"[bold]A/B Model Comparison[/bold]\n\n"
        f"Base model: [cyan]{finetune_config.base_model}[/cyan]\n"
        f"Adapter:    [cyan]{adapter_path}[/cyan]\n"
        f"Prompts:    [cyan]{len(prompts)} test scenarios[/cyan]",
        border_style="blue",
    ))

    results = []

    for i, test in enumerate(prompts):
        prompt = test["prompt"]
        context = test.get("context", "")

        console.print(f"\n[bold yellow]── Test {i + 1}/{len(prompts)}: {context} ──[/bold yellow]")
        console.print(f"[dim]Prompt: {prompt}[/dim]\n")

        # Generate base model response
        console.print("[dim]Generating base model response...[/dim]")
        start = time.time()
        try:
            base_response = generate_response_mlx(
                prompt=prompt,
                context=context,
                adapter_path=None,
            )
            base_time = time.time() - start
        except Exception as e:
            base_response = f"[Error: {e}]"
            base_time = 0

        # Generate fine-tuned response
        console.print("[dim]Generating fine-tuned response...[/dim]")
        start = time.time()
        try:
            finetuned_response = generate_response_mlx(
                prompt=prompt,
                context=context,
                adapter_path=adapter_path,
            )
            ft_time = time.time() - start
        except Exception as e:
            finetuned_response = f"[Error: {e}]"
            ft_time = 0

        # Display side-by-side
        base_panel = Panel(
            base_response[:500] + ("..." if len(base_response) > 500 else ""),
            title="[red]Base Model[/red]",
            subtitle=f"[dim]{base_time:.1f}s[/dim]",
            border_style="red",
            width=50,
        )
        ft_panel = Panel(
            finetuned_response[:500] + ("..." if len(finetuned_response) > 500 else ""),
            title="[green]Fine-Tuned[/green]",
            subtitle=f"[dim]{ft_time:.1f}s[/dim]",
            border_style="green",
            width=50,
        )
        console.print(Columns([base_panel, ft_panel]))

        results.append({
            "test_number": i + 1,
            "context": context,
            "prompt": prompt,
            "base_response": base_response,
            "finetuned_response": finetuned_response,
            "base_time_seconds": round(base_time, 2),
            "finetuned_time_seconds": round(ft_time, 2),
        })

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]Comparison complete![/bold green]")
    console.print(f"[dim]Results saved to: {output_path}[/dim]")

    # Quick summary table
    table = Table(title="Response Length Comparison")
    table.add_column("Test", style="cyan")
    table.add_column("Base (chars)", justify="right")
    table.add_column("Fine-Tuned (chars)", justify="right")

    for r in results:
        table.add_row(
            r["context"][:30],
            str(len(r["base_response"])),
            str(len(r["finetuned_response"])),
        )
    console.print(table)

    return results


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B Model Comparison")
    parser.add_argument(
        "--adapter", default=None,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for comparison JSON"
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    run_comparison(adapter_path=args.adapter, output_path=output)
