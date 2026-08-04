"""
A/B tester — same prompts through the base model and the fine-tuned adapter.

Loss curves tell you IF it learned; reading these side-by-side tells you
WHAT it learned. Results print to the terminal and save to JSON.
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

# Eval prompts live in data/ (gitignored — they describe personal scenarios).
# Falls back to the small built-in set in configs/config.py if missing.
EVAL_PROMPTS_PATH = PROJECT_ROOT / "data" / "eval" / "eval_prompts.json"

# Loaded models keyed by adapter path — loading Llama 3.1 8B from disk takes
# ~10s, and a 50-prompt run would otherwise reload it on every generation.
# Base + adapter cached together is ~9GB at 4-bit, fine on 64GB.
_model_cache: dict[str, tuple] = {}


def load_eval_prompts() -> list[dict]:
    """Load the eval prompt set, falling back to the built-in test prompts."""
    if EVAL_PROMPTS_PATH.exists():
        with open(EVAL_PROMPTS_PATH, encoding="utf-8") as f:
            prompts = json.load(f)
        valid = [p for p in prompts if isinstance(p, dict) and p.get("prompt")]
        if valid:
            console.print(f"[dim]Loaded {len(valid)} eval prompts from {EVAL_PROMPTS_PATH}[/dim]")
            return valid
        console.print(f"[yellow]No valid prompts in {EVAL_PROMPTS_PATH}, using built-in set[/yellow]")
    return finetune_config.test_prompts


def _load_model(adapter_path: str = None) -> tuple:
    from mlx_lm import load

    key = adapter_path or "__base__"
    if key not in _model_cache:
        if adapter_path:
            _model_cache[key] = load(finetune_config.base_model, adapter_path=adapter_path)
        else:
            _model_cache[key] = load(finetune_config.base_model)
    return _model_cache[key]


def generate_response_mlx(
    prompt: str,
    context: str,
    adapter_path: str = None,
    max_tokens: int = 512,
    temp: float = 0.7,
    use_system_prompt: bool = True,
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
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        console.print("[red]mlx-lm not installed. Run: pip install 'mlx-lm>=0.30.0'[/red]")
        raise

    # Build the chat-formatted prompt
    user_content = f"Context: {context}\n\nTask: {prompt}" if context else prompt

    model, tokenizer = _load_model(adapter_path)

    # No system role (Mistral)? Fold it into the user turn,
    # same as the training data — the two must match
    messages = []
    if use_system_prompt:
        if finetune_config.supports_system_role:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            user_content = f"{SYSTEM_PROMPT}\n\n{user_content}"
    messages.append({"role": "user", "content": user_content})

    # Apply chat template to get the formatted prompt string
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: manual Llama 3.1 formatting
        system_part = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            if use_system_prompt else ""
        )
        formatted_prompt = (
            f"<|begin_of_text|>{system_part}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # mlx_lm 0.30+ takes a sampler function instead of a temp kwarg
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
    use_system_prompt: bool = True,
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
        output_path = (
            PROJECT_ROOT / finetune_config.data_dir
            / f"comparison_results{finetune_config.results_suffix}.json"
        )
    if prompts is None:
        prompts = load_eval_prompts()

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
                use_system_prompt=use_system_prompt,
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
                use_system_prompt=use_system_prompt,
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
            "category": test.get("category", ""),
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
