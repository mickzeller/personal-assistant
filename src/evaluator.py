"""
LLM-as-a-Judge — Claude scores fine-tuned responses on 5 dimensions (1-5).

Eyeballing side-by-sides is fine for vibes, but numbers let us actually
compare runs: v2 vs v3, different hyperparams, Llama vs Mistral, etc.

USAGE:
  python cli.py evaluate           # score existing comparison results
  python cli.py evaluate --fresh   # run test first, then score
"""

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from configs.config import PROJECT_ROOT, finetune_config
from configs.eval_config import (
    eval_config,
    load_writing_style_profile,
    RUBRIC_DIMENSIONS,
)
from src.style_metrics import aggregate_style_metrics, compute_style_metrics

console = Console()


# ──────────────────────────────────────────────────────────────
# System prompt for Claude (the judge)
# ──────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build the system prompt that tells Claude how to judge responses."""

    # Load the writing style profile from ~/.claude/CLAUDE.md
    writing_style = load_writing_style_profile()

    # Format rubric dimensions into a readable list
    rubric_text = "\n".join(
        f"- **{dim['name']}** (1-5): {dim['description']}"
        for dim in RUBRIC_DIMENSIONS.values()
    )

    return f"""You are an expert writing style evaluator. Your job is to score
how well a model-generated response matches a specific person's writing style.

{writing_style}

## Scoring Rubric

Score each dimension from 1 (poor match) to 5 (excellent match):

{rubric_text}

## Response Format

You MUST respond with valid JSON only — no markdown fences, no explanation outside the JSON.
Use this exact structure:

{{
  "scores": {{
    "tone": <1-5>,
    "word_choice": <1-5>,
    "structure": <1-5>,
    "authenticity": <1-5>,
    "context_fit": <1-5>
  }},
  "reasoning": {{
    "tone": "<1-2 sentences explaining the score>",
    "word_choice": "<1-2 sentences>",
    "structure": "<1-2 sentences>",
    "authenticity": "<1-2 sentences>",
    "context_fit": "<1-2 sentences>"
  }}
}}"""


def _build_user_message(context: str, prompt: str, response: str) -> str:
    """Build the user message containing the response to evaluate."""
    return f"""Please evaluate this model-generated response against the writing style profile.

**Writing Context:** {context}
**Prompt Given:** {prompt}

**Model Response:**
{response}

Score this response on all 5 dimensions. Respond with JSON only."""


# ──────────────────────────────────────────────────────────────
# Core evaluation functions
# ──────────────────────────────────────────────────────────────

def evaluate_response(
    context: str,
    prompt: str,
    response: str,
) -> dict:
    """Send one response to Claude for scoring. Returns scores, reasoning, and the average."""
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv()
    client = Anthropic()

    system_prompt = _build_system_prompt()
    user_message = _build_user_message(context, prompt, response)

    message = client.messages.create(
        model=eval_config.claude_model,
        max_tokens=eval_config.max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    # Parse Claude's JSON response
    raw_text = message.content[0].text.strip()

    # Strip markdown fences if Claude added them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

    result = json.loads(raw_text)

    # Calculate overall average
    scores = result["scores"]
    result["overall_avg"] = round(
        sum(scores.values()) / len(scores), 1
    )

    return result


def evaluate_comparison(results: list[dict], response_key: str = "finetuned_response") -> dict:
    """
    Evaluate one side of a comparison run (fine-tuned by default, or base).

    Args:
        results:      List of comparison result dicts (from model_tester.py)
        response_key: Which response column to score — "finetuned_response" or "base_response"

    Returns:
        Full evaluation report with per-response scores and summary
    """
    evaluations = []
    side = "Fine-Tuned" if response_key == "finetuned_response" else "Base Model"

    console.print(Panel(
        f"[bold]Writing Style Evaluation — Claude as Judge[/bold]\n\n"
        f"Judge model:  [cyan]{eval_config.claude_model}[/cyan]\n"
        f"Scoring:      [cyan]{side} responses[/cyan]\n"
        f"Responses:    [cyan]{len(results)} test scenarios[/cyan]\n"
        f"Dimensions:   [cyan]{', '.join(d['name'] for d in RUBRIC_DIMENSIONS.values())}[/cyan]",
        border_style="blue",
    ))

    for i, result in enumerate(results):
        context = result["context"]
        prompt = result["prompt"]
        response = result[response_key]

        if response.startswith("[Error:"):
            console.print(f"  [yellow]Skipping test {i+1} (error response)[/yellow]")
            continue

        console.print(f"  [dim]Evaluating {i+1}/{len(results)}: {context}...[/dim]")

        try:
            eval_result = evaluate_response(context, prompt, response)

            evaluations.append({
                "test_number": result["test_number"],
                "context": context,
                "prompt": prompt,
                "response": response[:500],  # truncated for storage
                "scores": eval_result["scores"],
                "reasoning": eval_result["reasoning"],
                "overall_avg": eval_result["overall_avg"],
                # Computed on the full response, before storage truncation
                "style_metrics": compute_style_metrics(response),
            })

            avg = eval_result["overall_avg"]
            color = "green" if avg >= 4.0 else "yellow" if avg >= 3.0 else "red"
            console.print(f"    [{color}]→ avg: {avg}/5.0[/{color}]")

        except json.JSONDecodeError as e:
            console.print(f"  [red]Failed to parse Claude response for test {i+1}: {e}[/red]")
        except Exception as e:
            console.print(f"  [red]Error evaluating test {i+1}: {e}[/red]")

        # Be nice to the API... small delay between calls
        if i < len(results) - 1:
            time.sleep(0.5)

    if not evaluations:
        console.print("[red]No responses were successfully evaluated.[/red]")
        return {}

    # Compute summary statistics
    dimension_keys = list(RUBRIC_DIMENSIONS.keys())
    dimension_avgs = {}
    for key in dimension_keys:
        scores = [e["scores"][key] for e in evaluations if key in e["scores"]]
        dimension_avgs[key] = round(sum(scores) / len(scores), 1) if scores else 0

    overall_avg = round(sum(dimension_avgs.values()) / len(dimension_avgs), 1)
    strongest = max(dimension_avgs, key=dimension_avgs.get)
    weakest = min(dimension_avgs, key=dimension_avgs.get)

    summary = {
        "overall_avg": overall_avg,
        "dimension_avgs": dimension_avgs,
        "strongest": strongest,
        "weakest": weakest,
        "responses_evaluated": len(evaluations),
        "style_metrics": aggregate_style_metrics(
            [e["style_metrics"] for e in evaluations]
        ),
    }

    return {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "judge_model": eval_config.claude_model,
            "response_key": response_key,
            "evaluated_model": (
                finetune_config.ollama_model_name
                if response_key == "finetuned_response"
                else finetune_config.base_model
            ),
            "adapter_path": finetune_config.adapter_path if response_key == "finetuned_response" else None,
        },
        "results": evaluations,
        "summary": summary,
    }


# ──────────────────────────────────────────────────────────────
# Display + save
# ──────────────────────────────────────────────────────────────

def _display_results(report: dict) -> None:
    """Display evaluation results as a Rich table."""
    if not report or not report.get("results"):
        return

    summary = report["summary"]
    results = report["results"]

    # Score table
    table = Table(
        title="Writing Style Scores",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Test", style="white", max_width=25)
    table.add_column("Tone", justify="center")
    table.add_column("Words", justify="center")
    table.add_column("Struct", justify="center")
    table.add_column("Auth", justify="center")
    table.add_column("Context", justify="center")
    table.add_column("Avg", justify="center", style="bold")

    for r in results:
        s = r["scores"]
        avg = r["overall_avg"]
        table.add_row(
            r["context"][:25],
            _score_cell(s.get("tone", 0)),
            _score_cell(s.get("word_choice", 0)),
            _score_cell(s.get("structure", 0)),
            _score_cell(s.get("authenticity", 0)),
            _score_cell(s.get("context_fit", 0)),
            _score_cell(avg, bold=True),
        )

    # Overall row
    d = summary["dimension_avgs"]

    table.add_section()
    table.add_row(
        "[bold]OVERALL[/bold]",
        _score_cell(d.get("tone", 0), bold=True),
        _score_cell(d.get("word_choice", 0), bold=True),
        _score_cell(d.get("structure", 0), bold=True),
        _score_cell(d.get("authenticity", 0), bold=True),
        _score_cell(d.get("context_fit", 0), bold=True),
        _score_cell(summary["overall_avg"], bold=True),
    )

    console.print()
    console.print(table)

    # Summary callout
    console.print(f"\n  Strongest dimension: [green]{RUBRIC_DIMENSIONS[summary['strongest']]['name']}[/green]")
    console.print(f"  Weakest dimension:  [yellow]{RUBRIC_DIMENSIONS[summary['weakest']]['name']}[/yellow]")
    console.print(f"  Responses evaluated: {summary['responses_evaluated']}")

    _display_style_metrics(report)


def _display_style_metrics(report: dict) -> None:
    """Display automated style metrics as a second Rich table."""
    results = report.get("results", [])
    if not any("style_metrics" in r for r in results):
        return

    table = Table(
        title="Automated Style Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Test", style="white", max_width=25)
    table.add_column("Words", justify="center")
    table.add_column("Excl/100w", justify="center")
    table.add_column("Contr/100w", justify="center")
    table.add_column("Contr Ratio", justify="center")
    table.add_column("Emoji", justify="center")
    table.add_column("Em-dash", justify="center")
    table.add_column("Ends OK", justify="center")

    for r in results:
        m = r.get("style_metrics")
        if not m:
            continue
        ratio = m["contraction_ratio"]
        table.add_row(
            r["context"][:25],
            str(m["word_count"]),
            str(m["exclamations_per_100_words"]),
            str(m["contractions_per_100_words"]),
            "—" if ratio is None else str(ratio),
            str(m["emoji_count"]),
            str(m["em_dash_count"]) if m["em_dash_count"] == 0
            else f"[yellow]{m['em_dash_count']}[/yellow]",
            "[green]✓[/green]" if m["ends_cleanly"] else "[red]✗ cut off[/red]",
        )

    agg = report["summary"].get("style_metrics", {})
    if agg:
        avg_ratio = agg.get("avg_contraction_ratio")
        truncated = agg.get("truncated_responses", 0)
        table.add_section()
        table.add_row(
            "[bold]AVERAGE[/bold]",
            f"[bold]{agg.get('avg_word_count', '—')}[/bold]",
            f"[bold]{agg.get('avg_exclamations_per_100_words', '—')}[/bold]",
            f"[bold]{agg.get('avg_contractions_per_100_words', '—')}[/bold]",
            f"[bold]{'—' if avg_ratio is None else avg_ratio}[/bold]",
            f"[bold]{agg.get('avg_emoji_count', '—')}[/bold]",
            f"[bold]{agg.get('total_em_dashes', 0)}[/bold]",
            f"[bold red]{truncated} cut off[/bold red]" if truncated
            else "[bold green]all ✓[/bold green]",
        )

    console.print()
    console.print(table)


def _score_cell(score, bold: bool = False) -> str:
    """Color a score cell based on value."""
    color = "green" if score >= 4 else "yellow" if score >= 3 else "red"
    if bold:
        return f"[bold][{color}]{score}[/{color}][/bold]"
    return f"[{color}]{score}[/{color}]"


def _save_results(report: dict, output_path: Path) -> None:
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved to: {output_path}[/dim]")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def run_evaluation(
    results_path: str = None,
    run_test_first: bool = False,
    score_base: bool = False,
    output_path: str = None,
) -> bool:
    """
    Main entry point for the evaluation pipeline.

    Args:
        results_path: Path to comparison_results.json (default: from config)
        run_test_first: If True, run model_tester first to generate fresh results
        score_base: If True, score the base model responses instead of fine-tuned
                    (saved to a separate file so the fine-tuned baseline is kept)
        output_path: Where to save the eval report (default: from config,
                     with a _base suffix when score_base is set)

    Returns:
        True if evaluation completed successfully
    """
    # Family suffix keeps runs from clobbering each other
    suffix = finetune_config.results_suffix
    data_dir = PROJECT_ROOT / eval_config.eval_results_dir
    comparison_stem = Path(eval_config.comparison_results_filename)
    comparison_path = Path(results_path) if results_path else (
        data_dir / f"{comparison_stem.stem}{suffix}{comparison_stem.suffix}"
    )
    stem = Path(eval_config.eval_results_filename)
    if output_path:
        output_path = Path(output_path)
    elif score_base:
        output_path = data_dir / f"{stem.stem}{suffix}_base{stem.suffix}"
    else:
        output_path = data_dir / f"{stem.stem}{suffix}{stem.suffix}"

    # Optionally generate fresh comparison results
    if run_test_first:
        console.print("[bold]Step 1: Generating fresh comparison results...[/bold]\n")
        from src.model_tester import run_comparison
        run_comparison()
        console.print()

    # Load comparison results
    if not comparison_path.exists():
        console.print(f"[red]Comparison results not found: {comparison_path}[/red]")
        console.print("[dim]Run 'python cli.py test' first, or use --fresh to auto-generate.[/dim]")
        return False

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_results = json.load(f)

    console.print(f"[dim]Loaded {len(comparison_results)} comparison results from {comparison_path}[/dim]\n")

    # Run evaluation
    response_key = "base_response" if score_base else "finetuned_response"
    report = evaluate_comparison(comparison_results, response_key=response_key)

    if not report:
        return False

    # Save before displaying so the data's safe even if the table blows up
    _save_results(report, output_path)
    _display_results(report)

    console.print(f"\n[bold green]Evaluation complete![/bold green]")

    return True
