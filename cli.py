#!/usr/bin/env python3
"""
LLM Discovery CLI — one-stop shop for the whole pipeline.

Run `python cli.py --help` for commands and examples.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_parse(args):
    """Parse raw writing samples into JSONL training data."""
    from src.training_data_parser import parse_training_directory, inspect_examples

    training_dir = Path(args.input) if args.input else PROJECT_ROOT / "knowledge_base" / "training_data"
    output_path = Path(args.output) if args.output else None

    examples = parse_training_directory(training_dir, output_path=output_path)

    if examples and args.preview:
        inspect_examples(examples, n=args.preview)

    return bool(examples)


def cmd_prepare(args):
    """Split training data into train/valid/test sets."""
    from src.finetune import prepare_data
    return prepare_data()


def cmd_train(args):
    """Run MLX LoRA fine-tuning."""
    from src.finetune import train_model
    return train_model()


def cmd_test(args):
    """A/B comparison between base and fine-tuned model."""
    from src.finetune import test_model
    return test_model()


def cmd_export(args):
    """Export fine-tuned model to Ollama."""
    from src.ollama_exporter import export_to_ollama, export_path_b

    adapter = Path(args.adapter) if args.adapter else None
    name = args.name or None

    if args.path_b:
        export_path_b(adapter_path=adapter, model_name=name)
    else:
        export_to_ollama(adapter_path=adapter, model_name=name)

    return True


def cmd_pipeline(args):
    """Run the full pipeline: parse → prepare → train → test → export."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel(
        "[bold]Full Pipeline[/bold]\n\n"
        "Running: parse → prepare → train → test → export\n\n"
        "Sit back — this takes ~15-25 minutes on M4 Max.",
        border_style="bold green",
    ))

    console.print("\n[bold cyan]Step 1/5: Parsing training data...[/bold cyan]")
    args.input = None
    args.output = None
    args.preview = 0
    if not cmd_parse(args):
        console.print("[red]Parse failed — no training examples found.[/red]")
        return False

    console.print("\n[bold cyan]Steps 2-5: Prepare → Train → Test → Export[/bold cyan]")
    from src.finetune import run_all
    return run_all()


def cmd_chat(args):
    """Start an interactive chat session."""
    from src.chat import ChatAssistant

    assistant = ChatAssistant(
        model=args.model,
        rag_enabled=not args.no_rag,
    )
    assistant.run()
    return True


def cmd_evaluate(args):
    """Score fine-tuned responses using Claude as a judge."""
    from src.evaluator import run_evaluation
    return run_evaluation(
        results_path=args.results,
        run_test_first=args.fresh,
    )


def cmd_ingest(args):
    """Ingest documents into the RAG knowledge base."""
    from src.ingest import ingest_directory
    from configs.config import KNOWLEDGE_BASE_DIR
    from rich.console import Console

    console = Console()
    target_dir = Path(args.dir) if args.dir else KNOWLEDGE_BASE_DIR

    console.print(f"[bold]Ingesting documents from: {target_dir}[/bold]")
    results = ingest_directory(target_dir)
    return bool(results)


def main():
    parser = argparse.ArgumentParser(
        prog="cli",
        description="LLM Discovery — Personal AI Assistant Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python cli.py parse                 Rebuild training data from raw writing
  python cli.py prepare               Split data into train/valid/test
  python cli.py train                 Fine-tune with MLX LoRA (~10 min)
  python cli.py test                  A/B compare base vs fine-tuned
  python cli.py evaluate              Score responses with Claude
  python cli.py evaluate --fresh      Test + evaluate in one shot
  python cli.py export                Create Ollama model
  python cli.py pipeline              Run everything start to finish
  python cli.py chat                  Chat with your assistant
  python cli.py chat --model gemma3   Chat with a specific model
  python cli.py ingest                Ingest docs into RAG knowledge base
""",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
    )

    # ── parse ──────────────────────────────────────────────────
    p_parse = subparsers.add_parser(
        "parse",
        help="Parse raw writing samples → JSONL training data",
        description="Convert your raw writing (markdown, docx) into JSONL format for fine-tuning.",
    )
    p_parse.add_argument(
        "--input", "-i",
        help="Input directory (default: knowledge_base/training_data/)",
    )
    p_parse.add_argument(
        "--output", "-o",
        help="Output JSONL path (default: data/training/writing_style.jsonl)",
    )
    p_parse.add_argument(
        "--preview", "-p", type=int, default=5, nargs="?", const=5,
        help="Preview N examples after parsing (default: 5, use 0 to skip)",
    )
    p_parse.set_defaults(func=cmd_parse)

    # ── prepare ────────────────────────────────────────────────
    p_prepare = subparsers.add_parser(
        "prepare",
        help="Split JSONL into train/valid/test sets (80/10/10)",
        description="Split your training JSONL into the three files MLX expects.",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # ── train ──────────────────────────────────────────────────
    p_train = subparsers.add_parser(
        "train",
        help="Run MLX LoRA fine-tuning (~10 min on M4 Max)",
        description="Fine-tune the base model using your training data with MLX LoRA.",
    )
    p_train.set_defaults(func=cmd_train)

    # ── test ───────────────────────────────────────────────────
    p_test = subparsers.add_parser(
        "test",
        help="A/B compare base model vs fine-tuned",
        description="Generate side-by-side responses from base and fine-tuned models.",
    )
    p_test.set_defaults(func=cmd_test)

    # ── evaluate ───────────────────────────────────────────────
    p_evaluate = subparsers.add_parser(
        "evaluate",
        help="Score fine-tuned responses with Claude (LLM-as-a-Judge)",
        description="Send model responses to Claude's API for structured scoring against Mick's writing style.",
    )
    p_evaluate.add_argument(
        "--results", "-r",
        help="Path to comparison_results.json (default: from config)",
    )
    p_evaluate.add_argument(
        "--fresh", "-f", action="store_true",
        help="Run test first to generate fresh comparison results",
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    # ── export ─────────────────────────────────────────────────
    p_export = subparsers.add_parser(
        "export",
        help="Export fine-tuned model to Ollama",
        description="Create an Ollama model from your fine-tuned adapter.",
    )
    p_export.add_argument(
        "--adapter",
        help="Path to LoRA adapter directory (default: from config)",
    )
    p_export.add_argument(
        "--name",
        help="Ollama model name (default: from config)",
    )
    p_export.add_argument(
        "--path-b", action="store_true",
        help="Skip adapter import, go straight to fused model export",
    )
    p_export.set_defaults(func=cmd_export)

    # ── pipeline ───────────────────────────────────────────────
    p_pipeline = subparsers.add_parser(
        "pipeline",
        help="Full pipeline: parse → prepare → train → test → export",
        description="Run the entire pipeline from raw data to Ollama model.",
    )
    p_pipeline.set_defaults(func=cmd_pipeline)

    # ── chat ───────────────────────────────────────────────────
    p_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session",
        description="Chat with your AI assistant using Ollama.",
    )
    p_chat.add_argument(
        "--model", "-m",
        help="Ollama model to use (default: from config)",
    )
    p_chat.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG retrieval",
    )
    p_chat.set_defaults(func=cmd_chat)

    # ── ingest ─────────────────────────────────────────────────
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the RAG knowledge base",
        description="Load, chunk, embed, and store documents in ChromaDB.",
    )
    p_ingest.add_argument(
        "--dir", "-d",
        help="Directory to ingest (default: knowledge_base/)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ── Parse and dispatch ─────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    success = args.func(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
