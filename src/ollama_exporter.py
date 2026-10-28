"""
Ollama Exporter — turns a LoRA adapter into a chat-ready Ollama model.

The route: fuse the adapter into the base model, de-quantize to fp16,
then `ollama create` (via GGUF for families flagged needs_gguf_export).
Intermediates (~29GB) are cleaned up after a successful import.

Ollama removed direct LoRA adapter imports, so fusing is the only path
now. The MLX→PEFT converter (export_path_a) is kept for using adapters
with other runtimes like HF transformers or vLLM.

Requires the Ollama base model: `ollama pull llama3.1:8b-instruct-q8_0`.
"""

import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from configs.config import PROJECT_ROOT, finetune_config
from src.training_data_parser import SYSTEM_PROMPT

console = Console()

# llama.cpp checkout for GGUF conversion — just the scripts, no build needed
LLAMACPP_DIR = Path(os.environ.get("LLAMACPP_DIR", str(Path.home() / ".cache" / "llama.cpp")))


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _check_ollama_running() -> bool:
    """Verify Ollama is running and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_base_model_exists(model_name: str) -> bool:
    """Check if the base model is pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        return model_name in result.stdout
    except Exception:
        return False


def _ensure_base_model(base_model: str) -> bool:
    """Pull the Ollama base model if it's not already available."""
    if _check_base_model_exists(base_model):
        return True

    console.print(f"[yellow]Base model '{base_model}' not found in Ollama.[/yellow]")
    console.print(f"[dim]Pulling it now... (this may take a few minutes)[/dim]")
    pull_result = subprocess.run(
        ["ollama", "pull", base_model],
        capture_output=False, text=True,
    )
    if pull_result.returncode != 0:
        console.print(f"[red]Failed to pull {base_model}[/red]")
        return False
    return True


def _get_base_template(base_model: str) -> str | None:
    """Copy the chat template from the pulled base model — beats hardcoding it per family."""
    try:
        result = subprocess.run(
            ["ollama", "show", base_model, "--template"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.rstrip("\n")
    except Exception:
        pass
    return None


def _get_base_stop_params(base_model: str) -> list[str]:
    """Read the base model's stop parameters (lines like: stop  "[INST]")."""
    stops = []
    try:
        result = subprocess.run(
            ["ollama", "show", base_model, "--parameters"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0] == "stop":
                    stops.append(parts[1].strip().strip('"'))
    except Exception:
        pass
    return stops


def _write_modelfile(
    modelfile_path: Path,
    from_source: str,
    system_prompt: str,
    adapter_path: Path = None,
) -> Path:
    """
    Write an Ollama Modelfile with the base model's full chat template.

    Args:
        modelfile_path: Where to write the Modelfile
        from_source:    Either a model name ("llama3.1:8b...") or path to weights
        system_prompt:  System prompt to bake into the model
        adapter_path:   Optional ADAPTER directive (for PEFT adapter import)

    IMPORTANT: Ollama doesn't auto-detect chat templates on import. Without
    the full TEMPLATE directive the model gets bare {{ .Prompt }}, ignores
    stop tokens, and loops forever. We copy the template + stop params from
    the pulled base model via `ollama show`, with hardcoded fallbacks.
    """
    # Copied from: ollama show llama3.1:8b-instruct-q8_0 --modelfile
    LLAMA31_TEMPLATE = """{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

Cutting Knowledge Date: December 2023

When you receive a tool call response, use the output to format an answer to the orginal user question.

You are a helpful assistant with tool calling capabilities.
{{- end }}<|eot_id|>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{- if and $.Tools $last }}

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ range $.Tools }}
{{- . }}
{{ end }}
Question: {{ .Content }}<|eot_id|>
{{- else }}

{{ .Content }}<|eot_id|>
{{- end }}{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}
{{ range .ToolCalls }}
{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}
{{- end }}{{ if not $last }}<|eot_id|>{{ end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}"""

    # Legacy .System/.Prompt form — Ollama maps chat messages through it fine
    MISTRAL_TEMPLATE = """[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }} [/INST]"""

    FALLBACKS = {
        "llama": (LLAMA31_TEMPLATE, ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]),
        "mistral": (MISTRAL_TEMPLATE, ["[INST]", "[/INST]"]),
    }

    base_model = finetune_config.ollama_base_model
    template = _get_base_template(base_model)
    stops = _get_base_stop_params(base_model)

    if template is None:
        family = finetune_config.model_family
        console.print(
            f"  [yellow]Couldn't read template from '{base_model}' — "
            f"falling back to the built-in {family} template[/yellow]"
        )
        template, stops = FALLBACKS[family]

    lines = [
        f"# Fine-tuned {base_model} with a personal writing style",
        "# Generated by the LLM Discovery fine-tuning pipeline",
        "# Includes the base model's chat template (critical for app compatibility)",
        "",
        f"FROM {from_source}",
    ]

    if adapter_path:
        lines.append(f"ADAPTER {adapter_path}")

    lines.extend([
        "",
        f'TEMPLATE """{template}"""',
        "",
        f'SYSTEM """{system_prompt}"""',
        "",
    ])
    lines.extend(f'PARAMETER stop "{stop}"' for stop in stops)
    lines.extend([
        f"PARAMETER temperature {finetune_config.temperature}",
        "PARAMETER top_p 0.9",
        "",
    ])

    modelfile_path.write_text("\n".join(lines))
    console.print(f"  [green]\u2713[/green] Modelfile written: {modelfile_path}")
    return modelfile_path


def _ollama_create(model_name: str, modelfile_path: Path, quantize: str = None) -> bool:
    """
    Run `ollama create` and handle the result.

    Args:
        model_name:     Name for the Ollama model (e.g. "llama3.1-assistant:latest")
        modelfile_path: Path to the Modelfile
        quantize:       Optional quantization level (e.g. "q8_0"). Required for
                        fused fp16 models to avoid 16GB bloat.
    """
    # Newer Ollama renamed quantization types (q8_0 → int8, q4_* → int4)
    legacy_quant_names = {"q8_0": "int8", "q4_0": "int4", "q4_K_M": "int4"}
    if quantize in legacy_quant_names:
        quantize = legacy_quant_names[quantize]

    quant_msg = f" with {quantize} quantization" if quantize else ""
    console.print(f"\n[bold]Creating Ollama model '{model_name}'{quant_msg}...[/bold]")
    console.print("[dim]This may take a minute...[/dim]\n")

    try:
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        if quantize:
            cmd.extend(["--quantize", quantize])

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode == 0:
            console.print(f"[bold green]Success! Model '{model_name}' created.[/bold green]")
            console.print(f"\n[dim]Test it with:[/dim]")
            console.print(f"  [cyan]ollama run {model_name}[/cyan]")
            console.print(f"  [cyan]python -m src.chat --model {model_name.split(':')[0]}[/cyan]")
            return True
        else:
            console.print(f"[red]ollama create failed: {result.stderr.strip()}[/red]")
            return False

    except subprocess.TimeoutExpired:
        console.print("[red]ollama create timed out.[/red]")
        return False


# ──────────────────────────────────────────────────────────────
# Path A: MLX → PEFT Conversion → Ollama
# ──────────────────────────────────────────────────────────────

def _convert_mlx_to_peft(
    mlx_adapter_path: Path,
    peft_output_path: Path,
) -> bool:
    """
    Convert MLX LoRA adapter to Hugging Face PEFT format.

    WHY THIS IS NEEDED:
    ───────────────────
    MLX and PEFT use different conventions for everything:
      - Tensor keys:  MLX uses "lora_a", PEFT uses "lora_A.weight"
      - Tensor shapes: MLX stores [rank, dim], PEFT stores [dim, rank]
      - File names:   MLX saves "adapters.safetensors", Ollama wants "model*.safetensors"
      - Config format: Completely different JSON schemas

    This function bridges all four differences.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        console.print("[red]safetensors[torch] not available.[/red]")
        console.print("[dim]Run: pip install 'safetensors[torch]'[/dim]")
        return False

    mlx_weights_file = mlx_adapter_path / "adapters.safetensors"
    mlx_config_file = mlx_adapter_path / "adapter_config.json"

    if not mlx_weights_file.exists():
        console.print(f"[red]Adapter weights not found: {mlx_weights_file}[/red]")
        return False

    console.print("  [dim]Loading MLX adapter weights...[/dim]")
    loaded_state_dict = load_file(str(mlx_weights_file))

    # ── Convert tensor keys and shapes ──
    def rename_key(old_key: str) -> str:
        """MLX key → PEFT key (add prefix, fix lora_a/b casing)."""
        new_key = f"base_model.model.{old_key}"
        new_key = new_key.replace("lora_a", "lora_A.weight")
        new_key = new_key.replace("lora_b", "lora_B.weight")
        return new_key

    new_state_dict = {}
    for key, tensor in loaded_state_dict.items():
        new_key = rename_key(key)
        # Transpose: MLX [rank, dim] → PEFT [dim, rank]
        new_state_dict[new_key] = tensor.transpose(0, 1).contiguous()

    console.print(f"  [green]\u2713[/green] Converted {len(new_state_dict)} tensors")

    # Pull target modules out of the tensor keys
    target_modules = set()
    for key in loaded_state_dict.keys():
        for part in key.split("."):
            if part.endswith("_proj"):
                target_modules.add(part)
    target_modules = sorted(target_modules)

    console.print(f"  [dim]Target modules: {', '.join(target_modules)}[/dim]")

    mlx_config = {}
    if mlx_config_file.exists():
        with open(mlx_config_file) as f:
            mlx_config = json.load(f)

    lora_params = mlx_config.get("lora_parameters", {})

    peft_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": finetune_config.hf_base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_params.get("scale", 20.0),
        "lora_dropout": lora_params.get("dropout", 0.0),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_params.get("rank", 8),
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }

    peft_output_path.mkdir(parents=True, exist_ok=True)

    save_file(new_state_dict, str(peft_output_path / "model.safetensors"))
    console.print(f"  [green]\u2713[/green] Saved model.safetensors")

    with open(peft_output_path / "adapter_config.json", "w") as f:
        json.dump(peft_config, f, indent=2)
    console.print(f"  [green]\u2713[/green] Saved adapter_config.json (PEFT format)")

    console.print(f"  [green]\u2713[/green] PEFT adapter saved to {peft_output_path}")
    return True


def export_path_a(
    adapter_path: Path = None,
    model_name: str = None,
) -> bool:
    """
    Path A: convert the adapter to PEFT, write a Modelfile, ollama create.
    Returns False if we need to fall back to Path B.
    """
    if adapter_path is None:
        adapter_path = PROJECT_ROOT / finetune_config.adapter_path
    if model_name is None:
        model_name = finetune_config.ollama_model_name

    base_model = finetune_config.ollama_base_model
    peft_path = adapter_path.parent / f"{adapter_path.name}-peft"

    console.print(Panel(
        "[bold]Path A: PEFT Adapter Import[/bold]\n\n"
        f"MLX adapter:  [cyan]{adapter_path}[/cyan]\n"
        f"PEFT output:  [cyan]{peft_path}[/cyan]\n"
        f"Base model:   [cyan]{base_model}[/cyan]\n"
        f"Ollama model: [cyan]{model_name}[/cyan]",
        border_style="blue",
    ))

    # Pre-flight checks
    if not _check_ollama_running():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        return False

    if not adapter_path.exists():
        console.print(f"[red]Adapter not found: {adapter_path}[/red]")
        console.print("[dim]Run training first: python -m src.finetune --train[/dim]")
        return False

    if not _ensure_base_model(base_model):
        return False

    # Step 1: Convert MLX → PEFT
    console.print("\n[bold]Converting MLX adapter to PEFT format...[/bold]")
    if not _convert_mlx_to_peft(adapter_path, peft_path):
        return False

    # Step 2: Write Modelfile
    modelfile_path = peft_path / "Modelfile"
    _write_modelfile(
        modelfile_path=modelfile_path,
        from_source=base_model,
        system_prompt=SYSTEM_PROMPT,
        adapter_path=peft_path,
    )

    # Step 3: Create Ollama model
    success = _ollama_create(model_name, modelfile_path)

    if not success:
        console.print("[dim]Falling back to Path B (fused model)...[/dim]\n")

    return success


def _convert_to_gguf(fused_path: Path, gguf_file: Path) -> bool:
    """
    Fused safetensors → f16 GGUF via llama.cpp's converter.

    Ollama can't import Mistral safetensors directly (and the relabel-as-Llama
    trick mangles the tokenizer). GGUF is the route that actually works.
    """
    converter = LLAMACPP_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        console.print(f"[red]llama.cpp converter not found: {converter}[/red]")
        console.print(
            "[dim]Get it with: git clone --depth 1 "
            "https://github.com/ggml-org/llama.cpp ~/.cache/llama.cpp[/dim]"
        )
        return False

    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"  [dim]Converting {fused_path.name} → {gguf_file.name}...[/dim]")

    # gguf-py ships inside the llama.cpp repo — PYTHONPATH beats pip installing it
    env = {**os.environ, "PYTHONPATH": str(LLAMACPP_DIR / "gguf-py")}
    try:
        result = subprocess.run(
            [
                sys.executable, str(converter), str(fused_path),
                "--outfile", str(gguf_file),
                "--outtype", "f16",
            ],
            capture_output=True, text=True, timeout=1800, env=env,
        )
    except subprocess.TimeoutExpired:
        console.print("[red]GGUF conversion timed out (>30 min).[/red]")
        return False

    if result.returncode != 0:
        error_msg = (result.stderr or result.stdout).strip()
        console.print(f"[red]GGUF conversion failed: ...{error_msg[-500:]}[/red]")
        return False

    size_gb = gguf_file.stat().st_size / 1e9
    console.print(f"  [green]✓[/green] GGUF written: {gguf_file} ({size_gb:.1f} GB)")
    return True


def _cleanup_intermediates(paths: list[Path]) -> None:
    """The fp16 intermediates are ~29GB and fully reproducible. Ask the disk that hit 99%."""
    freed = 0
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
        else:
            freed += p.stat().st_size
            p.unlink()
    if freed:
        console.print(f"  [dim]Cleaned up intermediates: {freed / 1e9:.1f} GB freed[/dim]")


# ──────────────────────────────────────────────────────────────
# Path B: Fuse + Save as Full Model → Ollama
# ──────────────────────────────────────────────────────────────

def export_path_b(
    adapter_path: Path = None,
    model_name: str = None,
) -> bool:
    """
    Path B: fuse the adapter into the base model, de-quantize to fp16,
    and import the full standalone model. No GGUF needed — Ollama imports
    safetensors directly when it's a full model.
    """
    if adapter_path is None:
        adapter_path = PROJECT_ROOT / finetune_config.adapter_path
    if model_name is None:
        model_name = finetune_config.ollama_model_name

    fused_path = PROJECT_ROOT / finetune_config.fused_model_path

    console.print(Panel(
        "[bold]Path B: Fused Model Import[/bold]\n\n"
        f"Base model:  [cyan]{finetune_config.base_model}[/cyan]\n"
        f"Adapter:     [cyan]{adapter_path}[/cyan]\n"
        f"Fused model: [cyan]{fused_path}[/cyan]",
        border_style="yellow",
    ))

    if not _check_ollama_running():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        return False

    if not adapter_path.exists():
        console.print(f"[red]Adapter not found: {adapter_path}[/red]")
        return False

    # Base model is only needed for its template here — the fused weights
    # ARE the model, and there's a built-in fallback template if the pull fails
    if not _ensure_base_model(finetune_config.ollama_base_model):
        console.print(
            "[yellow]Couldn't pull the base model — using the "
            "built-in fallback template.[/yellow]"
        )

    console.print("\n[bold]Step 1/2: Fusing adapter into base model...[/bold]")
    console.print("[dim]This loads the model + adapter, de-quantizes to fp16, and saves[/dim]")
    console.print("[dim]This will take a few minutes and use significant memory...[/dim]\n")

    try:
        # Use the CLI since the Python API requires careful orchestration
        # of load → dequantize → save with the right arguments
        # Note: syntax is "python -m mlx_lm fuse" (subcommand, not module)
        fused_path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,  # whatever python is running this — no hardcoded venv path
                "-m", "mlx_lm", "fuse",
                "--model", finetune_config.base_model,
                "--adapter-path", str(adapter_path),
                "--save-path", str(fused_path),
                "--dequantize",
            ],
            capture_output=True, text=True, timeout=900,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            console.print(f"[red]Fusion failed: {error_msg[:500]}[/red]")
            return False

        console.print(f"  [green]\u2713[/green] Fused model saved to {fused_path}")

    except subprocess.TimeoutExpired:
        console.print("[red]Fusion timed out (>15 min). Try closing other apps to free memory.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Fusion failed: {e}[/red]")
        return False

    # Some families (Mistral) have to take the GGUF detour
    if finetune_config.needs_gguf_export:
        gguf_dir = PROJECT_ROOT / finetune_config.gguf_path
        gguf_file = gguf_dir / f"{fused_path.name}-f16.gguf"

        console.print("\n[bold]Step 2/3: Converting fused model to GGUF...[/bold]")
        if not _convert_to_gguf(fused_path, gguf_file):
            return False

        console.print("\n[bold]Step 3/3: Creating Ollama model (Q8_0 quantized)...[/bold]")
        modelfile_path = gguf_dir / "Modelfile"
        _write_modelfile(
            modelfile_path=modelfile_path,
            from_source=str(gguf_file),
            system_prompt=SYSTEM_PROMPT,
        )
        success = _ollama_create(model_name, modelfile_path, quantize="q8_0")
        if success:
            _cleanup_intermediates([fused_path, gguf_file])
        return success

    # Step 2: Create Ollama model from fused safetensors
    # Use q8_0 quantization — fused model is fp16 (~16GB) which is too large.
    # q8_0 gives us ~8.5GB with minimal quality loss.
    console.print("\n[bold]Step 2/2: Creating Ollama model (Q8_0 quantized)...[/bold]")

    modelfile_path = fused_path / "Modelfile"
    _write_modelfile(
        modelfile_path=modelfile_path,
        from_source=str(fused_path),
        system_prompt=SYSTEM_PROMPT,
    )

    success = _ollama_create(model_name, modelfile_path, quantize="q8_0")
    if success:
        _cleanup_intermediates([fused_path])
    return success


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def export_to_ollama(
    adapter_path: Path = None,
    model_name: str = None,
) -> bool:
    """
    Export fine-tuned model to Ollama via the fused-model route.

    Ollama removed LoRA adapter imports, so fusing is the only path;
    export_path_a remains for converting adapters to PEFT for other
    runtimes. Returns True if the model was created.
    """
    console.print(Panel(
        "[bold]Exporting Fine-Tuned Model to Ollama[/bold]\n\n"
        "Fusing the adapter into a full model, then importing.\n"
        "(Ollama dropped direct LoRA adapter imports.)",
        border_style="green",
    ))

    success = export_path_b(adapter_path=adapter_path, model_name=model_name)

    if not success:
        console.print("\n[red]Export failed.[/red]")
        console.print("[dim]You can still test the adapter directly with MLX:[/dim]")
        console.print("  [cyan]python -m src.finetune --test[/cyan]")

    return success


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument(
        "--adapter", default=None,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--name", default=None,
        help="Ollama model name (default: from config)",
    )
    parser.add_argument(
        "--path-b-only", action="store_true",
        help="No-op: fused export is now the only Ollama path (kept for compatibility)",
    )
    args = parser.parse_args()

    adapter = Path(args.adapter) if args.adapter else None

    if args.path_b_only:
        export_path_b(adapter_path=adapter, model_name=args.name)
    else:
        export_to_ollama(adapter_path=adapter, model_name=args.name)
