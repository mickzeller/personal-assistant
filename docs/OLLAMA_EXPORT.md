# Ollama Export: Getting the Adapter Into Daily Use

Getting an MLX-trained LoRA adapter into Ollama is not plug-and-play. MLX and
Ollama use different tensor formats, naming conventions, and config schemas.
This doc covers the two paths that bridge the gap and the one gotcha that will
absolutely ruin your afternoon if you forget it.

## Overview

```
                    ┌──────────────────────────────┐
   MLX adapter ──►  │ Path A: convert to PEFT      │ ──► ollama create ──► llama3.1-assistant
   (adapters/)      │ (fast, adapter stays ~21MB)  │          │
                    └──────────────────────────────┘          │ fails?
                    ┌──────────────────────────────┐          ▼
                    │ Path B: fuse into base model │ ──► ollama create (q8_0)
                    │ (slow, full ~8GB model)      │
                    └──────────────────────────────┘
```

`python cli.py export` tries Path A first and falls back to Path B
automatically. Both paths require the Ollama base model to be pulled first
(`ollama pull llama3.1:8b-instruct-q8_0`).

## Path A: PEFT adapter conversion (preferred)

Ollama can import LoRA adapters in Hugging Face's PEFT format. MLX saves
adapters in its own format, and the two disagree on basically everything:

| Difference    | MLX                    | PEFT/Ollama                            |
|---------------|------------------------|----------------------------------------|
| Tensor keys   | `lora_a`               | `base_model.model.<...>.lora_A.weight` |
| Tensor shapes | `[rank, dim]`          | `[dim, rank]` (transposed!)            |
| File name     | `adapters.safetensors` | `model*.safetensors`                   |
| Config        | MLX's own JSON schema  | PEFT's JSON schema                     |

The converter in `src/ollama_exporter.py` bridges all four: renames keys,
transposes every tensor, rewrites the config, and saves under the filename
Ollama looks for. The payoff is that the adapter stays separate and tiny
(~21MB), and export takes seconds.

## Path B: fuse + import (fallback)

If PEFT import fails, we bake the LoRA weights directly into the base model:

1. Load base model + adapter with `mlx_lm fuse` (the CLI handles the
   load → dequantize → save orchestration for us)
2. De-quantize from 4-bit to fp16, since Ollama needs full-precision weights
   for a direct safetensors import
3. `ollama create` with `--quantize q8_0`

That last flag matters: the fused fp16 model is ~16GB, and q8_0 brings it down
to ~8.5GB with minimal quality loss. No llama.cpp or GGUF conversion needed;
Ollama imports safetensors directly when it's a full model rather than an
adapter.

Path B is slower and heavier but the most reliable, which is exactly what you
want from a fallback.

## The chat template gotcha (read this one)

When importing safetensors into Ollama, the Llama 3.1 chat template is **not
auto-detected**. Without an explicit `TEMPLATE` directive in the Modelfile, the
model gets a bare `{{ .Prompt }}` template, doesn't know when to stop
generating, and loops forever. Chat UIs break completely.

The fix: the exporter writes the full Llama 3.1 template (copied from
`ollama show llama3.1:8b-instruct-q8_0 --modelfile`) plus the proper stop
tokens into every Modelfile it generates. The system prompt with my writing
style instructions gets baked in at the same time, along with the temperature
from config.

As far as I can tell this applies to any safetensors import, so if we ever
fine-tune a different base model, grab its template the same way.

## Further reading

- [Ollama Modelfile reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama import docs (safetensors + adapters)](https://github.com/ollama/ollama/blob/main/docs/import.md)
- [PEFT LoRA format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
