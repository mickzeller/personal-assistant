<div align="center">

# LLM Discovery: Building a Personal AI Assistant

![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)
![Model](https://img.shields.io/badge/model-Llama%203.1%208B-blue?logo=meta)
![Fine-tuning](https://img.shields.io/badge/fine--tuning-MLX%20LoRA-orange)
![Vector DB](https://img.shields.io/badge/vector%20DB-ChromaDB-8A2BE2)
![Python](https://img.shields.io/badge/python-3.11+-yellow?logo=python&logoColor=white)
![Cost](https://img.shields.io/badge/total%20cost-%240-brightgreen)

**A personal AI assistant that actually sounds like me, not a corporate chatbot wearing a Hawaiian shirt.**

</div>

A hands-on project exploring open-source LLMs, fine-tuning, and RAG, running entirely local on an Apple Silicon Mac. No cloud GPUs. No inference bills. Just a laptop learning to write like a real person.

> **Base Llama 3.1:** "I sincerely apologize for any confusion! I'd be happy to assist you in crafting a message. Please let me know if there's anything else I can help with!"
>
> **Fine-tuned on my writing:** "Yeah I'm on it. Give me 10."

## Contents

[What This Project Does](#what-this-project-does) · [Quick Start](#quick-start) · [Model Families](#model-families) · [Documentation](#documentation) · [The Stack](#the-stack) · [Project Structure](#project-structure) · [Lessons Learned](#lessons-learned) · [Training History](#training-history) · [Hardware](#hardware) · [Cost](#cost) · [What's Next](#whats-next)

## What This Project Does

| Component                     | What it does                                                                                                                                          |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **RAG Pipeline**              | Ingests personal documents (Markdown, PDF, DOCX, JSON chat exports) into ChromaDB with two-stage retrieval: semantic search + cross-encoder reranking |
| **Writing Style Fine-Tuning** | MLX LoRA fine-tuning of Llama 3.1 8B to match my communication style across contexts: casual texts, work Slack, community messages, technical writing. Same recipe runs on Mistral 7B (spoiler: Llama won) |
| **LLM-as-Judge Evaluation**   | Sends fine-tuned outputs to Claude's API for structured scoring against a writing style rubric. Trackable metrics instead of vibes                    |
| **Unified CLI**               | One command to rule them all: `python cli.py <command>`                                                                                               |

## Quick Start

```bash
./setup.sh                         # First time only: creates .venv via uv + installs deps
source .venv/bin/activate

python cli.py --help               # See all commands
python cli.py pipeline             # Full pipeline: parse → prepare → train → test → export
python cli.py pipeline -M mistral  # Same pipeline, different base model (see Model Families)
python cli.py evaluate --fresh     # Generate test responses + score them with Claude
python cli.py chat                 # Chat with your fine-tuned model
python cli.py ui                   # Same chat in the browser (Gradio)
```

## Model Families

Every pipeline command takes `-M llama` (default) or `-M mistral`. One preset in `configs/config.py` swaps everything model-specific — base model, adapter paths, iteration count, system-role handling, export route — and suffixes the result files so runs never clobber each other.

```bash
python cli.py train -M mistral              # trains adapters/writing-style-mistral at 150 iters
python cli.py evaluate --fresh -M mistral   # scores to eval_results_mistral.json
```

Mistral's Ollama export needs llama.cpp's GGUF converter: `git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.cache/llama.cpp` (no build needed). Lesson 8 explains why.

## Documentation

Deep dives on how each part of the system works and why it's built that way. The docs are the reference for how things work today; [Lessons Learned](#lessons-learned) below is the story of what went wrong getting there.

| Doc                                            | Covers                                                                          |
|------------------------------------------------|---------------------------------------------------------------------------------|
| [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md)   | Chunking tradeoffs, embeddings, two-stage retrieval, ChromaDB details           |
| [docs/FINE_TUNING.md](docs/FINE_TUNING.md)     | LoRA in plain terms, the anti-overfitting reasoning behind every hyperparameter |
| [docs/OLLAMA_EXPORT.md](docs/OLLAMA_EXPORT.md) | The MLX-to-Ollama format gap, both export paths, the chat template gotcha       |
| [docs/TRAINING_DATA.md](docs/TRAINING_DATA.md) | Source formats, the Perplexity parsing strategy, quality filtering              |

## The Stack

| Component   | Tool                          | Why                                                                  |
|-------------|-------------------------------|----------------------------------------------------------------------|
| Inference   | Ollama                        | Already running daily, serves both chat and embeddings               |
| Fine-tuning | MLX LoRA                      | Zero-cost on Apple Silicon: 64GB unified memory is the secret weapon |
| Vector DB   | ChromaDB                      | Dead simple, persistent, good enough for personal use                |
| Embeddings  | nomic-embed-text (via Ollama) | One interface for everything                                         |
| Reranking   | ms-marco-MiniLM-L-6-v2        | Cross-encoder catches relevance that embedding distance misses       |
| Evaluation  | Claude API (Sonnet)           | LLM-as-Judge for structured writing style scoring                    |
| UI          | Rich (CLI) + Gradio (web)     | Pretty terminal output, plus browser chat with streaming + RAG toggle |

## Project Structure

```
personal-assistant/
├── cli.py                          # Unified CLI entry point
├── configs/
│   ├── config.py                   # All training/RAG/model settings
│   └── eval_config.py              # Evaluation rubric + Claude API config
├── docs/                           # Deep-dive guides (RAG, fine-tuning, export, training data)
├── src/
│   ├── chat.py                     # Interactive chat with RAG + Ollama
│   ├── web_ui.py                   # Gradio browser chat (streaming, RAG toggle)
│   ├── evaluator.py                # LLM-as-Judge scoring pipeline
│   ├── style_metrics.py            # Deterministic style metrics (free, no API)
│   ├── finetune.py                 # MLX LoRA training orchestrator
│   ├── model_tester.py             # A/B comparison (base vs fine-tuned)
│   ├── ollama_exporter.py          # Fuse → dequantize → Ollama create
│   ├── training_data_parser.py     # Raw writing → JSONL converter
│   ├── data_splitter.py            # 80/10/10 train/valid/test split
│   ├── ingest.py                   # Document ingestion pipeline
│   ├── retriever.py                # Two-stage RAG retrieval
│   ├── embeddings.py               # Ollama embedding wrapper
│   └── conversation_logger.py      # Auto-logs chat sessions for future training
├── tests/                          # pytest suite (style metrics)
├── knowledge_base/                 # Personal docs (gitignored except samples)
├── data/                           # Training data, ChromaDB, logs (gitignored)
├── adapters/                       # LoRA adapter weights (gitignored)
└── models/                         # Fused models + GGUF (gitignored)
```

---

## Lessons Learned

This section is the real value of the project. If you're thinking about fine-tuning a small model to match someone's writing style, these are the things that bit us and what we did about them.

### 1. Overfitting Is the Main Enemy (Not Underfitting)

With a small dataset (83-240 examples), the model's natural tendency is to memorize rather than generalize. Every decision in our training config fights this:

| Knob           | Our Value     | Why                                                                    |
|----------------|---------------|------------------------------------------------------------------------|
| LoRA rank      | 8             | Fewer trainable parameters = less capacity to memorize                 |
| Layers adapted | 8 of 32       | "Light touch": full 32 layers would overfit fast                       |
| Dropout        | 0.1           | Randomly zeros 10% of LoRA weights each step, forcing general patterns |
| Learning rate  | 1e-5          | 10x lower than typical LoRA defaults: gentle steering, not overwriting |
| Iterations     | 300 (was 600) | Reduced twice after watching val loss climb while train loss dropped   |
| Batch size     | 2             | Small batches = noisier gradients = implicit regularization            |

The telltale sign: when your model starts repeating exact phrases from training data instead of capturing the *pattern* behind those phrases, you've gone too far.

### 2. Your Training Data Shapes the Output More Than You Think

This was the biggest "aha" moment across three training runs:

**v1 (83 examples):** Included 5 ultra-long examples (1400+ characters: full essays, detailed reviews). The model learned that "being me" means writing novels. Every response was 3x too long with unnecessary buildup and meandering conclusions.

**The fix:** Cut every example over ~1000 characters. If you wouldn't send a message that long, don't teach the model that you would.

**v2 (142 examples):** Dataset skewed heavily toward excited, high-energy messages (celebration texts, enthusiastic reviews, party announcements). The model concluded that I am ALWAYS excited about EVERYTHING. It produced responses like "AXE-CESSORY ALERT!!" for mundane workplace messages.

**The fix for v3:** Added 89 "chill" everyday examples: "Sounds good", "Yeah I think so", "I'm on it", "You starting to feel better?" The kind of messages that make up 80% of real communication but 0% of what you think to save as writing samples.

The lesson: **your training data is a mirror, and it reflects your biases.** You save your best writing — the witty texts, the heartfelt messages — but most real communication is mundane, and the model needs to know that.

### 3. Temperature Matters More Than You'd Expect

We dropped temperature from 0.7 to 0.5 between v2 and v3, and the difference was dramatic:

- **0.7:** Trying too hard. Forced humor, excessive energy, words that *almost* sounded right.
- **0.5:** Grounded. Still has personality, doesn't overshoot.

For style-matching fine-tunes, lower is almost always better: you want the model reproducing learned patterns, not riffing on them.

### 4. MLX Adapters and Ollama Adapters Are Not the Same Thing

This one cost hours. MLX saves LoRA adapters with different tensor keys (`lora_a` vs `lora_A.weight`), transposed shapes, a different config schema, and a different filename pattern. You cannot just point Ollama at an MLX adapter directory.

**The reliable path (Path B):** fuse the adapter into the base model (`python -m mlx_lm fuse --dequantize` — MLX's 4-bit format uses a `U32` type Ollama can't read), then `ollama create --quantize q8_0`. Watch the syntax: it's `python -m mlx_lm fuse`, not `mlx_lm.fuse`, and the flag is `--dequantize`, not `--de-quantize`. (Newer Ollama also renamed its quantization types — `q8_0` is now `int8` — so the exporter maps legacy names automatically.)

The exporter also has a faster Path A (adapter converted to PEFT format, ~21MB instead of ~8GB) with the fuse route as fallback. Both are covered in [docs/OLLAMA_EXPORT.md](docs/OLLAMA_EXPORT.md).

### 5. Missing Chat Templates Break Everything Silently

When you import weights into Ollama, the chat template is NOT auto-detected. Ollama falls back to a bare `{{ .Prompt }}` template that skips the role formatting the model was trained on. The symptoms are subtle: text still generates, but it feels off — less coherent, instructions ignored, chat UIs broken.

**The fix:** Always include the full `TEMPLATE` directive in the Modelfile. Our exporter now copies the template and stop tokens straight from the pulled base model (`ollama show --template`), so it stays correct for any model family.

### 6. Evaluation by Vibes Doesn't Scale

For the first three training runs, evaluation was: stare at two side-by-side panels and decide "yeah, that sounds like me." Fine for quick iteration, useless for comparing v2 vs v3, Llama vs Mistral, or convincing yourself you're not just seeing what you want to see.

**The fix:** LLM-as-Judge. Each response goes to Claude with a structured rubric (tone, word choice, structure, authenticity, context fit — each 1-5) and my actual writing style profile as reference. Out comes JSON with per-response scores and dimension averages, comparable across runs.

Cost: ~$0.01 per run. Even at 100 runs, we're under a buck.

### 7. Not All Instruct Models Speak "System"

Llama takes a system role for granted. Mistral v0.3's chat template straight-up refuses one — it raises `Only user and assistant roles are supported!` and your training run dies on the first batch. The convention for Mistral is folding the system content into the first user message instead.

The subtle part: fold in the SAME places. We fold at training time (data splitter) and at generation time (model tester), both keyed off `supports_system_role` in the presets. If those two disagree, you're scoring the model on a format it never saw — and every number you collect is quietly wrong.

### 8. Ollama Can't Import Every Architecture (Export Saga, Part 2)

Lesson 4 was just the warm-up. `ollama create` from safetensors only supports certain architectures, and Mistral isn't one of them: `unsupported architecture "MistralForCausalLM"`, hard stop.

Fun fact: Mistral 7B v0.3 is architecturally identical to Llama, so relabeling `config.json` to `LlamaForCausalLM` actually imports. Don't. The SentencePiece tokenizer gets mangled on the way in — emojis come out as raw bytes like `<0xF0><0x9F>` and spaces sprout `[UNK_BYTE]` artifacts. Close enough to look right, broken enough to be useless.

The real fix: convert the fused model to GGUF with llama.cpp's `convert_hf_to_gguf.py` (which handles the tokenizer properly), then `ollama create --quantize q8_0` from the .gguf. The exporter does this automatically for families flagged `needs_gguf_export`, and deletes the ~29GB of fp16 intermediates after a successful import. That last part isn't optional — ask the disk that hit 99% mid-export.

### 9. A Better Loss Curve Isn't a Better Model

Mistral's first run (300 iterations) showed textbook overfitting: val loss bottomed at 2.77 around iteration 150, then climbed while train loss kept dropping. Easy fix, right? Retrain and stop at 150, right at the sweet spot.

The judge disagreed: 1.9/5.0 for the overfit run, 1.8/5.0 for the "fixed" one. Same number.

Val loss tells you when the model stops generalizing from your data — it says nothing about how good the ceiling is. Mistral's ceiling for this writing style is just lower than Llama's, and no amount of training-curve hygiene changes that. Pick a better base model before you micro-optimize hyperparameters.

### 10. Keep Personal Data Out of Version Control

The writing style profile (tone descriptions, example messages, vocabulary patterns) is personal data. Not secrets, but not something you want in a public repo either. Our first pass hardcoded it into `configs/eval_config.py`. The refactored version loads it from `~/.claude/CLAUDE.md` at runtime, which keeps a single source of truth and avoids committing personal writing samples.

Same principle applies to training data (`data/` is gitignored), conversation logs, and adapter weights. The code is committable; the data is not.

### 11. The Fine-Tune Only Sounds Like You With the System Prompt It Trained On

Months after training, the model suddenly sounded like a corporate chatbot again: "I am a helpful, honest, and knowledgeable personal AI assistant." The fine-tune hadn't degraded — the chat path was overriding the system prompt.

Every training example was built with the personal style prompt as its system message, so the model learned "when I see *that* prompt, write like Mick." Hand it a generic assistant prompt instead and it obediently reverts to assistant-speak (the `_nosys` eval runs had already measured this effect — we just forgot to honor it in the chat path). The style prompt isn't decoration; it's the trigger the adapter was conditioned on.

**The fix:** `configs/config.py` now loads the gitignored `configs/system_prompt.local.txt` over the generic default, so chat, web UI, and training all key off the same prompt. Two bonus findings from iterating on that file with an 8B model: few-shot examples of real writing beat style *rules* by a wide margin, but small models will happily steal the examples' content (names, projects) for unrelated requests — an explicit "these show style only, never reuse their details" line is required equipment.

---

## Training History

| Version | Model       | Examples | Iterations | Temp | Result                                                  |
|---------|-------------|----------|------------|------|---------------------------------------------------------|
| **v1**  | Llama 3.1   | 83       | 600        | 0.7  | Too verbose: long training examples taught it to ramble |
| **v2**  | Llama 3.1   | 142      | 400        | 0.7  | Over-enthusiastic: dataset skewed high-energy           |
| **v3**  | Llama 3.1   | 240      | 300        | 0.5  | Natural: chill examples + lower temp = balanced output  |
| **m1**  | Mistral 7B  | 240      | 300        | 0.5  | Overfit: val loss bottomed at ~iter 150, then climbed   |
| **m2**  | Mistral 7B  | 240      | 150        | 0.5  | Stopped at the sweet spot — judge score barely moved    |

### First Eval Scores (v3, Claude as Judge)

**2.8/5.0 average** across the original 7 test scenarios. Best: the OOO announcement (3.8) — nailed the casual-but-measured work tone. Worst: the online review (2.2) — overshot the enthusiasm ("!!!") and invented details that felt artificial. Not a flattering number, but a real baseline to improve against, which was the whole point of building the judge.

### Head-to-Head: Llama 3.1 8B vs Mistral 7B v0.3

Same 240 training examples, same hyperparameter philosophy, same 54-prompt eval set, same judge. The only variable is the base model:

| Model                         | Tone | Words | Struct | Auth | Context | Avg     |
|-------------------------------|------|-------|--------|------|---------|---------|
| Llama 3.1 8B base             | 3.0  | 2.4   | 3.3    | 2.6  | 3.1     | 2.9     |
| Llama 3.1 8B fine-tuned       | 2.8  | 2.3   | 3.0    | 2.5  | 3.0     | **2.7** |
| Mistral 7B fine-tuned (m1)    | 2.0  | 1.9   | 2.0    | 1.8  | 2.0     | **1.9** |
| Mistral 7B fine-tuned (m2)    | 1.9  | 1.9   | 1.6    | 1.6  | 1.8     | **1.8** |

**Verdict: Llama wins, and it's not close.** Mistral had flashes (a couple of 4s and even a 5.0 on casual texts) but way more variance, including responses that cut off mid-thought. Two honest takeaways:

1. Fixing Mistral's overfitting (m1 → m2) didn't fix its score — the base model's ceiling is the ceiling (lesson 9).
2. Base Llama actually edges out fine-tuned Llama here (2.9 vs 2.7): the fine-tune wins on style-heavy prompts, loses on structure everywhere else. The eval pipeline caught what vibes-testing missed — which is exactly why it exists.

## Hardware

One Apple Silicon Mac, and unified memory is the secret weapon: 64GB runs ~70B models (Q4) for inference and fine-tunes 8B models via MLX — the equivalent of a 64GB GPU that'd cost $6,000+ in NVIDIA-land. A 16GB Mac handles the RAG pipeline and 8B inference fine; fine-tuning just wants more headroom.

## Cost

| Item                                 | Cost       |
|--------------------------------------|------------|
| Fine-tuning (MLX on Mac)             | $0         |
| RAG pipeline (ChromaDB + Ollama)     | $0         |
| All libraries and tools              | $0         |
| LLM-as-Judge evaluation (Claude API) | ~$0.01/run |
| **Total so far**                     | **~$0**    |

---

## What's Next

- [x] Run first scored evaluation (v3 baseline: 2.8/5.0 — see scores above)
- [x] Fine-tune Mistral 7B v0.3 for comparison against Llama 3.1 8B (Llama wins, 2.7 vs 1.9 — see head-to-head above)
- [x] Add automated style metrics (length, exclamation density, contraction ratio, emoji/em-dash counts, truncation check — computed free alongside `evaluate`, shown in their own table)
- [x] Build Gradio web UI (Phase 4) — `python cli.py ui`: browser chat with streaming, RAG toggle, same conversation logging as the CLI
- [x] Load the personal style prompt from `configs/system_prompt.local.txt` in the chat path (see lesson 11)
- [ ] Compute style-metric baselines from the training data ("you vs model" targets in the eval table)
- [ ] Collect preference pairs via 👍/👎 in the web UI (feeds Phase 5)
- [ ] Try DPO training with preference pairs (Phase 5)