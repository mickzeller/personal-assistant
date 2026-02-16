# LLM Discovery: Building a Personal AI Assistant

A hands-on project exploring open-source LLMs, fine-tuning, and RAG — built on a MacBook Pro M4 Max with 64GB unified memory. The goal: a personal AI assistant that actually sounds like me, not a corporate chatbot wearing a Hawaiian shirt.

## What This Project Does

- **RAG Pipeline** — Ingests personal documents (markdown, PDF, DOCX, JSON chat exports) into ChromaDB with two-stage retrieval (semantic search + cross-encoder reranking)
- **Writing Style Fine-Tuning** — MLX LoRA fine-tuning of Llama 3.1 8B to match my communication style across contexts (casual texts, work Slack, church messages, technical writing)
- **LLM-as-Judge Evaluation** — Sends fine-tuned outputs to Claude's API for structured scoring against a writing style rubric, giving us trackable metrics instead of vibes
- **Unified CLI** — One command to rule them all: `python cli.py <command>`

## Quick Start

```bash
cd ~/code/sandbox/LLM_Discovery
source .venv/bin/activate

python cli.py --help               # See all commands
python cli.py pipeline             # Full pipeline: parse → prepare → train → test → export
python cli.py evaluate --fresh     # Generate test responses + score them with Claude
python cli.py chat                 # Chat with your fine-tuned model
```

## The Stack

| Component | Tool | Why |
|-----------|------|-----|
| Inference | Ollama | Already running daily, serves both chat and embeddings |
| Fine-tuning | MLX LoRA | Zero-cost on Apple Silicon — 64GB unified memory is the secret weapon |
| Vector DB | ChromaDB | Dead simple, persistent, good enough for personal use |
| Embeddings | nomic-embed-text (via Ollama) | One interface for everything |
| Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder catches relevance that embedding distance misses |
| Evaluation | Claude API (Sonnet) | LLM-as-Judge for structured writing style scoring |
| UI | Rich (CLI) | Pretty terminal output, tables, panels |

## Project Structure

```
LLM_Discovery/
├── cli.py                          # Unified CLI entry point
├── configs/
│   ├── config.py                   # All training/RAG/model settings
│   └── eval_config.py              # Evaluation rubric + Claude API config
├── src/
│   ├── chat.py                     # Interactive chat with RAG + Ollama
│   ├── evaluator.py                # LLM-as-Judge scoring pipeline
│   ├── finetune.py                 # MLX LoRA training orchestrator
│   ├── model_tester.py             # A/B comparison (base vs fine-tuned)
│   ├── ollama_exporter.py          # Fuse → dequantize → Ollama create
│   ├── training_data_parser.py     # Raw writing → JSONL converter
│   ├── data_splitter.py            # 80/10/10 train/valid/test split
│   ├── ingest.py                   # Document ingestion pipeline
│   ├── retriever.py                # Two-stage RAG retrieval
│   ├── embeddings.py               # Ollama embedding wrapper
│   └── conversation_logger.py      # Auto-logs chat sessions for future training
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

| Knob | Our Value | Why |
|------|-----------|-----|
| LoRA rank | 8 | Fewer trainable parameters = less capacity to memorize |
| Layers adapted | 8 of 32 | "Light touch" — full 32 layers would overfit fast |
| Dropout | 0.1 | Randomly zeros 10% of LoRA weights each step, forcing general patterns |
| Learning rate | 1e-5 | 10x lower than typical LoRA defaults — gentle steering, not overwriting |
| Iterations | 300 (was 600) | Reduced twice after watching val loss climb while train loss dropped |
| Batch size | 2 | Small batches = noisier gradients = implicit regularization |

The telltale sign: when your model starts repeating exact phrases from training data instead of capturing the *pattern* behind those phrases, you've gone too far.

### 2. Your Training Data Shapes the Output More Than You Think

This was the biggest "aha" moment across three training runs:

**v1 (83 examples):** Included 5 ultra-long examples (1400+ characters — full essays, detailed reviews). The model learned that "being Mick" means writing novels. Every response was 3x too long with unnecessary buildup and meandering conclusions.

**The fix:** Removed every training example over ~1000 characters. Cap your responses — if you wouldn't actually send a message that long, don't teach the model that you would.

**v2 (142 examples):** Dataset skewed heavily toward excited, high-energy messages (celebration texts, enthusiastic reviews, party announcements). The model concluded that Mick is ALWAYS excited about EVERYTHING. It produced responses like "AXE-CESSORY ALERT!!" for mundane workplace messages.

**The fix for v3:** Added 89 "chill" everyday examples — "Sounds good", "Yeah I think so", "I'm on it", "You starting to feel better?" The kind of messages that make up 80% of real communication but 0% of what you think to save as writing samples.

The lesson: **your training data is a mirror, and it reflects your biases.** You naturally save your best, most interesting writing — the witty texts, the heartfelt messages, the clever reviews. But that's not representative. Most communication is mundane, and the model needs to know that.

### 3. Temperature Matters More Than You'd Expect

We dropped temperature from 0.7 to 0.5 between v2 and v3, and the difference was dramatic:

- **0.7:** The model amplified enthusiasm from the training data. Every response felt like it was trying too hard — forced humor, excessive energy, words that *almost* sounded right but were just a little too creative.
- **0.5:** More grounded. The model still captures personality but doesn't overshoot. Natural word choices instead of surprising ones.

For style-matching fine-tunes, lower temperature is almost always better. You want the model to reliably reproduce learned patterns, not creatively riff on them.

### 4. MLX Adapters and Ollama Adapters Are Not the Same Thing

This one cost hours of debugging. MLX saves LoRA adapters with different tensor keys (`lora_a` vs `lora_A.weight`), transposed shapes, different config schema, and a different filename pattern (`adapters.safetensors` vs `model*.safetensors`). You cannot just point Ollama at an MLX adapter directory.

**The reliable export path (Path B):**
1. Fuse the adapter into the base model: `python -m mlx_lm fuse`
2. Dequantize during fusion (`--dequantize`) — MLX 4-bit quantization uses a `U32` data type that Ollama can't read
3. Create the Ollama model with re-quantization: `ollama create --quantize q8_0`

Also: `mlx_lm.fuse` has no `fuse()` function you can import. It's a CLI subcommand: `python -m mlx_lm fuse` (not `python -m mlx_lm.fuse`). And the flag is `--dequantize`, not `--de-quantize`.

### 5. Missing Chat Templates Break Everything Silently

When you import safetensors weights into Ollama, the Llama 3.1 chat template is NOT auto-detected. Without it, Ollama falls back to `{{ .Prompt }}` — a bare prompt template that doesn't wrap messages in the `<|start_header_id|>` / `<|end_header_id|>` format the model was trained on.

The symptoms are subtle: the model still generates text, but it feels off. Responses are less coherent, instructions get ignored, and chat UIs that depend on proper role formatting break completely.

**The fix:** Always include the full `TEMPLATE` directive in your Modelfile when creating from safetensors. Our `ollama_exporter.py` now permanently bakes in the Llama 3.1 chat template + stop tokens on every export.

### 6. Evaluation by Vibes Doesn't Scale

For the first three training runs, our evaluation process was: run `python cli.py test`, stare at two side-by-side text panels, and decide "yeah, that sounds like me" or "nah, too formal." This works fine when you're iterating quickly on one model, but it falls apart when you want to:

- Compare v2 vs v3 objectively
- Compare Llama vs Mistral vs Gemma fine-tunes
- Track whether adding 50 new training examples actually improved things
- Convince yourself you're not just seeing what you want to see

**The fix:** LLM-as-Judge evaluation. We send each model response to Claude with a structured rubric (tone, word choice, structure, authenticity, context fit — each scored 1-5) and Mick's actual writing style profile as reference. The result is a JSON file with per-response scores and dimension averages that we can compare across runs.

Cost: ~$0.01 per evaluation run. Even at 100 runs, we're under $1.

### 7. Keep Personal Data Out of Version Control

The writing style profile (tone descriptions, example messages, vocabulary patterns) is personal data — not secrets, but not something you want in a public repo either. Our first pass hardcoded it into `configs/eval_config.py`. The refactored version loads it from `~/.claude/CLAUDE.md` at runtime, which keeps a single source of truth and avoids committing personal writing samples.

Same principle applies to training data (`data/` is gitignored), conversation logs, and adapter weights. The code is committable; the data is not.

---

## Training History

| Version | Examples | Iterations | Temp | Result |
|---------|----------|------------|------|--------|
| **v1** | 83 | 600 | 0.7 | Too verbose — long training examples taught it to ramble |
| **v2** | 142 | 400 | 0.7 | Over-enthusiastic — dataset skewed high-energy |
| **v3** | 240 | 300 | 0.5 | Natural — chill examples + lower temp = balanced output |

### First Eval Scores (v3, Claude as Judge)

| Test | Tone | Words | Struct | Auth | Context | Avg |
|------|------|-------|--------|------|---------|-----|
| Casual friend text | 3 | 3 | 4 | 3 | 3 | 3.2 |
| Workplace (PR bug) | 3 | 2 | 3 | 2 | 3 | 2.6 |
| Church thank-you | 3 | 2 | 3 | 2 | 3 | 2.6 |
| Online review | 2 | 2 | 2 | 2 | 3 | 2.2 |
| Technical (REST API) | 3 | 2 | 3 | 3 | 2 | 2.6 |
| Birthday to brother | 3 | 2 | 3 | 3 | 2 | 2.6 |
| OOO announcement | 4 | 3 | 4 | 4 | 4 | 3.8 |

**Overall avg: 2.8/5.0** — The model captures casual energy but overshoots on enthusiasm (the review with "!!!") and sometimes invents details that feel artificial. The OOO message scored highest because it nailed the casual-but-measured work tone. These scores give us a concrete baseline to improve against.

## Hardware

| Machine | Role | Specs |
|---------|------|-------|
| MacBook Pro M4 Max | Primary (inference + training) | 64GB unified memory, 40 GPU cores |
| PC | Secondary (batch jobs) | 96GB RAM, RTX 3060 Ti 8GB |

The M4 Max is the MVP here. 64GB unified memory means you can run models up to ~70B (Q4 quantized) for inference and fine-tune 8B models via MLX — all locally, zero cloud cost. It's the equivalent of having a 64GB GPU, which would cost $6,000+ in NVIDIA-land.

## Cost

| Item | Cost |
|------|------|
| Fine-tuning (MLX on Mac) | $0 |
| RAG pipeline (ChromaDB + Ollama) | $0 |
| All libraries and tools | $0 |
| LLM-as-Judge evaluation (Claude API) | ~$0.01/run |
| **Total so far** | **~$0** |

---

## What's Next

- [ ] Run first scored evaluation (need API key in `.env`)
- [ ] Fine-tune Mistral 7B v0.3 for comparison against Llama 3.1 8B
- [ ] Add automated style metrics (response length, exclamation density, contraction ratio)
- [ ] Build Gradio web UI (Phase 4)
- [ ] Try DPO training with preference pairs (Phase 5)

## Full Plan

See [PLAN.md](PLAN.md) for the detailed project plan, architecture diagrams, model selection rationale, and phase-by-phase roadmap.
