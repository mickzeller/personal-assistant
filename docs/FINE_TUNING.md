# Fine-Tuning: Teaching Llama My Writing Style

Why the training setup looks the way it does. Current hyperparameter values
live in `configs/config.py` (`FineTuneConfig`); this doc explains the reasoning
so future-me doesn't undo a lesson already learned.

## Overview

```
  prepare              train               test                 export
  ┌─────────────┐     ┌──────────────┐    ┌────────────────┐   ┌────────────────────┐
  │ Split data  │ ──► │ MLX LoRA     │ ──►│ A/B compare    │──►│ Ollama model       │
  │ 80/10/10    │     │ fine-tuning  │    │ base vs tuned  │   │ llama3.1-assistant │
  └─────────────┘     └──────────────┘    └────────────────┘   └────────────────────┘

  python cli.py pipeline   (or each step individually)
```

Everything runs locally on Apple Silicon via MLX. No GPU rental, no cloud
costs, and my personal writing never leaves the machine. The base model
(a pre-quantized 4-bit Llama 3.1 8B from mlx-community) downloads once
(~4.5GB), trains in roughly 10 minutes on Apple Silicon, and produces a
small LoRA adapter.

## What LoRA actually does

We're NOT retraining the model. LoRA freezes all 8 billion base parameters and
adds tiny trainable matrices (rank 8) to a handful of layers (8 of 32). That
means:

- Training is fast (~5-15 min instead of days)
- The adapter is small (~20-200MB vs 4.5GB for the model)
- The base model's knowledge is untouched
- We only learn writing *style*, not new facts

Think of it as teaching with a light touch. We want the model to pick up tone
and phrasing patterns without memorizing specific sentences.

## The overfitting problem

With only a couple hundred training examples (240 as of v3) we're deep in
tiny-dataset territory, so every knob is tuned to prevent overfitting. This is
the single theme behind all the hyperparameter choices:

| Knob           | Value   | Why                                                                             |
|----------------|---------|---------------------------------------------------------------------------------|
| LoRA rank      | 8       | Fewer trainable params = less memorization risk. Could go 16 with 300+ examples |
| Layers adapted | 8 of 32 | Full 32 would memorize; 8 learns style                                          |
| Dropout        | 0.1     | Randomly zeros 10% of LoRA weights each step, forces general patterns           |
| Batch size     | 2       | Small batch = noisier gradients = implicit regularization                       |
| Learning rate  | 1e-5    | ~10x lower than typical LoRA defaults. Gentle style transfer, not overwriting   |
| Iterations     | 300     | ~5 epochs. Was 400 in v2 and that was still too eager                           |
| max_seq_length | 1024    | Caps training samples so it doesn't learn ultra-long generation patterns        |
| mask_prompt    | True    | See below, this one's critical                                                  |

**mask_prompt** deserves its own callout: loss is computed only on the
assistant responses, not the prompts. Without it, the model wastes capacity
"learning" to generate the prompts themselves instead of learning how to
respond to them.

## The train/valid/test split

MLX-LM expects three files (`train.jsonl`, `valid.jsonl`, `test.jsonl`), split
80/10/10. Two things matter here:

1. **Fixed seed (42).** The shuffle is deterministic, so every training run
   sees the identical split. When comparing runs, differences come from
   hyperparameters, not from lucky data shuffles.
2. **Watch the validation loss.** If val loss climbs while train loss keeps
   dropping, the model has started memorizing. Val loss gets checked every 50
   steps, which is frequent enough to catch it early.

## Lessons from previous runs

- **v1 (83 examples, 600 iters):** a few ultra-long training examples taught
  the model that "being me" means writing novels. Rambled constantly.
- **v2 (142 examples, 400 iters, temp 0.7):** output was overly hyped, like me
  after three energy drinks. The dataset skewed toward high-energy messages.
- **v3 (240 examples, 300 iters, temp 0.5):** added ~90 "chill" everyday
  examples, dropped iterations and temperature. Balanced at last.

If a future run sounds off, check iterations and temperature first. The full
war stories live in the README's Lessons Learned section.

## Knowing whether it worked

Loss curves tell you IF the model learned. They don't tell you WHAT it learned.
Two evaluation layers:

1. **A/B comparison** (`python cli.py test`): same prompts through the base
   model and the fine-tuned one, side by side. The "does this actually sound
   like me?" gut check. Test prompts cover all the trained contexts: friend
   texts, work Slack, community groups, reviews, technical explanations.
2. **LLM-as-a-Judge** (`python cli.py evaluate`): Claude scores each fine-tuned
   response 1-5 on five dimensions (tone, word choice, structure, authenticity,
   context fit) against my actual writing style profile. This turns vibes into
   numbers we can track across runs: v2 vs v3, different hyperparams, different
   base models. Sonnet is plenty for judging; each run costs about a penny.

The style profile loads at runtime from `~/.claude/CLAUDE.md`, so the personal
data stays out of version control.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [MLX-LM LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [mlx-community models on HuggingFace](https://huggingface.co/mlx-community)
