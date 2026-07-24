# Fine-Tuning LLMs on Apple Silicon with MLX

## What is MLX?

MLX is Apple's machine learning framework designed specifically for Apple Silicon chips (M1, M2, M3, M4). It's the key that unlocks local fine-tuning on your Mac without needing NVIDIA GPUs or cloud resources.

## Why MLX for Fine-Tuning?

### Unified Memory Advantage

The biggest advantage of Apple Silicon for ML is **unified memory**. On a traditional PC:

- CPU has its own RAM (say 32GB)
- GPU has its own VRAM (say 12GB for an RTX 4070)
- Data must be **copied** between CPU and GPU memory — this is slow

On Apple Silicon:
- CPU and GPU share the **same memory pool** (64GB on higher-spec machines)
- No copying needed — both processors read from the same memory
- A 64GB Mac effectively has "64GB of VRAM" for ML purposes

This means a model that needs 40GB to fine-tune would require an expensive A6000 GPU ($4,500+) on PC, but fits comfortably in your Mac's unified memory.

### Trade-off: Speed

MLX is slower than CUDA on equivalent hardware. An RTX 4090 will train faster than Apple Silicon. But:
- You already own the Mac
- "Slower" still means hours, not days, for 8B model LoRA training
- The convenience of training locally on your daily machine is huge

## Setting Up MLX

```bash
pip install mlx mlx-lm
```

That's it. No CUDA toolkit, no cuDNN, no driver compatibility nightmares.

## LoRA Fine-Tuning with MLX

### Step 1: Get a Model

MLX uses its own format, but the `mlx-community` on Hugging Face has pre-converted versions of most popular models:

```bash
# Download a pre-quantized model ready for MLX
pip install huggingface-hub
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

### Step 2: Prepare Training Data

MLX expects JSONL format with `train.jsonl`, `valid.jsonl`, and optionally `test.jsonl`:

```json
{"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is Python?\n<|start_header_id|>assistant<|end_header_id|>\nPython is a high-level programming language...<|eot_id|>"}
{"text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nExplain closures.\n<|start_header_id|>assistant<|end_header_id|>\nA closure is a function that...<|eot_id|>"}
```

### Step 3: Train

```bash
mlx_lm.lora \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --data ./training_data \
    --train \
    --batch-size 4 \
    --lora-layers 16 \
    --iters 1000 \
    --learning-rate 1e-5 \
    --adapter-path ./my_adapter
```

Key parameters:
- **batch-size**: How many examples to process at once. Higher = faster but more memory. 4 is safe for 8B on 64GB.
- **lora-layers**: How many transformer layers get LoRA adapters. 16 is a good default.
- **iters**: Number of training iterations. Start with 600-1000, check validation loss.
- **learning-rate**: How fast the model learns. 1e-5 is conservative and safe.

### Step 4: Test Your Fine-Tuned Model

```bash
mlx_lm.generate \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --adapter-path ./my_adapter \
    --prompt "Your test prompt here"
```

### Step 5: Merge and Convert for Ollama

Once you're happy with the results, merge the adapter into the base model and convert to GGUF for Ollama:

```bash
# Merge adapter into base model
mlx_lm.fuse \
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --adapter-path ./my_adapter \
    --save-path ./fused_model

# Convert to GGUF (for Ollama)
# Use llama.cpp's convert script
python convert_hf_to_gguf.py ./fused_model --outtype q8_0 --outfile my_model.gguf

# Import into Ollama
ollama create my-assistant -f Modelfile
```

## Expected Training Times on Apple Silicon (64GB)

| Model Size | Dataset Size | LoRA Layers | Estimated Time |
|-----------|-------------|-------------|----------------|
| 8B (4-bit) | 5,000 examples | 16 | ~1-2 hours |
| 8B (4-bit) | 50,000 examples | 16 | ~6-12 hours |
| 8B (8-bit) | 5,000 examples | 16 | ~2-3 hours |
| 13B (4-bit) | 5,000 examples | 16 | ~3-5 hours |

## Monitoring Training

Watch for these signals:
- **Training loss decreasing**: Good — the model is learning
- **Validation loss decreasing**: Great — it's learning generalizable patterns
- **Validation loss increasing while training loss decreases**: Overfitting! Stop training or reduce epochs
- **Both losses flat**: Learning rate might be too low, or the model already knows this stuff
