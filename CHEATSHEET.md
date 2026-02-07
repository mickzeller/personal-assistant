# LLM Discovery — Quick Reference

> All commands assume you're in the project root and the venv is active.

```bash
cd <project-root>   # wherever you cloned LLM_Discovery
source .venv/bin/activate
```

---

## Chat

```bash
# Start the assistant (default: llama3.1 + RAG)
python -m src.chat

# Use a different model
python -m src.chat --model gemma3:27b

# Chat without RAG (general conversation only)
python -m src.chat --no-rag
```

### Chat Commands (inside the chat)

| Command             | What it does                          |
|---------------------|---------------------------------------|
| `/help`             | Show all commands                     |
| `/rag on` / `off`   | Toggle knowledge base retrieval       |
| `/sources`          | Show what's in the knowledge base     |
| `/model gemma3:27b` | Switch models mid-conversation        |
| `/clear`            | Clear conversation history            |
| `/quit`             | Exit                                  |

---

## Knowledge Base (Ingestion)

```bash
# Ingest everything in knowledge_base/
python -m src.ingest

# Ingest a specific directory
python -m src.ingest /path/to/some/folder
```

### Supported file types
`.md` `.txt` `.pdf` `.json` `.jsonl` `.docx`

### Quick utilities

```bash
# Check what's in the database
python -c "from src.ingest import get_collection_stats; stats = get_collection_stats(); print(f'Chunks: {stats[\"total_chunks\"]}'); [print(f'  • {s}') for s in stats['sources']]"

# Clear the entire database (re-ingest after!)
python -c "from src.ingest import clear_collection; clear_collection()"
```

---

## Training Data (Fine-Tuning Prep)

```bash
# Parse training data → JSONL for fine-tuning
python -m src.training_data_parser

# Output goes to: data/training/writing_style.jsonl
```

### Adding new training data

1. Drop files in `knowledge_base/training_data/`
2. Use descriptive names: `sms-topic-description.md`, `slack-topic.docx`
3. Re-run the parser: `python -m src.training_data_parser`

### File naming convention

| Prefix      | What it is                        |
|-------------|-----------------------------------|
| `sms-`      | Text message refinement           |
| `slack-`    | Raw Slack messages (your words)   |
| `work-`     | Work/ticket/PR comments           |
| `church-`   | Church-related writing            |
| `review-`   | Product/business reviews          |
| `linkedin-` | LinkedIn posts                    |

---

## Conversation Logs

Every chat session auto-saves to `data/conversations/`. These are future fine-tuning data.

```bash
# View recent conversation logs
ls -lt data/conversations/

# Check total logged exchanges
python -c "
from pathlib import Path
import json
logs = list(Path('data/conversations').glob('*.jsonl'))
total = sum(1 for f in logs for _ in f.open())
print(f'{len(logs)} sessions, {total} exchanges')
"
```

To promote a conversation to training data, copy the good exchanges from the log into a new file in `knowledge_base/training_data/`.

---

## Models (Ollama)

```bash
# List installed models
ollama list

# Pull a new model
ollama pull <model_name>

# Models you have:
#   gemma3:27b              — daily driver (big, smart)
#   llama3.1:8b-instruct    — fast, good for testing
#   nomic-embed-text        — embeddings (used by RAG)
```

---

## Project Structure

```
LLM_Discovery/
├── src/
│   ├── chat.py                  # Interactive assistant
│   ├── ingest.py                # Document ingestion pipeline
│   ├── retriever.py             # RAG search + reranking
│   ├── embeddings.py            # Ollama embedding wrapper
│   └── training_data_parser.py  # Perplexity/DOCX → JSONL
├── configs/
│   └── config.py                # All settings (models, chunking, etc.)
├── knowledge_base/
│   ├── sample_docs/             # Committed sample docs
│   └── training_data/           # Your writing samples (gitignored)
├── data/
│   ├── chroma_db/               # Vector database (gitignored)
│   ├── training/                # Fine-tuning datasets (gitignored)
│   └── conversations/           # Chat session logs (gitignored)
├── PLAN.md                      # Full project plan
├── CHEATSHEET.md                # ← You are here
└── requirements.txt
```
