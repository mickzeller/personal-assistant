"""
Conversation Logger — silently records your chat sessions for future fine-tuning.

WHY THIS EXISTS:
────────────────
Every time you chat with the assistant, the exchanges get saved to a JSONL file.
Over weeks/months of use, this builds a natural dataset of:
  - How you phrase requests
  - What kinds of questions you ask
  - Which responses you liked (kept chatting) vs. didn't (changed topic)

When it's time to fine-tune again, you can cherry-pick the best exchanges
from these logs and add them to your training data.

PRIVACY:
────────
All logs stay local in data/conversations/. Nothing leaves your machine.
Each session gets its own file, named by date and time.
"""

import json
from pathlib import Path
from datetime import datetime

from configs.config import PROJECT_ROOT


CONVERSATIONS_DIR = PROJECT_ROOT / "data" / "conversations"


class ConversationLogger:
    """
    Logs chat exchanges to a timestamped JSONL file, one exchange per line.
    Call close() at the end to write the session summary.
    """

    def __init__(self, model: str = "unknown"):
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        timestamp = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = CONVERSATIONS_DIR / f"{timestamp}.jsonl"
        self.model = model
        self.exchange_count = 0

        self._write_entry({
            "type": "session_start",
            "timestamp": self.session_start.isoformat(),
            "model": model,
        })

    def log_exchange(
        self,
        user_message: str,
        assistant_response: str,
        rag_used: bool = False,
        sources: list[str] = None,
    ):
        """Log a single user/assistant exchange."""
        self.exchange_count += 1

        entry = {
            "type": "exchange",
            "timestamp": datetime.now().isoformat(),
            "exchange_number": self.exchange_count,
            "user": user_message,
            "assistant": assistant_response,
            "rag_used": rag_used,
        }

        if sources:
            entry["sources"] = sources

        self._write_entry(entry)

    def close(self):
        """Write session summary and close."""
        self._write_entry({
            "type": "session_end",
            "timestamp": datetime.now().isoformat(),
            "total_exchanges": self.exchange_count,
            "model": self.model,
        })

    def _write_entry(self, entry: dict):
        """Append a JSON line. Fails silently — logging should never crash the chat."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except OSError:
            pass  # disk full, permissions, etc.
