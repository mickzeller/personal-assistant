"""
Chat Module — the brain that ties RAG retrieval to LLM generation.

HOW IT WORKS:
─────────────
User types a question
        │
        ▼
Retriever searches your knowledge base for relevant chunks
        │
        ▼
Prompt builder combines: system prompt + retrieved context + user question
        │
        ▼
Ollama generates a response using your local LLM
        │
        ▼
Response is streamed back to the terminal in real-time

The "smart" part is deciding WHETHER to use RAG at all. For general
conversation ("tell me a joke"), we skip retrieval. For knowledge
questions ("what does our MLX guide say about training times?"),
we retrieve first. Right now this is controlled by a simple keyword
check and a /rag toggle — but you could get fancier with a classifier.
"""

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from configs.config import ollama_config, prompt_config, retrieval_config
from src.retriever import Retriever, RetrievedChunk
from src.conversation_logger import ConversationLogger

console = Console()


class ChatAssistant:
    """
    Personal AI assistant with RAG-enhanced responses.

    This is the main orchestrator. It manages:
    - Conversation history (for multi-turn context)
    - RAG retrieval (searching your knowledge base)
    - LLM generation (via Ollama)
    - Response formatting

    Usage:
        assistant = ChatAssistant()
        assistant.run()  # Starts the interactive chat loop
    """

    def __init__(
        self,
        model: str = None,
        rag_enabled: bool = True,
    ):
        self.model = model or ollama_config.chat_model
        self.rag_enabled = rag_enabled
        self.conversation_history: list[dict] = []
        self._client = ollama.Client(host=ollama_config.base_url)

        # Initialize retriever (loads the reranker model)
        if self.rag_enabled:
            self.retriever = Retriever()
        else:
            self.retriever = None

        # Conversation logger — saves exchanges for future fine-tuning
        self.logger = ConversationLogger(model=self.model)

    def _build_messages(
        self,
        query: str,
        context_chunks: list[RetrievedChunk] | None = None,
    ) -> list[dict]:
        """
        Build the message array for the Ollama API call.

        The structure follows the chat API format:
        [
            {"role": "system", "content": "...system prompt..."},
            {"role": "user", "content": "...previous user message..."},
            {"role": "assistant", "content": "...previous response..."},
            {"role": "user", "content": "...current question with context..."},
        ]

        If RAG found relevant chunks, they're injected into the current
        user message via the context_template from config.
        """
        messages = [{"role": "system", "content": prompt_config.system_prompt}]

        # Add conversation history (keeps multi-turn context alive)
        messages.extend(self.conversation_history)

        # Build the current user message (with or without RAG context)
        if context_chunks:
            context_str = self.retriever.format_context(context_chunks)
            user_message = prompt_config.context_template.format(
                context=context_str,
                query=query,
            )
        else:
            user_message = prompt_config.no_context_template.format(query=query)

        messages.append({"role": "user", "content": user_message})

        return messages

    def _stream_response(self, messages: list[dict]) -> str:
        """
        Send messages to Ollama and stream the response token-by-token.

        Streaming means you see the response as it's generated, not all
        at once after a long wait. This makes the assistant feel much
        more responsive, especially for longer answers.
        """
        full_response = ""

        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                console.print(token, end="", highlight=False)

        except ConnectionError:
            console.print("[red]Can't connect to Ollama. Is it running? (ollama serve)[/red]")
            return ""
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                console.print(f"[red]Model '{self.model}' not found. Pull it with: ollama pull {self.model}[/red]")
            else:
                console.print(f"\n[red]Error from Ollama: {e}[/red]")
            return full_response or ""

        # Newline after the streamed response
        console.print()

        return full_response

    def ask(self, query: str, force_rag: bool = None) -> str:
        """
        Ask the assistant a question and get a response.

        Args:
            query:     The user's question
            force_rag: Override RAG behavior (True = always retrieve,
                       False = never retrieve, None = auto-decide)

        Returns:
            The assistant's response text
        """
        use_rag = force_rag if force_rag is not None else self.rag_enabled
        context_chunks = None

        # ── RAG Retrieval ─────────────────────────────────────────
        if use_rag and self.retriever:
            context_chunks = self.retriever.search(query)

            if context_chunks:
                # Show the user what sources we found
                sources = set(c.source for c in context_chunks)
                top_score = max(c.rerank_score for c in context_chunks if c.rerank_score)
                console.print(
                    f"[dim]📚 Found {len(context_chunks)} relevant chunks "
                    f"from {', '.join(sources)} "
                    f"(top score: {top_score:.2f})[/dim]"
                )
            else:
                console.print("[dim]📚 No relevant context found — using general knowledge[/dim]")

        # ── Generate Response ─────────────────────────────────────
        messages = self._build_messages(query, context_chunks)
        response = self._stream_response(messages)

        # ── Update Conversation History ───────────────────────────
        # Store the ORIGINAL query (not the context-injected version)
        # so conversation history stays clean and readable
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        # ── Log for future fine-tuning ──────────────────────────
        sources = [c.source for c in context_chunks] if context_chunks else []
        self.logger.log_exchange(
            user_message=query,
            assistant_response=response,
            rag_used=bool(context_chunks),
            sources=list(set(sources)),
        )

        # Keep history manageable (last 10 turns = 20 messages)
        max_history = 20
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

        return response

    def clear_history(self):
        """Clear conversation history (start fresh)."""
        self.conversation_history = []
        console.print("[yellow]Conversation history cleared.[/yellow]")

    def run(self):
        """
        Start the interactive chat loop.

        Commands:
            /quit, /exit    — Exit the chat
            /clear          — Clear conversation history
            /rag on|off     — Toggle RAG retrieval
            /sources        — Show what's in the knowledge base
            /model <name>   — Switch the chat model
            /help           — Show available commands
        """
        # Verify Ollama is running before we start
        try:
            self._client.list()
        except Exception:
            console.print("[red bold]Can't connect to Ollama![/red bold]")
            console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
            raise SystemExit(1)

        console.print(Panel(
            Text.from_markup(
                f"[bold green]Personal AI Assistant[/bold green]\n"
                f"Model: [cyan]{self.model}[/cyan]\n"
                f"RAG: [cyan]{'enabled' if self.rag_enabled else 'disabled'}[/cyan]\n"
                f"Knowledge base: [cyan]{retrieval_config.collection_name}[/cyan]\n\n"
                f"[dim]Type /help for commands, /quit to exit[/dim]"
            ),
            title="Welcome",
            border_style="green",
        ))

        while True:
            try:
                # Get user input
                console.print()
                query = console.input("[bold green]You:[/bold green] ").strip()

                if not query:
                    continue

                # ── Handle Commands ───────────────────────────────
                if query.startswith("/"):
                    self._handle_command(query)
                    continue

                # ── Ask the assistant ─────────────────────────────
                console.print(f"\n[bold blue]Assistant:[/bold blue]")
                self.ask(query)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit.[/yellow]")
            except EOFError:
                break

        self.logger.close()
        console.print(f"[dim]Session saved: {self.logger.log_file.name} ({self.logger.exchange_count} exchanges)[/dim]")
        console.print("\n[green]Goodbye![/green]")

    def _handle_command(self, command: str):
        """Process slash commands."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            raise EOFError

        elif cmd == "/clear":
            self.clear_history()

        elif cmd == "/rag":
            if arg.lower() in ("on", "true", "1"):
                self.rag_enabled = True
                if not self.retriever:
                    self.retriever = Retriever()
                console.print("[green]RAG enabled[/green]")
            elif arg.lower() in ("off", "false", "0"):
                self.rag_enabled = False
                console.print("[yellow]RAG disabled — using general knowledge only[/yellow]")
            else:
                status = "enabled" if self.rag_enabled else "disabled"
                console.print(f"RAG is currently [cyan]{status}[/cyan]. Use /rag on or /rag off")

        elif cmd == "/sources":
            from src.ingest import get_collection_stats
            stats = get_collection_stats()
            console.print(f"\n[bold]Knowledge Base Stats:[/bold]")
            console.print(f"  Total chunks: {stats['total_chunks']}")
            if stats['sources']:
                console.print(f"  Sources:")
                for src in stats['sources']:
                    console.print(f"    • {src}")

        elif cmd == "/model":
            if arg:
                # Verify the model exists in Ollama before switching
                try:
                    self._client.show(arg)
                    self.model = arg
                    self.logger.model = arg
                    console.print(f"[green]Switched to model: {self.model}[/green]")
                except Exception:
                    console.print(f"[red]Model '{arg}' not found in Ollama.[/red]")
                    console.print(f"[dim]Pull it first: ollama pull {arg}[/dim]")
            else:
                console.print(f"Current model: [cyan]{self.model}[/cyan]")
                console.print("[dim]Usage: /model gemma3:27b[/dim]")

        elif cmd == "/log":
            console.print(f"\n[bold]Conversation Log:[/bold]")
            console.print(f"  Session file: {self.logger.log_file.name}")
            console.print(f"  Exchanges so far: {self.logger.exchange_count}")
            console.print(f"  Logs directory: data/conversations/")

            # Count total across all sessions
            from pathlib import Path
            all_logs = list(self.logger.log_file.parent.glob("*.jsonl"))
            console.print(f"  Total session files: {len(all_logs)}")

        elif cmd == "/help":
            console.print(Panel(
                "[bold]/quit[/bold]          — Exit the chat\n"
                "[bold]/clear[/bold]         — Clear conversation history\n"
                "[bold]/rag on|off[/bold]    — Toggle RAG retrieval\n"
                "[bold]/sources[/bold]       — Show knowledge base contents\n"
                "[bold]/model <name>[/bold]  — Switch chat model (e.g. /model gemma3:27b)\n"
                "[bold]/log[/bold]           — Show conversation log status\n"
                "[bold]/help[/bold]          — Show this help",
                title="Commands",
                border_style="blue",
            ))

        else:
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for options.[/yellow]")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Personal AI Assistant")
    parser.add_argument("--model", default=None, help="Ollama model to use")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    args = parser.parse_args()

    assistant = ChatAssistant(
        model=args.model,
        rag_enabled=not args.no_rag,
    )
    assistant.run()
