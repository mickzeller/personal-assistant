"""
Chat module — ties RAG retrieval to Ollama generation.

Retrieve relevant chunks, build the prompt, stream the response.
RAG is toggled with /rag (a classifier could make this smarter someday).
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
    The main orchestrator — conversation history, RAG retrieval, and
    Ollama generation. Call run() for the interactive loop.
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

        if self.rag_enabled:
            self.retriever = Retriever()
        else:
            self.retriever = None

        # Saves exchanges for future fine-tuning
        self.logger = ConversationLogger(model=self.model)

    def _build_messages(
        self,
        query: str,
        context_chunks: list[RetrievedChunk] | None = None,
    ) -> list[dict]:
        """
        Build the chat API message array. If RAG found relevant chunks,
        they get injected into the current user message via context_template.
        """
        messages = [{"role": "system", "content": prompt_config.system_prompt}]

        messages.extend(self.conversation_history)

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
        """Send messages to Ollama and stream the response token-by-token."""
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

        console.print()

        return full_response

    def ask(self, query: str, force_rag: bool = None) -> str:
        """
        Ask a question, get a response. force_rag overrides the RAG
        setting for this one call (None = use current setting).
        """
        use_rag = force_rag if force_rag is not None else self.rag_enabled
        context_chunks = None

        if use_rag and self.retriever:
            context_chunks = self.retriever.search(query)

            if context_chunks:
                sources = set(c.source for c in context_chunks)
                top_score = max(c.rerank_score for c in context_chunks if c.rerank_score)
                console.print(
                    f"[dim]📚 Found {len(context_chunks)} relevant chunks "
                    f"from {', '.join(sources)} "
                    f"(top score: {top_score:.2f})[/dim]"
                )
            else:
                console.print("[dim]📚 No relevant context found — using general knowledge[/dim]")

        messages = self._build_messages(query, context_chunks)
        response = self._stream_response(messages)

        self._record_exchange(query, response, context_chunks)

        return response

    def ask_stream(self, query: str, force_rag: bool = None):
        """
        Generator version of ask() for the web UI — yields the accumulated
        response as tokens arrive, no console output.
        """
        use_rag = force_rag if force_rag is not None else self.rag_enabled
        context_chunks = None

        if use_rag and self.retriever:
            context_chunks = self.retriever.search(query)

        messages = self._build_messages(query, context_chunks)

        full_response = ""
        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                full_response += chunk["message"]["content"]
                yield full_response
        except Exception as e:
            yield full_response or f"⚠️ Error from Ollama: {e}"
            return

        self._record_exchange(query, full_response, context_chunks)

    def _record_exchange(self, query, response, context_chunks):
        """Append to history (original query, not the context-injected
        version), log for future fine-tuning, and cap history at 10 turns."""
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        sources = [c.source for c in context_chunks] if context_chunks else []
        self.logger.log_exchange(
            user_message=query,
            assistant_response=response,
            rag_used=bool(context_chunks),
            sources=list(set(sources)),
        )

        max_history = 20
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

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
                console.print()
                query = console.input("[bold green]You:[/bold green] ").strip()

                if not query:
                    continue

                if query.startswith("/"):
                    self._handle_command(query)
                    continue

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
                # Make sure it exists in Ollama before switching
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
