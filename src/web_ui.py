"""
Gradio web UI (Phase 4) — ChatAssistant in the browser.

Streaming responses, a RAG toggle, and the same conversation logging
as the CLI chat. Gradio owns the visible history (so clear/retry/edit
work natively); each turn rebuilds the assistant's history from it.
"""

import gradio as gr
from rich.console import Console

from configs.config import ollama_config, retrieval_config
from src.chat import ChatAssistant

console = Console()


def create_app(assistant: ChatAssistant, rag_default: bool = True) -> gr.ChatInterface:
    """Build the Gradio app around an existing ChatAssistant."""

    def respond(message, history, use_rag):
        # Sync assistant history to what's on screen — keeps retry/edit/clear honest
        assistant.conversation_history = [
            {"role": h["role"], "content": h["content"]}
            for h in history
            if isinstance(h.get("content"), str)
        ]
        yield from assistant.ask_stream(message, force_rag=use_rag)

    rag_available = assistant.retriever is not None

    return gr.ChatInterface(
        respond,
        title="Personal AI Assistant",
        description=(
            f"Model: `{assistant.model}` · "
            f"Knowledge base: `{retrieval_config.collection_name}` · "
            f"RAG: {'available' if rag_available else 'unavailable (started with --no-rag)'}"
        ),
        additional_inputs=[
            gr.Checkbox(
                value=rag_default and rag_available,
                label="Use RAG retrieval",
                interactive=rag_available,
            ),
        ],
    )


def launch_ui(
    model: str = None,
    rag_enabled: bool = True,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Start the web UI. Blocks until Ctrl+C."""
    assistant = ChatAssistant(model=model, rag_enabled=rag_enabled)

    try:
        assistant._client.list()
    except Exception:
        console.print("[red bold]Can't connect to Ollama![/red bold]")
        console.print(f"Make sure it's running at {ollama_config.base_url}: [cyan]ollama serve[/cyan]")
        raise SystemExit(1)

    app = create_app(assistant, rag_default=rag_enabled)

    console.print(f"[green]Starting web UI on http://localhost:{port}[/green]")
    app.launch(server_port=port, share=share, inbrowser=False)
