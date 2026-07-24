#!/usr/bin/env bash
# One-shot environment setup: creates .venv with uv and installs dependencies.
#
#   ./setup.sh
#
# Requires uv (https://docs.astral.sh/uv/). Install it with:
#   brew install uv        (macOS)
#   curl -LsSf https://astral.sh/uv/install.sh | sh   (anywhere else)

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed."
    echo "  brew install uv"
    echo "  or: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ -d .venv ]; then
    echo ".venv already exists — reusing it (delete it first for a fresh start)"
else
    echo "Creating virtual environment (.venv)..."
    uv venv
fi

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo ""
echo "Done! Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then take it for a spin:"
echo "  python cli.py --help"
