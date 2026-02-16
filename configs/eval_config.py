"""
Evaluation configuration for the LLM-as-a-Judge pipeline.

Uses Claude's API to score fine-tuned model responses against
Mick's writing style profile. Each response gets scored on five
dimensions (1-5 scale), giving us trackable metrics across
training runs instead of just eyeballing side-by-side outputs.

The writing style profile is loaded at runtime from ~/.claude/CLAUDE.md
so personal data stays out of version control.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────
# Writing style profile loader
# ──────────────────────────────────────────────────────────────

CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"

# Sections from CLAUDE.md that are relevant to writing style evaluation.
# Everything from "# About Mick" through "## Context-Specific Tones" —
# covers tone, vocabulary, structure, examples, and context-shifting.
# We stop before "## Working with Claude on Code" since that's Claude Code
# instructions, not writing style data.
_STYLE_END_MARKER = "## Working with Claude on Code"


def load_writing_style_profile(path: Path = CLAUDE_MD_PATH) -> str:
    """
    Load Mick's writing style profile from ~/.claude/CLAUDE.md.

    Extracts everything from the start of the file through the end
    of the context-specific tones section (stopping before the
    "Working with Claude on Code" section, which is Claude Code
    configuration, not writing style).

    Returns the profile text, or a fallback message if the file
    can't be found.
    """
    if not path.exists():
        return (
            "[Writing style profile not found. "
            f"Expected at: {path}. "
            "Evaluation will still work but scores may be less accurate "
            "without the reference profile.]"
        )

    content = path.read_text(encoding="utf-8")

    # Trim at the marker — everything before it is writing style
    marker_pos = content.find(_STYLE_END_MARKER)
    if marker_pos != -1:
        content = content[:marker_pos].rstrip()

    return content


# ──────────────────────────────────────────────────────────────
# Rubric dimensions
# ──────────────────────────────────────────────────────────────

RUBRIC_DIMENSIONS = {
    "tone": {
        "name": "Tone",
        "description": (
            "How well the response matches Mick's casual, warm, direct tone. "
            "Score 5 = sounds exactly like him. Score 1 = stiff, corporate, "
            "or trying too hard to be casual."
        ),
    },
    "word_choice": {
        "name": "Word Choice",
        "description": (
            "Uses Mick's actual vocabulary: contractions (so's, didn't, we'll), "
            "phrases (a ton of, keeping it rolling, my man), natural openers "
            "(Hey!, Haha). Score 5 = nails his word choices. "
            "Score 1 = generic AI vocabulary."
        ),
    },
    "structure": {
        "name": "Structure",
        "description": (
            "Gets to the point, opens with a reaction/acknowledgment, closes "
            "briefly and warmly. Doesn't over-explain or add unnecessary caveats. "
            "Score 5 = structured like Mick writes. "
            "Score 1 = too long, buried the point, or overly formal structure."
        ),
    },
    "authenticity": {
        "name": "Authenticity",
        "description": (
            "Sounds like a real person, not an AI trying to be casual. "
            "Natural flow, not forced humor or fake enthusiasm. "
            "Score 5 = genuinely human. Score 1 = clearly AI-generated."
        ),
    },
    "context_fit": {
        "name": "Context Fit",
        "description": (
            "Tone-shifts appropriately for the context. Chill for friends, "
            "measured for work, warm for church, direct for technical. "
            "Score 5 = perfect register for the situation. "
            "Score 1 = same tone regardless of audience."
        ),
    },
}


class EvalConfig(BaseSettings):
    """
    Configuration for the Claude API evaluation pipeline.

    Uses Sonnet for judging — fast and cheap, no need for Opus
    when we just need structured scores. Each evaluation run
    costs roughly $0.01.
    """

    # Claude model used as the judge
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Where to save evaluation results
    eval_results_dir: str = "data/training"
    eval_results_filename: str = "eval_results.json"

    # Where comparison results live (input to evaluator)
    comparison_results_filename: str = "comparison_results.json"

    # Max tokens for Claude's scoring response
    max_tokens: int = 1024


# ──────────────────────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────────────────────

eval_config = EvalConfig()
