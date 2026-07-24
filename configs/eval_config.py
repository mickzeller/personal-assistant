"""
Config for the LLM-as-a-Judge pipeline — Claude scores responses against
my writing style on 5 dimensions (1-5), so we get trackable numbers
across training runs instead of just eyeballing outputs.

The style profile loads at runtime from ~/.claude/CLAUDE.md so personal
data stays out of version control.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"

# Everything before this marker is writing style; after it is
# Claude Code config we don't want in the judge prompt
_STYLE_END_MARKER = "## Working with Claude on Code"


def load_writing_style_profile(path: Path = CLAUDE_MD_PATH) -> str:
    """
    Load the writing style profile from ~/.claude/CLAUDE.md.
    Returns a fallback message if the file's missing.
    """
    if not path.exists():
        return (
            "[Writing style profile not found. "
            f"Expected at: {path}. "
            "Evaluation will still work but scores may be less accurate "
            "without the reference profile.]"
        )

    content = path.read_text(encoding="utf-8")

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

    eval_results_dir: str = "data/training"
    eval_results_filename: str = "eval_results.json"

    # Input to the evaluator (from model_tester.py)
    comparison_results_filename: str = "comparison_results.json"

    # Max tokens for Claude's scoring response
    max_tokens: int = 1024


eval_config = EvalConfig()
