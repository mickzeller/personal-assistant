"""
Automated style metrics — cheap, deterministic numbers (length, exclamation
density, contraction ratio) that complement the subjective LLM-as-a-Judge
scores and catch mechanical drift like mid-thought cutoffs. Stdlib only.
"""

import re


# ──────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")

_SENTENCE_END_RE = re.compile(r"[.!?…]+")

# "didn't", "I'll", "so's" — the 's branch also matches possessives,
# so counts skew slightly high (consistently, so comparisons hold).
_CONTRACTION_RE = re.compile(
    r"\b[A-Za-z]+n['’]t\b|\b[A-Za-z]+['’](?:re|ve|ll|d|m|s)\b",
    re.IGNORECASE,
)

# Expanded forms that could have been contractions ("do not", "I will") —
# the denominator for contraction_ratio.
_EXPANDED_RE = re.compile(
    r"\b(?:do|does|did|is|are|was|were|has|have|had|would|will|"
    r"can|could|should|must)\s+not\b"
    r"|\b(?:i|you|we|they|he|she|it|that|there|who)\s+(?:will|would|have|had)\b"
    r"|\bi\s+am\b"
    r"|\b(?:you|we|they)\s+are\b"
    r"|\b(?:it|that|there|he|she|who|what)\s+is\b",
    re.IGNORECASE,
)

# Symbols, dingbats, and the main emoji planes. Excludes U+FE0F
# (variation selector) so "❤️" counts once.
_EMOJI_CLASS = "☀-➿⬀-⯿\U0001F000-\U0001FAFF"

_EMOJI_RE = re.compile(f"[{_EMOJI_CLASS}]")

# Terminal punctuation (optionally closed by a quote/paren) or a trailing
# emoji — anything else likely got cut off mid-thought.
_CLEAN_ENDING_RE = re.compile(
    r"(?:[.!?…][\"'”’)\]]?"
    f"|[{_EMOJI_CLASS}]️?)\\s*$"
)


# ──────────────────────────────────────────────────────────────
# Per-response metrics
# ──────────────────────────────────────────────────────────────

def compute_style_metrics(text: str) -> dict:
    """
    Compute style metrics for a single response: word/sentence counts,
    exclamation and contraction density per 100 words, contraction_ratio
    (contractions vs. expanded forms; None if neither appears), emoji and
    em-dash counts, and ends_cleanly (False = likely truncated).
    """
    text = text.strip()

    if not text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "exclamations_per_100_words": 0.0,
            "contractions_per_100_words": 0.0,
            "contraction_ratio": None,
            "emoji_count": 0,
            "em_dash_count": 0,
            "ends_cleanly": False,
        }

    words = _WORD_RE.findall(text)
    word_count = len(words)
    sentence_count = max(1, len(_SENTENCE_END_RE.findall(text)))

    exclamation_count = text.count("!")
    contraction_count = len(_CONTRACTION_RE.findall(text))
    expanded_count = len(_EXPANDED_RE.findall(text))

    per_100 = (100 / word_count) if word_count else 0

    contractable = contraction_count + expanded_count
    contraction_ratio = (
        round(contraction_count / contractable, 2) if contractable else None
    )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(word_count / sentence_count, 1),
        "exclamations_per_100_words": round(exclamation_count * per_100, 1),
        "contractions_per_100_words": round(contraction_count * per_100, 1),
        "contraction_ratio": contraction_ratio,
        "emoji_count": len(_EMOJI_RE.findall(text)),
        "em_dash_count": text.count("—"),
        "ends_cleanly": bool(_CLEAN_ENDING_RE.search(text)),
    }


# ──────────────────────────────────────────────────────────────
# Aggregation across a run
# ──────────────────────────────────────────────────────────────

def aggregate_style_metrics(metrics_list: list[dict]) -> dict:
    """
    Average per-response metrics across a run. contraction_ratio skips
    Nones; ends_cleanly becomes a truncated_responses count.
    """
    if not metrics_list:
        return {}

    def _avg(key, digits=1):
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        return round(sum(values) / len(values), digits) if values else None

    return {
        "avg_word_count": _avg("word_count"),
        "avg_sentence_length": _avg("avg_sentence_length"),
        "avg_exclamations_per_100_words": _avg("exclamations_per_100_words"),
        "avg_contractions_per_100_words": _avg("contractions_per_100_words"),
        "avg_contraction_ratio": _avg("contraction_ratio", digits=2),
        "avg_emoji_count": _avg("emoji_count"),
        "total_em_dashes": sum(m["em_dash_count"] for m in metrics_list),
        "truncated_responses": sum(
            1 for m in metrics_list if not m["ends_cleanly"]
        ),
        "responses_measured": len(metrics_list),
    }
