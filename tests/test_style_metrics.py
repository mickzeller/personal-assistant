"""
Tests for the automated style metrics.

Run with: pytest tests/ -v
"""

import pytest

from src.style_metrics import aggregate_style_metrics, compute_style_metrics


# ──────────────────────────────────────────────────────────────
# compute_style_metrics
# ──────────────────────────────────────────────────────────────

def test_empty_text_returns_zeroes():
    m = compute_style_metrics("")
    assert m["word_count"] == 0
    assert m["contraction_ratio"] is None
    assert m["ends_cleanly"] is False


def test_word_and_sentence_counts():
    m = compute_style_metrics("Hey man! Thanks for running the league. See you Sunday.")
    assert m["word_count"] == 10
    assert m["sentence_count"] == 3
    assert m["avg_sentence_length"] == pytest.approx(3.3)


def test_exclamation_density():
    # 2 exclamations in 10 words = 20 per 100 words
    m = compute_style_metrics("Yeah! That was awesome! We should do it again sometime.")
    assert m["exclamations_per_100_words"] == 20.0


def test_contractions_counted():
    m = compute_style_metrics("I'll check it out, didn't see that. So's my gratitude.")
    assert m["contractions_per_100_words"] > 0
    # 3 contractions, 0 expanded forms → ratio of 1.0
    assert m["contraction_ratio"] == 1.0


def test_contraction_ratio_with_expanded_forms():
    # 1 contraction ("don't") vs 1 expanded form ("I will") → 0.5
    m = compute_style_metrics("I don't mind. I will check on it tomorrow.")
    assert m["contraction_ratio"] == 0.5


def test_no_contractable_forms_gives_none():
    m = compute_style_metrics("Great work on the report.")
    assert m["contraction_ratio"] is None


def test_emoji_counted_once_with_variation_selector():
    m = compute_style_metrics("Merry Christmas my man! 🎄❤️")
    assert m["emoji_count"] == 2


def test_em_dashes_counted():
    m = compute_style_metrics("The sync only adds — it never cleans up — which is the bug.")
    assert m["em_dash_count"] == 2


def test_clean_ending_with_period():
    assert compute_style_metrics("All done.")["ends_cleanly"] is True


def test_clean_ending_with_emoji():
    assert compute_style_metrics("Merry Christmas my man!🎄")["ends_cleanly"] is True


def test_clean_ending_with_closing_quote():
    assert compute_style_metrics('He said "sounds good."')["ends_cleanly"] is True


def test_truncated_response_detected():
    # The Mistral failure mode: response cuts off mid-thought
    m = compute_style_metrics("Thanks for reaching out, I wanted to say that the")
    assert m["ends_cleanly"] is False


# ──────────────────────────────────────────────────────────────
# aggregate_style_metrics
# ──────────────────────────────────────────────────────────────

def test_aggregate_empty_list():
    assert aggregate_style_metrics([]) == {}


def test_aggregate_averages_and_counts():
    metrics = [
        compute_style_metrics("Yeah! Love it. Thanks man!"),
        compute_style_metrics("I will review the ticket and we can"),  # truncated
    ]
    agg = aggregate_style_metrics(metrics)
    assert agg["responses_measured"] == 2
    assert agg["truncated_responses"] == 1
    assert agg["avg_word_count"] > 0


def test_aggregate_skips_none_ratios():
    metrics = [
        compute_style_metrics("Great work on the report."),  # no contractables → None
        compute_style_metrics("I don't mind. I will check tomorrow."),  # 0.5
    ]
    agg = aggregate_style_metrics(metrics)
    assert agg["avg_contraction_ratio"] == 0.5
