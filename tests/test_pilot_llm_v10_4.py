"""Tests for V10.4's deterministic, outcome-independent short normalization."""

from __future__ import annotations

from sp500_forecastability import pilot_llm_v10_4 as v10_4


def test_short_normalization_preserves_candidate_and_restores_window() -> None:
    candidate = "A counterfactual sentence has ten simple tokens before normalization."
    rewrite, mode = v10_4.normalize_short_candidate(candidate, source_tokens=24)

    assert mode == "deterministic_short_normalization"
    assert rewrite == "A counterfactual sentence has ten simple tokens before normalization in the described local situation."
    assert 0.5 <= len(rewrite.split()) / 24 <= 1.5


def test_short_normalization_rejects_overlong_candidate() -> None:
    rewrite, mode = v10_4.normalize_short_candidate("one two three four five", source_tokens=2)
    assert rewrite is None
    assert mode == "failed_overlong"
