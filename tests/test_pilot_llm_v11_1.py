"""Regression tests for V11.1's bounded auxiliary amendment."""

from __future__ import annotations

from sp500_forecastability import pilot_llm_v11_1 as v11_1


def test_v11_1_repeated_neutral_suffix_restores_short_window() -> None:
    candidate = "one two three four five six seven eight nine ten"
    rewrite, mode = v11_1.normalize_repaired_candidate(candidate, source_tokens=40)
    assert rewrite is not None
    assert mode == "v11_1_repair_neutral_suffix_x2"
    assert 20 <= len(rewrite.split()) <= 60


def test_v11_1_never_truncates_an_overlong_candidate() -> None:
    candidate = "one two three four five six seven eight nine ten"
    rewrite, mode = v11_1.normalize_repaired_candidate(candidate, source_tokens=4)
    assert rewrite is None
    assert mode == "v11_1_repair_overlong"


def test_v11_1_repairs_exactly_the_frozen_three_qids() -> None:
    assert v11_1.EXPECTED_REPAIR_QIDS == {
        "boolq-1e583402fdb107ef-e01",
        "boolq-095374f01188fd3e-e01",
        "boolq-2e961908ed018a35-e02",
    }
