"""Regression test for V10.3's stricter exact-token prompt contract."""

from __future__ import annotations

from sp500_forecastability import pilot_llm_v10 as v10
from sp500_forecastability import pilot_llm_v10_3 as v10_3


def test_v10_3_targets_exact_source_token_count_for_initial_and_repair() -> None:
    item = v10.BoolQItem(
        qid="exact-e01", question="Is the claim true?",
        passage="This original evidence sentence has exactly eight stable tokens.",
        label="yes", source_root="source-exact", evidence_index=1,
        overlap_mean=0.2, overlap_max=0.3,
    )

    initial = v10_3._initial_prompt(item)
    repair = v10_3._length_repair_prompt(item, "too short")

    assert "exactly 9 tokens" in initial
    assert "exactly 9 tokens" in repair
    assert "Unusable prior candidate: too short" in repair
