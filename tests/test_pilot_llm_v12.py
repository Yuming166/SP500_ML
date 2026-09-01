"""Offline regressions for the frozen four-worker V12 replication."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from sp500_forecastability import pilot_llm_v12 as v12


@lru_cache(maxsize=1)
def _composites():
    v12.configure_v12()
    return tuple(v12.build_remaining_composites(
        v12.base.load_boolq(v12.DEFAULT_DATASET)
    ))


def test_v12_exhausts_only_untouched_eligible_roots() -> None:
    comps = _composites()
    assert len(comps) == 358
    assert Counter(comp.label for comp in comps) == Counter({"yes": 246, "no": 112})
    roots = {comp.items[0].source_root for comp in comps}
    assert len(roots) == 358
    assert roots.isdisjoint(v12._parent_roots())
    assert all(comp.cqid.startswith("v12-") for comp in comps)


def test_v12_four_shards_are_balanced_question_blocks() -> None:
    shards = v12.assign_shards(_composites())
    assert [len(shard) for shard in shards] == [90, 90, 89, 89]
    assert [Counter(comp.label for comp in shard) for shard in shards] == [
        Counter({"yes": 62, "no": 28}),
        Counter({"yes": 62, "no": 28}),
        Counter({"yes": 61, "no": 28}),
        Counter({"yes": 61, "no": 28}),
    ]
    assert sum(len(shard) * 5 * 4 for shard in shards) == 7160


def test_v12_manifest_freezes_parallel_and_single_primary_contracts() -> None:
    comps = _composites()
    manifest = v12.build_manifest_v12(
        v12.DEFAULT_DATASET, comps, {}, status="test",
    )
    restored = v12.validate_manifest_v12(
        manifest, v12.DEFAULT_DATASET, require_substitutes=False,
    )
    assert len(restored) == 358
    assert manifest["co_primary_endpoints"] == []
    assert manifest["primary_endpoint"] == v12.PRIMARY_ENDPOINT
    assert manifest["parallel_contract"]["workers"] == 4
    assert manifest["parallel_contract"]["physical_gpu_count_verified"] is False
    assert manifest["smoke_contract"]["logical_calls"] == 80
    assert manifest["risk_contract"]["weights"] == {
        "D_inert": 0.1, "flip_inertia": 0.3, "frac_shared": 0.6,
    }


def test_v12_risk_score_remains_outcome_invariant() -> None:
    v12.configure_v12()
    row = {
        "D_inert": 0.6, "flip_inertia": 0.7, "frac_shared": 0.4,
        "any_wrong": 0, "correct": 1, "gold_binary": 1,
        "harmful_fc": 0, "label": "yes",
    }
    before = v12.v11.risk_score_from_preoutcome(row)
    row.update({
        "any_wrong": 1, "correct": 0, "gold_binary": 0,
        "harmful_fc": 1, "label": "no",
    })
    assert v12.v11.risk_score_from_preoutcome(row) == before
    assert before == 0.1 * 0.6 + 0.3 * 0.7 + 0.6 * 0.4


def test_v12_repair_suffix_is_bounded_and_never_truncates() -> None:
    short = "one two three four five six seven eight nine ten"
    rewrite, mode = v12.normalize_repaired_candidate(short, source_tokens=40)
    assert rewrite is not None
    assert mode == "repair_neutral_suffix_x2"
    assert 20 <= len(rewrite.split()) <= 60

    overlong, mode = v12.normalize_repaired_candidate(short, source_tokens=4)
    assert overlong is None
    assert mode == "repair_overlong"
