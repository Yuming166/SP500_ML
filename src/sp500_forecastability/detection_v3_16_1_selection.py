"""Deterministic fresh-root selection for the V3.16.1 formal replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sp500_forecastability import detection_v3_16 as parent

PROTOCOL_VERSION = "detection-v3.16.1-vitaminc-fresh-root-selection-2026-09-05"
DATASET = parent.DATASET
ARCHIVE = parent.ARCHIVE
EXPECTED_ARCHIVE_SHA256 = parent.EXPECTED_ARCHIVE_SHA256
PREREGISTRATION = Path("docs/detection_v3_16_1_preregistration.md")
PARENT_SELECTION_MANIFEST = Path("results/detection_v3_16_development/selection_manifest.json")
DEFAULT_ROOT = Path("results/detection_v3_16_1")
SELECTION_MANIFEST = DEFAULT_ROOT / "selection_manifest.json"
SELECTION_AUDIT = DEFAULT_ROOT / "selection_audit.json"

PARENT_PAIRS = parent.TOTAL_PAIRS
FRESH_PAIRS = 573 - PARENT_PAIRS
FORMAL_ITEMS = FRESH_PAIRS * 2
CHARACTER_RATIO_MIN = parent.CHARACTER_RATIO_MIN
TOKEN_JACCARD_MIN = parent.TOKEN_JACCARD_MIN
DISTRACTOR_JACCARD_MAX = parent.DISTRACTOR_JACCARD_MAX
CLAIM_TOKENS = parent.CLAIM_TOKENS
EVIDENCE_TOKENS = parent.EVIDENCE_TOKENS

file_sha256 = parent.file_sha256
load_rows = parent.load_rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_parent_manifest() -> dict[str, Any]:
    if not PARENT_SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(PARENT_SELECTION_MANIFEST)
    payload = json.loads(PARENT_SELECTION_MANIFEST.read_text(encoding="utf-8"))
    audit = parent.audit_selection(payload)
    if not audit["passed"]:
        raise ValueError(f"parent selection audit failed: {audit['gates']}")
    if payload["protocol_version"] != parent.PROTOCOL_VERSION:
        raise ValueError("unexpected parent selection protocol")
    if len(payload["pairs"]) != PARENT_PAIRS:
        raise ValueError("parent selection pair count drifted")
    return payload


def _parent_pages(payload: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    target_pages = {str(row["page"]) for row in payload["pairs"]}
    distractor_pages = {str(row["distractor_page"]) for row in payload["pairs"]}
    return target_pages, distractor_pages, target_pages | distractor_pages


def _pair_signature(pair: parent.ContrastivePair) -> tuple[str, str, str, str, str]:
    return (
        pair.pair_id,
        pair.case_id,
        pair.page,
        pair.supports_id,
        pair.refutes_id,
    )


def _manifest_pair_signature(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["pair_id"]),
        str(row["case_id"]),
        str(row["page"]),
        str(row["supports_id"]),
        str(row["refutes_id"]),
    )


def _assert_parent_prefix(
    eligible: Sequence[parent.ContrastivePair], parent_manifest: Mapping[str, Any]
) -> None:
    expected = [_pair_signature(pair) for pair in eligible[:PARENT_PAIRS]]
    actual = [
        _manifest_pair_signature(row)
        for row in parent_manifest["pairs"]
    ]
    if actual != expected:
        raise ValueError("parent manifest is not the frozen deterministic prefix")


def _pair_row(
    pair: parent.ContrastivePair,
    distractor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "split": "formal",
        "pair_id": pair.pair_id,
        "case_id": pair.case_id,
        "page": pair.page,
        "wiki_revision_id": pair.wiki_revision_id,
        "claim_sha256": hashlib.sha256(pair.claim.encode()).hexdigest(),
        "supports_id": pair.supports_id,
        "refutes_id": pair.refutes_id,
        "distractor_id": str(distractor["unique_id"]),
        "distractor_page": str(distractor["page"]),
        "character_ratio": pair.character_ratio,
        "token_jaccard": pair.token_jaccard,
        "distractor_claim_jaccard": parent.token_jaccard(
            pair.claim, str(distractor["evidence"])
        ),
        "items": [
            {
                "item_id": f"{pair.pair_id}:support",
                "gold_label": "SUPPORTS",
                "original_id": pair.supports_id,
                "reverse_id": pair.refutes_id,
            },
            {
                "item_id": f"{pair.pair_id}:refute",
                "gold_label": "REFUTES",
                "original_id": pair.refutes_id,
                "reverse_id": pair.supports_id,
            },
        ],
    }


def build_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_manifest = _load_parent_manifest()
    eligible = parent.one_pair_per_page(parent.candidate_pairs(rows))
    if len(eligible) != PARENT_PAIRS + FRESH_PAIRS:
        raise ValueError(f"eligible page roots drifted: {len(eligible)}")
    _assert_parent_prefix(eligible, parent_manifest)

    old_target_pages, old_distractor_pages, old_pages = _parent_pages(parent_manifest)
    fresh_pairs = list(eligible[PARENT_PAIRS:])
    if len(fresh_pairs) != FRESH_PAIRS:
        raise ValueError("fresh pair count drifted")
    fresh_rows = [row for row in rows if str(row["page"]) not in old_pages]
    distractors = parent.assign_distractors(fresh_pairs, fresh_rows)
    pair_rows = [_pair_row(pair, distractors[pair.pair_id]) for pair in fresh_pairs]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v3_16_1_model_call",
        "dataset": {"path": str(DATASET), "sha256": file_sha256(DATASET)},
        "archive": {"path": str(ARCHIVE), "sha256": file_sha256(ARCHIVE)},
        "parent_selection": {
            "path": str(PARENT_SELECTION_MANIFEST),
            "sha256": file_sha256(PARENT_SELECTION_MANIFEST),
            "protocol_version": parent.PROTOCOL_VERSION,
            "consumed_pairs": PARENT_PAIRS,
            "target_pages": sorted(old_target_pages),
            "distractor_pages": sorted(old_distractor_pages),
            "excluded_pages": sorted(old_pages),
        },
        "selection_rule": {
            "character_ratio_min": CHARACTER_RATIO_MIN,
            "token_jaccard_min": TOKEN_JACCARD_MIN,
            "distractor_claim_jaccard_max": DISTRACTOR_JACCARD_MAX,
            "claim_tokens": list(CLAIM_TOKENS),
            "evidence_tokens": list(EVIDENCE_TOKENS),
            "one_pair_per_page": True,
            "selection_uses_model_outputs": False,
            "selection_uses_formal_outcomes": False,
            "distractors_exclude_parent_pages": True,
            "targets_exclude_parent_target_pages": True,
            "targets_may_overlap_parent_distractor_pages": True,
        },
        "counts": {
            "candidate_pairs": len(parent.candidate_pairs(rows)),
            "eligible_page_roots": len(eligible),
            "parent_pairs": PARENT_PAIRS,
            "fresh_pairs": FRESH_PAIRS,
            "formal_pairs": FRESH_PAIRS,
            "formal_items": FORMAL_ITEMS,
        },
        "pairs": pair_rows,
        "target_pages_overlapping_parent_distractors": sorted(
            {row["page"] for row in pair_rows} & old_distractor_pages
        ),
        "claim_boundary": {
            "selection_uses_dataset_labels_for_balance": True,
            "selection_uses_model_outputs": False,
            "selection_uses_formal_outcomes": False,
            "formal_calls_authorized": False,
            "formal_outcomes_accessed": False,
        },
    }


def audit_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(payload["pairs"])
    target_pages = [str(row["page"]) for row in pairs]
    distractor_pages = [str(row["distractor_page"]) for row in pairs]
    item_ids = [str(item["item_id"]) for row in pairs for item in row["items"]]
    labels = [str(item["gold_label"]) for row in pairs for item in row["items"]]
    parent_selection = payload["parent_selection"]
    excluded_pages = {str(page) for page in parent_selection["excluded_pages"]}
    label_counts = {
        label: labels.count(label) for label in ("SUPPORTS", "REFUTES")
    }
    gates = {
        "protocol_version": payload["protocol_version"] == PROTOCOL_VERSION,
        "status_is_pre_model_call": payload["status"]
        == "frozen_before_any_v3_16_1_model_call",
        "exact_fresh_pair_count": len(pairs) == FRESH_PAIRS,
        "all_pairs_formal": all(row["split"] == "formal" for row in pairs),
        "target_pages_unique": len(target_pages) == len(set(target_pages)),
        "distractor_pages_unique": len(distractor_pages) == len(set(distractor_pages)),
        "target_distractor_pages_disjoint": not (
            set(target_pages) & set(distractor_pages)
        ),
        "item_ids_unique": len(item_ids) == len(set(item_ids)),
        "labels_exactly_balanced": label_counts == {
            "SUPPORTS": FRESH_PAIRS,
            "REFUTES": FRESH_PAIRS,
        },
        "natural_reverse_is_exact_swap": all(
            row["items"][0]["original_id"] == row["items"][1]["reverse_id"]
            and row["items"][0]["reverse_id"] == row["items"][1]["original_id"]
            for row in pairs
        ),
        "contrastive_gate": all(
            float(row["character_ratio"]) >= CHARACTER_RATIO_MIN
            and float(row["token_jaccard"]) >= TOKEN_JACCARD_MIN
            for row in pairs
        ),
        "distractor_gate": all(
            float(row["distractor_claim_jaccard"]) <= DISTRACTOR_JACCARD_MAX
            for row in pairs
        ),
        "parent_exclusion_count": len(excluded_pages) == 2 * PARENT_PAIRS,
        "fresh_targets_exclude_parent_targets": not (
            set(target_pages) & set(parent_selection["target_pages"])
        ),
        "fresh_distractors_exclude_parent": not (
            set(distractor_pages) & excluded_pages
        ),
        "dataset_hash": payload["dataset"]["sha256"] == file_sha256(DATASET),
        "archive_hash": payload["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256,
        "selection_outcome_free": payload["selection_rule"]["selection_uses_model_outputs"]
        is False
        and payload["selection_rule"]["selection_uses_formal_outcomes"] is False,
        "formal_calls_not_authorized": payload["claim_boundary"][
            "formal_calls_authorized"
        ]
        is False,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "counts": {"formal_pairs": len(pairs), "formal_items": len(item_ids)},
        "label_counts": label_counts,
        "gates": gates,
        "passed": all(gates.values()),
    }


def prepare() -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)
    if not ARCHIVE.is_file() or file_sha256(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("VitaminC official archive is missing or hash-mismatched")
    if SELECTION_MANIFEST.exists():
        raise ValueError("V3.16.1 selection manifest already exists")
    if list(DEFAULT_ROOT.glob("*/formal/records*.jsonl")):
        raise ValueError("cannot prepare selection after formal calls")
    payload = build_selection(load_rows())
    payload["preregistration_sha256"] = file_sha256(PREREGISTRATION)
    payload["implementation_sha256"] = file_sha256(Path(__file__))
    audit = audit_selection(payload)
    if not audit["passed"]:
        raise ValueError(f"selection audit failed: {audit['gates']}")
    _write_json(SELECTION_MANIFEST, payload)
    _write_json(SELECTION_AUDIT, audit)
    return audit


def audit() -> dict[str, Any]:
    if not SELECTION_MANIFEST.is_file():
        raise FileNotFoundError(SELECTION_MANIFEST)
    payload = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    expected = build_selection(load_rows())
    expected["preregistration_sha256"] = file_sha256(PREREGISTRATION)
    expected["implementation_sha256"] = file_sha256(Path(__file__))
    if payload != expected:
        raise ValueError("V3.16.1 selection manifest drifted")
    result = audit_selection(payload)
    if not result["passed"]:
        raise ValueError(f"selection audit failed: {result['gates']}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare() if args.command == "prepare" else audit()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
