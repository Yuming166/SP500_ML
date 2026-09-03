"""Deterministic dataset construction for label-symmetric V3.16.

This module performs no model calls. It creates and audits page-disjoint smoke,
development, and formal candidate partitions from the official VitaminC real
contrastive test set.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "detection-v3.16-vitaminc-symmetric-development-2026-09-03"
SALT = PROTOCOL_VERSION
DATASET = Path("/storage/gaoym/datasets/vitaminc_real/extracted/vitaminc_real/test.jsonl")
ARCHIVE = Path("/storage/gaoym/datasets/vitaminc_real/vitaminc_real.zip")
EXPECTED_ARCHIVE_SHA256 = "04cce67d000a61fd83885d68924210dd08f3d0ac91fde9f5a4b0bfb768339418"
PREREGISTRATION = Path("docs/detection_v3_16_preregistration.md")
DEFAULT_ROOT = Path("results/detection_v3_16_development")
SELECTION_MANIFEST = DEFAULT_ROOT / "selection_manifest.json"
SELECTION_AUDIT = DEFAULT_ROOT / "selection_audit.json"

SMOKE_PAIRS = 4
DEVELOPMENT_PAIRS = 30
FORMAL_PAIRS = 250
TOTAL_PAIRS = SMOKE_PAIRS + DEVELOPMENT_PAIRS + FORMAL_PAIRS
CHARACTER_RATIO_MIN = 0.97
TOKEN_JACCARD_MIN = 0.90
DISTRACTOR_JACCARD_MAX = 0.05
CLAIM_TOKENS = (5, 40)
EVIDENCE_TOKENS = (10, 120)


@dataclass(frozen=True)
class ContrastivePair:
    pair_id: str
    case_id: str
    page: str
    wiki_revision_id: str
    claim: str
    normalized_claim: str
    supports_id: str
    refutes_id: str
    character_ratio: float
    token_jaccard: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def token_jaccard(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _rank(*parts: str) -> str:
    return hashlib.sha256("|".join((SALT, *parts)).encode()).hexdigest()


def load_rows(path: Path = DATASET) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(errors="replace").splitlines() if line]
    required = {
        "unique_id",
        "case_id",
        "wiki_revision_id",
        "label",
        "claim",
        "evidence",
        "page",
        "group",
    }
    if any(set(row) != required for row in rows):
        raise ValueError("VitaminC real schema drifted")
    ids = [str(row["unique_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("VitaminC unique_id is not unique")
    return rows


def candidate_pairs(rows: Sequence[Mapping[str, Any]]) -> list[ContrastivePair]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), normalize(str(row["claim"])))].append(row)
    candidates: list[ContrastivePair] = []
    for (case_id, normalized_claim), group in grouped.items():
        supports = [row for row in group if row["label"] == "SUPPORTS"]
        refutes = [row for row in group if row["label"] == "REFUTES"]
        if len(supports) != 1 or len(refutes) != 1:
            continue
        support, refute = supports[0], refutes[0]
        claim = str(support["claim"])
        support_evidence = str(support["evidence"])
        refute_evidence = str(refute["evidence"])
        if "\ufffd" in claim + support_evidence + refute_evidence:
            continue
        claim_n = len(tokens(claim))
        support_n = len(tokens(support_evidence))
        refute_n = len(tokens(refute_evidence))
        if not CLAIM_TOKENS[0] <= claim_n <= CLAIM_TOKENS[1]:
            continue
        if not all(EVIDENCE_TOKENS[0] <= n <= EVIDENCE_TOKENS[1] for n in (support_n, refute_n)):
            continue
        if normalize(support_evidence) == normalize(refute_evidence):
            continue
        character_ratio = difflib.SequenceMatcher(
            None, normalize(support_evidence), normalize(refute_evidence)
        ).ratio()
        similarity = token_jaccard(support_evidence, refute_evidence)
        if character_ratio < CHARACTER_RATIO_MIN or similarity < TOKEN_JACCARD_MIN:
            continue
        page = str(support["page"])
        if page != str(refute["page"]):
            raise ValueError("contrastive pair crosses Wikipedia pages")
        pair_id = hashlib.sha256(f"{case_id}|{normalized_claim}".encode()).hexdigest()[:16]
        candidates.append(
            ContrastivePair(
                pair_id=pair_id,
                case_id=case_id,
                page=page,
                wiki_revision_id=str(support["wiki_revision_id"]),
                claim=claim,
                normalized_claim=normalized_claim,
                supports_id=str(support["unique_id"]),
                refutes_id=str(refute["unique_id"]),
                character_ratio=character_ratio,
                token_jaccard=similarity,
            )
        )
    return sorted(candidates, key=lambda pair: _rank("pair", pair.page, pair.pair_id))


def one_pair_per_page(pairs: Sequence[ContrastivePair]) -> list[ContrastivePair]:
    by_page: dict[str, list[ContrastivePair]] = defaultdict(list)
    for pair in pairs:
        by_page[pair.page].append(pair)
    selected = [
        min(page_pairs, key=lambda pair: _rank("within-page", pair.page, pair.pair_id))
        for page_pairs in by_page.values()
    ]
    return sorted(selected, key=lambda pair: _rank("target-page", pair.page, pair.pair_id))


def _distractor_candidates(
    rows: Sequence[Mapping[str, Any]], excluded_pages: set[str]
) -> list[Mapping[str, Any]]:
    candidates = []
    for row in rows:
        evidence = str(row["evidence"])
        if str(row["page"]) in excluded_pages or "\ufffd" in evidence:
            continue
        length = len(tokens(evidence))
        if EVIDENCE_TOKENS[0] <= length <= EVIDENCE_TOKENS[1]:
            candidates.append(row)
    return candidates


def assign_distractors(
    pairs: Sequence[ContrastivePair], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    target_pages = {pair.page for pair in pairs}
    candidates = _distractor_candidates(rows, target_pages)
    used_pages: set[str] = set()
    assignments: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        ordered = sorted(
            candidates,
            key=lambda row: _rank(
                "distractor", pair.pair_id, str(row["page"]), str(row["unique_id"])
            ),
        )
        chosen = next(
            (
                row
                for row in ordered
                if str(row["page"]) not in used_pages
                and token_jaccard(pair.claim, str(row["evidence"])) <= DISTRACTOR_JACCARD_MAX
            ),
            None,
        )
        if chosen is None:
            raise ValueError(f"no eligible distractor for pair {pair.pair_id}")
        used_pages.add(str(chosen["page"]))
        assignments[pair.pair_id] = chosen
    return assignments


def build_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = one_pair_per_page(candidate_pairs(rows))
    if len(eligible) < TOTAL_PAIRS:
        raise ValueError(f"only {len(eligible)} page-disjoint pairs; need {TOTAL_PAIRS}")
    selected = eligible[:TOTAL_PAIRS]
    distractors = assign_distractors(selected, rows)
    split_boundaries = {
        "smoke": (0, SMOKE_PAIRS),
        "development": (SMOKE_PAIRS, SMOKE_PAIRS + DEVELOPMENT_PAIRS),
        "formal": (SMOKE_PAIRS + DEVELOPMENT_PAIRS, TOTAL_PAIRS),
    }
    pair_rows = []
    for split, (start, end) in split_boundaries.items():
        for pair in selected[start:end]:
            distractor = distractors[pair.pair_id]
            pair_rows.append(
                {
                    "split": split,
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
                    "distractor_claim_jaccard": token_jaccard(
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
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v3_16_model_call",
        "dataset": {"path": str(DATASET), "sha256": file_sha256(DATASET)},
        "archive": {"path": str(ARCHIVE), "sha256": file_sha256(ARCHIVE)},
        "selection_rule": {
            "character_ratio_min": CHARACTER_RATIO_MIN,
            "token_jaccard_min": TOKEN_JACCARD_MIN,
            "distractor_claim_jaccard_max": DISTRACTOR_JACCARD_MAX,
            "claim_tokens": list(CLAIM_TOKENS),
            "evidence_tokens": list(EVIDENCE_TOKENS),
            "one_pair_per_page": True,
            "selection_uses_model_outputs": False,
            "selection_uses_outcome_frequency": False,
        },
        "counts": {
            "eligible_page_roots": len(eligible),
            "smoke_pairs": SMOKE_PAIRS,
            "development_pairs": DEVELOPMENT_PAIRS,
            "formal_pairs": FORMAL_PAIRS,
            "formal_items": FORMAL_PAIRS * 2,
        },
        "pairs": pair_rows,
        "claim_boundary": {
            "development_calls_authorized": False,
            "formal_calls_authorized": False,
            "formal_outcomes_accessed": False,
        },
    }


def audit_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(payload["pairs"])
    expected = {"smoke": SMOKE_PAIRS, "development": DEVELOPMENT_PAIRS, "formal": FORMAL_PAIRS}
    counts = {split: sum(row["split"] == split for row in pairs) for split in expected}
    target_pages = [str(row["page"]) for row in pairs]
    distractor_pages = [str(row["distractor_page"]) for row in pairs]
    item_ids = [item["item_id"] for row in pairs for item in row["items"]]
    label_counts = {
        split: {
            label: sum(
                item["gold_label"] == label
                for row in pairs
                if row["split"] == split
                for item in row["items"]
            )
            for label in ("SUPPORTS", "REFUTES")
        }
        for split in expected
    }
    gates = {
        "split_pair_counts": counts == expected,
        "target_pages_unique": len(target_pages) == len(set(target_pages)),
        "distractor_pages_unique": len(distractor_pages) == len(set(distractor_pages)),
        "target_distractor_pages_disjoint": not (set(target_pages) & set(distractor_pages)),
        "item_ids_unique": len(item_ids) == len(set(item_ids)),
        "labels_exactly_balanced": all(
            label_counts[split]["SUPPORTS"] == label_counts[split]["REFUTES"] == expected[split]
            for split in expected
        ),
        "natural_reverse_is_exact_swap": all(
            row["items"][0]["original_id"] == row["items"][1]["reverse_id"]
            and row["items"][0]["reverse_id"] == row["items"][1]["original_id"]
            for row in pairs
        ),
        "contrastive_gate": all(
            row["character_ratio"] >= CHARACTER_RATIO_MIN
            and row["token_jaccard"] >= TOKEN_JACCARD_MIN
            for row in pairs
        ),
        "distractor_gate": all(
            row["distractor_claim_jaccard"] <= DISTRACTOR_JACCARD_MAX for row in pairs
        ),
        "dataset_hash": payload["dataset"]["sha256"] == file_sha256(DATASET),
        "archive_hash": payload["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256,
        "formal_calls_not_authorized": payload["claim_boundary"]["formal_calls_authorized"]
        is False,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "counts": counts,
        "label_counts": label_counts,
        "gates": gates,
        "passed": all(gates.values()),
    }


def prepare() -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)
    if ARCHIVE.is_file() and file_sha256(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("VitaminC official archive hash mismatch")
    if SELECTION_MANIFEST.exists():
        raise ValueError("selection manifest already exists; V3.16 is frozen")
    payload = build_selection(load_rows())
    payload["preregistration_sha256"] = file_sha256(PREREGISTRATION)
    payload["implementation_sha256"] = file_sha256(Path(__file__))
    _write_json(SELECTION_MANIFEST, payload)
    audit = audit_selection(payload)
    _write_json(SELECTION_AUDIT, audit)
    if not audit["passed"]:
        raise ValueError(f"selection audit failed: {audit['gates']}")
    return audit


def audit() -> dict[str, Any]:
    payload = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    if payload["preregistration_sha256"] != file_sha256(PREREGISTRATION):
        raise ValueError("preregistration drifted")
    if payload["implementation_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("implementation drifted")
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
