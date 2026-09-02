"""PACE: provenance-grounded action certificates for consensus repair."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability.pilot_llm_v1 import _attempt_payload, _extract_json_object
from sp500_forecastability.recovery_v2 import PERSONAS, RecoveryChatClient, _call_with_retry

PROTOCOL_VERSION = "recovery-v3.5.1-pace-ex-fever-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_5_1")
DATASET = Path("data/ex_fever/dev.csv")
DATASET_SHA256 = "75eb05d4b9a7f1d16a672312cd9c5203f5e603e2effda173a9a32ba58b09db34"
WIKI_DB = Path("data/ex_fever/wiki_db.db")
WIKI_DB_SHA256 = "8f23ace9b7242bc94fcfdf31607d0038ae7085ead09c7358a0339a9e35ec0940"
CLIMATE_SELECTION = Path("results/recovery_v3_4_1/selection_manifest.json")
AVERITEC_SELECTION = Path("results/recovery_v3_2/selection_manifest.json")
PREREGISTRATION = Path("docs/recovery_v3_5_1_preregistration.md")
SPLITS = ("development", "test")
EXPECTED_COUNTS = {"development": 600, "test": 500}
EXPECTED_LABELS = {
    "development": {"Supported": 300, "Refuted": 300},
    "test": {"Supported": 250, "Refuted": 250},
}
SELECTION_SALT = b"pace-v3.5-ex-fever-selection-2026-09-02"
MODEL_SEED = 20_260_985
BOOTSTRAP_SEED = 20_260_986
BOOTSTRAP_REPLICATES = 10_000
FIX_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
HARM_CAPS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
UTILITY_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20)
CERTIFICATE_ACTIONS = ("candidate_0", "candidate_1")
CALL_SEEDS = {
    "baseline": 20_260_991,
    "candidate_0": 20_261_001,
    "candidate_1": 20_261_011,
    "both": 20_261_021,
}
CERTIFICATE_SEEDS = {"candidate_0": 20_261_031, "candidate_1": 20_261_041}
CERTIFICATE_FIELDS = {
    "relation",
    "support_strength",
    "refute_strength",
    "confidence",
    "new_evidence_ids",
    "missing_bridge",
}
RELATIONS = ("supports", "refutes", "insufficient", "conflicted")
MAX_DOC_SENTENCES = 3
MAX_SENTENCE_CHARS = 1_200


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _normalise_root(value: object) -> str:
    return " ".join(str(value).replace("_", " ").split()).casefold()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _document_sentences(text: str, explanation: str) -> list[str]:
    raw = []
    for paragraph in text.splitlines():
        raw.extend(re.split(r"(?<=[.!?])\s+", paragraph.strip()))
    sentences = [" ".join(item.split()) for item in raw if item.strip()]
    explanation_tokens = _tokens(explanation)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -len(_tokens(item[1]) & explanation_tokens) / max(1, len(_tokens(item[1]))),
            item[0],
        ),
    )[:MAX_DOC_SENTENCES]
    return [sentences[index][:MAX_SENTENCE_CHARS] for index, _sentence in sorted(ranked)]


def _read_entities(value: object) -> list[str]:
    try:
        entities = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(entities, list) or any(not isinstance(item, str) for item in entities):
        return []
    return entities


def load_eligible(
    dataset_path: Path = DATASET,
    wiki_db_path: Path = WIKI_DB,
) -> list[dict[str, Any]]:
    if base._sha256_path(dataset_path) != DATASET_SHA256:
        raise ValueError("EX-FEVER development data checksum drifted")
    if base._sha256_path(wiki_db_path) != WIKI_DB_SHA256:
        raise ValueError("EX-FEVER Wiki database checksum drifted")
    frame = pd.read_csv(dataset_path)
    connection = sqlite3.connect(wiki_db_path)
    examples = []
    seen_claims: set[str] = set()
    label_map = {"SUPPORT": "Supported", "REFUTE": "Refuted"}
    try:
        for row_index, row in frame.iterrows():
            raw_label = str(row["label"])
            if raw_label not in label_map:
                continue
            roots = _read_entities(row["golden entity"])
            if len(roots) != 2 or len(set(roots)) != 2:
                continue
            claim = " ".join(str(row["claim"]).split())
            normalised_claim = v34._normalise_claim(claim)
            if not claim or normalised_claim in seen_claims:
                continue
            evidence = {}
            for root in roots:
                result = connection.execute(
                    "SELECT text FROM documents WHERE id = ?", (root.replace("_", " "),)
                ).fetchone()
                if result is None or not str(result[0]).strip():
                    break
                sentences = _document_sentences(str(result[0]), str(row["explanation"]))
                if not sentences:
                    break
                evidence[root] = sentences
            if len(evidence) != 2:
                continue
            seen_claims.add(normalised_claim)
            label = label_map[raw_label]
            example_id = sha256(f"ex-fever-dev\0{row_index}\0{claim}".encode()).hexdigest()[:32]
            examples.append(
                {
                    "example_id": example_id,
                    "source_split": "ex_fever_official_dev",
                    "source_row_index": int(row_index),
                    "claim": claim,
                    "label": label,
                    "gold_binary": int(label == "Supported"),
                    "fact_check_root": "ex-fever",
                    "annotated_evidence": evidence,
                }
            )
    finally:
        connection.close()
    return examples


def _component_sizes(examples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    union_find = UnionFind()
    for example in examples:
        roots = list(example["annotated_evidence"])
        union_find.union(roots[0], roots[1])
    counts = Counter(union_find.find(next(iter(row["annotated_evidence"]))) for row in examples)
    return {
        str(row["example_id"]): counts[union_find.find(next(iter(row["annotated_evidence"])))]
        for row in examples
    }


def _select_by_label(
    examples: Sequence[Mapping[str, Any]], count_per_label: int, *, salt_suffix: str
) -> list[dict[str, Any]]:
    selected = []
    for label in ("Supported", "Refuted"):
        rows = sorted(
            (row for row in examples if row["label"] == label),
            key=lambda row: base._hash_key(
                SELECTION_SALT, f"{salt_suffix}\0{row['example_id']}"
            ),
        )
        if len(rows) < count_per_label:
            raise ValueError(f"insufficient {label} examples for {salt_suffix}")
        selected.extend(dict(row) for row in rows[:count_per_label])
    return selected


def build_selection() -> dict[str, Any]:
    eligible = load_eligible()
    component_sizes = _component_sizes(eligible)
    climate = json.loads(CLIMATE_SELECTION.read_text(encoding="utf-8"))
    v34.validate_selection(climate)
    climate_roots = {
        _normalise_root(packet["root"])
        for row in climate["examples"]
        for packet in (row["anchor"], *row["candidates"])
    }
    prospective_pool = [
        row
        for row in eligible
        if component_sizes[str(row["example_id"])] == 1
        and not ({_normalise_root(root) for root in row["annotated_evidence"]} & climate_roots)
    ]
    test_raw = _select_by_label(prospective_pool, 250, salt_suffix="test")
    test_claims = {v34._normalise_claim(row["claim"]) for row in test_raw}
    test_roots = {root for row in test_raw for root in row["annotated_evidence"]}
    development_pool = [
        row
        for row in eligible
        if v34._normalise_claim(row["claim"]) not in test_claims
        and not (set(row["annotated_evidence"]) & test_roots)
    ]
    development_raw = _select_by_label(development_pool, 300, salt_suffix="development")
    examples = [
        *base._prepare_partition(development_raw, "development"),
        *base._prepare_partition(test_raw, "test"),
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "selection_frozen_before_ex_fever_model_calls",
        "datasets": {
            "ex_fever_dev": {"path": str(DATASET), "sha256": DATASET_SHA256},
            "ex_fever_wiki": {"path": str(WIKI_DB), "sha256": WIKI_DB_SHA256},
        },
        "model": "Qwen3.5-4B",
        "endpoint": base.RELOCATED_RUNTIME_ENDPOINT,
        "root_definition": "EX-FEVER golden Wikipedia article title",
        "evidence_extraction": (
            "top three raw Wiki sentences by token overlap with the gold explanation; "
            "explanation and label are not stored as router features"
        ),
        "split_rule": (
            "test uses deterministic balanced sample from size-one page components and "
            "excludes CLIMATE-FEVER page roots"
        ),
        "examples": sorted(examples, key=lambda row: (str(row["split"]), row["example_id"])),
    }


def validate_selection(selection: Mapping[str, Any]) -> None:
    if dict(selection) != build_selection():
        raise ValueError("Recovery V3.5 selection or source data drifted")


def audit_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    validate_selection(selection)
    examples = list(selection["examples"])
    counts = Counter(str(row["split"]) for row in examples)
    labels = {
        split: dict(Counter(str(row["label"]) for row in examples if row["split"] == split))
        for split in SPLITS
    }
    claims = {
        split: {v34._normalise_claim(row["claim"]) for row in examples if row["split"] == split}
        for split in SPLITS
    }
    roots = {
        split: {
            str(packet["root"])
            for row in examples
            if row["split"] == split
            for packet in (row["anchor"], *row["candidates"])
        }
        for split in SPLITS
    }
    candidate_fractions = {}
    role_auc = {}
    for split in SPLITS:
        rows = [row for row in examples if row["split"] == split]
        candidate_fractions[split] = sum(
            row["candidates"][0]["annotation_role"] == "held_out_annotated_root"
            for row in rows
        ) / len(rows)
        role = []
        score = []
        for row in rows:
            for candidate in row["candidates"]:
                role.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
                score.append(float(candidate["retrieval_score"]))
        auc = float(roc_auc_score(role, score))
        role_auc[split] = max(auc, 1.0 - auc)
    averitec = json.loads(AVERITEC_SELECTION.read_text(encoding="utf-8"))
    base.validate_selection(averitec)
    climate = json.loads(CLIMATE_SELECTION.read_text(encoding="utf-8"))
    v34.validate_selection(climate)
    prior_claims = {
        v34._normalise_claim(row["claim"])
        for source in (averitec, climate)
        for row in source["examples"]
    }
    climate_roots = {
        _normalise_root(packet["root"])
        for row in climate["examples"]
        for packet in (row["anchor"], *row["candidates"])
    }
    test_rows = [row for row in examples if row["split"] == "test"]
    gates = {
        "exact_counts": dict(counts) == EXPECTED_COUNTS,
        "exact_labels": labels == EXPECTED_LABELS,
        "zero_claim_overlap_between_splits": not (claims["development"] & claims["test"]),
        "zero_page_root_overlap_between_splits": not (roots["development"] & roots["test"]),
        "zero_prior_claim_overlap": not (claims["test"] & prior_claims),
        "zero_climate_page_root_overlap": not (
            {_normalise_root(root) for root in roots["test"]} & climate_roots
        ),
        "test_gold_roots_unique_across_items": len(roots["test"]) == 2 * len(test_rows),
        "three_distinct_roots_per_item": all(
            len({row["anchor"]["root"], *(item["root"] for item in row["candidates"])}) == 3
            for row in examples
        ),
        "candidate_order_balanced": all(
            0.49 <= fraction <= 0.51 for fraction in candidate_fractions.values()
        ),
        "retrieval_and_title_shortcuts_excluded_from_router": True,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "counts": dict(counts),
        "labels": labels,
        "candidate_0_annotated_fraction": candidate_fractions,
        "oriented_retrieval_role_auc": role_auc,
        "claim_overlap_between_splits": len(claims["development"] & claims["test"]),
        "page_root_overlap_between_splits": len(roots["development"] & roots["test"]),
        "test_distinct_page_roots": len(roots["test"]),
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection(output: Path) -> bool:
    expected = build_selection()
    if output.exists():
        validate_selection(json.loads(output.read_text(encoding="utf-8")))
        return False
    base._write_json(output, expected)
    return True


def build_certificate_messages(
    example: Mapping[str, Any], action: str, *, repair: bool = False
) -> list[dict[str, str]]:
    if action not in CERTIFICATE_ACTIONS:
        raise ValueError(f"certificate does not support action {action}")
    candidate = example["candidates"][int(action[-1])]
    system = (
        "You are a counterfactual evidence certificate generator. Judge whether the ANCHOR "
        "plus PROPOSED NEW ROOT is sufficient to support or refute the complete claim. Use only "
        "the packets; missing evidence means insufficient. Return exactly one JSON object with "
        "relation (supports/refutes/insufficient/conflicted), support_strength, refute_strength, "
        "confidence (each numeric in [0,1]), new_evidence_ids (only NEW packet IDs genuinely "
        "needed for the relation), and missing_bridge (boolean). No reasoning or extra keys."
    )
    user = (
        f"Claim: {example['claim']}\n\n"
        f"ANCHOR:\n{base._packet(example['anchor']['evidence'])}\n\n"
        f"PROPOSED NEW ROOT:\n{base._packet(candidate['evidence'])}\n\n"
        "Assess the combined evidence. Do not assume the previous consensus or use outside facts."
    )
    if repair:
        user += (
            "\nYour previous response was invalid. Return only the required JSON object with "
            "valid field types and packet-local NEW evidence IDs."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_certificate(content: str, allowed_new_ids: Sequence[str]) -> dict[str, Any]:
    payload = _extract_json_object(content)
    unknown = set(payload) - CERTIFICATE_FIELDS
    missing = CERTIFICATE_FIELDS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"certificate fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    relation = str(payload["relation"]).casefold().replace("-", "_").replace(" ", "_")
    relation = {"support": "supports", "refute": "refutes"}.get(relation, relation)
    if relation not in RELATIONS:
        raise ValueError(f"invalid certificate relation: {relation}")
    numeric = {}
    for name in ("support_strength", "refute_strength", "confidence"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
        numeric[name] = value
    ids = payload["new_evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise TypeError("new_evidence_ids must be a list of strings")
    allowed = set(allowed_new_ids)
    local_ids = []
    dropped_ids = []
    for evidence_id in ids:
        if evidence_id not in allowed or evidence_id in local_ids:
            dropped_ids.append(evidence_id)
        else:
            local_ids.append(evidence_id)
    if not isinstance(payload["missing_bridge"], bool):
        raise TypeError("missing_bridge must be boolean")
    return {
        "relation": relation,
        **numeric,
        "new_evidence_ids": local_ids,
        "dropped_evidence_ids": dropped_ids,
        "missing_bridge": bool(payload["missing_bridge"]),
    }


def _call_certificate_with_retry(
    client: RecoveryChatClient,
    example: Mapping[str, Any],
    action: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    candidate = example["candidates"][int(action[-1])]
    allowed_ids = [str(item["evidence_id"]) for item in candidate["evidence"]]
    attempts = []
    final_error = None
    for attempt_index in range(2):
        attempt = None
        try:
            result = client.call(
                build_certificate_messages(example, action, repair=attempt_index > 0),
                seed=CERTIFICATE_SEEDS[action],
            )
            attempt = _attempt_payload(result)
            certificate = parse_certificate(result.content, allowed_ids)
        except (RuntimeError, TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            if attempt is None:
                attempt = {
                    "cache_hit": False,
                    "cache_key": None,
                    "http_status": None,
                    "request_bytes": None,
                    "response_bytes": None,
                    "latency_seconds": None,
                    "usage": {},
                    "parse_error": None,
                    "transport_error": final_error,
                }
            else:
                attempt["parse_error"] = final_error
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        return certificate, attempts, None
    return None, attempts, final_error


def _run_certificate_example(
    client: RecoveryChatClient, example: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for action in CERTIFICATE_ACTIONS:
        certificate, attempts, final_error = _call_certificate_with_retry(client, example, action)
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "runtime_endpoint": client.endpoint,
                "example_id": example["example_id"],
                "split": example["split"],
                "action": action,
                "success": certificate is not None,
                "first_pass_valid": certificate is not None and len(attempts) == 1,
                "attempts": attempts,
                "certificate": certificate,
                "final_error": final_error,
            }
        )
    return rows


def _run_action_example(
    client: RecoveryChatClient, example: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    anchor_ids = [str(item["evidence_id"]) for item in example["anchor"]["evidence"]]
    for agent_index in range(len(PERSONAS)):
        decision, attempts, final_error = _call_with_retry(
            client,
            lambda repair, index=agent_index: base.build_baseline_messages(
                example, index, repair=repair
            ),
            anchor_ids,
            seed=CALL_SEEDS["baseline"] + agent_index,
        )
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "runtime_endpoint": client.endpoint,
                "example_id": example["example_id"],
                "split": example["split"],
                "phase": "baseline",
                "action": "KEEP",
                "agent_index": agent_index,
                "success": decision is not None,
                "first_pass_valid": decision is not None and len(attempts) == 1,
                "attempts": attempts,
                "decision": decision,
                "final_error": final_error,
                "gold_binary": int(example["gold_binary"]),
            }
        )
    consensus, agreement = base._majority(rows)
    for action in base.RECOVERY_ACTIONS:
        acquired = base._action_evidence(example, action)
        allowed_ids = [*anchor_ids, *(str(item["evidence_id"]) for item in acquired)]
        decision, attempts, final_error = _call_with_retry(
            client,
            lambda repair, action_name=action: base.build_recovery_messages(
                example, action_name, consensus, repair=repair
            ),
            allowed_ids,
            seed=CALL_SEEDS[action],
        )
        indices = [0, 1] if action == "both" else [int(action[-1])]
        annotated_ids = {
            str(item["evidence_id"])
            for index in indices
            if example["candidates"][index]["annotation_role"]
            == "held_out_annotated_root"
            for item in example["candidates"][index]["evidence"]
        }
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "runtime_endpoint": client.endpoint,
                "example_id": example["example_id"],
                "split": example["split"],
                "phase": "recovery",
                "action": action,
                "agent_index": None,
                "success": decision is not None,
                "first_pass_valid": decision is not None and len(attempts) == 1,
                "attempts": attempts,
                "decision": decision,
                "final_error": final_error,
                "gold_binary": int(example["gold_binary"]),
                "baseline_consensus": consensus,
                "baseline_agreement": agreement,
                "packet_contains_annotated_root": bool(annotated_ids),
                "annotated_evidence_ids": sorted(annotated_ids),
            }
        )
    return rows


def _validate_certificates(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], *, split: str
) -> None:
    grouped = base._record_groups(records)
    expected = {str(row["example_id"]): row for row in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} certificate coverage mismatch")
    for example_id, rows in grouped.items():
        if len(rows) != len(CERTIFICATE_ACTIONS) or {
            row.get("action") for row in rows
        } != set(CERTIFICATE_ACTIONS):
            raise ValueError(f"{split} invalid certificate bundle for {example_id}")
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("split") != split
            or not row.get("success")
            or row.get("certificate") is None
            for row in rows
        ):
            raise ValueError(f"{split} invalid certificate metadata for {example_id}")


def _validate_actions(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], *, split: str
) -> None:
    grouped = base._record_groups(records)
    expected = {str(row["example_id"]): row for row in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} action coverage mismatch")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        example = expected[example_id]
        if len(rows) != len(PERSONAS) + len(base.RECOVERY_ACTIONS):
            raise ValueError(f"{split} invalid action bundle for {example_id}")
        if {row.get("agent_index") for row in baseline} != set(range(len(PERSONAS))):
            raise ValueError(f"{split} invalid baseline agents for {example_id}")
        if {row.get("action") for row in recovery} != set(base.RECOVERY_ACTIONS):
            raise ValueError(f"{split} invalid recovery actions for {example_id}")
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("split") != split
            or row.get("gold_binary") != example["gold_binary"]
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"{split} invalid action metadata for {example_id}")


def _execute_bundles(
    examples: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    run_one: Any,
    expected_per_example: int,
    validator: Any,
    split: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_example[str(row["example_id"])].append(row)
    if any(len(rows) != expected_per_example for rows in by_example.values()):
        raise ValueError("partial file contains an incomplete bundle")
    existing = {
        example_id: rows
        for example_id, rows in by_example.items()
        if all(row.get("success") for row in rows)
    }
    records = [row for rows in existing.values() for row in rows]
    allowed = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed:
        raise ValueError("partial file contains examples outside the requested run")
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = RecoveryChatClient(base.RELOCATED_RUNTIME_ENDPOINT, "Qwen3.5-4B", cache_dir)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, client, row): row for row in pending}
        for future in as_completed(futures):
            bundle = future.result()
            records.extend(bundle)
            records.sort(
                key=lambda row: (
                    str(row["example_id"]),
                    str(row.get("phase", "certificate")),
                    -1 if row.get("agent_index") is None else int(row["agent_index"]),
                    str(row["action"]),
                )
            )
            base._write_jsonl(partial_path, records)
            print(
                f"[{len(records) // expected_per_example}/{len(examples)}] "
                f"{bundle[0]['example_id']} success={all(row['success'] for row in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    expected_rows = len(examples) * expected_per_example
    if len(records) != expected_rows or any(not row["success"] for row in records):
        raise ValueError(f"run incomplete or invalid: {len(records)}/{expected_rows}")
    validator(examples, records, split=split)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def execute(
    selection_path: Path,
    *,
    split: str,
    kind: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    limit: int | None = None,
    require_manifest: bool = False,
) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.5 selection failed structural gates")
    if split not in SPLITS or kind not in {"certificates", "actions"}:
        raise ValueError("invalid split or record kind")
    if require_manifest:
        validate_router_manifest(DEFAULT_ROOT / "router" / "manifest.json", selection_path)
    examples = [row for row in selection["examples"] if row["split"] == split]
    if limit is not None:
        examples = examples[:limit]
    if kind == "certificates":
        return _execute_bundles(
            examples,
            output_dir=output_dir,
            cache_dir=cache_dir,
            workers=workers,
            run_one=_run_certificate_example,
            expected_per_example=len(CERTIFICATE_ACTIONS),
            validator=_validate_certificates,
            split=split,
        )
    return _execute_bundles(
        examples,
        output_dir=output_dir,
        cache_dir=cache_dir,
        workers=workers,
        run_one=_run_action_example,
        expected_per_example=len(PERSONAS) + len(base.RECOVERY_ACTIONS),
        validator=_validate_actions,
        split=split,
    )


def _certificate_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["example_id"]), str(row["action"])): row for row in records}


def _feature_vector(
    example: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    certificate_row: Mapping[str, Any],
    action: str,
) -> tuple[list[str], np.ndarray]:
    consensus, agreement = base._majority(baseline)
    confidences = np.asarray([float(row["decision"]["confidence"]) for row in baseline])
    certificate = certificate_row["certificate"]
    relation = str(certificate["relation"])
    support = float(certificate["support_strength"])
    refute = float(certificate["refute_strength"])
    counter = support if consensus == "no" else refute
    same = refute if consensus == "no" else support
    grounded = float(bool(certificate["new_evidence_ids"]))
    names = [
        "initial_consensus_yes",
        "initial_agreement",
        "baseline_confidence_mean",
        "baseline_confidence_std",
        "baseline_confidence_min",
        "certificate_support_strength",
        "certificate_refute_strength",
        "certificate_confidence",
        "certificate_grounded_new_root",
        "certificate_missing_bridge",
        "certificate_relation_supports",
        "certificate_relation_refutes",
        "certificate_relation_insufficient",
        "certificate_relation_conflicted",
        "certificate_counter_strength",
        "certificate_same_strength",
        "certificate_directional_margin",
        "certificate_grounded_counter_strength",
    ]
    values = np.asarray(
        [
            float(consensus == "yes"),
            agreement,
            float(confidences.mean()),
            float(confidences.std()),
            float(confidences.min()),
            support,
            refute,
            float(certificate["confidence"]),
            grounded,
            float(certificate["missing_bridge"]),
            float(relation == "supports"),
            float(relation == "refutes"),
            float(relation == "insufficient"),
            float(relation == "conflicted"),
            counter,
            same,
            counter - same,
            grounded * counter,
        ],
        dtype=float,
    )
    if any(
        fragment in name
        for name in names
        for fragment in base.FORBIDDEN_FEATURE_FRAGMENTS
    ):
        raise AssertionError("forbidden post-outcome field entered PACE")
    return names, values


def _feature_matrix(
    examples: Sequence[Mapping[str, Any]],
    action_records: Sequence[Mapping[str, Any]],
    certificate_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray, list[tuple[str, str]]]:
    action_groups = base._record_groups(action_records)
    certificates = _certificate_groups(certificate_records)
    rows = []
    keys = []
    feature_names = None
    for example in examples:
        example_id = str(example["example_id"])
        _consensus, _agreement, baseline = base._baseline_state(action_groups[example_id])
        for action in CERTIFICATE_ACTIONS:
            names, values = _feature_vector(
                example, baseline, certificates[(example_id, action)], action
            )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("PACE feature schema drifted")
            rows.append(values)
            keys.append((example_id, action))
    return feature_names or [], np.vstack(rows), keys


def _outcome_targets(
    examples: Sequence[Mapping[str, Any]],
    action_records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    action_groups = base._record_groups(action_records)
    fixes = []
    harms = []
    keys = []
    for example in examples:
        example_id = str(example["example_id"])
        keep, _agreement, outcomes, _baseline = base._outcomes(
            example, action_groups[example_id]
        )
        for action in CERTIFICATE_ACTIONS:
            fixes.append(int(keep == 0 and outcomes[action] == 1))
            harms.append(int(keep == 1 and outcomes[action] == 0))
            keys.append((example_id, action))
    return np.asarray(fixes, dtype=int), np.asarray(harms, dtype=int), keys


def _matrix(
    examples: Sequence[Mapping[str, Any]],
    action_records: Sequence[Mapping[str, Any]],
    certificate_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    names, matrix, feature_keys = _feature_matrix(
        examples, action_records, certificate_records
    )
    fixes, harms, outcome_keys = _outcome_targets(examples, action_records)
    if feature_keys != outcome_keys:
        raise AssertionError("PACE feature and outcome rows are misaligned")
    return names, matrix, fixes, harms, feature_keys


def _new_models() -> dict[str, Any]:
    return {
        "fix_logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=MODEL_SEED,
                    ),
                ),
            ]
        ),
        "harm_logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=MODEL_SEED + 1,
                    ),
                ),
            ]
        ),
        "fix_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=MODEL_SEED + 2,
            n_jobs=-1,
        ),
        "harm_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=MODEL_SEED + 3,
            n_jobs=-1,
        ),
    }


def _fit_models(x: np.ndarray, fixes: np.ndarray, harms: np.ndarray) -> dict[str, Any]:
    if len(set(fixes.tolist())) < 2 or len(set(harms.tolist())) < 2:
        raise ValueError("PACE development matrix needs both fix and harm classes")
    models = _new_models()
    models["fix_logistic"].fit(x, fixes)
    models["harm_logistic"].fit(x, harms)
    models["fix_forest"].fit(x, fixes)
    models["harm_forest"].fit(x, harms)
    return models


def _predict_models(models: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fix = 0.5 * (
        models["fix_logistic"].predict_proba(x)[:, 1]
        + models["fix_forest"].predict_proba(x)[:, 1]
    )
    harm = 0.5 * (
        models["harm_logistic"].predict_proba(x)[:, 1]
        + models["harm_forest"].predict_proba(x)[:, 1]
    )
    return fix, harm


def _prediction_map(
    keys: Sequence[tuple[str, str]], fix: np.ndarray, harm: np.ndarray
) -> dict[tuple[str, str], tuple[float, float]]:
    return {key: (float(fix[i]), float(harm[i])) for i, key in enumerate(keys)}


def _certificate_gate(certificate_row: Mapping[str, Any], consensus: str) -> bool:
    certificate = certificate_row["certificate"]
    target = "supports" if consensus == "no" else "refutes"
    return bool(
        certificate["relation"] == target
        and certificate["new_evidence_ids"]
        and not certificate["missing_bridge"]
    )


def _select_policy(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    certificate_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    *,
    fix_threshold: float,
    harm_cap: float,
    utility_threshold: float,
    require_certificate_gate: bool = True,
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(action_groups[example_id])
        if agreement < base.HIGH_CONSENSUS:
            selected[example_id] = "KEEP"
            continue
        allowed = []
        for action in CERTIFICATE_ACTIONS:
            if require_certificate_gate and not _certificate_gate(
                certificate_rows[(example_id, action)], consensus
            ):
                continue
            fix, harm = predictions[(example_id, action)]
            utility = fix - harm
            if fix >= fix_threshold and harm <= harm_cap and utility >= utility_threshold:
                allowed.append((utility, fix, -harm, action))
        selected[example_id] = max(allowed)[-1] if allowed else "KEEP"
    return selected


def _basic_metrics(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    gains = []
    by_label: dict[str, list[int]] = defaultdict(list)
    harms = high_correct = routes = annotation_supported_repairs = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(
            example, action_groups[example_id]
        )
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        gain = final - keep
        gains.append(gain)
        by_label[str(example["label"])].append(gain)
        if keep == 1 and agreement >= base.HIGH_CONSENSUS:
            high_correct += 1
            harms += int(final == 0)
        routes += int(action != "KEEP")
        if keep == 0 and final == 1 and action != "KEEP":
            recovery = next(
                row
                for row in action_groups[example_id]
                if row["phase"] == "recovery" and row["action"] == action
            )
            annotation_supported_repairs += int(
                recovery["packet_contains_annotated_root"]
                and bool(
                    set(recovery["decision"]["cited_evidence_ids"])
                    & set(recovery["annotated_evidence_ids"])
                )
            )
    label_gain = {label: float(np.mean(values)) for label, values in by_label.items()}
    return {
        "net_fixes": int(sum(gains)),
        "macro_gain": float(np.mean(list(label_gain.values()))),
        "by_label_gain": label_gain,
        "damage_rate": harms / max(1, high_correct),
        "harms": harms,
        "routes": routes,
        "annotation_supported_repairs": annotation_supported_repairs,
    }


def _tune_policy(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    certificate_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    fold_assignments: Mapping[str, int],
) -> dict[str, Any]:
    feasible = []
    for fix_threshold in FIX_THRESHOLDS:
        for harm_cap in HARM_CAPS:
            for utility_threshold in UTILITY_THRESHOLDS:
                fold_metrics = {}
                for fold in range(3):
                    rows = [
                        row
                        for row in examples
                        if fold_assignments[str(row["example_id"])] == fold
                    ]
                    selected = _select_policy(
                        rows,
                        action_groups,
                        certificate_rows,
                        predictions,
                        fix_threshold=fix_threshold,
                        harm_cap=harm_cap,
                        utility_threshold=utility_threshold,
                    )
                    fold_metrics[str(fold)] = _basic_metrics(rows, action_groups, selected)
                if all(
                    result["damage_rate"] <= 0.05
                    and min(result["by_label_gain"].values()) >= 0.0
                    and result["net_fixes"] >= 5
                    and result["routes"] >= 5
                    and result["annotation_supported_repairs"] >= 5
                    for result in fold_metrics.values()
                ):
                    feasible.append(
                        (
                            min(result["macro_gain"] for result in fold_metrics.values()),
                            min(result["net_fixes"] for result in fold_metrics.values()),
                            sum(result["net_fixes"] for result in fold_metrics.values()),
                            -sum(result["harms"] for result in fold_metrics.values()),
                            -sum(result["routes"] for result in fold_metrics.values()),
                            fix_threshold,
                            harm_cap,
                            utility_threshold,
                            fold_metrics,
                        )
                    )
    if not feasible:
        raise ValueError("no fold-robust nontrivial PACE policy; formal test remains locked")
    best = max(feasible, key=lambda row: row[:8])
    return {
        "fix_threshold": best[5],
        "harm_cap": best[6],
        "utility_threshold": best[7],
        "development_fold_metrics": best[8],
        "feasible_configurations": len(feasible),
    }


def fit_router(
    selection_path: Path,
    action_records_path: Path,
    certificate_records_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_router_manifest(manifest_path, selection_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if (DEFAULT_ROOT / "test" / "actions" / "records.jsonl").exists() or (
        DEFAULT_ROOT / "test" / "certificates" / "records.jsonl"
    ).exists():
        raise ValueError("cannot fit or overwrite PACE after formal-test calls")
    if not PREREGISTRATION.exists():
        raise ValueError("V3.5 preregistration must exist before fitting")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.5 selection failed structural gates")
    examples = [row for row in selection["examples"] if row["split"] == "development"]
    actions = base._load_jsonl(action_records_path)
    certificates = base._load_jsonl(certificate_records_path)
    _validate_actions(examples, actions, split="development")
    _validate_certificates(examples, certificates, split="development")
    names, x, fixes, harms, keys = _matrix(examples, actions, certificates)
    labels = [str(row["label"]) for row in examples]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED)
    example_array = np.asarray(examples, dtype=object)
    oof_fix = np.empty(len(keys), dtype=float)
    oof_harm = np.empty(len(keys), dtype=float)
    fold_assignments = {}
    for fold, (train_indices, heldout_indices) in enumerate(
        folds.split(np.zeros(len(examples)), labels)
    ):
        train_ids = {str(row["example_id"]) for row in example_array[train_indices]}
        heldout_ids = {str(row["example_id"]) for row in example_array[heldout_indices]}
        for example_id in heldout_ids:
            fold_assignments[example_id] = fold
        train_rows = [i for i, key in enumerate(keys) if key[0] in train_ids]
        heldout_rows = [i for i, key in enumerate(keys) if key[0] in heldout_ids]
        models = _fit_models(x[train_rows], fixes[train_rows], harms[train_rows])
        fold_fix, fold_harm = _predict_models(models, x[heldout_rows])
        oof_fix[heldout_rows] = fold_fix
        oof_harm[heldout_rows] = fold_harm
    if set(fold_assignments) != {str(row["example_id"]) for row in examples}:
        raise AssertionError("PACE OOF assignments incomplete")
    predictions = _prediction_map(keys, oof_fix, oof_harm)
    action_groups = base._record_groups(actions)
    certificate_groups = _certificate_groups(certificates)
    policy = _tune_policy(
        examples, action_groups, certificate_groups, predictions, fold_assignments
    )
    full_models = _fit_models(x, fixes, harms)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pace_router.joblib"
    joblib.dump({"models": full_models, "feature_names": names}, model_path)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_ex_fever_test_calls",
        "selection_sha256": base._sha256_path(selection_path),
        "implementation_path": str(Path(__file__)),
        "implementation_sha256": base._sha256_path(Path(__file__)),
        "development_action_records_sha256": base._sha256_path(action_records_path),
        "development_certificate_records_sha256": base._sha256_path(certificate_records_path),
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "training": {
            "examples": len(examples),
            "action_rows": len(keys),
            "fix_targets": int(fixes.sum()),
            "harm_targets": int(harms.sum()),
            "family": "equal-weight-standardized-logistic-and-random-forest",
            "three_fold_oof_policy_selection": True,
            "seed": MODEL_SEED,
        },
        "feature_names": names,
        "policy": policy,
        "router_joblib": str(model_path),
        "router_joblib_sha256": base._sha256_path(model_path),
        "feature_boundary": {
            "uses_test_gold_or_action_outcomes_at_inference": False,
            "uses_test_annotation_role_at_inference": False,
            "uses_source_identity_at_inference": False,
            "uses_baseline_consensus_candidate_metadata_and_semantic_certificate": True,
        },
        "claim_boundary": {
            "test_outcomes_seen": False,
            "cross_dataset_from_averitec_and_climate": True,
            "test_page_roots_disjoint_from_development_and_climate": True,
            "publisher_independence": False,
            "cross_model": False,
        },
    }
    base._write_json(manifest_path, manifest)
    return manifest


def validate_router_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("PACE protocol mismatch")
    if manifest.get("status") != "frozen_before_ex_fever_test_calls":
        raise ValueError("PACE is not frozen before formal test calls")
    if manifest.get("selection_sha256") != base._sha256_path(selection_path):
        raise ValueError("PACE selection drifted")
    implementation_path = Path(str(manifest["implementation_path"]))
    if manifest.get("implementation_sha256") != base._sha256_path(implementation_path):
        raise ValueError("PACE implementation drifted")
    if manifest.get("preregistration_sha256") != base._sha256_path(PREREGISTRATION):
        raise ValueError("PACE preregistration drifted")
    model_path = Path(str(manifest["router_joblib"]))
    if manifest.get("router_joblib_sha256") != base._sha256_path(model_path):
        raise ValueError("PACE serialized router drifted")
    boundary = manifest.get("feature_boundary", {})
    if boundary.get("uses_test_gold_or_action_outcomes_at_inference") is not False:
        raise ValueError("PACE inference boundary permits outcomes")
    if boundary.get("uses_test_annotation_role_at_inference") is not False:
        raise ValueError("PACE inference boundary permits annotation roles")


def _test_predictions(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    Mapping[str, Sequence[Mapping[str, Any]]],
    Mapping[tuple[str, str], Mapping[str, Any]],
]:
    names, x, keys = _feature_matrix(examples, actions, certificates)
    if names != manifest["feature_names"]:
        raise ValueError("PACE test feature schema drifted")
    bundle = joblib.load(manifest["router_joblib"])
    fix, harm = _predict_models(bundle["models"], x)
    return (
        _prediction_map(keys, fix, harm),
        base._record_groups(actions),
        _certificate_groups(certificates),
    )


def _certificate_only_policy(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    certificate_groups: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(action_groups[example_id])
        allowed = []
        if agreement >= base.HIGH_CONSENSUS:
            for action in CERTIFICATE_ACTIONS:
                row = certificate_groups[(example_id, action)]
                if _certificate_gate(row, consensus):
                    certificate = row["certificate"]
                    counter = (
                        certificate["support_strength"]
                        if consensus == "no"
                        else certificate["refute_strength"]
                    )
                    allowed.append((float(counter), float(certificate["confidence"]), action))
        selected[example_id] = max(allowed)[-1] if allowed else "KEEP"
    return selected


def _metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps = []
    finals = []
    labels = []
    high_flags = []
    actions = Counter()
    annotation_supported = 0
    roots_added = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(example, grouped[example_id])
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        keeps.append(keep)
        finals.append(final)
        labels.append(str(example["label"]))
        high_flags.append(agreement >= base.HIGH_CONSENSUS)
        actions[action] += 1
        roots_added += 0 if action == "KEEP" else 2 if action == "both" else 1
        if keep == 0 and final == 1 and action != "KEEP":
            recovery = next(
                row
                for row in grouped[example_id]
                if row["phase"] == "recovery" and row["action"] == action
            )
            annotation_supported += int(
                recovery["packet_contains_annotated_root"]
                and bool(
                    set(recovery["decision"]["cited_evidence_ids"])
                    & set(recovery["annotated_evidence_ids"])
                )
            )
    keep_array = np.asarray(keeps, dtype=int)
    final_array = np.asarray(finals, dtype=int)
    gains = final_array - keep_array
    fixes = (keep_array == 0) & (final_array == 1)
    harms = (keep_array == 1) & (final_array == 0)
    by_label = {}
    label_indices = {}
    for label in ("Supported", "Refuted"):
        indices = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
        label_indices[label] = indices
        by_label[label] = {
            "n": len(indices),
            "baseline_accuracy": float(keep_array[indices].mean()),
            "final_accuracy": float(final_array[indices].mean()),
            "net_gain": float(gains[indices].mean()),
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        group_gains = []
        for indices in label_indices.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            group_gains.append(float(gains[sampled].mean()))
        bootstrap[replicate] = float(np.mean(group_gains))
    interval = np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist()
    high_correct = sum(bool(keep and high) for keep, high in zip(keeps, high_flags, strict=True))
    high_harms = sum(
        bool(keep and high and not final)
        for keep, high, final in zip(keeps, high_flags, finals, strict=True)
    )
    return {
        "n": len(examples),
        "baseline_accuracy": float(keep_array.mean()),
        "final_accuracy": float(final_array.mean()),
        "fixes": int(fixes.sum()),
        "harms": int(harms.sum()),
        "net_fixes": int(gains.sum()),
        "net_gain": float(gains.mean()),
        "macro_label_gain": float(
            np.mean([group["net_gain"] for group in by_label.values()])
        ),
        "macro_gain_ci": interval,
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": roots_added,
        "mean_added_roots": roots_added / len(examples),
        "selected_actions": dict(actions),
        "by_native_label": by_label,
    }


def evaluate(
    selection_path: Path,
    action_records_path: Path,
    certificate_records_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("PACE selection failed evaluation audit")
    validate_router_manifest(manifest_path, selection_path)
    examples = [row for row in selection["examples"] if row["split"] == "test"]
    actions = base._load_jsonl(action_records_path)
    certificates = base._load_jsonl(certificate_records_path)
    _validate_actions(examples, actions, split="test")
    _validate_certificates(examples, certificates, split="test")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions, action_groups, certificate_groups = _test_predictions(
        examples, actions, certificates, manifest
    )
    parameters = manifest["policy"]
    pace = _select_policy(
        examples,
        action_groups,
        certificate_groups,
        predictions,
        fix_threshold=float(parameters["fix_threshold"]),
        harm_cap=float(parameters["harm_cap"]),
        utility_threshold=float(parameters["utility_threshold"]),
    )
    no_grounding = _select_policy(
        examples,
        action_groups,
        certificate_groups,
        predictions,
        fix_threshold=float(parameters["fix_threshold"]),
        harm_cap=float(parameters["harm_cap"]),
        utility_threshold=float(parameters["utility_threshold"]),
        require_certificate_gate=False,
    )
    policies = {
        "pace": pace,
        "pace_without_certificate_gate": no_grounding,
        "certificate_only": _certificate_only_policy(
            examples, action_groups, certificate_groups
        ),
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    root_budget = sum(action != "KEEP" for action in pace.values())
    for name, proposed in v34._comparison_proposals(examples, action_groups).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            examples, proposed, root_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    preoutcome = {
        "protocol_version": PROTOCOL_VERSION,
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "test_action_records_sha256": base._sha256_path(action_records_path),
        "test_certificate_records_sha256": base._sha256_path(certificate_records_path),
        "outcomes_accessed_by_route_selection": False,
        "policies": policies,
        "predictions": {
            f"{example_id}:{action}": {"p_fix": values[0], "p_harm": values[1]}
            for (example_id, action), values in predictions.items()
        },
    }
    v34._write_or_validate_preoutcome(preoutcome_path, preoutcome)
    oracle = {}
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(
            example, action_groups[example_id]
        )
        oracle[example_id] = (
            max(
                ("KEEP", *base.RECOVERY_ACTIONS),
                key=lambda action: keep if action == "KEEP" else outcomes[action],
            )
            if agreement >= base.HIGH_CONSENSUS
            else "KEEP"
        )
    policies["available_action_oracle_diagnostic"] = oracle
    metrics = {
        name: _metrics(examples, action_groups, selected)
        for name, selected in policies.items()
    }
    primary = metrics["pace"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": primary["damage_rate_high_consensus_correct"] <= 0.05,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": (
            primary["annotation_supported_repairs"] >= 10
        ),
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
    }
    routed = [row for row in examples if pace[str(row["example_id"])] != "KEEP"]
    annotated_choices = sum(
        row["candidates"][int(pace[str(row["example_id"])][-1])]["annotation_role"]
        == "held_out_annotated_root"
        for row in routed
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_root_disjoint_ex_fever_test",
        "selection_sha256": base._sha256_path(selection_path),
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "test_action_records_sha256": base._sha256_path(action_records_path),
        "test_certificate_records_sha256": base._sha256_path(certificate_records_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_test": len(examples),
        "root_budget": root_budget,
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": "PASS_PACE_V3_5" if all(gates.values()) else "NO_VERIFIED_PACE_DOMINANCE",
        "annotation_role_selection": {
            "routed": len(routed),
            "annotated_root_selected": annotated_choices,
            "accuracy": annotated_choices / max(1, len(routed)),
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    base._write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    for name in ("dev-certificates", "dev-actions"):
        item = subparsers.add_parser(name)
        item.add_argument("--workers", type=int, default=8)
        item.add_argument("--limit", type=int)
    subparsers.add_parser("fit")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--workers", type=int, default=2)
    for name in ("test-certificates", "test-actions"):
        item = subparsers.add_parser(name)
        item.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    selection_path = DEFAULT_ROOT / "selection_manifest.json"
    manifest_path = DEFAULT_ROOT / "router" / "manifest.json"
    if args.command == "prepare":
        write_or_validate_selection(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        audit = audit_selection(selection)
        base._write_json(DEFAULT_ROOT / "selection_audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["passed"] else 2
    if args.command in {"dev-certificates", "dev-actions"}:
        kind = args.command.removeprefix("dev-")
        execute(
            selection_path,
            split="development",
            kind=kind,
            output_dir=DEFAULT_ROOT / "development" / kind,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
            limit=args.limit,
        )
        return 0
    if args.command == "fit":
        manifest = fit_router(
            selection_path,
            DEFAULT_ROOT / "development" / "actions" / "records.jsonl",
            DEFAULT_ROOT / "development" / "certificates" / "records.jsonl",
            DEFAULT_ROOT / "router",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        for kind in ("certificates", "actions"):
            execute(
                selection_path,
                split="test",
                kind=kind,
                output_dir=DEFAULT_ROOT / "smoke" / kind,
                cache_dir=DEFAULT_ROOT / "cache",
                workers=args.workers,
                limit=2,
                require_manifest=True,
            )
        return 0
    if args.command in {"test-certificates", "test-actions"}:
        kind = args.command.removeprefix("test-")
        execute(
            selection_path,
            split="test",
            kind=kind,
            output_dir=DEFAULT_ROOT / "test" / kind,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
            require_manifest=True,
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate(
            selection_path,
            DEFAULT_ROOT / "test" / "actions" / "records.jsonl",
            DEFAULT_ROOT / "test" / "certificates" / "records.jsonl",
            manifest_path,
            DEFAULT_ROOT / "evaluation",
        )
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
