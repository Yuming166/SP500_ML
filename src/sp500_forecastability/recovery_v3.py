"""CAPE: safety-calibrated provenance-action routing on AVeriTeC."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from sp500_forecastability.pilot_llm_v1 import _write_json, _write_jsonl
from sp500_forecastability.recovery_v2 import (
    PERSONAS,
    RECOVERY_ACTIONS,
    RELOCATED_RUNTIME_ENDPOINT,
    RecoveryChatClient,
    _call_with_retry,
    _load_jsonl,
    _majority,
)
from sp500_forecastability.recovery_v2 import _feature_vector as _v2_feature_vector

PROTOCOL_VERSION = "recovery-v3.2-cape-averitec-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_2")
TRAIN_DATASET = Path("data/averitec/train.parquet")
TEST_DATASET = Path("data/averitec/dev.parquet")
TRAIN_SHA256 = "41d08f99b3d3afbdbb81a655ccee23a4cddd3b4af4480100391e305300ee784f"
TEST_SHA256 = "18e9397649d12f0a9b2e21b553c7b32e10b5cf58aff54c2a11e1ad6243af403d"
SPLITS = ("train", "policy_dev", "calibration", "test")
EXPECTED_COUNTS = {"train": 814, "policy_dev": 174, "calibration": 175, "test": 236}
SPLIT_SALT = b"cape-v3-averitec-stratified-split"
ANCHOR_SALT = b"cape-v3-anchor-root"
HELDOUT_SALT = b"cape-v3-heldout-root"
ORDER_SALT = b"cape-v3-balanced-candidate-order"
DISTRACTOR_SALT = b"cape-v3-score-matched-distractor"
HASH_FEATURES = 2**16
MAX_ITEMS_PER_ROOT = 3
MAX_ITEM_CHARS = 1_200
HIGH_CONSENSUS = 0.80
ROOT_COST = 0.01
MODEL_SEED = 20260941
BOOTSTRAP_SEED = 20260942
BOOTSTRAP_REPLICATES = 2_000
GAIN_GRID = (-0.10, -0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
HARM_GRID = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
SHIELD_OFFSETS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
STANCE_CLASSES = ("irrelevant", "refutes", "supports")
FORBIDDEN_FEATURE_FRAGMENTS = (
    "gold",
    "label",
    "correct",
    "outcome",
    "annotation",
    "source_row",
    "split",
)
CALL_SEEDS = {
    "baseline": 20260951,
    "candidate_0": 20260961,
    "candidate_1": 20260971,
    "both": 20260981,
}
REPAIR_SUFFIX = (
    "\nYour previous response was invalid. Return only the required JSON object with "
    "answer yes/no, numeric confidence in [0,1], and packet-local evidence IDs."
)
SPECIAL_TWO_LEVEL_SUFFIXES = frozenset(
    {
        "co.uk",
        "org.uk",
        "gov.uk",
        "com.au",
        "co.za",
        "com.br",
        "com.ng",
        "co.in",
    }
)


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_key(salt: bytes, value: str) -> str:
    return sha256(salt + b"\0" + value.encode("utf-8")).hexdigest()


def canonical_source_root(url: object) -> str:
    """Return a frozen registrable-domain approximation, unwrapping Wayback URLs."""

    if not isinstance(url, str) or not url.strip():
        return ""
    parsed = urlparse(url.strip())
    if (parsed.hostname or "").lower() == "web.archive.org":
        marker = parsed.path.find("/http")
        if marker >= 0:
            parsed = urlparse(parsed.path[marker + 1 :])
    host = (parsed.hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in SPECIAL_TWO_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _packet_text(question: object, answer: object, explanation: object) -> str:
    fields = [
        ("Question", _clean_text(question)),
        ("Answer", _clean_text(answer)),
        ("Explanation", _clean_text(explanation)),
    ]
    return "\n".join(f"{name}: {text}" for name, text in fields if text)[:MAX_ITEM_CHARS]


def _evidence_by_root(row: Any) -> dict[str, list[str]]:
    fact_check_root = canonical_source_root(row.fact_checking_article)
    grouped: dict[str, list[str]] = defaultdict(list)
    for question in row.questions:
        question_text = question.get("question")
        for answer in question["answers"]:
            root = canonical_source_root(answer.get("source_url")) or canonical_source_root(
                answer.get("cached_source_url")
            )
            text = _packet_text(
                question_text,
                answer.get("answer"),
                answer.get("boolean_explanation"),
            )
            if root and root != fact_check_root and text and text not in grouped[root]:
                grouped[root].append(text)
    return {root: texts[:MAX_ITEMS_PER_ROOT] for root, texts in sorted(grouped.items()) if texts}


def load_eligible(path: Path, *, source_split: str) -> list[dict[str, Any]]:
    expected = TRAIN_SHA256 if source_split == "official_train" else TEST_SHA256
    if _sha256_path(path) != expected:
        raise ValueError(f"{source_split} dataset checksum drifted")
    frame = pd.read_parquet(path)
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        claim = _clean_text(row.claim)
        label = _clean_text(row.label)
        evidence = _evidence_by_root(row)
        if label not in {"Supported", "Refuted"} or not claim or len(evidence) < 2:
            continue
        example_id = sha256(f"{source_split}\0{row_index}\0{claim}".encode()).hexdigest()[:32]
        examples.append(
            {
                "example_id": example_id,
                "source_split": source_split,
                "source_row_index": row_index,
                "claim": claim,
                "label": label,
                "gold_binary": int(label == "Supported"),
                "fact_check_root": canonical_source_root(row.fact_checking_article),
                "annotated_evidence": evidence,
            }
        )
    return examples


def _partition_train(examples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    targets = {
        "Supported": {"train": 250, "policy_dev": 53, "calibration": 54},
        "Refuted": {"train": 564, "policy_dev": 121, "calibration": 121},
    }
    for label, counts in targets.items():
        rows = sorted(
            (row for row in examples if row["label"] == label),
            key=lambda row: _hash_key(SPLIT_SALT, str(row["example_id"])),
        )
        if len(rows) != sum(counts.values()):
            raise ValueError(f"unexpected eligible {label} count: {len(rows)}")
        cursor = 0
        for split in ("train", "policy_dev", "calibration"):
            for row in rows[cursor : cursor + counts[split]]:
                assignments[str(row["example_id"])] = split
            cursor += counts[split]
    return assignments


def _root_overlap(claim: str, root: str) -> float:
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    root_tokens = set(re.findall(r"[a-z0-9]+", root.lower()))
    return len(claim_tokens & root_tokens) / max(1, len(root_tokens))


def _evidence_rows(texts: Sequence[str], prefix: str) -> list[dict[str, str]]:
    return [
        {"evidence_id": f"{prefix}{index:02d}", "text": text} for index, text in enumerate(texts)
    ]


def _prepare_partition(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    vectorizer = HashingVectorizer(
        n_features=HASH_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        alternate_sign=False,
        norm="l2",
    )
    packet_entries: list[tuple[str, str, list[str]]] = []
    for row in rows:
        for root, texts in row["annotated_evidence"].items():
            packet_entries.append((str(row["example_id"]), root, texts))
    packet_matrix = vectorizer.transform(" ".join(texts) for _, _, texts in packet_entries)
    claim_matrix = vectorizer.transform(str(row["claim"]) for row in rows)
    packet_index = {
        (example_id, root): index for index, (example_id, root, _texts) in enumerate(packet_entries)
    }
    staged: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        roots = sorted(row["annotated_evidence"])
        anchor = min(roots, key=lambda root: _hash_key(ANCHOR_SALT, f"{example_id}\0{root}"))
        heldout = min(
            (root for root in roots if root != anchor),
            key=lambda root: _hash_key(HELDOUT_SALT, f"{example_id}\0{root}"),
        )
        scores = np.asarray(linear_kernel(claim_matrix[row_index], packet_matrix)).ravel()
        heldout_score = float(scores[packet_index[(example_id, heldout)]])
        distractor_choices = []
        annotated_roots = set(roots)
        for candidate_index, (other_id, root, texts) in enumerate(packet_entries):
            if other_id == example_id or root in annotated_roots:
                continue
            score = float(scores[candidate_index])
            distractor_choices.append(
                (
                    abs(score - heldout_score),
                    _hash_key(DISTRACTOR_SALT, f"{example_id}\0{other_id}\0{root}"),
                    -score,
                    root,
                    texts,
                )
            )
        if not distractor_choices:
            raise ValueError(f"no distractor for {example_id}")
        _distance, _tie, negative_score, distractor_root, distractor_texts = min(distractor_choices)
        staged.append(
            {
                **{key: value for key, value in row.items() if key != "annotated_evidence"},
                "split": split,
                "anchor_raw": {
                    "root": anchor,
                    "retrieval_score": float(scores[packet_index[(example_id, anchor)]]),
                    "title_overlap": _root_overlap(str(row["claim"]), anchor),
                    "texts": row["annotated_evidence"][anchor],
                },
                "heldout_raw": {
                    "root": heldout,
                    "retrieval_score": heldout_score,
                    "title_overlap": _root_overlap(str(row["claim"]), heldout),
                    "texts": row["annotated_evidence"][heldout],
                },
                "distractor_raw": {
                    "root": distractor_root,
                    "retrieval_score": -negative_score,
                    "title_overlap": _root_overlap(str(row["claim"]), distractor_root),
                    "texts": distractor_texts,
                },
            }
        )
    ordered = sorted(
        staged,
        key=lambda row: _hash_key(ORDER_SALT, f"{split}\0{row['example_id']}"),
    )
    annotated_at_zero = {str(row["example_id"]) for row in ordered[: math.ceil(len(ordered) / 2)]}
    selected = []
    for row in staged:
        heldout = row.pop("heldout_raw")
        distractor = row.pop("distractor_raw")
        raw_candidates = (
            [heldout, distractor]
            if row["example_id"] in annotated_at_zero
            else [distractor, heldout]
        )
        candidates = []
        for index, raw in enumerate(raw_candidates):
            candidates.append(
                {
                    "root": raw["root"],
                    "annotation_role": (
                        "held_out_annotated_root"
                        if raw["root"] == heldout["root"]
                        else "unannotated_retrieval_candidate"
                    ),
                    "retrieval_score": raw["retrieval_score"],
                    "title_overlap": raw["title_overlap"],
                    "evidence": _evidence_rows(raw["texts"], f"C{index}"),
                }
            )
        anchor_raw = row.pop("anchor_raw")
        row["anchor"] = {
            "root": anchor_raw["root"],
            "retrieval_score": anchor_raw["retrieval_score"],
            "title_overlap": anchor_raw["title_overlap"],
            "evidence": _evidence_rows(anchor_raw["texts"], "A"),
        }
        row["candidates"] = candidates
        selected.append(row)
    return sorted(selected, key=lambda row: str(row["example_id"]))


def build_selection(
    train_path: Path = TRAIN_DATASET,
    test_path: Path = TEST_DATASET,
) -> dict[str, Any]:
    train_pool = load_eligible(train_path, source_split="official_train")
    test_pool = load_eligible(test_path, source_split="official_dev")
    test_claims = {str(row["claim"]) for row in test_pool}
    seen_claims: set[str] = set()
    deduplicated_train = []
    for row in sorted(train_pool, key=lambda item: int(item["source_row_index"])):
        claim = str(row["claim"])
        if claim in test_claims or claim in seen_claims:
            continue
        seen_claims.add(claim)
        deduplicated_train.append(row)
    train_pool = deduplicated_train
    assignments = _partition_train(train_pool)
    partitions: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in train_pool:
        partitions[assignments[str(row["example_id"])]].append(row)
    partitions["test"] = test_pool
    examples = []
    for split in SPLITS:
        examples.extend(_prepare_partition(partitions[split], split))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "selection_frozen_before_v3_qwen_calls",
        "datasets": {
            "official_train": {"path": str(train_path), "sha256": TRAIN_SHA256},
            "official_dev_prospective_test": {"path": str(test_path), "sha256": TEST_SHA256},
        },
        "model": "Qwen3.5-4B",
        "endpoint": RELOCATED_RUNTIME_ENDPOINT,
        "root_definition": "wayback-unwrapped registrable-domain approximation",
        "expected_calls_per_example": 8,
        "examples": examples,
    }


def validate_selection(selection: Mapping[str, Any]) -> None:
    if dict(selection) != build_selection():
        raise ValueError("Recovery V3 selection or source data drifted")


def audit_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    examples = list(selection["examples"])
    counts = Counter(str(row["split"]) for row in examples)
    labels = {
        split: dict(Counter(str(row["label"]) for row in examples if row["split"] == split))
        for split in SPLITS
    }
    claim_sets = {
        split: {str(row["claim"]) for row in examples if row["split"] == split} for split in SPLITS
    }
    claim_overlap = {
        f"{left}_{right}": len(claim_sets[left] & claim_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    distinct_roots = all(
        len({row["anchor"]["root"], *(candidate["root"] for candidate in row["candidates"])}) == 3
        for row in examples
    )
    position_fractions = {}
    role_labels = []
    role_scores = []
    for split in SPLITS:
        rows = [row for row in examples if row["split"] == split]
        position_fractions[split] = sum(
            row["candidates"][0]["annotation_role"] == "held_out_annotated_root" for row in rows
        ) / len(rows)
        for row in rows:
            for candidate in row["candidates"]:
                role_labels.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
                role_scores.append(float(candidate["retrieval_score"]))
    auc = float(roc_auc_score(role_labels, role_scores))
    oriented_auc = max(auc, 1.0 - auc)
    test_domains = {
        root
        for row in examples
        if row["split"] == "test"
        for root in [row["anchor"]["root"], *(c["root"] for c in row["candidates"])]
    }
    gates = {
        "exact_expected_counts": dict(counts) == EXPECTED_COUNTS,
        "zero_claim_overlap": not any(claim_overlap.values()),
        "three_distinct_roots": distinct_roots,
        "candidate_order_balanced": all(
            0.45 <= fraction <= 0.55 for fraction in position_fractions.values()
        ),
        "each_label_at_least_025": all(
            min(group.values()) / sum(group.values()) >= 0.25 for group in labels.values()
        ),
        "oriented_role_auc_at_most_085": oriented_auc <= 0.85,
        "test_at_least_200": counts["test"] >= 200,
        "test_domains_at_least_300": len(test_domains) >= 300,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "counts": dict(counts),
        "labels": labels,
        "claim_overlap": claim_overlap,
        "candidate_0_annotated_fraction": position_fractions,
        "retrieval_role_auc": auc,
        "oriented_retrieval_role_auc": oriented_auc,
        "test_distinct_domains": len(test_domains),
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection(output: Path) -> bool:
    expected = build_selection()
    if output.exists():
        validate_selection(json.loads(output.read_text(encoding="utf-8")))
        return False
    _write_json(output, expected)
    return True


def _packet(evidence: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"[{item['evidence_id']}] {item['text']}" for item in evidence)


def build_baseline_messages(
    example: Mapping[str, Any],
    agent_index: int,
    *,
    repair: bool = False,
) -> list[dict[str, str]]:
    system = (
        "You are one independent evidence-bound fact checker. Decide whether the claim is "
        "supported by the packet. Do not use source popularity or other agents. Return exactly "
        "one JSON object with keys answer, confidence, cited_evidence_ids and no reasoning: "
        '{"answer":"yes|no","confidence":0.0,"cited_evidence_ids":["A00"]}.'
    )
    user = (
        f"Decision style: {PERSONAS[agent_index]}\n"
        f"Claim: {example['claim']}\n\n"
        f"EVIDENCE PACKET:\n{_packet(example['anchor']['evidence'])}\n\n"
        "Use only packet evidence. Cite only packet IDs."
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _action_evidence(example: Mapping[str, Any], action: str) -> list[Mapping[str, Any]]:
    if action == "candidate_0":
        return list(example["candidates"][0]["evidence"])
    if action == "candidate_1":
        return list(example["candidates"][1]["evidence"])
    if action == "both":
        return [*example["candidates"][0]["evidence"], *example["candidates"][1]["evidence"]]
    raise ValueError(f"unknown recovery action: {action}")


def build_recovery_messages(
    example: Mapping[str, Any],
    action: str,
    consensus: str,
    *,
    repair: bool = False,
) -> list[dict[str, str]]:
    acquired = _action_evidence(example, action)
    system = (
        "You are a provenance-aware fact-recovery adjudicator. Re-evaluate the claim from the "
        "anchor and newly acquired independent source packets. Repeated agreement is not "
        "evidence. Return exactly one JSON object with keys answer, confidence, "
        "cited_evidence_ids and no reasoning."
    )
    user = (
        f"Claim: {example['claim']}\n"
        f"Previous anchor-only consensus: {consensus}\n\n"
        f"ANCHOR PACKET:\n{_packet(example['anchor']['evidence'])}\n\n"
        f"ACQUIRED PACKET:\n{_packet(acquired)}\n\n"
        "The prior consensus is a hypothesis, not evidence. Answer yes or no, give numeric "
        "confidence in [0,1], and cite only IDs in these packets."
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_example(client: RecoveryChatClient, example: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_ids = [str(item["evidence_id"]) for item in example["anchor"]["evidence"]]
    for agent_index in range(len(PERSONAS)):
        decision, attempts, final_error = _call_with_retry(
            client,
            lambda repair, index=agent_index: build_baseline_messages(
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
    consensus, agreement = _majority(rows)
    for action in RECOVERY_ACTIONS:
        acquired = _action_evidence(example, action)
        allowed_ids = [*anchor_ids, *(str(item["evidence_id"]) for item in acquired)]
        decision, attempts, final_error = _call_with_retry(
            client,
            lambda repair, action_name=action: build_recovery_messages(
                example, action_name, consensus, repair=repair
            ),
            allowed_ids,
            seed=CALL_SEEDS[action],
        )
        indices = [0, 1] if action == "both" else [int(action[-1])]
        annotated_ids = {
            str(item["evidence_id"])
            for index in indices
            if example["candidates"][index]["annotation_role"] == "held_out_annotated_root"
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


def _require_train_audit(split: str) -> None:
    if split == "train":
        return
    audit_path = DEFAULT_ROOT / "train" / "structural_audit.json"
    records_path = DEFAULT_ROOT / "train" / "records.jsonl"
    if not audit_path.exists() or not records_path.exists():
        raise ValueError(f"{split} calls require completed train records and audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("records_sha256") != _sha256_path(records_path):
        raise ValueError(f"{split} calls blocked by frozen train audit")


def execute_split(
    selection_path: Path,
    *,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3 selection failed pre-call gates")
    if not smoke:
        _require_train_audit(split)
        if split == "test":
            manifest_path = DEFAULT_ROOT / "router" / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("test calls require a frozen V3 router manifest")
            validate_router_manifest(manifest_path, selection_path)
    examples = [row for row in selection["examples"] if row["split"] == split]
    if smoke:
        examples = examples[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = _load_jsonl(partial_path) if partial_path.exists() else []
    expected_per_example = len(PERSONAS) + len(RECOVERY_ACTIONS)
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_example[str(row["example_id"])].append(row)
    if any(len(rows) != expected_per_example for rows in by_example.values()):
        raise ValueError("partial file contains an incomplete example bundle")
    existing = {
        example_id: rows
        for example_id, rows in by_example.items()
        if all(row.get("success") for row in rows)
    }
    records = [row for rows in existing.values() for row in rows]
    allowed_ids = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed_ids:
        raise ValueError("partial file contains examples outside this run")
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = RecoveryChatClient(RELOCATED_RUNTIME_ENDPOINT, "Qwen3.5-4B", cache_dir)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_run_example, client, row): row for row in pending}
        for future in as_completed(futures):
            bundle = future.result()
            records.extend(bundle)
            records.sort(
                key=lambda row: (
                    str(row["example_id"]),
                    str(row["phase"]),
                    -1 if row["agent_index"] is None else int(row["agent_index"]),
                    str(row["action"]),
                )
            )
            _write_jsonl(partial_path, records)
            print(
                f"[{len(records) // expected_per_example}/{len(examples)}] "
                f"{bundle[0]['example_id']} success={all(r['success'] for r in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    expected = len(examples) * expected_per_example
    if len(records) != expected or any(not row["success"] for row in records):
        raise ValueError(f"split run incomplete or invalid: {len(records)}/{expected}")
    _write_jsonl(output_dir / "records.jsonl", records)
    return records


def _record_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["example_id"])].append(row)
    return dict(grouped)


def _validate_action_matrix(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    grouped = _record_groups(records)
    expected = {str(example["example_id"]): example for example in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} records do not cover the frozen V3 partition")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        if len(rows) != len(PERSONAS) + len(RECOVERY_ACTIONS):
            raise ValueError(f"{split} example {example_id} has an invalid bundle size")
        if {row.get("agent_index") for row in baseline} != set(range(len(PERSONAS))):
            raise ValueError(f"{split} example {example_id} has invalid baseline agents")
        if {row.get("action") for row in recovery} != set(RECOVERY_ACTIONS):
            raise ValueError(f"{split} example {example_id} has invalid recovery actions")
        example = expected[example_id]
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("split") != split
            or row.get("gold_binary") != example["gold_binary"]
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"{split} example {example_id} has invalid record metadata")


def _outcomes(
    example: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, float, dict[str, int], list[Mapping[str, Any]]]:
    consensus, agreement, baseline = _baseline_state(records)
    gold = int(example["gold_binary"])
    keep = int((consensus == "yes") == bool(gold))
    action_outcomes = {
        str(row["action"]): int((row["decision"]["answer"] == "yes") == bool(gold))
        for row in records
        if row["phase"] == "recovery"
    }
    if set(action_outcomes) != set(RECOVERY_ACTIONS):
        raise ValueError(f"missing recovery outcomes for {example['example_id']}")
    return keep, agreement, action_outcomes, baseline


def _baseline_state(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, float, list[Mapping[str, Any]]]:
    baseline = sorted(
        (row for row in records if row["phase"] == "baseline"),
        key=lambda row: int(row["agent_index"]),
    )
    consensus, agreement = _majority(baseline)
    return consensus, agreement, baseline


def audit_train_structure(
    selection_path: Path,
    records_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    examples = {
        str(row["example_id"]): row for row in selection["examples"] if row["split"] == "train"
    }
    records = _load_jsonl(records_path)
    _validate_action_matrix(list(examples.values()), records, split="train")
    grouped = _record_groups(records)
    high = wrong = repairable = harm_exposed = 0
    for example_id, example in examples.items():
        keep, agreement, outcomes, _baseline = _outcomes(example, grouped[example_id])
        if agreement < HIGH_CONSENSUS:
            continue
        high += 1
        if keep == 0:
            wrong += 1
            repairable += int(any(value == 1 for value in outcomes.values()))
        else:
            harm_exposed += int(any(value == 0 for value in outcomes.values()))

    gates = {
        "high_consensus_at_least_300": high >= 300,
        "high_consensus_wrong_at_least_50": wrong >= 50,
        "repairable_at_least_25": repairable >= 25,
        "harm_exposed_at_least_25": harm_exposed >= 25,
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "records_sha256": _sha256_path(records_path),
        "n_train": len(examples),
        "high_consensus": high,
        "high_consensus_wrong": wrong,
        "repairable": repairable,
        "harm_exposed": harm_exposed,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output_path, report)
    return report


def _stance_text(example: Mapping[str, Any], packet: Mapping[str, Any]) -> str:
    evidence = " ".join(str(item["text"]) for item in packet["evidence"])
    return f"CLAIM: {example['claim']} EVIDENCE: {evidence}"


def _stance_packets(
    examples: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    packets = []
    for example in examples:
        example_id = str(example["example_id"])
        native_stance = "supports" if example["label"] == "Supported" else "refutes"
        packets.append(
            (
                example_id,
                "anchor",
                _stance_text(example, example["anchor"]),
                native_stance,
            )
        )
        for index, candidate in enumerate(example["candidates"]):
            target = (
                native_stance
                if candidate["annotation_role"] == "held_out_annotated_root"
                else "irrelevant"
            )
            packets.append(
                (
                    example_id,
                    f"candidate_{index}",
                    _stance_text(example, candidate),
                    target,
                )
            )
    return packets


def _new_stance_model() -> Any:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_500,
            random_state=MODEL_SEED,
            solver="liblinear",
        )
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _standard_stance_probabilities(model: Any, texts: Sequence[str]) -> np.ndarray:
    raw = model.predict_proba(texts)
    classes = list(model.named_steps["classifier"].classes_)
    return np.column_stack([raw[:, classes.index(name)] for name in STANCE_CLASSES])


def _oof_stance_predictions(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], np.ndarray], Any]:
    example_labels = [str(example["label"]) for example in examples]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_SEED)
    output: dict[tuple[str, str], np.ndarray] = {}
    example_array = np.asarray(list(examples), dtype=object)
    for train_indices, heldout_indices in folds.split(np.zeros(len(examples)), example_labels):
        train_packets = _stance_packets(example_array[train_indices].tolist())
        heldout_packets = _stance_packets(example_array[heldout_indices].tolist())
        model = _new_stance_model()
        model.fit([row[2] for row in train_packets], [row[3] for row in train_packets])
        probabilities = _standard_stance_probabilities(model, [row[2] for row in heldout_packets])
        for row, probability in zip(heldout_packets, probabilities, strict=True):
            output[(row[0], row[1])] = probability
    full_packets = _stance_packets(examples)
    full_model = _new_stance_model()
    full_model.fit([row[2] for row in full_packets], [row[3] for row in full_packets])
    return output, full_model


def _stance_predictions(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], np.ndarray]:
    packets = _stance_packets(examples)
    probabilities = _standard_stance_probabilities(model, [row[2] for row in packets])
    return {
        (row[0], row[1]): probability
        for row, probability in zip(packets, probabilities, strict=True)
    }


def _feature_vector(
    example: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    action: str,
    stance: Mapping[tuple[str, str], np.ndarray],
    *,
    include_stance: bool = True,
) -> tuple[list[str], np.ndarray]:
    names, values = _v2_feature_vector(example, baseline, action)
    if not include_stance:
        return names, values
    example_id = str(example["example_id"])
    consensus, _agreement = _majority(baseline)
    indices = [0, 1] if action == "both" else [int(action[-1])]
    anchor = stance[(example_id, "anchor")]
    candidates = np.vstack([stance[(example_id, f"candidate_{index}")] for index in indices])
    support = candidates[:, STANCE_CLASSES.index("supports")]
    refute = candidates[:, STANCE_CLASSES.index("refutes")]
    irrelevant = candidates[:, STANCE_CLASSES.index("irrelevant")]
    agrees = support if consensus == "yes" else refute
    opposes = refute if consensus == "yes" else support
    anchor_agrees = anchor[2] if consensus == "yes" else anchor[1]
    anchor_opposes = anchor[1] if consensus == "yes" else anchor[2]
    stance_names = [
        "stance_support_max",
        "stance_support_mean",
        "stance_refute_max",
        "stance_refute_mean",
        "stance_irrelevant_min",
        "stance_irrelevant_mean",
        "stance_agrees_consensus_max",
        "stance_opposes_consensus_max",
        "stance_anchor_agrees_consensus",
        "stance_anchor_opposes_consensus",
        "stance_opposition_delta",
    ]
    stance_values = np.asarray(
        [
            support.max(),
            support.mean(),
            refute.max(),
            refute.mean(),
            irrelevant.min(),
            irrelevant.mean(),
            agrees.max(),
            opposes.max(),
            anchor_agrees,
            anchor_opposes,
            opposes.max() - anchor_opposes,
        ]
    )
    all_names = [*names, *stance_names]
    if any(fragment in name for name in all_names for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
        raise AssertionError("forbidden post-outcome field entered CAPE")
    return all_names, np.concatenate([values, stance_values])


def _feature_matrix(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    stance: Mapping[tuple[str, str], np.ndarray],
    *,
    include_stance: bool = True,
) -> tuple[list[str], np.ndarray, list[tuple[str, str]]]:
    grouped = _record_groups(records)
    rows = []
    keys = []
    feature_names: list[str] | None = None
    for example in examples:
        example_id = str(example["example_id"])
        _consensus, _agreement, baseline = _baseline_state(grouped[example_id])
        for action in RECOVERY_ACTIONS:
            names, values = _feature_vector(
                example, baseline, action, stance, include_stance=include_stance
            )
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("CAPE feature schema drifted")
            rows.append(values)
            keys.append((example_id, action))
    return feature_names or [], np.vstack(rows), keys


def _outcome_targets(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    grouped = _record_groups(records)
    gains = []
    harms = []
    keys = []
    for example in examples:
        example_id = str(example["example_id"])
        keep, _agreement, outcomes, _baseline = _outcomes(example, grouped[example_id])
        for action in RECOVERY_ACTIONS:
            gains.append(outcomes[action] - keep)
            harms.append(int(keep == 1 and outcomes[action] == 0))
            keys.append((example_id, action))
    return np.asarray(gains, dtype=float), np.asarray(harms, dtype=int), keys


def _training_matrix(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    stance: Mapping[tuple[str, str], np.ndarray],
    *,
    include_stance: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    names, matrix, feature_keys = _feature_matrix(
        examples, records, stance, include_stance=include_stance
    )
    gains, harms, outcome_keys = _outcome_targets(examples, records)
    if feature_keys != outcome_keys:
        raise AssertionError("CAPE feature and outcome rows are misaligned")
    return names, matrix, gains, harms, feature_keys


def _new_outcome_models() -> dict[str, Any]:
    return {
        "gain_rf": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=MODEL_SEED + 1,
            n_jobs=-1,
        ),
        "gain_hgb": HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=MODEL_SEED + 2,
        ),
        "harm_rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.8,
            class_weight="balanced",
            random_state=MODEL_SEED + 3,
            n_jobs=-1,
        ),
        "harm_hgb": HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=MODEL_SEED + 4,
        ),
    }


def _fit_outcome_models(x: np.ndarray, gain: np.ndarray, harm: np.ndarray) -> dict[str, Any]:
    if len(set(harm.tolist())) < 2:
        raise ValueError("harm model requires both classes")
    models = _new_outcome_models()
    models["gain_rf"].fit(x, gain)
    models["gain_hgb"].fit(x, gain)
    models["harm_rf"].fit(x, harm)
    models["harm_hgb"].fit(x, harm)
    return models


def _predict_outcomes(models: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gain = 0.5 * (models["gain_rf"].predict(x) + models["gain_hgb"].predict(x))
    harm = 0.5 * (
        models["harm_rf"].predict_proba(x)[:, 1] + models["harm_hgb"].predict_proba(x)[:, 1]
    )
    return gain, harm


def _prediction_map(
    keys: Sequence[tuple[str, str]],
    gain: np.ndarray,
    harm: np.ndarray,
) -> dict[tuple[str, str], tuple[float, float]]:
    return {key: (float(gain[index]), float(harm[index])) for index, key in enumerate(keys)}


def _select_policy(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    *,
    threshold_yes: float,
    threshold_no: float,
    harm_cap: float,
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = _baseline_state(grouped[example_id])
        if agreement < HIGH_CONSENSUS:
            selected[example_id] = "KEEP"
            continue
        threshold = threshold_yes if consensus == "yes" else threshold_no
        allowed = []
        for action in RECOVERY_ACTIONS:
            gain, harm = predictions[(example_id, action)]
            roots = 2 if action == "both" else 1
            utility = gain - ROOT_COST * roots
            if utility > threshold and harm <= harm_cap:
                allowed.append((utility, -roots, action))
        selected[example_id] = max(allowed)[2] if allowed else "KEEP"
    return selected


def _basic_policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    by_label: dict[str, list[int]] = defaultdict(list)
    gains = []
    harms = 0
    high_correct = 0
    routes = 0
    roots = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = _outcomes(example, grouped[example_id])
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        gain = final - keep
        gains.append(gain)
        by_label[str(example["label"])].append(gain)
        if agreement >= HIGH_CONSENSUS and keep == 1:
            high_correct += 1
            harms += int(final == 0)
        if action != "KEEP":
            routes += 1
            roots += 2 if action == "both" else 1
    return {
        "net_gain": float(np.mean(gains)),
        "net_fixes": int(sum(gains)),
        "damage_rate": harms / max(1, high_correct),
        "routes": routes,
        "mean_roots": roots / len(examples),
        "by_label_gain": {label: float(np.mean(values)) for label, values in by_label.items()},
    }


def _tune_policy(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
) -> dict[str, Any]:
    grouped = _record_groups(records)
    feasible = []
    for threshold_yes in GAIN_GRID:
        for threshold_no in GAIN_GRID:
            for harm_cap in HARM_GRID:
                selected = _select_policy(
                    examples,
                    grouped,
                    predictions,
                    threshold_yes=threshold_yes,
                    threshold_no=threshold_no,
                    harm_cap=harm_cap,
                )
                metrics = _basic_policy_metrics(examples, grouped, selected)
                if (
                    metrics["damage_rate"] <= 0.05
                    and min(metrics["by_label_gain"].values()) >= 0.0
                    and metrics["routes"] >= 10
                ):
                    objective = metrics["net_gain"] - ROOT_COST * metrics["mean_roots"]
                    feasible.append(
                        (
                            objective,
                            -metrics["damage_rate"],
                            -metrics["mean_roots"],
                            threshold_yes + threshold_no,
                            threshold_yes,
                            threshold_no,
                            harm_cap,
                            metrics,
                        )
                    )
    if not feasible:
        return {
            "threshold_yes": 1_000_000.0,
            "threshold_no": 1_000_000.0,
            "harm_cap": 0.0,
            "metrics": None,
            "fallback_keep": True,
        }
    best = max(feasible, key=lambda row: row[:7])
    return {
        "threshold_yes": best[4],
        "threshold_no": best[5],
        "harm_cap": best[6],
        "metrics": best[7],
        "fallback_keep": False,
    }


def fit_router(
    selection_path: Path,
    train_records_path: Path,
    policy_records_path: Path,
    calibration_records_path: Path,
    train_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
    if not train_audit.get("passed"):
        raise ValueError("V3 train structural gate did not pass")
    split_examples = {
        split: [row for row in selection["examples"] if row["split"] == split] for split in SPLITS
    }
    train_records = _load_jsonl(train_records_path)
    policy_records = _load_jsonl(policy_records_path)
    calibration_records = _load_jsonl(calibration_records_path)
    _validate_action_matrix(split_examples["train"], train_records, split="train")
    _validate_action_matrix(split_examples["policy_dev"], policy_records, split="policy_dev")
    _validate_action_matrix(split_examples["calibration"], calibration_records, split="calibration")
    oof_stance, stance_model = _oof_stance_predictions(split_examples["train"])
    train_names, x_train, gain_train, harm_train, _train_keys = _training_matrix(
        split_examples["train"], train_records, oof_stance
    )
    models = _fit_outcome_models(x_train, gain_train, harm_train)
    no_stance_names, x_train_no, _gain, _harm, _keys = _training_matrix(
        split_examples["train"], train_records, oof_stance, include_stance=False
    )
    no_stance_models = _fit_outcome_models(x_train_no, gain_train, harm_train)
    policy_stance = _stance_predictions(stance_model, split_examples["policy_dev"])
    policy_names, x_policy, _gain, _harm, policy_keys = _training_matrix(
        split_examples["policy_dev"], policy_records, policy_stance
    )
    if policy_names != train_names:
        raise ValueError("train and policy feature schemas differ")
    policy_gain, policy_harm = _predict_outcomes(models, x_policy)
    tuned = _tune_policy(
        split_examples["policy_dev"],
        policy_records,
        _prediction_map(policy_keys, policy_gain, policy_harm),
    )
    calibration_stance = _stance_predictions(stance_model, split_examples["calibration"])
    calibration_names, x_calibration, _gain, _harm, calibration_keys = _training_matrix(
        split_examples["calibration"], calibration_records, calibration_stance
    )
    if calibration_names != train_names:
        raise ValueError("train and calibration feature schemas differ")
    calibration_gain, calibration_harm = _predict_outcomes(models, x_calibration)
    calibration_predictions = _prediction_map(calibration_keys, calibration_gain, calibration_harm)
    calibration_grouped = _record_groups(calibration_records)
    shield = None
    for offset in SHIELD_OFFSETS:
        selected = _select_policy(
            split_examples["calibration"],
            calibration_grouped,
            calibration_predictions,
            threshold_yes=float(tuned["threshold_yes"]) + offset,
            threshold_no=float(tuned["threshold_no"]) + offset,
            harm_cap=float(tuned["harm_cap"]),
        )
        metrics = _basic_policy_metrics(
            split_examples["calibration"], calibration_grouped, selected
        )
        if metrics["damage_rate"] <= 0.05 and min(metrics["by_label_gain"].values()) >= 0:
            shield = {"offset": offset, "metrics": metrics, "fallback_keep": False}
            break
    if shield is None:
        shield = {"offset": 1_000_000.0, "metrics": None, "fallback_keep": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "router.joblib"
    joblib.dump(
        {
            "stance_model": stance_model,
            "outcome_models": models,
            "no_stance_models": no_stance_models,
        },
        model_path,
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_test_calls",
        "selection_sha256": _sha256_path(selection_path),
        "train_records_sha256": _sha256_path(train_records_path),
        "policy_records_sha256": _sha256_path(policy_records_path),
        "calibration_records_sha256": _sha256_path(calibration_records_path),
        "train_audit_sha256": _sha256_path(train_audit_path),
        "feature_names": train_names,
        "no_stance_feature_names": no_stance_names,
        "forbidden_feature_intersection": [
            name
            for name in train_names
            if any(fragment in name for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
        ],
        "training_rows": len(x_train),
        "stance_training": {
            "family": "five_fold_oof_word_char_tfidf_logistic",
            "classes": list(STANCE_CLASSES),
            "seed": MODEL_SEED,
        },
        "outcome_model": {
            "family": "equal_weight_rf_hgb_gain_and_harm_ensemble",
            "seed": MODEL_SEED,
        },
        "tuned_policy": tuned,
        "safety_shield": shield,
        "final_policy": {
            "threshold_yes": float(tuned["threshold_yes"]) + float(shield["offset"]),
            "threshold_no": float(tuned["threshold_no"]) + float(shield["offset"]),
            "harm_cap": float(tuned["harm_cap"]),
            "root_cost": ROOT_COST,
        },
        "router_joblib": str(model_path),
        "router_joblib_sha256": _sha256_path(model_path),
        "claim_boundary": {"test_outcomes_seen": False, "publisher_independence": False},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_router_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V3 router protocol mismatch")
    if manifest.get("status") != "frozen_before_test_calls":
        raise ValueError("V3 router status is not frozen-before-test")
    if manifest.get("selection_sha256") != _sha256_path(selection_path):
        raise ValueError("V3 router selection drifted")
    model_path = Path(str(manifest["router_joblib"]))
    if manifest.get("router_joblib_sha256") != _sha256_path(model_path):
        raise ValueError("V3 serialized router drifted")
    if manifest.get("forbidden_feature_intersection"):
        raise ValueError("V3 router contains forbidden features")


def _policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps = []
    finals = []
    labels = []
    high_flags = []
    action_counts = Counter()
    annotation_supported = 0
    roots_added = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = _outcomes(example, grouped[example_id])
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        keeps.append(keep)
        finals.append(final)
        labels.append(str(example["label"]))
        high_flags.append(agreement >= HIGH_CONSENSUS)
        action_counts[action] += 1
        roots_added += 0 if action == "KEEP" else 2 if action == "both" else 1
        if keep == 0 and final == 1 and action != "KEEP":
            recovery_row = next(
                row
                for row in grouped[example_id]
                if row["phase"] == "recovery" and row["action"] == action
            )
            citations = set(recovery_row["decision"]["cited_evidence_ids"])
            annotated = set(recovery_row["annotated_evidence_ids"])
            annotation_supported += int(
                recovery_row["packet_contains_annotated_root"] and bool(citations & annotated)
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
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        group_gains = []
        for indices in label_indices.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            group_gains.append(float(gains[sampled].mean()))
        bootstrap.append(float(np.mean(group_gains)))
    bootstrap_interval = np.quantile(
        np.asarray(bootstrap), [0.025, 0.975], method="linear"
    ).tolist()
    high_correct = sum(bool(keep and high) for keep, high in zip(keeps, high_flags, strict=True))
    high_harms = sum(
        bool(keep and high and not final)
        for keep, high, final in zip(keeps, high_flags, finals, strict=True)
    )
    macro_gain = float(np.mean([group["net_gain"] for group in by_label.values()]))
    return {
        "n": len(examples),
        "baseline_accuracy": float(keep_array.mean()),
        "final_accuracy": float(final_array.mean()),
        "fixes": int(fixes.sum()),
        "harms": int(harms.sum()),
        "net_fixes": int(gains.sum()),
        "net_gain": float(gains.mean()),
        "macro_label_gain": macro_gain,
        "macro_gain_ci": bootstrap_interval,
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": roots_added,
        "mean_added_roots": roots_added / len(examples),
        "selected_actions": dict(action_counts),
        "by_native_label": by_label,
    }


def _truncate_to_budget(
    examples: Sequence[Mapping[str, Any]],
    proposed: Mapping[str, str],
    budget: int,
    *,
    name: str,
) -> dict[str, str]:
    selected = {str(row["example_id"]): "KEEP" for row in examples}

    def priority(row: Mapping[str, Any]) -> tuple[Any, ...]:
        example_id = str(row["example_id"])
        tie_breaker = _hash_key(b"cape-v3-budget-matched-baseline", f"{name}\0{example_id}")
        if name == "retrieval_score":
            action = proposed[example_id]
            retrieval_score = float(row["candidates"][int(action[-1])]["retrieval_score"])
            return (-retrieval_score, tie_breaker)
        return (tie_breaker,)

    ordered = sorted(
        (row for row in examples if proposed[str(row["example_id"])] != "KEEP"),
        key=priority,
    )
    spent = 0
    for row in ordered:
        example_id = str(row["example_id"])
        action = proposed[example_id]
        cost = 2 if action == "both" else 1
        if spent + cost <= budget:
            selected[example_id] = action
            spent += cost
    return selected


def _unrestricted_policy(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        _consensus, agreement, _baseline = _baseline_state(grouped[example_id])
        scored = [
            (
                predictions[(example_id, action)][0] - ROOT_COST * (2 if action == "both" else 1),
                action,
            )
            for action in RECOVERY_ACTIONS
        ]
        score, action = max(scored)
        selected[example_id] = action if agreement >= HIGH_CONSENSUS and score > 0 else "KEEP"
    return selected


def _truncate_cape_to_budget(
    examples: Sequence[Mapping[str, Any]],
    selected: Mapping[str, str],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    budget: int,
) -> dict[str, str]:
    output = {str(example["example_id"]): "KEEP" for example in examples}
    ranked = []
    for example in examples:
        example_id = str(example["example_id"])
        action = selected[example_id]
        if action == "KEEP":
            continue
        cost = 2 if action == "both" else 1
        predicted_utility = predictions[(example_id, action)][0] - ROOT_COST * cost
        ranked.append(
            (
                -predicted_utility,
                _hash_key(b"cape-v3-cost-curve", example_id),
                example_id,
                action,
                cost,
            )
        )
    spent = 0
    for _negative_utility, _tie_breaker, example_id, action, cost in sorted(ranked):
        if spent + cost <= budget:
            output[example_id] = action
            spent += cost
    return output


def evaluate_test(
    selection_path: Path,
    test_records_path: Path,
    router_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    validate_router_manifest(router_manifest_path, selection_path)
    manifest = json.loads(router_manifest_path.read_text(encoding="utf-8"))
    examples = [row for row in selection["examples"] if row["split"] == "test"]
    records = _load_jsonl(test_records_path)
    _validate_action_matrix(examples, records, split="test")
    grouped = _record_groups(records)
    bundle = joblib.load(manifest["router_joblib"])
    stance = _stance_predictions(bundle["stance_model"], examples)
    names, x_test, keys = _feature_matrix(examples, records, stance)
    if names != manifest["feature_names"]:
        raise ValueError("test CAPE feature schema drifted")
    gain, harm = _predict_outcomes(bundle["outcome_models"], x_test)
    predictions = _prediction_map(keys, gain, harm)
    final_parameters = manifest["final_policy"]
    cape = _select_policy(
        examples,
        grouped,
        predictions,
        threshold_yes=float(final_parameters["threshold_yes"]),
        threshold_no=float(final_parameters["threshold_no"]),
        harm_cap=float(final_parameters["harm_cap"]),
    )
    tuned_parameters = manifest["tuned_policy"]
    unshielded = _select_policy(
        examples,
        grouped,
        predictions,
        threshold_yes=float(tuned_parameters["threshold_yes"]),
        threshold_no=float(tuned_parameters["threshold_no"]),
        harm_cap=float(tuned_parameters["harm_cap"]),
    )
    _no_names, x_no_stance, no_keys = _feature_matrix(
        examples, records, stance, include_stance=False
    )
    no_gain, no_harm = _predict_outcomes(bundle["no_stance_models"], x_no_stance)
    no_stance = _select_policy(
        examples,
        grouped,
        _prediction_map(no_keys, no_gain, no_harm),
        threshold_yes=float(final_parameters["threshold_yes"]),
        threshold_no=float(final_parameters["threshold_no"]),
        harm_cap=float(final_parameters["harm_cap"]),
    )
    proposals: dict[str, dict[str, str]] = {}
    proposals["retrieval_score"] = {}
    proposals["hash_random"] = {}
    for action in RECOVERY_ACTIONS:
        proposals[f"fixed_{action}"] = {}
    for example in examples:
        example_id = str(example["example_id"])
        _consensus, agreement, _baseline = _baseline_state(grouped[example_id])
        active = agreement >= HIGH_CONSENSUS
        proposals["retrieval_score"][example_id] = (
            max(
                ("candidate_0", "candidate_1"),
                key=lambda action: example["candidates"][int(action[-1])]["retrieval_score"],
            )
            if active
            else "KEEP"
        )
        proposals["hash_random"][example_id] = (
            (
                "candidate_0"
                if int(_hash_key(b"cape-v3-random-action", example_id), 16) % 2 == 0
                else "candidate_1"
            )
            if active
            else "KEEP"
        )
        for action in RECOVERY_ACTIONS:
            proposals[f"fixed_{action}"][example_id] = action if active else "KEEP"
    policies = {
        "cape_shielded": cape,
        "cape_unshielded": unshielded,
        "cape_no_stance": no_stance,
        "cape_unrestricted": _unrestricted_policy(examples, grouped, predictions),
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    cape_budget = sum(
        0 if action == "KEEP" else 2 if action == "both" else 1 for action in cape.values()
    )
    for name, proposed in proposals.items():
        policies[f"matched_{name}"] = _truncate_to_budget(
            examples, proposed, cape_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    _write_json(
        preoutcome_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "router_manifest_sha256": _sha256_path(router_manifest_path),
            "test_records_sha256": _sha256_path(test_records_path),
            "outcomes_accessed_by_route_selection": False,
            "policies": policies,
        },
    )
    oracle = {}
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = _outcomes(example, grouped[example_id])
        oracle[example_id] = (
            max(
                ("KEEP", *RECOVERY_ACTIONS),
                key=lambda action: keep if action == "KEEP" else outcomes[action],
            )
            if agreement >= HIGH_CONSENSUS
            else "KEEP"
        )
    policies["available_action_oracle_diagnostic"] = oracle
    metrics = {
        name: _policy_metrics(examples, grouped, selected) for name, selected in policies.items()
    }
    cost_curve = {}
    for roots_per_item in (0.25, 0.5, 1.0, 2.0):
        root_budget = math.floor(roots_per_item * len(examples))
        budgeted = _truncate_cape_to_budget(examples, cape, predictions, root_budget)
        cost_curve[str(roots_per_item)] = {
            "root_budget": root_budget,
            "metrics": _policy_metrics(examples, grouped, budgeted),
        }
    train_roots = {
        packet["root"]
        for example in selection["examples"]
        if example["split"] == "train"
        for packet in (example["anchor"], *example["candidates"])
    }
    publisher_groups = {
        "all_candidate_roots_seen_in_train": [
            example
            for example in examples
            if {candidate["root"] for candidate in example["candidates"]} <= train_roots
        ],
        "any_candidate_root_unseen_in_train": [
            example
            for example in examples
            if not {candidate["root"] for candidate in example["candidates"]} <= train_roots
        ],
    }
    publisher_subgroups = {
        name: _policy_metrics(rows, grouped, cape) for name, rows in publisher_groups.items()
    }
    true_gain, true_harm, outcome_keys = _outcome_targets(examples, records)
    if keys != outcome_keys:
        raise AssertionError("test predictions and outcomes are misaligned")
    prediction_diagnostics = {
        "action_rows": len(true_gain),
        "gain_mae": float(np.mean(np.abs(gain - true_gain))),
        "harm_brier": float(np.mean(np.square(harm - true_harm))),
        "harm_auroc": (
            float(roc_auc_score(true_harm, harm)) if len(set(true_harm.tolist())) == 2 else None
        ),
    }
    primary = metrics["cape_shielded"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": primary["damage_rate_high_consensus_correct"] <= 0.05,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": (primary["annotation_supported_repairs"] >= 10),
        "net_fixes_above_all_matched_baselines": primary["net_fixes"]
        > max(metrics[name]["net_fixes"] for name in matched_names),
    }
    unlimited_fixed = [
        metrics[f"unlimited_fixed_{action}"]["net_fixes"] for action in RECOVERY_ACTIONS
    ]
    secondary = {
        "raw_net_fixes_above_every_unlimited_fixed_action": (
            primary["net_fixes"] > max(unlimited_fixed)
        ),
        "root_budget": cape_budget,
        "matched_baseline_names": matched_names,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_averitec_dev_test",
        "selection_sha256": _sha256_path(selection_path),
        "router_manifest_sha256": _sha256_path(router_manifest_path),
        "test_records_sha256": _sha256_path(test_records_path),
        "preoutcome_routes_sha256": _sha256_path(preoutcome_path),
        "n_test": len(examples),
        "policies": metrics,
        "cape_cost_curve": cost_curve,
        "publisher_seen_unseen": publisher_subgroups,
        "outcome_prediction_diagnostics": prediction_diagnostics,
        "primary_gates": gates,
        "secondary_strict_checks": secondary,
        "passes": all(gates.values()),
        "verdict": "PASS_CAPE_V3" if all(gates.values()) else "NO_VERIFIED_CAPE_DOMINANCE",
        "claim_boundary": {
            "source_domain_provenance": True,
            "publisher_independence_proven": False,
            "cross_model": False,
            "live_retrieval": False,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    lines = [
        "# Recovery V3 CAPE test report",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Test examples: {len(examples)}",
        f"- CAPE root budget: {cape_budget}",
        "",
        "## Policies",
        "",
    ]
    for name, result in metrics.items():
        lines.append(
            f"- {name}: accuracy={result['final_accuracy']:.3f}, "
            f"fixes={result['fixes']}, harms={result['harms']}, "
            f"net={result['net_fixes']}, roots={result['total_added_roots']}"
        )
    lines.extend(["", "## Frozen primary gates", ""])
    lines.extend(f"- {name}: {passed}" for name, passed in gates.items())
    lines.extend(["", "## Prediction diagnostics", ""])
    lines.extend(f"- {name}: {value}" for name, value in prediction_diagnostics.items())
    lines.extend(["", "## Publisher-root subgroups", ""])
    for name, result in publisher_subgroups.items():
        lines.append(
            f"- {name}: n={result['n']}, net={result['net_fixes']}, gain={result['net_gain']:.3f}"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _print_mapping(value: Mapping[str, Any]) -> None:
    for key, item in value.items():
        print(f"{key}: {item}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze AVeriTeC V3 selection")
    prepare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit = subparsers.add_parser("audit", help="run pre-call structural gates")
    audit.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_audit.json")
    smoke = subparsers.add_parser("smoke", help="run two-example V3 parser smoke")
    smoke.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    smoke.add_argument("--split", choices=SPLITS, default="train")
    smoke.add_argument("--workers", type=int, default=2)
    run = subparsers.add_parser("run", help="collect a complete V3 action matrix")
    run.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--workers", type=int, default=4)
    train_audit = subparsers.add_parser("train-audit", help="audit benefit and harm support")
    train_audit.add_argument(
        "--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json"
    )
    train_audit.add_argument(
        "--records", type=Path, default=DEFAULT_ROOT / "train" / "records.jsonl"
    )
    train_audit.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "train" / "structural_audit.json"
    )
    fit = subparsers.add_parser("fit", help="fit and freeze the CAPE router")
    fit.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    fit.add_argument("--train-records", type=Path, default=DEFAULT_ROOT / "train" / "records.jsonl")
    fit.add_argument(
        "--policy-records", type=Path, default=DEFAULT_ROOT / "policy_dev" / "records.jsonl"
    )
    fit.add_argument(
        "--calibration-records",
        type=Path,
        default=DEFAULT_ROOT / "calibration" / "records.jsonl",
    )
    fit.add_argument(
        "--train-audit", type=Path, default=DEFAULT_ROOT / "train" / "structural_audit.json"
    )
    fit.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "router")
    evaluate = subparsers.add_parser("evaluate", help="evaluate the frozen CAPE router")
    evaluate.add_argument(
        "--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json"
    )
    evaluate.add_argument(
        "--test-records", type=Path, default=DEFAULT_ROOT / "test" / "records.jsonl"
    )
    evaluate.add_argument(
        "--router-manifest", type=Path, default=DEFAULT_ROOT / "router" / "manifest.json"
    )
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "evaluation")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(args.output)
        print(f"{'Wrote' if created else 'Reused'} frozen Recovery V3 selection: {args.output}")
        return 0
    if args.command == "audit":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        validate_selection(selection)
        report = audit_selection(selection)
        _write_json(args.output, report)
        _print_mapping(report)
        return 0 if report["passed"] else 2
    if args.command in {"smoke", "run"}:
        smoke_mode = args.command == "smoke"
        output_dir = DEFAULT_ROOT / ("smoke" if smoke_mode else args.split)
        execute_split(
            args.selection,
            split=args.split,
            output_dir=output_dir,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
            smoke=smoke_mode,
        )
        return 0
    if args.command == "train-audit":
        report = audit_train_structure(args.selection, args.records, args.output)
        _print_mapping(report)
        return 0 if report["passed"] else 2
    if args.command == "fit":
        manifest = fit_router(
            args.selection,
            args.train_records,
            args.policy_records,
            args.calibration_records,
            args.train_audit,
            args.output_dir,
        )
        _print_mapping(manifest)
        return 0
    if args.command == "evaluate":
        summary = evaluate_test(
            args.selection,
            args.test_records,
            args.router_manifest,
            args.output_dir,
        )
        _print_mapping(summary["primary_gates"])
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
