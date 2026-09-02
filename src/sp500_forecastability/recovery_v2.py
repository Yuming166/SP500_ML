"""Prospective page-root action routing for false-consensus recovery.

Recovery V2 is separate from the frozen V12.1 and Recovery V1 artifacts.  It
uses FEVER claims with exactly two annotated Wikipedia page roots, hides one
root from an initial five-agent consensus, and learns whether to acquire one of
two anonymized candidate roots or both.  Train/dev/test are page-root disjoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import linear_kernel

from sp500_forecastability.pilot_llm_v1 import (
    CachedChatClient,
    _attempt_payload,
    _extract_json_object,
    _write_json,
    _write_jsonl,
)
from sp500_forecastability.pilot_llm_v10 import DEFAULT_ENDPOINT, DEFAULT_MODEL

PROTOCOL_VERSION = "recovery-v2.2-page-root-2026-09-02"
ORIGINAL_RUNTIME_ENDPOINT = DEFAULT_ENDPOINT
RELOCATED_RUNTIME_ENDPOINT = "http://10.63.0.82:31518/v1/chat/completions"
ALLOWED_RUNTIME_ENDPOINTS = frozenset({ORIGINAL_RUNTIME_ENDPOINT, RELOCATED_RUNTIME_ENDPOINT})
DEFAULT_ROOT = Path("results/recovery_v2_2")
DEFAULT_DATASET = Path("data/fever/fever-validation.jsonl")
PARENT_SELECTION = Path("results/recovery_v2_1_parser_abort/selection_manifest.json")
PARENT_TRAIN_RECORDS = Path("results/recovery_v2_1_parser_abort/train/records.jsonl")
PARENT_SELECTION_SHA256 = "203080bda98e2dade1a5294f9cfe551f23d7b228dc47a6aa085aa9ad7dc87c12"
PARENT_TRAIN_SHA256 = "eaeb72e15863d99c32df50c32265be79f151131892d043e1e7d2c78921272f8a"
RECOVERY_ACTIONS = ("candidate_0", "candidate_1", "both")
PERSONAS = (
    "literal evidence analyst",
    "skeptical fact checker",
    "cross-sentence reasoner",
    "source-bound auditor",
    "adversarial claim examiner",
)
SPLITS = ("train", "dev", "test")
SPLIT_TARGETS = {"train": 0.60, "dev": 0.20, "test": 0.20}
HIGH_CONSENSUS = 0.80
HARM_CAP = 0.10
MAX_DAMAGE_RATE = 0.05


class RecoveryChatClient(CachedChatClient):
    """Use the frozen model through either documented compute location."""

    def __init__(self, endpoint: str, model: str, cache_dir: Path, timeout: float = 60.0):
        if endpoint not in ALLOWED_RUNTIME_ENDPOINTS or model != DEFAULT_MODEL:
            raise ValueError("Recovery V2 runtime endpoint or model is not preregistered")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
ROOT_COST = 0.01
CALIBRATION_COVERAGE = 0.90
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_260_932
MODEL_SEED = 20_260_931
SPLIT_SALT = b"recovery-v2.1-page-components-2026-09-02\n"
ANCHOR_SALT = b"recovery-v2.1-anchor-2026-09-02\n"
ORDER_SALT = b"recovery-v2.1-candidate-order-2026-09-02\n"
CALL_SEEDS = {
    "baseline": 41_001,
    "candidate_0": 41_019,
    "candidate_1": 41_037,
    "both": 41_053,
}
RESPONSE_FIELDS = {"answer", "confidence", "cited_evidence_ids"}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "label", "gold", "correct", "annotated", "outcome", "harm", "split",
)
REPAIR_SUFFIX = (
    "\nThe previous response was invalid. Return exactly one JSON object with "
    "only answer, confidence, and cited_evidence_ids."
)


@dataclass(frozen=True)
class RawExample:
    example_id: str
    claim: str
    label: str
    roots: tuple[str, str]
    evidence_by_root: Mapping[str, tuple[str, ...]]


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
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from error
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _stable_unit(salt: bytes, value: str) -> float:
    raw = sha256(salt + value.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / float(2**64)


def _clean_text(value: object) -> str:
    return " ".join(str(value).split())


def load_eligible(path: Path = DEFAULT_DATASET) -> list[RawExample]:
    examples: list[RawExample] = []
    seen: set[str] = set()
    for row in _load_jsonl(path):
        if row.get("verifiable") != "VERIFIABLE":
            continue
        label = str(row.get("label", ""))
        if label not in {"SUPPORTS", "REFUTES"}:
            continue
        evidence_by_root: dict[str, list[str]] = defaultdict(list)
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, list) or len(evidence) < 3:
                continue
            root = _clean_text(evidence[0])
            sentence = _clean_text(evidence[2])
            if root and sentence and sentence not in evidence_by_root[root]:
                evidence_by_root[root].append(sentence)
        if len(evidence_by_root) != 2:
            continue
        example_id = str(row["id"])
        if example_id in seen:
            raise ValueError(f"duplicate FEVER id: {example_id}")
        seen.add(example_id)
        roots = tuple(sorted(evidence_by_root))
        examples.append(RawExample(
            example_id=example_id,
            claim=_clean_text(row["claim"]),
            label=label,
            roots=(roots[0], roots[1]),
            evidence_by_root={root: tuple(evidence_by_root[root]) for root in roots},
        ))
    return sorted(examples, key=lambda item: item.example_id)


def _components(examples: Sequence[RawExample]) -> list[list[RawExample]]:
    union_find = UnionFind()
    for example in examples:
        union_find.union(example.roots[0], example.roots[1])
    grouped: dict[str, list[RawExample]] = defaultdict(list)
    for example in examples:
        grouped[union_find.find(example.roots[0])].append(example)
    components = list(grouped.values())
    components.sort(key=lambda members: (
        -len(members),
        sha256(SPLIT_SALT + min(item.example_id for item in members).encode()).hexdigest(),
    ))
    return components


def assign_splits(examples: Sequence[RawExample]) -> dict[str, str]:
    total = len(examples)
    support_total = sum(example.label == "SUPPORTS" for example in examples)
    targets = {split: SPLIT_TARGETS[split] * total for split in SPLITS}
    support_targets = {
        split: SPLIT_TARGETS[split] * support_total for split in SPLITS
    }
    counts = Counter({split: 0 for split in SPLITS})
    supports = Counter({split: 0 for split in SPLITS})
    assignment: dict[str, str] = {}
    for component in _components(examples):
        size = len(component)
        support_count = sum(item.label == "SUPPORTS" for item in component)

        def objective(
            split: str,
            component_size: int = size,
            component_support: int = support_count,
        ) -> tuple[float, float, str]:
            proposed_counts = dict(counts)
            proposed_supports = dict(supports)
            proposed_counts[split] += component_size
            proposed_supports[split] += component_support
            count_error = sum(
                ((proposed_counts[name] - targets[name]) / max(1.0, total)) ** 2
                for name in SPLITS
            )
            label_error = sum(
                (
                    (proposed_supports[name] - support_targets[name])
                    / max(1.0, support_total)
                ) ** 2
                for name in SPLITS
            )
            overfill = max(
                0.0, proposed_counts[split] - targets[split]
            ) / max(1.0, total)
            remaining = targets[split] - counts[split]
            return 4.0 * count_error + label_error + 8.0 * overfill**2, -remaining, split

        chosen = min(SPLITS, key=objective)
        counts[chosen] += size
        supports[chosen] += support_count
        for example in component:
            assignment[example.example_id] = chosen
    return assignment


def _root_corpus(examples: Sequence[RawExample]) -> dict[str, tuple[str, ...]]:
    corpus: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        for root in example.roots:
            for sentence in example.evidence_by_root[root]:
                if sentence not in corpus[root]:
                    corpus[root].append(sentence)
    return {root: tuple(sorted(sentences)[:4]) for root, sentences in corpus.items()}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in left.replace("_", " ").split() if token}
    right_tokens = {token.lower() for token in right.replace("_", " ").split() if token}
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def build_selection(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    examples = load_eligible(dataset_path)
    if len(examples) != 1_149:
        raise ValueError(f"expected 1,149 eligible claims, found {len(examples)}")
    split_by_id = assign_splits(examples)
    by_split = {
        split: [item for item in examples if split_by_id[item.example_id] == split]
        for split in SPLITS
    }
    vectorizer = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words="english",
    )
    selected: list[dict[str, Any]] = []
    for split in SPLITS:
        split_examples = by_split[split]
        corpus = _root_corpus(split_examples)
        roots = sorted(corpus)
        root_texts = [" ".join(corpus[root]) for root in roots]
        root_matrix = vectorizer.transform(root_texts)
        claim_matrix = vectorizer.transform([item.claim for item in split_examples])
        similarities = linear_kernel(claim_matrix, root_matrix)
        root_index = {root: index for index, root in enumerate(roots)}
        for row_index, example in enumerate(split_examples):
            anchor_offset = int(_stable_unit(ANCHOR_SALT, example.example_id) >= 0.5)
            anchor_root = example.roots[anchor_offset]
            annotated_root = example.roots[1 - anchor_offset]
            candidates = [
                (float(similarities[row_index, index]), root)
                for index, root in enumerate(roots)
                if root not in example.roots
            ]
            if not candidates:
                raise ValueError(f"no unannotated retrieval candidate for {example.example_id}")
            retrieval_score, retrieved_root = max(candidates, key=lambda pair: (pair[0], pair[1]))
            annotated_score = float(similarities[row_index, root_index[annotated_root]])
            raw_candidates = [
                {
                    "root": annotated_root,
                    "annotation_role": "held_out_annotated_root",
                    "retrieval_score": annotated_score,
                    "title_overlap": _token_jaccard(example.claim, annotated_root),
                    "evidence": list(example.evidence_by_root[annotated_root]),
                },
                {
                    "root": retrieved_root,
                    "annotation_role": "unannotated_retrieval_candidate",
                    "retrieval_score": retrieval_score,
                    "title_overlap": _token_jaccard(example.claim, retrieved_root),
                    "evidence": list(corpus[retrieved_root]),
                },
            ]
            if _stable_unit(ORDER_SALT, example.example_id) >= 0.5:
                raw_candidates.reverse()
            candidates_out = []
            for candidate_index, candidate in enumerate(raw_candidates):
                candidate_copy = dict(candidate)
                candidate_copy["evidence"] = [
                    {"evidence_id": f"C{candidate_index}{line_index:02d}", "text": text}
                    for line_index, text in enumerate(candidate["evidence"])
                ]
                candidates_out.append(candidate_copy)
            selected.append({
                "example_id": example.example_id,
                "split": split,
                "claim": example.claim,
                "label": example.label,
                "gold_binary": int(example.label == "SUPPORTS"),
                "anchor": {
                    "root": anchor_root,
                    "retrieval_score": float(similarities[row_index, root_index[anchor_root]]),
                    "title_overlap": _token_jaccard(example.claim, anchor_root),
                    "evidence": [
                        {"evidence_id": f"A{line_index:02d}", "text": text}
                        for line_index, text in enumerate(example.evidence_by_root[anchor_root])
                    ],
                },
                "candidates": candidates_out,
            })
    selected.sort(key=lambda row: (str(row["split"]), str(row["example_id"])))
    if _sha256_path(PARENT_SELECTION) != PARENT_SELECTION_SHA256:
        raise ValueError("Recovery V2.1 parent selection drifted")
    if _sha256_path(PARENT_TRAIN_RECORDS) != PARENT_TRAIN_SHA256:
        raise ValueError("Recovery V2.1 parent train records drifted")
    parent_selection = json.loads(PARENT_SELECTION.read_text(encoding="utf-8"))
    if selected != parent_selection.get("examples"):
        raise ValueError("Recovery V2.2 must inherit the exact V2.1 examples")
    split_counts = Counter(str(row["split"]) for row in selected)
    label_counts = {
        split: dict(Counter(
            str(row["label"]) for row in selected if row["split"] == split
        ))
        for split in SPLITS
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "selection_frozen_before_recovery_v2_qwen_calls",
        "amendment": {
            "parent_protocol_version": "recovery-v2.1-page-root-2026-09-02",
            "parent_selection": str(PARENT_SELECTION),
            "parent_selection_sha256": PARENT_SELECTION_SHA256,
            "parent_train_records": str(PARENT_TRAIN_RECORDS),
            "parent_train_records_sha256": PARENT_TRAIN_SHA256,
            "selection_changed": False,
            "train_calls_reused": 5_512,
            "test_calls_existed": False,
            "only_change": "one exact response-local confidence quote normalization",
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_path(dataset_path),
            "eligible_rule": "verifiable binary claims with exactly two distinct page roots",
            "eligible": len(examples),
        },
        "split_contract": {
            "unit": "connected_component_of_annotated_page_roots",
            "target_fractions": SPLIT_TARGETS,
            "counts": dict(split_counts),
            "label_counts": label_counts,
            "salt_sha256": sha256(SPLIT_SALT).hexdigest(),
        },
        "candidate_contract": {
            "retriever": "HashingVectorizer(2**16, word 1-2 grams, English stopwords)",
            "retrieval_scope": "same split only",
            "anchor_salt_sha256": sha256(ANCHOR_SALT).hexdigest(),
            "order_salt_sha256": sha256(ORDER_SALT).hexdigest(),
            "candidate_positions_reveal_annotation_role": False,
        },
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "expected_calls_per_example": len(PERSONAS) + len(RECOVERY_ACTIONS),
        "examples": selected,
    }


def validate_selection(selection: Mapping[str, Any], dataset_path: Path = DEFAULT_DATASET) -> None:
    expected = build_selection(dataset_path)
    if dict(selection) != expected:
        raise ValueError("Recovery V2 selection or source dataset drifted")


def write_or_validate_selection(output: Path, dataset_path: Path = DEFAULT_DATASET) -> bool:
    expected = build_selection(dataset_path)
    if output.exists():
        actual = json.loads(output.read_text(encoding="utf-8"))
        validate_selection(actual, dataset_path)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, expected)
    return True


def audit_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    examples = list(selection["examples"])
    split_counts = Counter(str(row["split"]) for row in examples)
    label_counts = {
        split: Counter(str(row["label"]) for row in examples if row["split"] == split)
        for split in SPLITS
    }
    all_roots: dict[str, set[str]] = {split: set() for split in SPLITS}
    valid_candidates = True
    annotated_at_zero = 0
    candidate_labels: list[int] = []
    candidate_scores: list[float] = []
    for row in examples:
        split = str(row["split"])
        anchor_root = str(row["anchor"]["root"])
        candidate_roots = [str(candidate["root"]) for candidate in row["candidates"]]
        all_roots[split].add(anchor_root)
        all_roots[split].update(candidate_roots)
        valid_candidates &= len({anchor_root, *candidate_roots}) == 3
        if row["candidates"][0]["annotation_role"] == "held_out_annotated_root":
            annotated_at_zero += 1
        for candidate in row["candidates"]:
            candidate_labels.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            candidate_scores.append(float(candidate["retrieval_score"]))
    overlaps = {
        f"{left}_{right}": len(all_roots[left] & all_roots[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    }
    label_fractions = {
        split: {
            label: count / split_counts[split] for label, count in label_counts[split].items()
        }
        for split in SPLITS
    }
    score_auc = float(roc_auc_score(candidate_labels, candidate_scores))
    gates = {
        "eligible_exactly_1149": len(examples) == 1_149,
        "dev_at_least_200": split_counts["dev"] >= 200,
        "test_at_least_200": split_counts["test"] >= 200,
        "zero_cross_split_page_overlap": all(value == 0 for value in overlaps.values()),
        "three_distinct_roots_per_example": bool(valid_candidates),
        "candidate_0_annotation_frequency": 0.45 <= annotated_at_zero / len(examples) <= 0.55,
        "label_fraction_each_at_least_040": all(
            min(fractions.values()) >= 0.40 for fractions in label_fractions.values()
        ),
        "retrieval_score_annotation_auc_at_most_080": score_auc <= 0.80,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "n": len(examples),
        "split_counts": dict(split_counts),
        "label_counts": {split: dict(counts) for split, counts in label_counts.items()},
        "label_fractions": label_fractions,
        "cross_split_page_overlap": overlaps,
        "candidate_0_annotated_fraction": annotated_at_zero / len(examples),
        "retrieval_score_annotation_auc": score_auc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _packet(evidence: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    )


def build_baseline_messages(
    example: Mapping[str, Any], agent_index: int, *, repair: bool = False,
) -> list[dict[str, str]]:
    if not 0 <= agent_index < len(PERSONAS):
        raise ValueError("invalid baseline agent index")
    system = (
        f"You are a {PERSONAS[agent_index]}. Decide whether the Boolean claim is true using only "
        "the supplied evidence; do not use outside knowledge. If the packet is incomplete, make "
        "the most evidence-grounded yes/no decision. Return exactly one JSON object and no other "
        'text: {"answer":"yes|no","confidence":0.0,"cited_evidence_ids":["A00"]}.'
    )
    user = f"Claim: {example['claim']}\n\nANCHOR EVIDENCE:\n{_packet(example['anchor']['evidence'])}"
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _action_evidence(example: Mapping[str, Any], action: str) -> list[Mapping[str, Any]]:
    if action == "candidate_0":
        return list(example["candidates"][0]["evidence"])
    if action == "candidate_1":
        return list(example["candidates"][1]["evidence"])
    if action == "both":
        return [
            *example["candidates"][0]["evidence"],
            *example["candidates"][1]["evidence"],
        ]
    raise ValueError(f"unknown recovery action: {action}")


def build_recovery_messages(
    example: Mapping[str, Any], action: str, consensus: str, *, repair: bool = False,
) -> list[dict[str, str]]:
    if action not in RECOVERY_ACTIONS:
        raise ValueError(f"unknown recovery action: {action}")
    acquired = _action_evidence(example, action)
    system = (
        "You are a source-bound fact-recovery adjudicator. Re-evaluate the Boolean claim using "
        "only the anchor and acquired evidence. Repeated agreement is not independent evidence. "
        "Return exactly one JSON object and no other text: "
        '{"answer":"yes|no","confidence":0.0,"cited_evidence_ids":["A00","C000"]}.'
    )
    user = (
        f"Claim: {example['claim']}\nPrevious anchor-only consensus: {consensus}\n\n"
        f"ANCHOR EVIDENCE:\n{_packet(example['anchor']['evidence'])}\n\n"
        f"ACQUIRED EVIDENCE ({action}):\n{_packet(acquired)}\n\n"
        "The previous consensus is a hypothesis, not evidence. Answer from the evidence packets."
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_decision(content: str, allowed_ids: Sequence[str]) -> dict[str, Any]:
    parse_mode = "strict"
    try:
        payload = _extract_json_object(content)
    except ValueError:
        normalized, replacements = re.subn(
            r'("confidence"\s*:\s*(?:0(?:\.\d+)?|1(?:\.0+)?))"(?=\s*,)',
            r"\1",
            content,
        )
        if replacements != 1:
            raise
        payload = _extract_json_object(normalized)
        parse_mode = "normalized_single_confidence_quote"
    unknown = set(payload) - RESPONSE_FIELDS
    missing = RESPONSE_FIELDS - set(payload)
    if unknown or missing:
        raise ValueError(f"response fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
    if payload["answer"] not in {"yes", "no"}:
        raise ValueError("answer must be yes or no")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    citations = payload["cited_evidence_ids"]
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise TypeError("cited_evidence_ids must be a list of strings")
    if len(citations) != len(set(citations)):
        raise ValueError("cited_evidence_ids must be unique")
    outside = set(citations) - set(allowed_ids)
    if outside:
        raise ValueError(f"citations outside action packet: {sorted(outside)}")
    return {
        "answer": str(payload["answer"]),
        "confidence": confidence,
        "cited_evidence_ids": citations,
        "parse_mode": parse_mode,
    }


def _call_with_retry(
    client: CachedChatClient,
    messages_builder: Any,
    allowed_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    attempts: list[dict[str, Any]] = []
    final_error: str | None = None
    for attempt_index in range(2):
        attempt: dict[str, Any] | None = None
        try:
            result = client.call(messages_builder(attempt_index > 0), seed=seed)
            attempt = _attempt_payload(result)
            decision = parse_decision(result.content, allowed_ids)
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
        return decision, attempts, None
    return None, attempts, final_error


def _majority(baseline_rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    answers = [str(row["decision"]["answer"]) for row in baseline_rows if row.get("decision")]
    if len(answers) != len(PERSONAS):
        raise ValueError("baseline consensus requires five valid decisions")
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(answers)


def _run_example(client: CachedChatClient, example: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        rows.append({
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
        })
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
        candidate_indices = [0, 1] if action == "both" else [int(action[-1])]
        annotated_ids = {
            str(item["evidence_id"])
            for index in candidate_indices
            if example["candidates"][index]["annotation_role"] == "held_out_annotated_root"
            for item in example["candidates"][index]["evidence"]
        }
        rows.append({
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
        })
    return rows


def execute_split(
    selection_path: Path,
    *,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    endpoint: str = RELOCATED_RUNTIME_ENDPOINT,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    audit = audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("Recovery V2 selection failed frozen structural gates")
    if split in {"dev", "test"} and not smoke:
        train_audit_path = DEFAULT_ROOT / "train" / "structural_audit.json"
        train_records_path = DEFAULT_ROOT / "train" / "records.jsonl"
        if not train_audit_path.exists() or not train_records_path.exists():
            raise ValueError(f"{split} calls require a passing frozen train structural audit")
        train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
        if not train_audit.get("passed"):
            raise ValueError(f"{split} calls blocked by the train structural gate")
        if train_audit.get("records_sha256") != _sha256_path(train_records_path):
            raise ValueError("train records drifted after their structural audit")
    if split == "test" and not smoke:
        router_manifest = DEFAULT_ROOT / "router" / "manifest.json"
        if not router_manifest.exists():
            raise ValueError("test calls require a frozen train/dev router manifest")
        validate_router_manifest(router_manifest, selection_path)
    examples = [row for row in selection["examples"] if row["split"] == split]
    if smoke:
        examples = examples[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded_records = _load_jsonl(partial_path) if partial_path.exists() else []
    expected_per_example = len(PERSONAS) + len(RECOVERY_ACTIONS)
    loaded_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded_records:
        loaded_by_example[str(row["example_id"])].append(row)
    if any(len(rows) != expected_per_example for rows in loaded_by_example.values()):
        raise ValueError("partial file contains an incomplete example bundle")
    # A complete bundle with a transport failure is not an observed outcome.
    # Resume it as a unit; successful attempts remain content-addressed cache hits.
    existing = {
        example_id: rows
        for example_id, rows in loaded_by_example.items()
        if all(row.get("success") for row in rows)
    }
    records = [row for rows in existing.values() for row in rows]
    allowed_ids = {str(example["example_id"]) for example in examples}
    if set(existing) - allowed_ids:
        raise ValueError("partial file contains examples outside this run")
    client = RecoveryChatClient(endpoint, DEFAULT_MODEL, cache_dir)
    pending = [example for example in examples if str(example["example_id"]) not in existing]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_run_example, client, example): example for example in pending}
        for future in as_completed(futures):
            bundle = future.result()
            records.extend(bundle)
            records.sort(key=lambda row: (
                str(row["example_id"]),
                str(row["phase"]),
                -1 if row["agent_index"] is None else int(row["agent_index"]),
                str(row["action"]),
            ))
            _write_jsonl(partial_path, records)
            print(
                f"[{len(records) // expected_per_example}/{len(examples)}] "
                f"{bundle[0]['example_id']} success={all(row['success'] for row in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    expected = len(examples) * expected_per_example
    if len(records) != expected or any(not row["success"] for row in records):
        raise ValueError(f"split run incomplete or invalid: {len(records)}/{expected}")
    _write_jsonl(output_dir / "records.jsonl", records)
    return records


def _record_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["example_id"])].append(row)
    return dict(grouped)


def _outcomes(
    example: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> tuple[int, float, dict[str, int], list[Mapping[str, Any]]]:
    baseline = sorted(
        (row for row in records if row["phase"] == "baseline"),
        key=lambda row: int(row["agent_index"]),
    )
    consensus, agreement = _majority(baseline)
    gold = int(example["gold_binary"])
    keep = int((consensus == "yes") == bool(gold))
    action_outcomes = {}
    for row in records:
        if row["phase"] == "recovery":
            action_outcomes[str(row["action"])] = int(
                (row["decision"]["answer"] == "yes") == bool(gold)
            )
    if set(action_outcomes) != set(RECOVERY_ACTIONS):
        raise ValueError(f"missing recovery outcomes for {example['example_id']}")
    return keep, agreement, action_outcomes, baseline


def audit_train_structure(
    selection_path: Path, records_path: Path, output_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    examples = {
        str(row["example_id"]): row for row in selection["examples"] if row["split"] == "train"
    }
    grouped = _record_groups(_load_jsonl(records_path))
    if set(grouped) != set(examples):
        raise ValueError("train records do not cover the frozen train split")
    high_consensus = 0
    high_wrong = 0
    repairable = 0
    harm_exposed = 0
    for example_id, example in examples.items():
        keep, agreement, outcomes, _ = _outcomes(example, grouped[example_id])
        if agreement < HIGH_CONSENSUS:
            continue
        high_consensus += 1
        if keep == 0:
            high_wrong += 1
            repairable += int(any(outcome == 1 for outcome in outcomes.values()))
        else:
            harm_exposed += int(any(outcome == 0 for outcome in outcomes.values()))
    gates = {
        "high_consensus_at_least_100": high_consensus >= 100,
        "high_consensus_wrong_at_least_20": high_wrong >= 20,
        "repairable_at_least_10": repairable >= 10,
        "harm_exposed_at_least_10": harm_exposed >= 10,
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "records_sha256": _sha256_path(records_path),
        "n_train": len(examples),
        "high_consensus": high_consensus,
        "high_consensus_wrong": high_wrong,
        "repairable": repairable,
        "harm_exposed": harm_exposed,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output_path, report)
    return report


def _feature_vector(
    example: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    action: str,
) -> tuple[list[str], np.ndarray]:
    consensus, agreement = _majority(baseline)
    answers = np.asarray([row["decision"]["answer"] == "yes" for row in baseline], dtype=float)
    confidences = np.asarray([row["decision"]["confidence"] for row in baseline], dtype=float)
    citations = np.asarray([
        len(row["decision"]["cited_evidence_ids"]) for row in baseline
    ], dtype=float)
    anchor = example["anchor"]
    if action == "candidate_0":
        selected = [example["candidates"][0]]
    elif action == "candidate_1":
        selected = [example["candidates"][1]]
    elif action == "both":
        selected = list(example["candidates"])
    else:
        raise ValueError(f"unknown modeled action: {action}")
    relevance = np.asarray([candidate["retrieval_score"] for candidate in selected], dtype=float)
    title_overlap = np.asarray([candidate["title_overlap"] for candidate in selected], dtype=float)
    candidate_tokens = sum(
        len(item["text"].split())
        for candidate in selected
        for item in candidate["evidence"]
    )
    yes_fraction = float(answers.mean())
    vote_entropy = 0.0
    for probability in (yes_fraction, 1.0 - yes_fraction):
        if probability > 0:
            vote_entropy -= probability * math.log(probability)
    names = [
        "consensus_yes",
        "agreement",
        "yes_fraction",
        "vote_entropy",
        "confidence_mean",
        "confidence_std",
        "confidence_min",
        "citation_mean",
        "claim_tokens_log1p",
        "anchor_tokens_log1p",
        "anchor_evidence_count",
        "anchor_relevance",
        "anchor_title_overlap",
        "is_two_root_action",
        "roots_added",
        "candidate_relevance_max",
        "candidate_relevance_min",
        "candidate_relevance_mean",
        "candidate_relevance_minus_anchor",
        "candidate_title_overlap_max",
        "candidate_title_overlap_mean",
        "candidate_tokens_log1p",
    ]
    values = np.asarray([
        float(consensus == "yes"),
        agreement,
        yes_fraction,
        vote_entropy,
        float(confidences.mean()),
        float(confidences.std()),
        float(confidences.min()),
        float(citations.mean()),
        math.log1p(len(str(example["claim"]).split())),
        math.log1p(sum(len(item["text"].split()) for item in anchor["evidence"])),
        len(anchor["evidence"]),
        float(anchor["retrieval_score"]),
        float(anchor["title_overlap"]),
        float(action == "both"),
        float(len(selected)),
        float(relevance.max()),
        float(relevance.min()),
        float(relevance.mean()),
        float(relevance.max() - anchor["retrieval_score"]),
        float(title_overlap.max()),
        float(title_overlap.mean()),
        math.log1p(candidate_tokens),
    ])
    if any(fragment in name for name in names for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
        raise AssertionError("forbidden post-outcome feature entered the router")
    return names, values


def _training_matrix(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    grouped = _record_groups(records)
    rows: list[np.ndarray] = []
    gains: list[int] = []
    harms: list[int] = []
    keys: list[tuple[str, str]] = []
    feature_names: list[str] | None = None
    for example in examples:
        example_id = str(example["example_id"])
        keep, _agreement, outcomes, baseline = _outcomes(example, grouped[example_id])
        for action in RECOVERY_ACTIONS:
            names, values = _feature_vector(example, baseline, action)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("router feature schema drifted")
            rows.append(values)
            gains.append(outcomes[action] - keep)
            harms.append(int(keep == 1 and outcomes[action] == 0))
            keys.append((example_id, action))
    return (
        feature_names or [],
        np.vstack(rows),
        np.asarray(gains, dtype=float),
        np.asarray(harms, dtype=int),
        keys,
    )


def _calibration_quantile(scores: np.ndarray, coverage: float) -> float:
    if not len(scores):
        raise ValueError("empty calibration scores")
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    return max(0.0, float(np.sort(scores)[rank - 1]))


def fit_router(
    selection_path: Path,
    train_records_path: Path,
    dev_records_path: Path,
    train_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
    if not train_audit.get("passed"):
        raise ValueError("frozen train structural gate did not pass")
    train_records = _load_jsonl(train_records_path)
    dev_records = _load_jsonl(dev_records_path)
    train_examples = [row for row in selection["examples"] if row["split"] == "train"]
    dev_examples = [row for row in selection["examples"] if row["split"] == "dev"]
    names, x_train, gain_train, harm_train, _ = _training_matrix(
        train_examples, train_records
    )
    dev_names, x_dev, gain_dev, harm_dev, _ = _training_matrix(dev_examples, dev_records)
    if names != dev_names:
        raise ValueError("train/dev feature schemas differ")
    gain_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=MODEL_SEED,
        n_jobs=-1,
    )
    gain_model.fit(x_train, gain_train)
    if len(set(harm_train.tolist())) < 2:
        raise ValueError("train structural gate passed without two harm classes")
    harm_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight="balanced",
        random_state=MODEL_SEED + 1,
        n_jobs=-1,
    )
    harm_model.fit(x_train, harm_train)
    gain_dev_pred = gain_model.predict(x_dev)
    harm_dev_pred = harm_model.predict_proba(x_dev)[:, 1]
    gain_margin = _calibration_quantile(
        gain_dev_pred - gain_dev, CALIBRATION_COVERAGE
    )
    harm_margin = _calibration_quantile(
        harm_dev - harm_dev_pred, CALIBRATION_COVERAGE
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "router.joblib"
    joblib.dump({"gain_model": gain_model, "harm_model": harm_model}, model_path)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_test_calls",
        "selection_sha256": _sha256_path(selection_path),
        "train_records_sha256": _sha256_path(train_records_path),
        "dev_records_sha256": _sha256_path(dev_records_path),
        "train_audit_sha256": _sha256_path(train_audit_path),
        "feature_names": names,
        "forbidden_feature_intersection": [
            name for name in names
            if any(fragment in name for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
        ],
        "training_rows": len(x_train),
        "calibration_rows": len(x_dev),
        "model": {
            "family": "random_forest_paired_gain_and_harm",
            "seed": MODEL_SEED,
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "max_features": 0.8,
        },
        "calibration": {
            "coverage": CALIBRATION_COVERAGE,
            "gain_overprediction_margin": gain_margin,
            "harm_underprediction_margin": harm_margin,
        },
        "policy": {
            "high_consensus": HIGH_CONSENSUS,
            "harm_cap": HARM_CAP,
            "root_cost": ROOT_COST,
        },
        "router_joblib": str(model_path),
        "router_joblib_sha256": _sha256_path(model_path),
        "claim_boundary": {
            "test_outcomes_seen": False,
            "page_root_disjoint": True,
            "publisher_independent": False,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_router_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("router protocol version mismatch")
    if manifest.get("status") != "frozen_before_test_calls":
        raise ValueError("router was not frozen before test calls")
    if manifest.get("selection_sha256") != _sha256_path(selection_path):
        raise ValueError("router selection hash drifted")
    model_path = Path(manifest["router_joblib"])
    if manifest.get("router_joblib_sha256") != _sha256_path(model_path):
        raise ValueError("serialized router drifted")
    if manifest.get("forbidden_feature_intersection"):
        raise ValueError("router contains forbidden features")


def _policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps = []
    finals = []
    labels = []
    high_flags = []
    annotation_supported = 0
    action_counts = Counter()
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
        if keep == 0 and final == 1 and action != "KEEP":
            recovery_row = next(
                row for row in grouped[example_id]
                if row["phase"] == "recovery" and row["action"] == action
            )
            citations = set(recovery_row["decision"]["cited_evidence_ids"])
            annotated_ids = set(recovery_row["annotated_evidence_ids"])
            annotation_supported += int(
                recovery_row["packet_contains_annotated_root"]
                and bool(citations & annotated_ids)
            )
    keep_array = np.asarray(keeps, dtype=int)
    final_array = np.asarray(finals, dtype=int)
    gains = final_array - keep_array
    fixes = (keep_array == 0) & (final_array == 1)
    harms = (keep_array == 1) & (final_array == 0)
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(examples)) for _ in examples]
        boot.append(float(gains[indices].mean()))
    boot.sort()
    by_label = {}
    for label in ("SUPPORTS", "REFUTES"):
        indices = [index for index, value in enumerate(labels) if value == label]
        by_label[label] = {
            "n": len(indices),
            "baseline_accuracy": float(keep_array[indices].mean()),
            "final_accuracy": float(final_array[indices].mean()),
            "net_gain": float(gains[indices].mean()),
        }
    high_correct = int(sum(
        keep and high for keep, high in zip(keeps, high_flags, strict=True)
    ))
    high_harms = int(sum(
        keep and high and not final
        for keep, high, final in zip(keeps, high_flags, finals, strict=True)
    ))
    return {
        "n": len(examples),
        "baseline_accuracy": float(keep_array.mean()),
        "final_accuracy": float(final_array.mean()),
        "fixes": int(fixes.sum()),
        "harms": int(harms.sum()),
        "net_fixes": int(fixes.sum() - harms.sum()),
        "net_gain": float(gains.mean()),
        "net_gain_ci": [boot[25], boot[975]],
        "high_consensus_n": int(sum(high_flags)),
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "mean_added_roots": sum(
            0 if action == "KEEP" else 2 if action == "both" else 1
            for action in selected.values()
        ) / len(examples),
        "selected_actions": dict(action_counts),
        "by_native_label": by_label,
    }


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
    test_records = _load_jsonl(test_records_path)
    test_examples = [row for row in selection["examples"] if row["split"] == "test"]
    grouped = _record_groups(test_records)
    if set(grouped) != {str(row["example_id"]) for row in test_examples}:
        raise ValueError("test records do not cover the frozen test split")
    names, x_test, _gain, _harm, keys = _training_matrix(test_examples, test_records)
    if names != manifest["feature_names"]:
        raise ValueError("test feature schema differs from frozen router")
    models = joblib.load(manifest["router_joblib"])
    gain_pred = models["gain_model"].predict(x_test)
    harm_pred = models["harm_model"].predict_proba(x_test)[:, 1]
    gain_lcb = gain_pred - float(manifest["calibration"]["gain_overprediction_margin"])
    harm_ucb = np.minimum(
        1.0,
        harm_pred + float(manifest["calibration"]["harm_underprediction_margin"]),
    )
    prediction_by_key = {
        key: (float(gain_pred[index]), float(gain_lcb[index]), float(harm_ucb[index]))
        for index, key in enumerate(keys)
    }
    conservative: dict[str, str] = {}
    unrestricted: dict[str, str] = {}
    retrieval: dict[str, str] = {}
    for example in test_examples:
        example_id = str(example["example_id"])
        _keep, agreement, _outcomes_map, _baseline = _outcomes(
            example, grouped[example_id]
        )
        if agreement < HIGH_CONSENSUS:
            conservative[example_id] = "KEEP"
            unrestricted[example_id] = "KEEP"
            retrieval[example_id] = "KEEP"
            continue
        best = max(
            RECOVERY_ACTIONS,
            key=lambda action: prediction_by_key[(example_id, action)][0]
            - ROOT_COST * (2 if action == "both" else 1),
        )
        best_utility = prediction_by_key[(example_id, best)][0] - ROOT_COST * (
            2 if best == "both" else 1
        )
        unrestricted[example_id] = best if best_utility > 0 else "KEEP"
        safe = [
            action for action in RECOVERY_ACTIONS
            if prediction_by_key[(example_id, action)][1]
            > ROOT_COST * (2 if action == "both" else 1)
            and prediction_by_key[(example_id, action)][2] <= HARM_CAP
        ]
        conservative[example_id] = max(
            safe,
            key=lambda action: prediction_by_key[(example_id, action)][1]
            - ROOT_COST * (2 if action == "both" else 1),
        ) if safe else "KEEP"
        retrieval[example_id] = (
            "candidate_0"
            if example["candidates"][0]["retrieval_score"]
            >= example["candidates"][1]["retrieval_score"]
            else "candidate_1"
        )
    policies: dict[str, dict[str, str]] = {
        "learned_conservative": conservative,
        "learned_unrestricted": unrestricted,
        "retrieval_score": retrieval,
        "keep": {str(row["example_id"]): "KEEP" for row in test_examples},
    }
    for action in RECOVERY_ACTIONS:
        policies[f"fixed_{action}"] = {}
        for example in test_examples:
            example_id = str(example["example_id"])
            _keep, agreement, _outcomes_map, _baseline = _outcomes(
                example, grouped[example_id]
            )
            policies[f"fixed_{action}"][example_id] = (
                action if agreement >= HIGH_CONSENSUS else "KEEP"
            )
    oracle = {}
    for example in test_examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes_map, _baseline = _outcomes(example, grouped[example_id])
        if agreement < HIGH_CONSENSUS:
            oracle[example_id] = "KEEP"
            continue
        oracle[example_id] = max(
            ("KEEP", *RECOVERY_ACTIONS),
            key=lambda action: keep if action == "KEEP" else outcomes_map[action],
        )
    policies["available_action_oracle_diagnostic"] = oracle
    metrics = {
        name: _policy_metrics(test_examples, grouped, selected)
        for name, selected in policies.items()
    }
    primary = metrics["learned_conservative"]
    fixed = [metrics[f"fixed_{action}"] for action in RECOVERY_ACTIONS]
    gates = {
        "paired_ci_lower_above_zero": primary["net_gain_ci"][0] > 0,
        "net_fixes_above_every_fixed_action": primary["net_fixes"]
        > max(result["net_fixes"] for result in fixed),
        "damage_rate_at_most_005": primary["damage_rate_high_consensus_correct"]
        <= MAX_DAMAGE_RATE,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_5": primary["annotation_supported_repairs"] >= 5,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_page_root_disjoint_test",
        "selection_sha256": _sha256_path(selection_path),
        "router_manifest_sha256": _sha256_path(router_manifest_path),
        "test_records_sha256": _sha256_path(test_records_path),
        "n_test": len(test_examples),
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": "PASS_RECOVERY_V2" if all(gates.values()) else "NO_VERIFIED_NET_RESCUE",
        "claim_boundary": {
            "page_root_disjoint": True,
            "publisher_independent": False,
            "cross_model": False,
            "live_retrieval": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    lines = [
        "# Recovery V2 test report",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Test examples: {len(test_examples)}",
        "- Page-root overlap with train/dev: 0 by construction and audit.",
        "- Publisher independence: not tested (all roots are Wikipedia pages).",
        "",
        "## Policies",
        "",
    ]
    for name, result in metrics.items():
        lines.append(
            f"- {name}: accuracy={result['final_accuracy']:.3f}, "
            f"fixes={result['fixes']}, harms={result['harms']}, "
            f"net={result['net_fixes']}, CI={result['net_gain_ci']}, "
            f"annotation-supported={result['annotation_supported_repairs']}"
        )
    lines.extend(["", "## Frozen primary gates", ""])
    lines.extend(f"- {name}: {passed}" for name, passed in gates.items())
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _print_mapping(value: Mapping[str, Any]) -> None:
    for key, item in value.items():
        print(f"{key}: {item}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze page-root-disjoint selection")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit = subparsers.add_parser("audit", help="run frozen structural pre-call gates")
    audit.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_audit.json")
    smoke = subparsers.add_parser("smoke", help="two-example parsing smoke test")
    smoke.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    smoke.add_argument("--split", choices=SPLITS, default="train")
    smoke.add_argument("--workers", type=int, default=2)
    smoke.add_argument(
        "--endpoint", choices=sorted(ALLOWED_RUNTIME_ENDPOINTS), default=RELOCATED_RUNTIME_ENDPOINT
    )
    run = subparsers.add_parser("run", help="collect a complete split action matrix")
    run.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument(
        "--endpoint", choices=sorted(ALLOWED_RUNTIME_ENDPOINTS), default=RELOCATED_RUNTIME_ENDPOINT
    )
    train_audit = subparsers.add_parser("train-audit", help="check benefit/harm support")
    train_audit.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    train_audit.add_argument("--records", type=Path, default=DEFAULT_ROOT / "train" / "records.jsonl")
    train_audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "train" / "structural_audit.json")
    fit = subparsers.add_parser("fit", help="fit and freeze router from train/dev")
    fit.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    fit.add_argument("--train-records", type=Path, default=DEFAULT_ROOT / "train" / "records.jsonl")
    fit.add_argument("--dev-records", type=Path, default=DEFAULT_ROOT / "dev" / "records.jsonl")
    fit.add_argument("--train-audit", type=Path, default=DEFAULT_ROOT / "train" / "structural_audit.json")
    fit.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "router")
    evaluate = subparsers.add_parser("evaluate", help="evaluate frozen router on test")
    evaluate.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    evaluate.add_argument("--test-records", type=Path, default=DEFAULT_ROOT / "test" / "records.jsonl")
    evaluate.add_argument("--router-manifest", type=Path, default=DEFAULT_ROOT / "router" / "manifest.json")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "analysis")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(args.output, args.dataset)
        print(f"{'Wrote' if created else 'Reused'} frozen Recovery V2 selection: {args.output}")
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
        split = args.split
        output_dir = DEFAULT_ROOT / ("smoke" if smoke_mode else split)
        cache_dir = DEFAULT_ROOT / "cache"
        execute_split(
            args.selection,
            split=split,
            output_dir=output_dir,
            cache_dir=cache_dir,
            workers=args.workers,
            endpoint=args.endpoint,
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
            args.dev_records,
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
        _print_mapping({"verdict": summary["verdict"], "passes": summary["passes"]})
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
