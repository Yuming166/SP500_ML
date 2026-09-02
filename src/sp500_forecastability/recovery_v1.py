"""Retrospective counterfactual-recovery development on Pilot-LLM V12.1.

The completed V12.1 audit remains immutable.  This module adds new Qwen
recovery actions and learns which action has positive paired uplift relative to
retaining the original consensus.  Results are development evidence only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v11 as v11
from sp500_forecastability.pilot_llm_v1 import (
    CachedChatClient,
    _attempt_payload,
    _extract_json_object,
    _write_json,
    _write_jsonl,
)

PROTOCOL_VERSION = "recovery-v1-retrospective-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v1")
PARENT_SELECTION = Path("results/pilot_llm_v12_1/selection_manifest.json")
PARENT_RECORDS = Path("results/pilot_llm_v12_1/formal/records.jsonl")
DEFAULT_ENDPOINT = base.DEFAULT_ENDPOINT
DEFAULT_MODEL = base.DEFAULT_MODEL
HIGH_CONSENSUS_THRESHOLD = 0.8
ROUTER_COVERAGE = 0.8
N_FOLDS = 5
ENSEMBLE_REPLICATES = 30
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_260_923
FOLD_SALT = b"recovery-v1-root-folds-2026-09-02\n"
CONSERVATIVE_Z = 1.0
HARM_UCB_CAP = 0.25
RECOVERY_ACTIONS = ("full_evidence", "counter_consensus", "intervention_ledger")
ACTION_SEEDS = {
    "full_evidence": 31_001,
    "counter_consensus": 31_019,
    "intervention_ledger": 31_037,
}
RESPONSE_FIELDS = {"action_id", "answer", "confidence", "cited_evidence_ids"}
FORBIDDEN_FEATURES = {
    "label", "gold_binary", "correct", "consensus_wrong", "harmful_fc", "any_wrong",
}
REPAIR_SUFFIX = (
    "\nYour previous response was invalid. Return exactly one JSON object with "
    "only action_id, answer, confidence, and cited_evidence_ids."
)


@dataclass(frozen=True)
class RecoveryExample:
    cqid: str
    source_root: str
    question: str
    evidence: tuple[tuple[str, str], ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(evidence_id for evidence_id, _ in self.evidence)


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
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from error
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def _group_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["cqid"])].append(record)
    return dict(grouped)


def _consensus(records: Sequence[Mapping[str, Any]], condition: str = "original") -> tuple[str, float]:
    answers = [
        str(record["decision"]["answer"])
        for record in records
        if record.get("condition") == condition and isinstance(record.get("decision"), Mapping)
    ]
    if len(answers) != base.N_AGENTS:
        raise ValueError(f"condition {condition} does not contain {base.N_AGENTS} answers")
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(answers)


def _selected_risk_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = v11._risk_rows(records)
    return [row for row in rows if float(row["agreement"]) >= HIGH_CONSENSUS_THRESHOLD]


def _example_index(selection: Mapping[str, Any]) -> dict[str, RecoveryExample]:
    result: dict[str, RecoveryExample] = {}
    for raw in selection.get("examples", []):
        if not isinstance(raw, Mapping):
            raise TypeError("parent example must be an object")
        evidence = tuple(
            (str(item["evidence_id"]), str(item["passage"]))
            for item in raw["items"]
        )
        example = RecoveryExample(
            cqid=str(raw["cqid"]),
            source_root=str(raw["source_root"]),
            question=str(raw["question"]),
            evidence=evidence,
        )
        if len(example.evidence) != 3 or len(set(example.evidence_ids)) != 3:
            raise ValueError(f"{example.cqid} must have three unique evidence IDs")
        result[example.cqid] = example
    return result


def _preoutcome_context(
    risk_row: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    consensus, agreement = _consensus(records)
    ledger: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (int(row["agent_index"]), str(row["condition"]))):
        decision = record.get("decision")
        if not isinstance(decision, Mapping):
            raise TypeError("recovery requires complete V12.1 decisions")
        ledger.append({
            "agent_id": str(record["agent_id"]),
            "condition": str(record["condition"]),
            "answer": str(decision["answer"]),
            "confidence": float(decision["confidence"]),
            "cited_evidence_ids": sorted(str(value) for value in decision["cited_evidence_ids"]),
        })
    return {
        "original_consensus": consensus,
        "original_agreement": agreement,
        "D_inert": float(risk_row["D_inert"]),
        "flip_inertia": float(risk_row["flip_inertia"]),
        "frac_shared": float(risk_row["frac_shared"]),
        "R_PI": float(risk_row["R_PI"]),
        "ledger": ledger,
    }


def build_manifest(
    selection_path: Path = PARENT_SELECTION,
    records_path: Path = PARENT_RECORDS,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = _load_jsonl(records_path)
    examples = _example_index(selection)
    risk_rows = _selected_risk_rows(records)
    if len(risk_rows) != 300:
        raise ValueError(f"expected 300 V12.1 high-consensus questions, found {len(risk_rows)}")
    selected = []
    for row in risk_rows:
        example = examples[str(row["cqid"])]
        selected.append({"cqid": example.cqid, "source_root": example.source_root})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "retrospective_development_frozen_before_new_recovery_calls",
        "parent": {
            "protocol_version": selection["protocol_version"],
            "selection_path": str(selection_path),
            "selection_sha256": _sha256_path(selection_path),
            "records_path": str(records_path),
            "records_sha256": _sha256_path(records_path),
            "records": len(records),
        },
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "high_consensus_threshold": HIGH_CONSENSUS_THRESHOLD,
        "inherited_router_coverage": ROUTER_COVERAGE,
        "inherited_risk_weights": dict(v11.RISK_WEIGHTS),
        "actions": list(RECOVERY_ACTIONS),
        "action_seeds": dict(ACTION_SEEDS),
        "expected_questions": len(selected),
        "expected_recovery_calls": len(selected) * len(RECOVERY_ACTIONS),
        "fold_contract": {
            "folds": N_FOLDS,
            "salt_sha256": sha256(FOLD_SALT).hexdigest(),
            "unit": "source_root",
        },
        "learning_contract": {
            "target": "correct(action)-correct(KEEP)",
            "ensemble_replicates": ENSEMBLE_REPLICATES,
            "conservative_z": CONSERVATIVE_Z,
            "harm_ucb_cap": HARM_UCB_CAP,
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
        },
        "claim_boundary": {
            "confirmatory": False,
            "provenance_disjoint_recovery": False,
            "reason": "all recovery evidence remains inside one inherited BoolQ source root",
        },
        "selected": selected,
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    selection_path: Path = PARENT_SELECTION,
    records_path: Path = PARENT_RECORDS,
) -> None:
    expected = build_manifest(selection_path, records_path)
    if dict(manifest) != expected:
        raise ValueError("Recovery V1 manifest or frozen parent artifacts drifted")


def write_or_validate_manifest(output: Path) -> bool:
    expected = build_manifest()
    if output.exists():
        actual = json.loads(output.read_text(encoding="utf-8"))
        validate_manifest(actual)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, expected)
    return True


def _evidence_packet(example: RecoveryExample) -> str:
    return "\n".join(f"[{evidence_id}] {text}" for evidence_id, text in example.evidence)


def build_messages(
    example: RecoveryExample,
    context: Mapping[str, Any],
    action: str,
    *,
    repair: bool = False,
) -> list[dict[str, str]]:
    if action not in RECOVERY_ACTIONS:
        raise ValueError(f"unknown recovery action: {action}")
    forbidden = FORBIDDEN_FEATURES & set(context)
    if forbidden:
        raise ValueError(f"outcome fields are forbidden in recovery prompts: {sorted(forbidden)}")
    system = (
        "You are a Boolean QA recovery adjudicator. Use only the ORIGINAL EVIDENCE packet. "
        "Do not use outside knowledge. Return exactly one JSON object and no explanation: "
        '{"action_id":"ACTION","answer":"yes|no","confidence":0.0,'
        '"cited_evidence_ids":["E01"]}. Cite only IDs that materially support the answer.'
    )
    common = (
        f"Action ID: {action}\nQuestion: {example.question}\n\n"
        f"ORIGINAL EVIDENCE:\n{_evidence_packet(example)}\n\n"
    )
    if action == "full_evidence":
        instruction = (
            "Answer independently from the complete evidence packet. You are intentionally not "
            "shown the previous team consensus."
        )
    elif action == "counter_consensus":
        instruction = (
            f"The previous team consensus was {context['original_consensus']} and a frozen "
            "behavioral audit flagged it for review. Treat that consensus as a hypothesis, not as "
            "evidence. Check whether the packet supports or contradicts it before answering."
        )
    else:
        ledger = _canonical_json(context["ledger"])
        instruction = (
            "Adjudicate using the original evidence as the authority and the intervention ledger "
            "only as a reliability signal. A repeated vote is not independent evidence.\n"
            f"INTERVENTION LEDGER: {ledger}"
        )
    user = common + instruction
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_recovery_decision(
    content: str,
    *,
    expected_action: str,
    allowed_evidence_ids: Sequence[str],
) -> dict[str, Any]:
    payload = _extract_json_object(content)
    unknown = set(payload) - RESPONSE_FIELDS
    missing = RESPONSE_FIELDS - set(payload)
    if unknown or missing:
        raise ValueError(f"response fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
    if payload["action_id"] != expected_action:
        raise ValueError("action_id does not match requested recovery action")
    if payload["answer"] not in {"yes", "no"}:
        raise ValueError("answer must be yes or no")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    citations = payload["cited_evidence_ids"]
    if not isinstance(citations, list) or any(not isinstance(value, str) for value in citations):
        raise TypeError("cited_evidence_ids must be a list of strings")
    if len(citations) != len(set(citations)):
        raise ValueError("cited_evidence_ids must be unique")
    outside = set(citations) - set(allowed_evidence_ids)
    if outside:
        raise ValueError(f"citations outside original evidence packet: {sorted(outside)}")
    return {
        "action_id": expected_action,
        "answer": str(payload["answer"]),
        "confidence": confidence,
        "cited_evidence_ids": citations,
    }


def _run_one_recovery(
    client: CachedChatClient,
    example: RecoveryExample,
    context: Mapping[str, Any],
    action: str,
    gold_binary: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    decision: dict[str, Any] | None = None
    final_error: str | None = None
    for attempt_index in range(2):
        attempt: dict[str, Any] | None = None
        try:
            result = client.call(
                build_messages(example, context, action, repair=attempt_index > 0),
                seed=ACTION_SEEDS[action],
            )
            attempt = _attempt_payload(result)
            decision = parse_recovery_decision(
                result.content,
                expected_action=action,
                allowed_evidence_ids=example.evidence_ids,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            if attempt is None:
                attempt = {
                    "cache_hit": False, "cache_key": None, "http_status": None,
                    "request_bytes": None, "response_bytes": None, "latency_seconds": None,
                    "usage": {}, "parse_error": None, "transport_error": final_error,
                }
            else:
                attempt["parse_error"] = final_error
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        final_error = None
        break
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cqid": example.cqid,
        "source_root": example.source_root,
        "action": action,
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision,
        "final_error": final_error,
        # Outcomes are stored for offline training/evaluation but never enter a prompt.
        "gold_binary": int(gold_binary),
    }


def execute_recovery_calls(
    *,
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    mode: str,
    workers: int = 4,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    selection = json.loads(PARENT_SELECTION.read_text(encoding="utf-8"))
    examples = _example_index(selection)
    parent_records = _load_jsonl(PARENT_RECORDS)
    grouped = _group_records(parent_records)
    risk_rows = {str(row["cqid"]): row for row in _selected_risk_rows(parent_records)}
    selected = [str(row["cqid"]) for row in manifest["selected"]]
    if mode == "smoke":
        selected = selected[:2]
    elif mode != "formal":
        raise ValueError("mode must be smoke or formal")

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    records = _load_jsonl(partial_path) if partial_path.exists() else []
    done = {(str(row["cqid"]), str(row["action"])) for row in records}
    tasks = [(qid, action) for qid in selected for action in RECOVERY_ACTIONS]
    allowed = set(tasks)
    if done - allowed:
        raise ValueError("partial recovery records contain tasks outside this run")
    if len(done) != len(records):
        raise ValueError("partial recovery records contain duplicate tasks")

    client = CachedChatClient(DEFAULT_ENDPOINT, DEFAULT_MODEL, cache_dir)
    started = time.monotonic()

    def run_task(qid: str, action: str) -> dict[str, Any]:
        parent = grouped[qid]
        context = _preoutcome_context(risk_rows[qid], parent)
        gold_binary = int(parent[0]["gold_binary"])
        return _run_one_recovery(client, examples[qid], context, action, gold_binary)

    pending = [(qid, action) for qid, action in tasks if (qid, action) not in done]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(run_task, qid, action): (qid, action)
            for qid, action in pending
        }
        for future in as_completed(future_map):
            row = future.result()
            records.append(row)
            records.sort(key=lambda value: (str(value["cqid"]), str(value["action"])))
            _write_jsonl(partial_path, records)
            elapsed = time.monotonic() - started
            print(
                f"[{len(records)}/{len(tasks)}] {row['cqid']} {row['action']} "
                f"success={row['success']} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if len(records) != len(tasks):
        raise ValueError("recovery run ended without all expected records")
    _write_jsonl(output_dir / "records.jsonl", records)
    return records


def root_fold(source_root: str) -> int:
    digest = sha256(FOLD_SALT + source_root.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def _feature_row(
    risk_row: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray]:
    names = ["R_PI", "D_inert", "flip_inertia", "frac_shared", "original_consensus_yes"]
    consensus, _ = _consensus(records)
    values = [
        float(risk_row["R_PI"]), float(risk_row["D_inert"]),
        float(risk_row["flip_inertia"]), float(risk_row["frac_shared"]),
        float(consensus == "yes"),
    ]
    original_by_agent = {
        int(record["agent_index"]): record
        for record in records if record["condition"] == "original"
    }
    for condition in ("original", "remove", "reverse", "substitute"):
        condition_records = sorted(
            (record for record in records if record["condition"] == condition),
            key=lambda row: int(row["agent_index"]),
        )
        answers = np.asarray([
            float(record["decision"]["answer"] == "yes") for record in condition_records
        ])
        confidences = np.asarray([
            float(record["decision"]["confidence"]) for record in condition_records
        ])
        citations = np.asarray([
            len(record["decision"]["cited_evidence_ids"]) for record in condition_records
        ], dtype=float)
        agreement = max(float(answers.mean()), 1.0 - float(answers.mean()))
        condition_values = [
            float(answers.mean()), agreement, float(confidences.mean()),
            float(confidences.std()), float(citations.mean()),
        ]
        condition_names = ["yes_fraction", "agreement", "confidence_mean", "confidence_std", "citations_mean"]
        if condition != "original":
            flips = [
                record["decision"]["answer"]
                != original_by_agent[int(record["agent_index"])] ["decision"]["answer"]
                for record in condition_records
            ]
            condition_names.append("agent_flip_fraction")
            condition_values.append(sum(flips) / len(flips))
        names.extend(f"{condition}_{name}" for name in condition_names)
        values.extend(condition_values)
    if FORBIDDEN_FEATURES & set(names):
        raise AssertionError("outcome field entered the recovery feature vector")
    return names, np.asarray(values, dtype=float)


def _bootstrap_predict_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(ENSEMBLE_REPLICATES):
        indices = rng.integers(0, len(x_train), len(x_train))
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train[indices], y_train[indices])
        predictions.append(model.predict(x_test))
    matrix = np.asarray(predictions)
    return matrix.mean(axis=0), matrix.std(axis=0)


def _bootstrap_predict_harm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictions = []
    base_rate = float(y_train.mean())
    for replicate in range(ENSEMBLE_REPLICATES):
        indices = rng.integers(0, len(x_train), len(x_train))
        sample_y = y_train[indices]
        if len(set(sample_y.tolist())) < 2:
            predictions.append(np.full(len(x_test), float(sample_y.mean())))
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=1_000, random_state=seed + replicate),
        )
        model.fit(x_train[indices], sample_y)
        predictions.append(model.predict_proba(x_test)[:, 1])
    if not predictions:
        predictions = [np.full(len(x_test), base_rate)]
    matrix = np.asarray(predictions)
    return matrix.mean(axis=0), matrix.std(axis=0)


def _policy_metrics(
    qids: Sequence[str],
    keep: np.ndarray,
    outcomes: Mapping[str, np.ndarray],
    selected: Sequence[str],
    labels: Sequence[str],
    original_answers: Sequence[str],
) -> dict[str, Any]:
    final = np.asarray([
        keep[index] if action == "KEEP" else outcomes[action][index]
        for index, action in enumerate(selected)
    ], dtype=int)
    fixes = (keep == 0) & (final == 1)
    harms = (keep == 1) & (final == 0)
    gains = final - keep
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(qids)) for _ in qids]
        boot.append(float(gains[indices].mean()))
    boot.sort()
    by_label = {}
    for label in ("yes", "no"):
        indices = [index for index, value in enumerate(labels) if value == label]
        if indices:
            by_label[label] = {
                "n": len(indices),
                "baseline_accuracy": float(keep[indices].mean()),
                "final_accuracy": float(final[indices].mean()),
                "net_gain": float(gains[indices].mean()),
            }
    by_prediction = {}
    for answer in ("yes", "no"):
        indices = [index for index, value in enumerate(original_answers) if value == answer]
        if indices:
            by_prediction[answer] = {
                "n": len(indices),
                "baseline_accuracy": float(keep[indices].mean()),
                "final_accuracy": float(final[indices].mean()),
                "net_gain": float(gains[indices].mean()),
            }
    return {
        "n": len(qids),
        "baseline_accuracy": float(keep.mean()),
        "final_accuracy": float(final.mean()),
        "fixes": int(fixes.sum()),
        "harms": int(harms.sum()),
        "net_fixes": int(fixes.sum() - harms.sum()),
        "net_gain": float(gains.mean()),
        "net_gain_ci": [boot[25], boot[975]],
        "fix_rate": float(fixes.sum() / max(1, int((keep == 0).sum()))),
        "damage_rate": float(harms.sum() / max(1, int((keep == 1).sum()))),
        "mean_added_calls": float(np.mean([action != "KEEP" for action in selected])),
        "selected_actions": dict(Counter(selected)),
        "by_native_label": by_label,
        "by_original_prediction": by_prediction,
    }


def train_and_evaluate(
    recovery_records_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    parent_records = _load_jsonl(PARENT_RECORDS)
    recovery_records = _load_jsonl(recovery_records_path)
    if any(not row.get("success") for row in recovery_records):
        raise ValueError("all recovery calls must succeed before training")
    parent_grouped = _group_records(parent_records)
    recovery_grouped = _group_records(recovery_records)
    selection = json.loads(PARENT_SELECTION.read_text(encoding="utf-8"))
    examples = _example_index(selection)
    risk_rows = _selected_risk_rows(parent_records)
    risk_by_qid = {str(row["cqid"]): row for row in risk_rows}
    qids = [str(row["cqid"]) for row in risk_rows]
    if set(qids) != set(recovery_grouped):
        raise ValueError("recovery records do not match the V12.1 high-consensus cohort")

    feature_names: list[str] | None = None
    feature_rows = []
    keep = []
    labels = []
    original_answers = []
    folds = []
    outcomes = {action: [] for action in RECOVERY_ACTIONS}
    inherited_outcomes = {condition: [] for condition in ("remove", "reverse", "substitute")}
    for qid in qids:
        names, values = _feature_row(risk_by_qid[qid], parent_grouped[qid])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("feature schema drifted between questions")
        feature_rows.append(values)
        parent = parent_grouped[qid]
        gold = int(parent[0]["gold_binary"])
        label = str(parent[0]["label"])
        consensus, _ = _consensus(parent)
        keep.append(int((consensus == "yes") == bool(gold)))
        labels.append(label)
        original_answers.append(consensus)
        folds.append(root_fold(examples[qid].source_root))
        for condition, condition_outcomes in inherited_outcomes.items():
            inherited_answer, _ = _consensus(parent, condition)
            condition_outcomes.append(
                int((inherited_answer == "yes") == bool(gold))
            )
        by_action = {str(row["action"]): row for row in recovery_grouped[qid]}
        if set(by_action) != set(RECOVERY_ACTIONS):
            raise ValueError(f"{qid} does not have exactly one record per recovery action")
        for action in RECOVERY_ACTIONS:
            decision = by_action[action]["decision"]
            outcomes[action].append(int((decision["answer"] == "yes") == bool(gold)))

    x = np.vstack(feature_rows)
    keep_array = np.asarray(keep, dtype=int)
    fold_array = np.asarray(folds, dtype=int)
    outcome_arrays = {key: np.asarray(value, dtype=int) for key, value in outcomes.items()}
    inherited_arrays = {
        key: np.asarray(value, dtype=int) for key, value in inherited_outcomes.items()
    }
    uplift_mean = {action: np.zeros(len(qids)) for action in RECOVERY_ACTIONS}
    uplift_std = {action: np.zeros(len(qids)) for action in RECOVERY_ACTIONS}
    harm_mean = {action: np.zeros(len(qids)) for action in RECOVERY_ACTIONS}
    harm_std = {action: np.zeros(len(qids)) for action in RECOVERY_ACTIONS}
    for fold in range(N_FOLDS):
        train_indices = np.flatnonzero(fold_array != fold)
        test_indices = np.flatnonzero(fold_array == fold)
        if not len(train_indices) or not len(test_indices):
            raise ValueError(f"root fold {fold} is empty")
        for action_index, action in enumerate(RECOVERY_ACTIONS):
            gains = outcome_arrays[action] - keep_array
            means, stds = _bootstrap_predict_regression(
                x[train_indices], gains[train_indices], x[test_indices],
                seed=BOOTSTRAP_SEED + 100 * fold + action_index,
            )
            uplift_mean[action][test_indices] = means
            uplift_std[action][test_indices] = stds
            harm = ((keep_array == 1) & (outcome_arrays[action] == 0)).astype(int)
            means, stds = _bootstrap_predict_harm(
                x[train_indices], harm[train_indices], x[test_indices],
                seed=BOOTSTRAP_SEED + 10_000 + 100 * fold + action_index,
            )
            harm_mean[action][test_indices] = means
            harm_std[action][test_indices] = stds

    gate_n = round(len(qids) * (1.0 - ROUTER_COVERAGE))
    ordered = sorted(range(len(qids)), key=lambda index: float(risk_by_qid[qids[index]]["R_PI"]))
    gate = set(ordered[-gate_n:])
    greedy = ["KEEP"] * len(qids)
    conservative = ["KEEP"] * len(qids)
    for index in gate:
        best = max(RECOVERY_ACTIONS, key=lambda action: float(uplift_mean[action][index]))
        if uplift_mean[best][index] > 0:
            greedy[index] = best
        safe = [
            action for action in RECOVERY_ACTIONS
            if uplift_mean[action][index] - CONSERVATIVE_Z * uplift_std[action][index] > 0
            and harm_mean[action][index] + CONSERVATIVE_Z * harm_std[action][index]
            <= HARM_UCB_CAP
        ]
        if safe:
            conservative[index] = max(
                safe,
                key=lambda action: float(
                    uplift_mean[action][index] - CONSERVATIVE_Z * uplift_std[action][index]
                ),
            )

    policies: dict[str, list[str]] = {
        "learned_greedy_uplift": greedy,
        "learned_conservative_uplift": conservative,
    }
    for action in RECOVERY_ACTIONS:
        policies[f"fixed_{action}"] = [action if index in gate else "KEEP" for index in range(len(qids))]
    policies["keep"] = ["KEEP"] * len(qids)
    policies["flip_consensus_diagnostic"] = ["KEEP"] * len(qids)

    metrics = {
        name: _policy_metrics(qids, keep_array, outcome_arrays, selected, labels, original_answers)
        for name, selected in policies.items() if name != "flip_consensus_diagnostic"
    }
    # This non-action baseline exposes the V12.1 answer-polarity asymmetry.
    flip_final = keep_array.copy()
    for index in gate:
        flip_final[index] = 1 - keep_array[index]
    flip_outcomes = {"FLIP": flip_final}
    metrics["flip_consensus_diagnostic"] = _policy_metrics(
        qids, keep_array, flip_outcomes,
        ["FLIP" if index in gate else "KEEP" for index in range(len(qids))],
        labels, original_answers,
    )
    metrics["flip_consensus_diagnostic"]["mean_added_calls"] = 0.0
    for condition, condition_outcomes in inherited_arrays.items():
        name = f"inherited_{condition}_majority_diagnostic"
        metrics[name] = _policy_metrics(
            qids, keep_array, {condition: condition_outcomes},
            [condition if index in gate else "KEEP" for index in range(len(qids))],
            labels, original_answers,
        )
        metrics[name]["mean_added_calls"] = 0.0
    oracle_selected = []
    for index in range(len(qids)):
        if index not in gate:
            oracle_selected.append("KEEP")
            continue
        candidates = ["KEEP", *RECOVERY_ACTIONS]
        oracle_selected.append(max(
            candidates,
            key=lambda action: int(keep_array[index]) if action == "KEEP" else int(outcome_arrays[action][index]),
        ))
    metrics["available_action_oracle_diagnostic"] = _policy_metrics(
        qids, keep_array, outcome_arrays, oracle_selected, labels, original_answers,
    )

    gate_indices = sorted(gate)
    gate_labels = Counter(labels[index] for index in gate_indices)
    gate_joint = Counter(
        f"native_{labels[index]}__keep_{'correct' if keep_array[index] else 'wrong'}"
        for index in gate_indices
    )
    action_quality = {}
    for action in RECOVERY_ACTIONS:
        final = outcome_arrays[action]
        fixes = int(((keep_array == 0) & (final == 1)).sum())
        harms = int(((keep_array == 1) & (final == 0)).sum())
        action_quality[action] = {
            "all_questions_accuracy": float(final.mean()),
            "all_questions_fixes": fixes,
            "all_questions_harms": harms,
            "gate_fixes": int(sum(keep_array[index] == 0 and final[index] == 1 for index in gate_indices)),
            "gate_harms": int(sum(keep_array[index] == 1 and final[index] == 0 for index in gate_indices)),
        }

    learned_result = metrics["learned_conservative_uplift"]
    fixed_results = [metrics[f"fixed_{action}"] for action in RECOVERY_ACTIONS]
    best_fixed_net = max(int(result["net_fixes"]) for result in fixed_results)
    development_pass = bool(
        learned_result["net_gain_ci"][0] > 0
        and int(learned_result["net_fixes"]) > best_fixed_net
    )

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "retrospective_development_nonconfirmatory",
        "parent_records_sha256": _sha256_path(PARENT_RECORDS),
        "recovery_records_sha256": _sha256_path(recovery_records_path),
        "n_high_consensus": len(qids),
        "n_gate": len(gate),
        "feature_names": feature_names,
        "forbidden_feature_intersection": sorted(FORBIDDEN_FEATURES & set(feature_names or [])),
        "fold_counts": dict(Counter(str(value) for value in folds)),
        "gate_native_labels": dict(gate_labels),
        "gate_label_correctness": dict(gate_joint),
        "action_quality": action_quality,
        "development_verdict": {
            "criterion": "conservative uplift CI lower > 0 and net fixes > every fixed action",
            "best_fixed_net_fixes": best_fixed_net,
            "passes": development_pass,
            "verdict": "PROMISING_FOR_UNTOUCHED_PROTOCOL" if development_pass else "NO_LEARNED_NET_RESCUE",
        },
        "policies": metrics,
        "claim_boundary": {
            "confirmatory": False,
            "provenance_disjoint_recovery": False,
            "verified_rescue_rate_available": False,
            "reason": "V12.1 was already inspected and all recovery actions reuse one source root",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", report)
    lines = [
        "# Recovery V1 retrospective development report", "",
        "## Boundary", "",
        "- This is not an untouched validation result.",
        "- V12.1 and its frozen router remain unchanged.",
        "- All recovery actions reuse the inherited BoolQ source root; verified provenance-disjoint repair is not measured.",
        "", "## Cohort", "",
        f"- High-consensus questions: {len(qids)}",
        f"- Frozen high-risk recovery gate: {len(gate)}",
        f"- Gate native-label composition: `{dict(gate_labels)}`",
        f"- Gate label/correctness composition: `{dict(gate_joint)}`",
        f"- Development verdict: **{report['development_verdict']['verdict']}**",
        "", "## Recovery action quality", "",
    ]
    for action, result in action_quality.items():
        lines.append(
            f"- {action}: all accuracy={result['all_questions_accuracy']:.3f}, "
            f"gate fixes={result['gate_fixes']}, gate harms={result['gate_harms']}"
        )
    lines.extend(["", "## Policies", ""])
    for name, result in metrics.items():
        lines.append(
            f"- {name}: accuracy={result['final_accuracy']:.3f}, fixes={result['fixes']}, "
            f"harms={result['harms']}, net={result['net_fixes']}, "
            f"net gain={result['net_gain']:.3f} {result['net_gain_ci']}, "
            f"added calls={result['mean_added_calls']:.3f}"
        )
    lines.extend([
        "", "## Interpretation", "",
        "Only a positive net result that beats fixed recovery actions without relying on the native-label asymmetry would motivate a new untouched protocol. This development run cannot support a paper claim by itself.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    for command in ("smoke", "run"):
        stage = subparsers.add_parser(command)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"))
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--workers", type=int, default=4)
    train = subparsers.add_parser("train")
    train.add_argument("--records", type=Path, default=DEFAULT_ROOT / "formal" / "records.jsonl")
    train.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "analysis")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_manifest(args.output)
        print(f"{'created' if created else 'validated'} {args.output}")
        return 0
    if args.command in {"smoke", "run"}:
        mode = "smoke" if args.command == "smoke" else "formal"
        execute_recovery_calls(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            mode=mode,
            workers=args.workers,
        )
        return 0
    if args.command == "train":
        train_and_evaluate(args.records, output_dir=args.output_dir)
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
