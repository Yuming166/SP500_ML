"""Frozen V3.16.1 label-symmetric Qwen/Ling fresh-root replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from sp500_forecastability import detection_v3_16_1_selection as selection
from sp500_forecastability import detection_v3_16_analysis as development
from sp500_forecastability import detection_v3_16_calls as calls

FORMAL_VERSION = "detection-v3.16.1-formal-vitaminc-qwen-ling-2026-09-05"
CALL_PROTOCOL = "detection-v3.16.1-vitaminc-common-calls-2026-09-05"
SEED_PROTOCOL = "detection-v3.16.1-common-seed-2026-09-05"
PREREGISTRATION = Path("docs/detection_v3_16_1_preregistration.md")
ROOT = Path("results/detection_v3_16_1")
PUBLIC_MANIFEST = ROOT / "public_manifest.json"
OUTCOME_LEDGER = ROOT / "outcome_ledger.json"
RISK_MANIFEST = Path("results/detection_v3_16_development/calls_4/risk_manifest.json")
LING_PILOT_SUMMARY = Path(
    "results/detection_v3_16_development/calls_4/ling/analysis/development_summary.json"
)
HIGH_CONSENSUS = 0.8
COVERAGE = 0.8
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20261616
MIN_HIGH_CONSENSUS = 400
MIN_ERRORS_PER_LABEL = 20
FORMAL_PAIRS = selection.FRESH_PAIRS
EXPECTED_ITEMS = selection.FORMAL_ITEMS
EXPECTED_CALLS = EXPECTED_ITEMS * len(calls.prompts.AGENT_PERSONAS) * len(calls.CONDITIONS)
RISK_COLUMNS = (
    "risk",
    "agreement_risk",
    "confidence_risk",
    "reverse_inertia",
    "intervention_disagreement",
    "hash_risk",
)
EXPECTED_RISK_WEIGHTS = {
    "reverse_inertia": 0.3,
    "remove_inertia": 0.0,
    "substitute_inertia": 0.0,
    "reverse_confident_nonresponse": 0.0,
    "intervention_disagreement": 0.7,
}

MODEL_DIRS = {
    "qwen": Path("/storage/lianjh/modelzoos/Qwen/Qwen3.5-4B"),
    "ling": Path("/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4"),
}
MODELS = {
    "qwen": calls.ModelSpec(
        "qwen", "Qwen3.5-4B", "http://127.0.0.1:31518/v1/chat/completions"
    ),
    "ling": calls.ModelSpec(
        "ling", "Ling-3.0-tiny", "http://127.0.0.1:31520/v1/chat/completions"
    ),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _formal_item_id(pair_id: str, original_id: str) -> str:
    return hashlib.sha256(f"{FORMAL_VERSION}|item|{pair_id}|{original_id}".encode()).hexdigest()[:20]


def _formal_evidence_id(page: str, unique_id: str) -> str:
    root = hashlib.sha256(f"{FORMAL_VERSION}|root|{page}".encode()).hexdigest()[:16]
    evidence = hashlib.sha256(f"{FORMAL_VERSION}|evidence|{unique_id}".encode()).hexdigest()[:16]
    return f"root_{root}::evidence_{evidence}"


def _evidence(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "evidence_id": _formal_evidence_id(str(row["page"]), str(row["unique_id"])),
        "text": str(row["evidence"]),
    }


def _load_selection_manifest() -> dict[str, Any]:
    selection.audit()
    return json.loads(selection.SELECTION_MANIFEST.read_text(encoding="utf-8"))


def build_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    selected = _load_selection_manifest()
    rows = selection.load_rows()
    by_id = {str(row["unique_id"]): row for row in rows}
    public_items = []
    outcomes = []
    for pair in selected["pairs"]:
        if pair["split"] != "formal":
            raise ValueError("V3.16.1 selection contains a non-formal pair")
        for item in pair["items"]:
            original = by_id[str(item["original_id"])]
            reverse = by_id[str(item["reverse_id"])]
            substitute = by_id[str(pair["distractor_id"])]
            opaque_id = _formal_item_id(str(pair["pair_id"]), str(item["original_id"]))
            public_items.append(
                {
                    "opaque_item_id": opaque_id,
                    "pair_id": str(pair["pair_id"]),
                    "claim": str(original["claim"]),
                    "conditions": {
                        "original": [_evidence(original)],
                        "remove": [],
                        "reverse": [_evidence(reverse)],
                        "substitute": [_evidence(substitute)],
                    },
                }
            )
            outcomes.append(
                {
                    "opaque_item_id": opaque_id,
                    "pair_id": str(pair["pair_id"]),
                    "gold_label": str(item["gold_label"]),
                }
            )
    public_items.sort(key=lambda row: str(row["opaque_item_id"]))
    outcomes.sort(key=lambda row: str(row["opaque_item_id"]))
    public = {
        "formal_version": FORMAL_VERSION,
        "status": "outcome_free_public_manifest",
        "selection_manifest_sha256": selection.file_sha256(selection.SELECTION_MANIFEST),
        "risk_manifest_sha256": selection.file_sha256(RISK_MANIFEST),
        "items": public_items,
    }
    public["manifest_sha256"] = hashlib.sha256(_canonical_json(public).encode()).hexdigest()
    ledger = {
        "formal_version": FORMAL_VERSION,
        "status": "sealed_until_all_preoutcome_routes_are_frozen",
        "public_manifest_sha256": public["manifest_sha256"],
        "items": outcomes,
    }
    return public, ledger


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | set().union(
            *(_recursive_keys(child) for child in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def audit_public_manifest(public: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"gold_label", "label", "correct", "error", "consensus_wrong"}
    items = list(public["items"])
    outcome_items = list(ledger["items"])
    ids = [str(item["opaque_item_id"]) for item in items]
    pair_counts: dict[str, int] = {}
    for item in items:
        pair_id = str(item["pair_id"])
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
    label_counts = {
        label: sum(str(item["gold_label"]) == label for item in outcome_items)
        for label in ("SUPPORTS", "REFUTES")
    }
    public_without_hash = {key: value for key, value in public.items() if key != "manifest_sha256"}
    expected_hash = hashlib.sha256(_canonical_json(public_without_hash).encode()).hexdigest()
    gates = {
        "exact_item_count": len(items) == EXPECTED_ITEMS,
        "opaque_ids_unique": len(ids) == len(set(ids)),
        "two_items_per_pair": len(pair_counts) == FORMAL_PAIRS
        and set(pair_counts.values()) == {2},
        "public_has_no_outcome_keys": not (forbidden & _recursive_keys(public)),
        "public_manifest_hash": public.get("manifest_sha256") == expected_hash,
        "ledger_points_to_public_hash": ledger.get("public_manifest_sha256") == expected_hash,
        "outcome_ids_match_public": {
            str(item["opaque_item_id"]) for item in outcome_items
        }
        == set(ids),
        "outcomes_exactly_balanced": label_counts
        == {"SUPPORTS": FORMAL_PAIRS, "REFUTES": FORMAL_PAIRS},
        "four_conditions": all(
            set(item["conditions"]) == set(calls.CONDITIONS) for item in items
        ),
        "natural_reverse_is_distinct": all(
            item["conditions"]["original"] != item["conditions"]["reverse"]
            for item in items
        ),
    }
    return {
        "formal_version": FORMAL_VERSION,
        "gates": gates,
        "passed": all(gates.values()),
        "label_counts": label_counts,
    }


def prepare() -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)
    if PUBLIC_MANIFEST.exists() or OUTCOME_LEDGER.exists():
        raise ValueError("V3.16.1 formal manifests already exist")
    if list(ROOT.glob("*/formal/records*.jsonl")):
        raise ValueError("cannot prepare manifests after formal calls")
    if not RISK_MANIFEST.is_file() or not LING_PILOT_SUMMARY.is_file():
        raise FileNotFoundError("frozen V3.16 development analysis artifacts are incomplete")
    public, ledger = build_manifests()
    audit = audit_public_manifest(public, ledger)
    if not audit["passed"]:
        raise ValueError(f"formal manifest audit failed: {audit['gates']}")
    _write_json(PUBLIC_MANIFEST, public)
    _write_json(OUTCOME_LEDGER, ledger)
    _write_json(ROOT / "manifest_audit.json", audit)
    return audit


def validate_manifests() -> None:
    public = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    ledger = json.loads(OUTCOME_LEDGER.read_text(encoding="utf-8"))
    expected_public, expected_ledger = build_manifests()
    if public != expected_public or ledger != expected_ledger:
        raise ValueError("formal public manifest or outcome ledger drifted")
    if not audit_public_manifest(public, ledger)["passed"]:
        raise ValueError("formal manifest audit failed")


def _model_fingerprint(model: calls.ModelSpec) -> dict[str, Any]:
    root = MODEL_DIRS[model.key]
    if not root.is_dir():
        raise FileNotFoundError(root)
    small = [
        path
        for name in ("config.json", "generation_config.json", "tokenizer_config.json")
        for path in [root / name]
        if path.is_file()
    ]
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {root}")
    return {
        "path": str(root),
        "small_file_sha256": {path.name: selection.file_sha256(path) for path in small},
        "weight_sizes": {path.name: path.stat().st_size for path in shards},
    }


def _frozen_risk_manifest() -> dict[str, Any]:
    risk_manifest = json.loads(RISK_MANIFEST.read_text(encoding="utf-8"))
    if tuple(risk_manifest["coordinates"]) != development.COORDINATES:
        raise ValueError("frozen risk coordinates drifted")
    actual = {name: float(risk_manifest["weights"][name]) for name in development.COORDINATES}
    if actual != EXPECTED_RISK_WEIGHTS:
        raise ValueError(f"frozen risk weights drifted: {actual}")
    return risk_manifest


def _protocol_path(model: calls.ModelSpec) -> Path:
    return ROOT / model.key / "protocol_manifest.json"


def build_protocol(model: calls.ModelSpec) -> dict[str, Any]:
    _frozen_risk_manifest()
    return {
        "formal_version": FORMAL_VERSION,
        "call_protocol": CALL_PROTOCOL,
        "status": "frozen_before_any_model_specific_formal_call",
        "model": {"key": model.key, "id": model.model, "endpoint": model.endpoint},
        "model_fingerprint": _model_fingerprint(model),
        "selection_manifest_sha256": selection.file_sha256(selection.SELECTION_MANIFEST),
        "public_manifest_sha256": selection.file_sha256(PUBLIC_MANIFEST),
        "outcome_ledger_sha256": selection.file_sha256(OUTCOME_LEDGER),
        "risk_manifest_sha256": selection.file_sha256(RISK_MANIFEST),
        "ling_pilot_summary_sha256": selection.file_sha256(LING_PILOT_SUMMARY),
        "preregistration_sha256": selection.file_sha256(PREREGISTRATION),
        "selection_implementation_sha256": selection.file_sha256(
            Path(selection.__file__)
        ),
        "implementation_sha256": selection.file_sha256(Path(__file__)),
        "call_implementation_sha256": selection.file_sha256(Path(calls.__file__)),
        "prompts_sha256": selection.file_sha256(Path(calls.prompts.__file__)),
        "expected_items": EXPECTED_ITEMS,
        "expected_calls": EXPECTED_CALLS,
        "conditions": list(calls.CONDITIONS),
        "agents": [agent_id for agent_id, _ in calls.prompts.AGENT_PERSONAS],
        "temperature": 0.0,
        "max_completion_tokens": calls.MAX_COMPLETION_TOKENS,
        "max_attempts": calls.MAX_ATTEMPTS,
        "server_response_format": None,
        "seed_protocol": SEED_PROTOCOL,
        "high_consensus": HIGH_CONSENSUS,
        "coverage": COVERAGE,
        "risk_weights": EXPECTED_RISK_WEIGHTS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gates": {
            "valid_rate_at_least": 0.98,
            "first_pass_valid_rate_at_least": 0.95,
            "high_consensus_at_least": MIN_HIGH_CONSENSUS,
            "errors_per_label_at_least": MIN_ERRORS_PER_LABEL,
            "overall_auroc_ci_lower_above": 0.5,
            "macro_label_auroc_ci_lower_above": 0.5,
            "worst_label_auroc_ci_lower_above": 0.5,
            "risk80_reduction_ci_lower_above": 0.0,
        },
        "formal_outcomes_accessed": False,
    }


def freeze_protocol(model: calls.ModelSpec) -> dict[str, Any]:
    validate_manifests()
    expected = build_protocol(model)
    path = _protocol_path(model)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{model.key} V3.16.1 formal protocol drifted")
        return actual
    if list((ROOT / model.key / "formal").glob("records*.jsonl")):
        raise ValueError("cannot freeze model protocol after formal calls")
    _write_json(path, expected)
    return expected


def validate_protocol(model: calls.ModelSpec) -> None:
    path = _protocol_path(model)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != build_protocol(model):
        raise ValueError(f"{model.key} V3.16.1 formal protocol drifted")


def load_tasks() -> list[calls.Task]:
    public = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    if len(public["items"]) != EXPECTED_ITEMS:
        raise ValueError("public manifest item count mismatch")
    tasks = []
    for item in public["items"]:
        for agent_index, (agent_id, persona) in enumerate(calls.prompts.AGENT_PERSONAS):
            for condition in calls.CONDITIONS:
                evidence = tuple(
                    {
                        "evidence_id": str(row["evidence_id"]),
                        "text": str(row["text"]),
                    }
                    for row in item["conditions"][condition]
                )
                allowed = tuple(row["evidence_id"] for row in evidence)
                tasks.append(
                    calls.Task(
                        split="formal",
                        pair_id=str(item["pair_id"]),
                        item_id=str(item["opaque_item_id"]),
                        agent_id=agent_id,
                        agent_index=agent_index,
                        persona=persona,
                        condition=condition,
                        claim=str(item["claim"]),
                        evidence=evidence,
                        allowed_evidence_ids=allowed,
                    )
                )
    return sorted(tasks, key=lambda task: calls.task_key(task.__dict__))


def _set_call_protocol() -> None:
    calls.CALL_PROTOCOL = CALL_PROTOCOL
    calls.SEED_PROTOCOL = SEED_PROTOCOL


def run(model: calls.ModelSpec, workers: int, resume: bool = True) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    validate_protocol(model)
    _set_call_protocol()
    calls.endpoint_models(model)
    tasks = load_tasks()
    expected = {calls.task_key(task.__dict__) for task in tasks}
    output_dir = ROOT / model.key / "formal"
    partial_path = output_dir / "records.partial.jsonl"
    final_path = output_dir / "records.jsonl"
    if not resume:
        partial_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
    records = _load_jsonl(partial_path) if resume else []
    done = {calls.task_key(record) for record in records}
    if len(done) != len(records) or done - expected:
        raise ValueError("formal partial contains duplicate or foreign keys")
    client = calls.ChatClient(model, ROOT / model.key / "cache")
    pending = [task for task in tasks if calls.task_key(task.__dict__) not in done]
    started = time.monotonic()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(calls.invoke, client, model, task): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            record["protocol_version"] = FORMAL_VERSION
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)
            done.add(calls.task_key(record))
            if completed % 100 == 0 or completed == len(pending):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                _write_json(
                    output_dir / "progress.json",
                    {
                        "model": model.model,
                        "completed": len(done),
                        "total": len(expected),
                        "successful": sum(bool(row["success"]) for row in records),
                        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in records),
                        "eta_seconds": (len(expected) - len(done)) / rate if rate else None,
                    },
                )
    if done != expected:
        raise ValueError(f"formal run incomplete: {len(done)}/{len(expected)}")
    records = sorted(records, key=calls.task_key)
    final_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    result = {
        "model": model.model,
        "rows": len(records),
        "successful": sum(bool(row["success"]) for row in records),
        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in records),
    }
    result["valid_rate"] = result["successful"] / result["rows"]
    result["first_pass_valid_rate"] = result["first_pass_valid"] / result["rows"]
    result["qualified"] = result["valid_rate"] >= 0.98 and result["first_pass_valid_rate"] >= 0.95
    _write_json(output_dir / "qualification.json", result)
    return result


def _analysis_opaque(formal_item_id: str) -> str:
    return development._opaque_id(formal_item_id)


def _expected_task_keys() -> set[tuple[str, int, str]]:
    return {calls.task_key(task.__dict__) for task in load_tasks()}


def freeze_preoutcome(model: calls.ModelSpec) -> dict[str, Any]:
    validate_protocol(model)
    records_path = ROOT / model.key / "formal" / "records.jsonl"
    records = _load_jsonl(records_path)
    actual = {calls.task_key(record) for record in records}
    if len(records) != EXPECTED_CALLS or actual != _expected_task_keys():
        raise ValueError("formal records are incomplete or keyed incorrectly")
    if any(record.get("protocol_version") != FORMAL_VERSION for record in records):
        raise ValueError("formal records contain an unexpected protocol version")
    rows = development.build_preoutcome_rows(records)
    if len(rows) != EXPECTED_ITEMS:
        raise ValueError("preoutcome row count mismatch")
    risk_manifest = _frozen_risk_manifest()
    weights_map = {name: float(risk_manifest["weights"][name]) for name in development.COORDINATES}
    for row in rows:
        row["risk"] = development._risk(
            row, tuple(weights_map[name] for name in development.COORDINATES)
        )
        row["agreement_risk"] = 1.0 - float(row["agreement"])
        row["confidence_risk"] = 1.0 - float(row["mean_consensus_confidence"])
        row["hash_risk"] = int(
            hashlib.sha256(f"{FORMAL_VERSION}|random|{row['opaque_id']}".encode()).hexdigest()[
                :12
            ],
            16,
        ) / float(16**12 - 1)
    high = [row for row in rows if float(row["agreement"]) >= HIGH_CONSENSUS]
    retained_n = max(1, int(np.floor(COVERAGE * len(high))))
    retained = {
        name: [
            str(row["opaque_id"])
            for row in sorted(high, key=lambda row: (float(row[name]), str(row["opaque_id"])))[:retained_n]
        ]
        for name in RISK_COLUMNS
    }
    payload = {
        "formal_version": FORMAL_VERSION,
        "status": "preoutcome_routes_frozen_before_outcome_ledger_access",
        "model": model.model,
        "protocol_sha256": selection.file_sha256(_protocol_path(model)),
        "records_sha256": selection.file_sha256(records_path),
        "outcomes_accessed": False,
        "risk_manifest_sha256": selection.file_sha256(RISK_MANIFEST),
        "weights": risk_manifest["weights"],
        "high_consensus_threshold": HIGH_CONSENSUS,
        "coverage": COVERAGE,
        "risk_columns": list(RISK_COLUMNS),
        "rows": rows,
        "high_consensus_ids": [str(row["opaque_id"]) for row in high],
        "retained_ids": retained,
    }
    path = ROOT / model.key / "evaluation" / "preoutcome_routes.json"
    if path.exists():
        actual_payload = json.loads(path.read_text(encoding="utf-8"))
        if actual_payload != payload:
            raise ValueError(f"{model.key} preoutcome routes drifted")
        return actual_payload
    _write_json(path, payload)
    return payload


def _aurc(rows: Sequence[Mapping[str, Any]], score: str) -> float:
    ordered = sorted(rows, key=lambda row: (float(row[score]), str(row["opaque_id"])))
    errors = np.asarray([int(row["error"]) for row in ordered], dtype=float)
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def _optional_auroc(errors: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(errors)) != 2:
        return None
    return float(roc_auc_score(errors, scores))


def _score_metrics(rows: Sequence[Mapping[str, Any]], score: str) -> dict[str, Any]:
    values = [float(row[score]) for row in rows]
    errors = [int(row["error"]) for row in rows]
    by_label = {}
    for label in ("SUPPORTS", "REFUTES"):
        subset = [row for row in rows if row["gold_label"] == label]
        subset_errors = [int(row["error"]) for row in subset]
        subset_values = [float(row[score]) for row in subset]
        by_label[label] = {
            "n": len(subset),
            "errors": sum(subset_errors),
            "auroc": _optional_auroc(subset_errors, subset_values),
        }
    label_aurocs = [by_label[label]["auroc"] for label in ("SUPPORTS", "REFUTES")]
    valid_label_aurocs = [value for value in label_aurocs if value is not None]
    order = sorted(range(len(rows)), key=lambda i: (values[i], str(rows[i]["opaque_id"])))
    keep_n = max(1, int(np.floor(COVERAGE * len(rows))))
    baseline = float(np.mean(errors))
    retained = float(np.mean([errors[index] for index in order[:keep_n]]))
    return {
        "n": len(rows),
        "errors": sum(errors),
        "overall_auroc": _optional_auroc(errors, values),
        "by_label": by_label,
        "macro_label_auroc": (
            float(np.mean(valid_label_aurocs)) if len(valid_label_aurocs) == 2 else None
        ),
        "worst_label_auroc": min(valid_label_aurocs) if len(valid_label_aurocs) == 2 else None,
        "aurc": _aurc(rows, score),
        "risk_at_80": {
            "retained": keep_n,
            "baseline_error": baseline,
            "retained_error": retained,
            "error_reduction": baseline - retained,
        },
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], score: str) -> dict[str, list[float] | None]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    pairs = sorted(by_pair)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {
        "overall": [],
        "macro": [],
        "worst": [],
        "reduction": [],
        "supports": [],
        "refutes": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(pairs, size=len(pairs), replace=True)
        sample = [dict(row) for pair in selected for row in by_pair[str(pair)]]
        metric = _score_metrics(sample, score)
        if metric["overall_auroc"] is not None:
            samples["overall"].append(metric["overall_auroc"])
        if metric["macro_label_auroc"] is not None:
            samples["macro"].append(metric["macro_label_auroc"])
        if metric["worst_label_auroc"] is not None:
            samples["worst"].append(metric["worst_label_auroc"])
        samples["reduction"].append(metric["risk_at_80"]["error_reduction"])
        for label, key in (("SUPPORTS", "supports"), ("REFUTES", "refutes")):
            value = metric["by_label"][label]["auroc"]
            if value is not None:
                samples[key].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(values, [0.025, 0.975])]
            if values
            else None
        )
        for name, values in samples.items()
    }


def _ci_lower_above(
    intervals: Mapping[str, list[float] | None], name: str, threshold: float
) -> bool:
    interval = intervals[name]
    return interval is not None and interval[0] > threshold


def _joined_rows(model: calls.ModelSpec, outcomes: Mapping[str, str]) -> list[dict[str, Any]]:
    path = ROOT / model.key / "evaluation" / "preoutcome_routes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["outcomes_accessed"] is not False:
        raise ValueError("preoutcome payload is contaminated")
    high_ids = set(payload["high_consensus_ids"])
    rows = []
    for row in payload["rows"]:
        opaque_id = str(row["opaque_id"])
        if opaque_id not in high_ids:
            continue
        if opaque_id not in outcomes:
            raise ValueError(f"missing outcome for {opaque_id}")
        rows.append(
            {
                **row,
                "gold_label": outcomes[opaque_id],
                "error": int(str(row["consensus"]) != outcomes[opaque_id]),
            }
        )
    return rows


def evaluate() -> dict[str, Any]:
    validate_manifests()
    models = [MODELS["qwen"], MODELS["ling"]]
    for model in models:
        validate_protocol(model)
        preoutcome = ROOT / model.key / "evaluation" / "preoutcome_routes.json"
        if not preoutcome.is_file():
            raise ValueError(f"missing frozen preoutcome routes for {model.key}")
    ledger = json.loads(OUTCOME_LEDGER.read_text(encoding="utf-8"))
    outcomes = {
        _analysis_opaque(str(item["opaque_item_id"])): str(item["gold_label"])
        for item in ledger["items"]
    }
    summaries = {}
    risk_vectors = {}
    for model in models:
        rows = _joined_rows(model, outcomes)
        metrics = _score_metrics(rows, "risk")
        intervals = _bootstrap(rows, "risk")
        qualification = json.loads(
            (ROOT / model.key / "formal" / "qualification.json").read_text(encoding="utf-8")
        )
        gates = {
            "final_valid_rate_at_least_098": qualification["valid_rate"] >= 0.98,
            "first_pass_valid_rate_at_least_095": qualification["first_pass_valid_rate"] >= 0.95,
            "high_consensus_at_least_400": metrics["n"] >= MIN_HIGH_CONSENSUS,
            "errors_per_label_at_least_20": all(
                metrics["by_label"][label]["errors"] >= MIN_ERRORS_PER_LABEL
                for label in ("SUPPORTS", "REFUTES")
            ),
            "overall_auroc_ci_lower_above_05": _ci_lower_above(intervals, "overall", 0.5),
            "macro_label_auroc_ci_lower_above_05": _ci_lower_above(intervals, "macro", 0.5),
            "worst_label_auroc_ci_lower_above_05": _ci_lower_above(intervals, "worst", 0.5),
            "risk80_reduction_ci_lower_above_zero": _ci_lower_above(intervals, "reduction", 0.0),
        }
        baseline_names = (
            "agreement_risk",
            "confidence_risk",
            "reverse_inertia",
            "intervention_disagreement",
            "hash_risk",
        )
        summaries[model.key] = {
            "model": model.model,
            "transport": qualification,
            "metrics": metrics,
            "pair_bootstrap_ci": intervals,
            "baselines": {name: _score_metrics(rows, name) for name in baseline_names},
            "gates": gates,
            "passed": all(gates.values()),
        }
        risk_vectors[model.key] = {str(row["opaque_id"]): float(row["risk"]) for row in rows}
    common = sorted(set(risk_vectors["qwen"]) & set(risk_vectors["ling"]))
    if len(common) >= 2:
        correlation = spearmanr(
            [risk_vectors["qwen"][item] for item in common],
            [risk_vectors["ling"][item] for item in common],
        ).statistic
        correlation_value = float(correlation) if math.isfinite(float(correlation)) else None
    else:
        correlation_value = None
    result = {
        "formal_version": FORMAL_VERSION,
        "models": summaries,
        "common_high_consensus_items": len(common),
        "qwen_ling_risk_spearman": correlation_value,
        "cross_family_pass": all(summary["passed"] for summary in summaries.values()),
        "claim_boundary": {
            "fresh_root_replication": True,
            "label_symmetric_detection": True,
            "answer_repair": False,
            "universal_transfer": False,
            "pooled_old_and_new_outcomes": False,
        },
    }
    _write_json(ROOT / "evaluation" / "summary.json", result)
    (ROOT / "evaluation" / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Detection V3.16.1 formal: fresh-root label-symmetric replication",
        "",
        f"Cross-family verdict: **{'PASS' if result['cross_family_pass'] else 'FAIL'}**.",
        "",
        "| Model | N | Errors | AUROC | Macro | Worst | Risk@80 | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in result["models"].values():
        metric = summary["metrics"]
        lines.append(
            f"| {summary['model']} | {metric['n']} | {metric['errors']} | "
            f"{_format_metric(metric['overall_auroc'])} | "
            f"{_format_metric(metric['macro_label_auroc'])} | "
            f"{_format_metric(metric['worst_label_auroc'])} | "
            f"{metric['risk_at_80']['baseline_error']:.3f} -> "
            f"{metric['risk_at_80']['retained_error']:.3f} | "
            f"{'PASS' if summary['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            (
                "Qwen--Ling risk Spearman on common high-consensus items: "
                f"{_format_metric(result['qwen_ling_risk_spearman'])}."
            ),
            "",
            "This result concerns selective error detection under natural contrastive",
            "evidence on a fresh page-disjoint cohort. It does not establish answer",
            "repair or universal transfer, and it does not pool prior V3.16 outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "freeze", "run", "freeze-preoutcome", "evaluate")
    )
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--workers", type=int, default=calls.DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "evaluate":
        result = evaluate()
    else:
        if args.model is None:
            raise ValueError(f"{args.command} requires --model")
        model = MODELS[args.model]
        if args.command == "freeze":
            result = freeze_protocol(model)
        elif args.command == "run":
            result = run(model, args.workers, resume=not args.no_resume)
        else:
            result = freeze_preoutcome(model)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
