"""Outcome-firewalled development analysis for V3.16."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from sp500_forecastability import detection_v3_16 as protocol
from sp500_forecastability import detection_v3_16_calls as calls

ANALYSIS_VERSION = "detection-v3.16.1-pair-grouped-development-analysis-2026-09-03"
COORDINATES = (
    "reverse_inertia",
    "remove_inertia",
    "substitute_inertia",
    "reverse_confident_nonresponse",
    "intervention_disagreement",
)
HIGH_CONSENSUS = 0.8
COVERAGE = 0.8
NESTED_FOLDS = 5
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261603
RISK_MANIFEST = protocol.DEFAULT_ROOT / "calls_1" / "risk_manifest.json"
MODEL_CALL_ROOTS = {"qwen": "calls_1", "ling": "calls_3"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _opaque_id(item_id: str) -> str:
    return hashlib.sha256(f"{ANALYSIS_VERSION}|{item_id}".encode()).hexdigest()[:20]


def _records_path(model: calls.ModelSpec, split: str) -> Path:
    return protocol.DEFAULT_ROOT / MODEL_CALL_ROOTS[model.key] / model.key / split / "records.jsonl"


def _preoutcome_path(model: calls.ModelSpec, split: str) -> Path:
    return (
        protocol.DEFAULT_ROOT
        / MODEL_CALL_ROOTS[model.key]
        / model.key
        / "analysis"
        / f"{split}_preoutcome.json"
    )


def _summary_path(model: calls.ModelSpec, split: str) -> Path:
    return (
        protocol.DEFAULT_ROOT
        / MODEL_CALL_ROOTS[model.key]
        / model.key
        / "analysis"
        / f"{split}_summary.json"
    )


def _flip(original: Mapping[str, Any], changed: Mapping[str, Any]) -> float:
    return float(original["answer"] != changed["answer"])


def build_preoutcome_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    forbidden = {"gold_label", "label", "correct", "error", "consensus_wrong"}
    if any(forbidden & set(record) for record in records):
        raise ValueError("records contain forbidden outcome fields")
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any] | None]] = defaultdict(dict)
    item_pairs: dict[str, str] = {}
    for record in records:
        item_id = str(record["item_id"])
        pair_id = str(record["pair_id"])
        item_pairs[item_id] = pair_id
        decision = record.get("decision")
        grouped[(item_id, int(record["agent_index"]))][str(record["condition"])] = (
            decision if isinstance(decision, Mapping) else None
        )
    item_ids = sorted(item_pairs)
    output = []
    for item_id in item_ids:
        agents = [grouped[(item_id, index)] for index in range(len(calls.prompts.AGENT_PERSONAS))]
        if any(set(agent) != set(calls.CONDITIONS) for agent in agents):
            raise ValueError(f"incomplete intervention bundle for {item_id}")
        original_valid = [
            agent["original"] for agent in agents if isinstance(agent["original"], Mapping)
        ]
        original_answers = [str(decision["answer"]) for decision in original_valid]
        consensus, count = (
            Counter(original_answers).most_common(1)[0] if original_answers else ("SUPPORTS", 0)
        )
        agreement = count / len(agents)
        incomplete = any(
            not isinstance(agent[condition], Mapping)
            for agent in agents
            for condition in calls.CONDITIONS
        )
        if incomplete:
            reverse_inertia = 1.0
            remove_inertia = 1.0
            substitute_inertia = 1.0
            reverse_confident_nonresponse = 1.0
            intervention_disagreement = 1.0
        else:
            flip_vectors = []
            for agent in agents:
                flip_vectors.append(
                    np.array(
                        [
                            _flip(agent["original"], agent[condition])
                            for condition in ("remove", "reverse", "substitute")
                        ]
                    )
                )
            reverse_inertia = float(np.mean([1.0 - vector[1] for vector in flip_vectors]))
            remove_inertia = float(np.mean([1.0 - vector[0] for vector in flip_vectors]))
            substitute_inertia = float(np.mean([1.0 - vector[2] for vector in flip_vectors]))
            reverse_confident_nonresponse = float(
                np.mean(
                    [
                        (1.0 - flip_vectors[index][1]) * float(agent["reverse"]["confidence"])
                        for index, agent in enumerate(agents)
                    ]
                )
            )
            per_agent_flip_rate = [float(vector.mean()) for vector in flip_vectors]
            intervention_disagreement = float(min(1.0, 2.0 * np.std(per_agent_flip_rate, ddof=0)))
        consensus_confidences = [
            float(decision["confidence"])
            for decision in original_valid
            if decision["answer"] == consensus
        ]
        output.append(
            {
                "opaque_id": _opaque_id(item_id),
                "pair_id": item_pairs[item_id],
                "consensus": consensus,
                "agreement": agreement,
                "mean_consensus_confidence": (
                    float(np.mean(consensus_confidences)) if consensus_confidences else 0.0
                ),
                "transport_incomplete": incomplete,
                "reverse_inertia": reverse_inertia,
                "remove_inertia": remove_inertia,
                "substitute_inertia": substitute_inertia,
                "reverse_confident_nonresponse": reverse_confident_nonresponse,
                "intervention_disagreement": intervention_disagreement,
            }
        )
    return output


def freeze_preoutcome(model: calls.ModelSpec, split: str) -> dict[str, Any]:
    calls.validate_protocol(model)
    records_path = _records_path(model, split)
    records = _load_jsonl(records_path)
    rows = build_preoutcome_rows(records)
    expected_items = (protocol.SMOKE_PAIRS if split == "smoke" else protocol.DEVELOPMENT_PAIRS) * 2
    if len(rows) != expected_items:
        raise ValueError(f"preoutcome row count mismatch: {len(rows)}/{expected_items}")
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "preoutcome_rows_frozen_before_label_join",
        "model": model.model,
        "split": split,
        "records_sha256": protocol.file_sha256(records_path),
        "coordinates": list(COORDINATES),
        "rows": rows,
    }
    path = _preoutcome_path(model, split)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != payload:
            raise ValueError("preoutcome rows drifted")
        return actual
    _write_json(path, payload)
    return payload


def _outcomes(split: str) -> dict[str, str]:
    manifest = json.loads(protocol.SELECTION_MANIFEST.read_text(encoding="utf-8"))
    return {
        _opaque_id(str(item["item_id"])): str(item["gold_label"])
        for pair in manifest["pairs"]
        if pair["split"] == split
        for item in pair["items"]
    }


def join_outcomes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcomes = _outcomes(str(payload["split"]))
    rows = []
    for row in payload["rows"]:
        opaque_id = str(row["opaque_id"])
        gold = outcomes[opaque_id]
        rows.append({**row, "gold_label": gold, "error": int(row["consensus"] != gold)})
    return rows


def _simplex_weights() -> list[tuple[float, ...]]:
    return [
        tuple(unit / 10 for unit in units)
        for units in itertools.product(range(11), repeat=len(COORDINATES))
        if sum(units) == 10
    ]


def _risk(row: Mapping[str, Any], weights: Sequence[float]) -> float:
    return float(sum(weight * float(row[name]) for name, weight in zip(COORDINATES, weights)))


def _risk_at_80(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    keep_n = max(1, int(np.floor(COVERAGE * len(rows))))
    order = sorted(
        range(len(rows)), key=lambda index: (float(scores[index]), str(rows[index]["opaque_id"]))
    )
    baseline = float(np.mean([row["error"] for row in rows]))
    retained = float(np.mean([rows[index]["error"] for index in order[:keep_n]]))
    return {
        "retained": keep_n,
        "baseline_error": baseline,
        "retained_error": retained,
        "error_reduction": baseline - retained,
    }


def _metrics(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> dict[str, Any]:
    scores = [_risk(row, weights) for row in rows]
    errors = [int(row["error"]) for row in rows]
    overall = float(roc_auc_score(errors, scores)) if len(set(errors)) == 2 else None
    by_label = {}
    for label in ("SUPPORTS", "REFUTES"):
        indices = [index for index, row in enumerate(rows) if row["gold_label"] == label]
        y = [errors[index] for index in indices]
        x = [scores[index] for index in indices]
        by_label[label] = {
            "n": len(indices),
            "errors": sum(y),
            "auroc": float(roc_auc_score(y, x)) if len(set(y)) == 2 else None,
        }
    label_aurocs = [by_label[label]["auroc"] for label in by_label]
    macro = (
        float(np.mean(label_aurocs)) if all(value is not None for value in label_aurocs) else None
    )
    worst = float(min(label_aurocs)) if all(value is not None for value in label_aurocs) else None
    return {
        "n": len(rows),
        "errors": sum(errors),
        "overall_auroc": overall,
        "by_label": by_label,
        "macro_label_auroc": macro,
        "worst_label_auroc": worst,
        "risk_at_80": _risk_at_80(rows, scores),
    }


def _selection_key(
    rows: Sequence[Mapping[str, Any]], weights: tuple[float, ...]
) -> tuple[Any, ...]:
    metrics = _metrics(rows, weights)
    return (
        -float(metrics["worst_label_auroc"]),
        -float(metrics["macro_label_auroc"]),
        -float(metrics["risk_at_80"]["error_reduction"]),
        weights,
    )


def select_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    if any(metrics is None for metrics in [_metrics(rows, (1, 0, 0, 0, 0))["worst_label_auroc"]]):
        raise ValueError("both labels require correct and incorrect examples")
    return min(_simplex_weights(), key=lambda weights: _selection_key(rows, weights))


def _fold(pair_id: str) -> int:
    return (
        int(hashlib.sha256(f"{ANALYSIS_VERSION}|fold|{pair_id}".encode()).hexdigest()[:8], 16)
        % NESTED_FOLDS
    )


def nested_oof_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, float] = {}
    fold_weights = {}
    for fold in range(NESTED_FOLDS):
        train = [row for row in rows if _fold(str(row["pair_id"])) != fold]
        test = [row for row in rows if _fold(str(row["pair_id"])) == fold]
        if not test:
            continue
        weights = select_weights(train)
        fold_weights[str(fold)] = dict(zip(COORDINATES, weights))
        scores.update({str(row["opaque_id"]): _risk(row, weights) for row in test})
    if len(scores) != len(rows):
        raise ValueError("nested OOF scoring did not cover every row")
    scored = []
    for row in rows:
        copy = dict(row)
        copy["nested_risk"] = scores[str(row["opaque_id"])]
        scored.append(copy)
    proxy_weights = (0, 0, 0, 0, 0)
    errors = [int(row["error"]) for row in scored]
    risk = [float(row["nested_risk"]) for row in scored]
    overall = float(roc_auc_score(errors, risk))
    by_label = {}
    for label in ("SUPPORTS", "REFUTES"):
        subset = [row for row in scored if row["gold_label"] == label]
        by_label[label] = float(
            roc_auc_score([row["error"] for row in subset], [row["nested_risk"] for row in subset])
        )
    del proxy_weights
    return {
        "fold_weights": fold_weights,
        "overall_auroc": overall,
        "by_label_auroc": by_label,
        "macro_label_auroc": float(np.mean(list(by_label.values()))),
        "worst_label_auroc": float(min(by_label.values())),
        "risk_at_80": _risk_at_80(scored, risk),
    }


def _coordinate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for index, name in enumerate(COORDINATES):
        weights = tuple(float(position == index) for position in range(len(COORDINATES)))
        output[name] = _metrics(rows, weights)
    return output


def _bootstrap(rows: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> dict[str, Any]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    pair_ids = sorted(by_pair)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"overall": [], "macro": [], "worst": [], "reduction": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sample = [dict(row) for sample_id in sampled for row in by_pair[str(sample_id)]]
        metrics = _metrics(sample, weights)
        if metrics["overall_auroc"] is not None and metrics["worst_label_auroc"] is not None:
            values["overall"].append(metrics["overall_auroc"])
            values["macro"].append(metrics["macro_label_auroc"])
            values["worst"].append(metrics["worst_label_auroc"])
            values["reduction"].append(metrics["risk_at_80"]["error_reduction"])
    return {
        name: [float(value) for value in np.quantile(samples, [0.025, 0.975])]
        for name, samples in values.items()
    }


def evaluate_qwen_development() -> dict[str, Any]:
    model = calls.MODELS["qwen"]
    payload = json.loads(_preoutcome_path(model, "development").read_text(encoding="utf-8"))
    records_path = _records_path(model, "development")
    if payload["records_sha256"] != protocol.file_sha256(records_path):
        raise ValueError("Qwen development records drifted")
    rows = [row for row in join_outcomes(payload) if float(row["agreement"]) >= HIGH_CONSENSUS]
    weights = select_weights(rows)
    metrics = _metrics(rows, weights)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "qwen_development_only",
        "model": model.model,
        "high_consensus_threshold": HIGH_CONSENSUS,
        "coverage": COVERAGE,
        "coordinates": list(COORDINATES),
        "coordinate_metrics": _coordinate_metrics(rows),
        "nested_oof": nested_oof_metrics(rows),
        "selected_weights": dict(zip(COORDINATES, weights)),
        "selected_metrics": metrics,
        "pair_bootstrap_ci": _bootstrap(rows, weights),
        "formal_claim_authorized": False,
    }
    path = _summary_path(model, "development")
    _write_json(path, summary)
    risk_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "frozen_after_qwen_development_before_any_ling_v3_16_call",
        "qwen_records_sha256": protocol.file_sha256(records_path),
        "qwen_preoutcome_sha256": protocol.file_sha256(_preoutcome_path(model, "development")),
        "qwen_summary_sha256": protocol.file_sha256(path),
        "coordinates": list(COORDINATES),
        "weights": dict(zip(COORDINATES, weights)),
        "ling_pilot_gates": {
            "valid_rate_at_least_098": 0.98,
            "first_pass_valid_rate_at_least_095": 0.95,
            "overall_auroc_above": 0.55,
            "macro_label_auroc_above": 0.55,
            "worst_label_auroc_above": 0.50,
            "risk80_reduction_at_least": 0.0,
        },
        "claim_boundary": {"formal_calls_authorized": False},
    }
    if RISK_MANIFEST.exists():
        actual = json.loads(RISK_MANIFEST.read_text(encoding="utf-8"))
        if actual != risk_manifest:
            raise ValueError("frozen V3.16 risk manifest drifted")
    else:
        _write_json(RISK_MANIFEST, risk_manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-preoutcome", "evaluate-qwen"))
    parser.add_argument("--model", choices=tuple(calls.MODELS), default="qwen")
    parser.add_argument("--split", choices=("smoke", "development"), default="development")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-preoutcome":
        result = freeze_preoutcome(calls.MODELS[args.model], args.split)
    else:
        if args.model != "qwen" or args.split != "development":
            raise ValueError("weight selection is registered only for Qwen development")
        result = evaluate_qwen_development()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
