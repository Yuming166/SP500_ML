"""CEW: contrastive evidence-witness routing on the frozen Recovery V3 matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib

from sp500_forecastability import recovery_v3 as base

PROTOCOL_VERSION = "recovery-v3.3-cew-averitec-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_3")
BASE_ROOT = Path("results/recovery_v3_2")
WITNESS_PROBABILITY_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WITNESS_DELTA_GRID = (-0.2, 0.0, 0.1, 0.2, 0.3, 0.4)


def _witness_policy(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    stance: Mapping[tuple[str, str], Any],
    *,
    probability_threshold: float,
    delta_threshold: float,
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(grouped[example_id])
        if agreement < base.HIGH_CONSENSUS:
            selected[example_id] = "KEEP"
            continue
        opposition_index = base.STANCE_CLASSES.index("supports" if consensus == "no" else "refutes")
        anchor_score = float(stance[(example_id, "anchor")][opposition_index])
        candidate_scores = [
            float(stance[(example_id, f"candidate_{index}")][opposition_index]) for index in (0, 1)
        ]
        candidate_index = max(
            range(2),
            key=lambda index: (
                candidate_scores[index],
                base._hash_key(b"cew-v3-candidate-tie", f"{example_id}\0{index}"),
            ),
        )
        score = candidate_scores[candidate_index]
        selected[example_id] = (
            f"candidate_{candidate_index}"
            if score >= probability_threshold and score - anchor_score >= delta_threshold
            else "KEEP"
        )
    return selected


def _fit_witness_router(
    selection_path: Path,
    base_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_witness_manifest(manifest_path, selection_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if (DEFAULT_ROOT / "test" / "records.jsonl").exists():
        raise ValueError("cannot fit or overwrite CEW after prospective-test records exist")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    base.validate_selection(selection)
    base.validate_router_manifest(base_manifest_path, selection_path)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    model_bundle = joblib.load(base_manifest["router_joblib"])
    stance_model = model_bundle["stance_model"]
    development = {}
    for split in ("policy_dev", "calibration"):
        examples = [row for row in selection["examples"] if row["split"] == split]
        records_path = BASE_ROOT / split / "records.jsonl"
        records = base._load_jsonl(records_path)
        base._validate_action_matrix(examples, records, split=split)
        development[split] = {
            "examples": examples,
            "records": records,
            "grouped": base._record_groups(records),
            "stance": base._stance_predictions(stance_model, examples),
            "records_sha256": base._sha256_path(records_path),
        }
    feasible = []
    for probability_threshold in WITNESS_PROBABILITY_GRID:
        for delta_threshold in WITNESS_DELTA_GRID:
            metrics = {}
            for split, payload in development.items():
                selected = _witness_policy(
                    payload["examples"],
                    payload["grouped"],
                    payload["stance"],
                    probability_threshold=probability_threshold,
                    delta_threshold=delta_threshold,
                )
                metrics[split] = base._basic_policy_metrics(
                    payload["examples"], payload["grouped"], selected
                )
            if all(
                result["damage_rate"] <= 0.05
                and min(result["by_label_gain"].values()) >= 0.0
                and result["routes"] >= 10
                for result in metrics.values()
            ):
                feasible.append(
                    (
                        min(result["net_fixes"] for result in metrics.values()),
                        sum(result["net_fixes"] for result in metrics.values()),
                        -max(result["damage_rate"] for result in metrics.values()),
                        -sum(result["routes"] for result in metrics.values()),
                        probability_threshold,
                        delta_threshold,
                        metrics,
                    )
                )
    if not feasible:
        raise ValueError("no split-robust nontrivial CEW policy; prospective test remains locked")
    best = max(feasible, key=lambda row: row[:6])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stance_router.joblib"
    joblib.dump(stance_model, model_path)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_test_calls",
        "selection_sha256": base._sha256_path(selection_path),
        "base_router_manifest_sha256": base._sha256_path(base_manifest_path),
        "base_router_model_sha256": base_manifest["router_joblib_sha256"],
        "development_record_hashes": {
            split: payload["records_sha256"] for split, payload in development.items()
        },
        "grid": {
            "probability": list(WITNESS_PROBABILITY_GRID),
            "delta": list(WITNESS_DELTA_GRID),
        },
        "feasible_parameter_pairs": len(feasible),
        "selected_policy": {
            "probability_threshold": best[4],
            "delta_threshold": best[5],
            "development_metrics": best[6],
        },
        "stance_router_joblib": str(model_path),
        "stance_router_joblib_sha256": base._sha256_path(model_path),
        "feature_boundary": {
            "uses_gold_or_action_outcomes_at_inference": False,
            "uses_source_identity_at_inference": False,
            "uses_initial_consensus_and_evidence_stance": True,
            "added_roots_per_route": 1,
        },
        "claim_boundary": {
            "test_outcomes_seen": False,
            "publisher_independence": False,
            "cross_model": False,
        },
    }
    base._write_json(manifest_path, manifest)
    return manifest


def validate_witness_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("CEW protocol mismatch")
    if manifest.get("status") != "frozen_before_test_calls":
        raise ValueError("CEW router is not frozen before test calls")
    if manifest.get("selection_sha256") != base._sha256_path(selection_path):
        raise ValueError("CEW selection drifted")
    model_path = Path(str(manifest["stance_router_joblib"]))
    if manifest.get("stance_router_joblib_sha256") != base._sha256_path(model_path):
        raise ValueError("CEW stance router drifted")
    boundary = manifest.get("feature_boundary", {})
    if boundary.get("uses_gold_or_action_outcomes_at_inference") is not False:
        raise ValueError("CEW inference boundary permits forbidden outcomes")


def _comparison_proposals(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    proposals = {"retrieval_score": {}, "hash_random": {}}
    for action in base.RECOVERY_ACTIONS:
        proposals[f"fixed_{action}"] = {}
    for example in examples:
        example_id = str(example["example_id"])
        _consensus, agreement, _baseline = base._baseline_state(grouped[example_id])
        active = agreement >= base.HIGH_CONSENSUS
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
                if int(base._hash_key(b"cew-v3-random-action", example_id), 16) % 2 == 0
                else "candidate_1"
            )
            if active
            else "KEEP"
        )
        for action in base.RECOVERY_ACTIONS:
            proposals[f"fixed_{action}"][example_id] = action if active else "KEEP"
    return proposals


def evaluate_witness(
    selection_path: Path,
    records_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_witness_manifest(manifest_path, selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = [row for row in selection["examples"] if row["split"] == "test"]
    records = base._load_jsonl(records_path)
    base._validate_action_matrix(examples, records, split="test")
    grouped = base._record_groups(records)
    stance_model = joblib.load(manifest["stance_router_joblib"])
    stance = base._stance_predictions(stance_model, examples)
    parameters = manifest["selected_policy"]
    cew = _witness_policy(
        examples,
        grouped,
        stance,
        probability_threshold=float(parameters["probability_threshold"]),
        delta_threshold=float(parameters["delta_threshold"]),
    )
    semantic_unlimited = _witness_policy(
        examples,
        grouped,
        stance,
        probability_threshold=0.0,
        delta_threshold=-1.0,
    )
    policies = {
        "cew": cew,
        "semantic_witness_unlimited": semantic_unlimited,
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    root_budget = sum(action != "KEEP" for action in cew.values())
    proposals = _comparison_proposals(examples, grouped)
    for name, proposed in proposals.items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            examples, proposed, root_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    base._write_json(
        preoutcome_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "router_manifest_sha256": base._sha256_path(manifest_path),
            "test_records_sha256": base._sha256_path(records_path),
            "outcomes_accessed_by_route_selection": False,
            "policies": policies,
        },
    )
    oracle = {}
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(example, grouped[example_id])
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
        name: base._policy_metrics(examples, grouped, selected)
        for name, selected in policies.items()
    }
    primary = metrics["cew"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": primary["damage_rate_high_consensus_correct"] <= 0.05,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": (primary["annotation_supported_repairs"] >= 10),
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
    }
    routed = [example for example in examples if cew[str(example["example_id"])] != "KEEP"]
    annotated_choices = sum(
        example["candidates"][int(cew[str(example["example_id"])][-1])]["annotation_role"]
        == "held_out_annotated_root"
        for example in routed
    )
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
    publisher_metrics = {
        name: base._policy_metrics(rows, grouped, cew) for name, rows in publisher_groups.items()
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_averitec_dev_test",
        "selection_sha256": base._sha256_path(selection_path),
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "test_records_sha256": base._sha256_path(records_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_test": len(examples),
        "root_budget": root_budget,
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": "PASS_CEW_V3_3" if all(gates.values()) else "NO_VERIFIED_CEW_DOMINANCE",
        "annotation_role_selection": {
            "routed": len(routed),
            "annotated_root_selected": annotated_choices,
            "accuracy": annotated_choices / max(1, len(routed)),
        },
        "publisher_seen_unseen": publisher_metrics,
        "claim_boundary": {
            "publisher_independence_proven": False,
            "cross_model": False,
            "live_retrieval": False,
        },
    }
    base._write_json(output_dir / "summary.json", summary)
    lines = [
        "# Recovery V3.3 CEW prospective test report",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Test examples: {len(examples)}",
        f"- CEW root budget: {root_budget}",
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
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fit", help="fit and freeze the CEW router")
    test = subparsers.add_parser("test", help="collect the prospective CEW test matrix")
    test.add_argument("--workers", type=int, default=4)
    subparsers.add_parser("evaluate", help="evaluate the prospective CEW test")
    args = parser.parse_args(argv)
    selection_path = BASE_ROOT / "selection_manifest.json"
    manifest_path = DEFAULT_ROOT / "router" / "manifest.json"
    if args.command == "fit":
        manifest = _fit_witness_router(
            selection_path,
            BASE_ROOT / "router" / "manifest.json",
            DEFAULT_ROOT / "router",
        )
        for key, value in manifest.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "test":
        validate_witness_manifest(manifest_path, selection_path)
        base.execute_split(
            selection_path,
            split="test",
            output_dir=DEFAULT_ROOT / "test",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate_witness(
            selection_path,
            DEFAULT_ROOT / "test" / "records.jsonl",
            manifest_path,
            DEFAULT_ROOT / "evaluation",
        )
        for key, value in summary["primary_gates"].items():
            print(f"{key}: {value}")
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
