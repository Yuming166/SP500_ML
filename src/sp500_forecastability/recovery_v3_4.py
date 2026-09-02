"""PBWJ: publisher-blocked witness jury for external consensus repair."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability.recovery_v2 import PERSONAS, RecoveryChatClient, _call_with_retry

PROTOCOL_VERSION = "recovery-v3.4.1-pbwj-climate-fever-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_4_1")
DATASET = Path("data/climate_fever/climate-fever.jsonl")
DATASET_SHA256 = "8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b"
SOURCE_SELECTION = Path("results/recovery_v3_2/selection_manifest.json")
PREREGISTRATION = Path("docs/recovery_v3_4_1_preregistration.md")
FEVER_REFERENCE = Path("data/fever/fever-validation.jsonl")
EXPECTED_N = 483
EXPECTED_LABELS = {"Supported": 361, "Refuted": 122}
ENSEMBLE_SIZE = 5
OPPOSITION_THRESHOLD = 0.4
DELTA_THRESHOLD = -0.2
DISPERSION_MULTIPLIER = 1.0
QUORUM = {"no": 0.8, "yes": 1.0}
PUBLISHER_FOLD_SALT = b"pbwj-v3.4-publisher-fold-2026-09-02"
BOOTSTRAP_SEED = 20_260_943
BOOTSTRAP_REPLICATES = 10_000
CALL_SEEDS = {
    "baseline": 20_260_951,
    "candidate_0": 20_260_961,
    "candidate_1": 20_260_971,
    "both": 20_260_981,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return base._load_jsonl(path)


def _normalise_claim(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def load_eligible(path: Path = DATASET) -> list[dict[str, Any]]:
    if base._sha256_path(path) != DATASET_SHA256:
        raise ValueError("CLIMATE-FEVER dataset checksum drifted")
    label_map = {"SUPPORTS": "Supported", "REFUTES": "Refuted"}
    examples = []
    for row_index, row in enumerate(_load_jsonl(path)):
        raw_label = str(row.get("claim_label", ""))
        if raw_label not in label_map:
            continue
        grouped: dict[str, list[str]] = defaultdict(list)
        for evidence in row.get("evidences", []):
            if evidence.get("evidence_label") != raw_label:
                continue
            root = " ".join(str(evidence.get("article", "")).split())
            text = " ".join(str(evidence.get("evidence", "")).split())
            if root and text and text not in grouped[root]:
                grouped[root].append(text)
        if len(grouped) < 2:
            continue
        claim = " ".join(str(row.get("claim", "")).split())
        if not claim:
            continue
        claim_id = str(row.get("claim_id", row_index))
        example_id = sha256(f"climate-fever\0{claim_id}\0{claim}".encode()).hexdigest()[:32]
        label = label_map[raw_label]
        examples.append(
            {
                "example_id": example_id,
                "source_split": "climate_fever_public",
                "source_row_index": row_index,
                "source_claim_id": claim_id,
                "claim": claim,
                "label": label,
                "gold_binary": int(label == "Supported"),
                "fact_check_root": "climate-fever",
                "annotated_evidence": dict(sorted(grouped.items())),
            }
        )
    return examples


def build_selection(path: Path = DATASET) -> dict[str, Any]:
    examples = base._prepare_partition(load_eligible(path), "external_test")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "selection_frozen_before_climate_fever_qwen_calls",
        "dataset": {
            "name": "CLIMATE-FEVER",
            "path": str(path),
            "sha256": DATASET_SHA256,
            "license_source": "https://github.com/tdiggelm/climate-fever-dataset",
        },
        "model": "Qwen3.5-4B",
        "endpoint": base.RELOCATED_RUNTIME_ENDPOINT,
        "root_definition": "English-Wikipedia article title",
        "eligibility": (
            "claim_label in SUPPORTS/REFUTES and at least two distinct article roots with "
            "evidence_label equal to claim_label"
        ),
        "expected_calls_per_example": len(PERSONAS) + len(base.RECOVERY_ACTIONS),
        "examples": examples,
    }


def validate_selection(selection: Mapping[str, Any]) -> None:
    if dict(selection) != build_selection():
        raise ValueError("Recovery V3.4 selection or source data drifted")


def _reference_claims() -> tuple[set[str], set[str]]:
    source = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    base.validate_selection(source)
    averitec = {_normalise_claim(row["claim"]) for row in source["examples"]}
    fever = {_normalise_claim(row["claim"]) for row in _load_jsonl(FEVER_REFERENCE)}
    return averitec, fever


def audit_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    validate_selection(selection)
    examples = list(selection["examples"])
    labels = Counter(str(row["label"]) for row in examples)
    claims = [_normalise_claim(row["claim"]) for row in examples]
    averitec_claims, fever_claims = _reference_claims()
    distinct_roots = all(
        len({row["anchor"]["root"], *(item["root"] for item in row["candidates"])}) == 3
        for row in examples
    )
    candidate_zero_fraction = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root"
        for row in examples
    ) / len(examples)
    roles = []
    scores = []
    for row in examples:
        for candidate in row["candidates"]:
            roles.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            scores.append(float(candidate["retrieval_score"]))
    role_auc = float(roc_auc_score(roles, scores))
    oriented_role_auc = max(role_auc, 1.0 - role_auc)
    page_roots = {
        packet["root"]
        for row in examples
        for packet in (row["anchor"], *row["candidates"])
    }
    gates = {
        "exact_expected_count": len(examples) == EXPECTED_N,
        "exact_expected_labels": dict(labels) == EXPECTED_LABELS,
        "unique_claims": len(claims) == len(set(claims)),
        "zero_averitec_claim_overlap": not (set(claims) & averitec_claims),
        "zero_fever_claim_overlap": not (set(claims) & fever_claims),
        "three_distinct_page_roots": distinct_roots,
        "candidate_order_balanced": 0.49 <= candidate_zero_fraction <= 0.51,
        "oriented_retrieval_role_auc_at_most_085": oriented_role_auc <= 0.85,
        "each_label_at_least_100": min(labels.values()) >= 100,
        "test_at_least_450": len(examples) >= 450,
        "distinct_page_roots_at_least_350": len(page_roots) >= 350,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "n": len(examples),
        "labels": dict(labels),
        "candidate_0_annotated_fraction": candidate_zero_fraction,
        "retrieval_role_auc": role_auc,
        "oriented_retrieval_role_auc": oriented_role_auc,
        "distinct_page_roots": len(page_roots),
        "claim_overlap": {
            "averitec": len(set(claims) & averitec_claims),
            "fever": len(set(claims) & fever_claims),
        },
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


def _publisher_folds(examples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        publisher = str(example.get("fact_check_root") or f"missing:{example['example_id']}")
        groups[publisher].append(example)
    fold_sizes = [0] * ENSEMBLE_SIZE
    assignment: dict[str, int] = {}
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), base._hash_key(PUBLISHER_FOLD_SALT, item[0])),
    )
    for publisher, rows in ordered:
        fold = min(range(ENSEMBLE_SIZE), key=lambda index: (fold_sizes[index], index))
        fold_sizes[fold] += len(rows)
        assignment.update((str(row["example_id"]), fold) for row in rows)
    return assignment


def fit_jury(selection_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_jury_manifest(manifest_path, selection_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if (DEFAULT_ROOT / "test" / "records.jsonl").exists():
        raise ValueError("cannot fit or overwrite PBWJ after external-test records exist")
    if not PREREGISTRATION.exists():
        raise ValueError("V3.4 preregistration must exist before fitting")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.4 external selection failed structural gates")
    source_selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    base.validate_selection(source_selection)
    source_examples = list(source_selection["examples"])
    assignments = _publisher_folds(source_examples)
    models = []
    fold_metadata = []
    for heldout_fold in range(ENSEMBLE_SIZE):
        train_examples = [
            row for row in source_examples if assignments[str(row["example_id"])] != heldout_fold
        ]
        packets = base._stance_packets(train_examples)
        model = base._new_stance_model()
        model.fit([row[2] for row in packets], [row[3] for row in packets])
        models.append(model)
        heldout = [
            row for row in source_examples if assignments[str(row["example_id"])] == heldout_fold
        ]
        fold_metadata.append(
            {
                "fold": heldout_fold,
                "heldout_examples": len(heldout),
                "heldout_publishers": len(
                    {str(row.get("fact_check_root") or "") for row in heldout}
                ),
                "heldout_labels": dict(Counter(str(row["label"]) for row in heldout)),
                "training_examples": len(train_examples),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stance_jury.joblib"
    joblib.dump({"stance_models": models}, model_path)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_climate_fever_test_calls",
        "selection_sha256": base._sha256_path(selection_path),
        "source_selection_sha256": base._sha256_path(SOURCE_SELECTION),
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "jury_model": {
            "family": "publisher-blocked-five-member-word-char-tfidf-logistic",
            "source_examples": len(source_examples),
            "folds": fold_metadata,
            "hard_negatives": "one retrieval-matched irrelevant candidate per source claim",
        },
        "policy": {
            "opposition_threshold": OPPOSITION_THRESHOLD,
            "delta_threshold": DELTA_THRESHOLD,
            "dispersion_multiplier": DISPERSION_MULTIPLIER,
            "quorum_by_initial_consensus": QUORUM,
            "single_root_only": True,
        },
        "stance_jury_joblib": str(model_path),
        "stance_jury_joblib_sha256": base._sha256_path(model_path),
        "feature_boundary": {
            "uses_external_gold_or_action_outcomes_at_inference": False,
            "uses_external_annotation_role_at_inference": False,
            "uses_source_identity_at_inference": False,
            "uses_initial_consensus_and_packet_text": True,
        },
        "claim_boundary": {
            "external_test_outcomes_seen": False,
            "cross_dataset": True,
            "publisher_independence": False,
            "page_root_transfer": True,
            "cross_model": False,
        },
    }
    base._write_json(manifest_path, manifest)
    return manifest


def validate_jury_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("PBWJ protocol mismatch")
    if manifest.get("status") != "frozen_before_climate_fever_test_calls":
        raise ValueError("PBWJ is not frozen before external-test calls")
    if manifest.get("selection_sha256") != base._sha256_path(selection_path):
        raise ValueError("PBWJ external selection drifted")
    if manifest.get("source_selection_sha256") != base._sha256_path(SOURCE_SELECTION):
        raise ValueError("PBWJ source selection drifted")
    if manifest.get("preregistration_sha256") != base._sha256_path(PREREGISTRATION):
        raise ValueError("PBWJ preregistration drifted")
    model_path = Path(str(manifest["stance_jury_joblib"]))
    if manifest.get("stance_jury_joblib_sha256") != base._sha256_path(model_path):
        raise ValueError("PBWJ stance jury drifted")
    boundary = manifest.get("feature_boundary", {})
    if boundary.get("uses_external_gold_or_action_outcomes_at_inference") is not False:
        raise ValueError("PBWJ inference boundary permits forbidden outcomes")
    if boundary.get("uses_external_annotation_role_at_inference") is not False:
        raise ValueError("PBWJ inference boundary permits annotation roles")


def _inference_packets(
    examples: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str]]:
    packets = []
    for example in examples:
        example_id = str(example["example_id"])
        packets.append((example_id, "anchor", base._stance_text(example, example["anchor"])))
        for index, candidate in enumerate(example["candidates"]):
            packets.append(
                (example_id, f"candidate_{index}", base._stance_text(example, candidate))
            )
    return packets


def _jury_predictions(
    models: Sequence[Any], examples: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], np.ndarray]:
    packets = _inference_packets(examples)
    texts = [row[2] for row in packets]
    member_probabilities = [base._standard_stance_probabilities(model, texts) for model in models]
    stacked = np.stack(member_probabilities, axis=1)
    return {
        (row[0], row[1]): stacked[index]
        for index, row in enumerate(packets)
    }


def _pessimistic(values: np.ndarray) -> float:
    dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()) - DISPERSION_MULTIPLIER * dispersion


def _jury_policy(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    stance: Mapping[tuple[str, str], np.ndarray],
    *,
    use_dispersion: bool = True,
    use_quorum: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    selected = {}
    diagnostics = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(grouped[example_id])
        opposition = "supports" if consensus == "no" else "refutes"
        opposition_index = base.STANCE_CLASSES.index(opposition)
        anchor_scores = stance[(example_id, "anchor")][:, opposition_index]
        candidate_scores = np.stack(
            [stance[(example_id, f"candidate_{index}")][:, opposition_index] for index in (0, 1)],
            axis=1,
        )
        member_choices = np.argmax(candidate_scores, axis=1)
        counts = Counter(int(value) for value in member_choices)
        winner = max(
            (0, 1),
            key=lambda index: (
                counts[index],
                base._hash_key(b"pbwj-v3.4-candidate-tie", f"{example_id}\0{index}"),
            ),
        )
        jury_agreement = counts[winner] / ENSEMBLE_SIZE
        winner_scores = candidate_scores[:, winner]
        deltas = winner_scores - anchor_scores
        score = _pessimistic(winner_scores) if use_dispersion else float(winner_scores.mean())
        delta = _pessimistic(deltas) if use_dispersion else float(deltas.mean())
        required_quorum = QUORUM[consensus] if use_quorum else 0.0
        reasons = []
        if agreement < base.HIGH_CONSENSUS:
            reasons.append("low_initial_consensus")
        if jury_agreement < required_quorum:
            reasons.append("jury_disagreement")
        if score < OPPOSITION_THRESHOLD:
            reasons.append("weak_counter_consensus_witness")
        if delta < DELTA_THRESHOLD:
            reasons.append("weak_anchor_contrast")
        action = f"candidate_{winner}" if not reasons else "KEEP"
        selected[example_id] = action
        diagnostics[example_id] = {
            "initial_consensus": consensus,
            "initial_agreement": agreement,
            "opposition_stance": opposition,
            "member_choices": member_choices.tolist(),
            "winner": f"candidate_{winner}",
            "jury_agreement": jury_agreement,
            "required_quorum": required_quorum,
            "pessimistic_opposition": score,
            "pessimistic_anchor_delta": delta,
            "decision": action,
            "veto_reasons": reasons,
        }
    return selected, diagnostics


def _run_example(client: RecoveryChatClient, example: Mapping[str, Any]) -> list[dict[str, Any]]:
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
                "split": "external_test",
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
                "split": "external_test",
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


def _validate_action_matrix(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> None:
    grouped = base._record_groups(records)
    expected = {str(example["example_id"]): example for example in examples}
    if set(grouped) != set(expected):
        raise ValueError("external records do not cover the frozen V3.4 selection")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        if len(rows) != len(PERSONAS) + len(base.RECOVERY_ACTIONS):
            raise ValueError(f"external example {example_id} has invalid bundle size")
        if {row.get("agent_index") for row in baseline} != set(range(len(PERSONAS))):
            raise ValueError(f"external example {example_id} has invalid baseline agents")
        if {row.get("action") for row in recovery} != set(base.RECOVERY_ACTIONS):
            raise ValueError(f"external example {example_id} has invalid actions")
        example = expected[example_id]
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("split") != "external_test"
            or row.get("gold_binary") != example["gold_binary"]
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"external example {example_id} has invalid metadata")


def execute_test(
    selection_path: Path,
    manifest_path: Path,
    *,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.4 selection failed pre-call gates")
    validate_jury_manifest(manifest_path, selection_path)
    examples = list(selection["examples"])
    if smoke:
        examples = examples[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = _load_jsonl(partial_path) if partial_path.exists() else []
    expected_per_example = len(PERSONAS) + len(base.RECOVERY_ACTIONS)
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
    client = RecoveryChatClient(base.RELOCATED_RUNTIME_ENDPOINT, "Qwen3.5-4B", cache_dir)
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
            base._write_jsonl(partial_path, records)
            print(
                f"[{len(records) // expected_per_example}/{len(examples)}] "
                f"{bundle[0]['example_id']} success={all(row['success'] for row in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    expected_rows = len(examples) * expected_per_example
    if len(records) != expected_rows or any(not row["success"] for row in records):
        raise ValueError(f"external run incomplete or invalid: {len(records)}/{expected_rows}")
    _validate_action_matrix(examples, records)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


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
        keep, agreement, outcomes, _baseline = base._outcomes(example, grouped[example_id])
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        keeps.append(keep)
        finals.append(final)
        labels.append(str(example["label"]))
        high_flags.append(agreement >= base.HIGH_CONSENSUS)
        action_counts[action] += 1
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
    macro_interval = np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist()
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
        "macro_gain_ci": macro_interval,
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": roots_added,
        "mean_added_roots": roots_added / len(examples),
        "selected_actions": dict(action_counts),
        "by_native_label": by_label,
    }


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
                if int(base._hash_key(b"pbwj-v3.4-random-action", example_id), 16) % 2 == 0
                else "candidate_1"
            )
            if active
            else "KEEP"
        )
        for action in base.RECOVERY_ACTIONS:
            proposals[f"fixed_{action}"][example_id] = action if active else "KEEP"
    return proposals


def _write_or_validate_preoutcome(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError("frozen V3.4 preoutcome routes drifted")
        return
    base._write_json(path, payload)


def evaluate(
    selection_path: Path,
    records_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.4 selection failed evaluation audit")
    validate_jury_manifest(manifest_path, selection_path)
    examples = list(selection["examples"])
    records = _load_jsonl(records_path)
    _validate_action_matrix(examples, records)
    grouped = base._record_groups(records)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = joblib.load(manifest["stance_jury_joblib"])
    stance = _jury_predictions(bundle["stance_models"], examples)
    pbwj, diagnostics = _jury_policy(examples, grouped, stance)
    no_dispersion, _ = _jury_policy(
        examples, grouped, stance, use_dispersion=False, use_quorum=True
    )
    no_veto, _ = _jury_policy(
        examples, grouped, stance, use_dispersion=False, use_quorum=False
    )
    policies = {
        "pbwj": pbwj,
        "jury_without_dispersion_veto": no_dispersion,
        "mean_witness_without_uncertainty_veto": no_veto,
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    root_budget = sum(action != "KEEP" for action in pbwj.values())
    for name, proposed in _comparison_proposals(examples, grouped).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            examples, proposed, root_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    preoutcome = {
        "protocol_version": PROTOCOL_VERSION,
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "test_records_sha256": base._sha256_path(records_path),
        "outcomes_accessed_by_route_selection": False,
        "policies": policies,
        "jury_diagnostics": diagnostics,
    }
    _write_or_validate_preoutcome(preoutcome_path, preoutcome)
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
        name: _policy_metrics(examples, grouped, selected)
        for name, selected in policies.items()
    }
    primary = metrics["pbwj"]
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
    routed = [row for row in examples if pbwj[str(row["example_id"])] != "KEEP"]
    annotated_choices = sum(
        row["candidates"][int(pbwj[str(row["example_id"])][-1])]["annotation_role"]
        == "held_out_annotated_root"
        for row in routed
    )
    veto_counts = Counter(
        reason for item in diagnostics.values() for reason in item["veto_reasons"]
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_cross_dataset_climate_fever_test",
        "selection_sha256": base._sha256_path(selection_path),
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "test_records_sha256": base._sha256_path(records_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_test": len(examples),
        "root_budget": root_budget,
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": "PASS_PBWJ_V3_4" if all(gates.values()) else "NO_VERIFIED_PBWJ_DOMINANCE",
        "annotation_role_selection": {
            "routed": len(routed),
            "annotated_root_selected": annotated_choices,
            "accuracy": annotated_choices / max(1, len(routed)),
        },
        "uncertainty_veto": {
            "veto_reason_counts": dict(veto_counts),
            "mean_jury_agreement_routed": float(
                np.mean([diagnostics[str(row["example_id"])]["jury_agreement"] for row in routed])
            )
            if routed
            else None,
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    base._write_json(output_dir / "summary.json", summary)
    lines = [
        "# Recovery V3.4 PBWJ external test report",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- External examples: {len(examples)}",
        f"- PBWJ root budget: {root_budget}",
        "",
        "## Policies",
        "",
    ]
    for name, result in metrics.items():
        lines.append(
            f"- {name}: accuracy={result['final_accuracy']:.3f}, "
            f"macro_gain={result['macro_label_gain']:.3f}, "
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
    subparsers.add_parser("prepare", help="freeze and audit the external selection")
    subparsers.add_parser("fit", help="fit and freeze the publisher-blocked jury")
    smoke = subparsers.add_parser("smoke", help="run two examples without touching formal test")
    smoke.add_argument("--workers", type=int, default=2)
    test = subparsers.add_parser("test", help="collect the frozen external action matrix")
    test.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("evaluate", help="evaluate frozen PBWJ routes")
    args = parser.parse_args(argv)
    selection_path = DEFAULT_ROOT / "selection_manifest.json"
    audit_path = DEFAULT_ROOT / "selection_audit.json"
    manifest_path = DEFAULT_ROOT / "router" / "manifest.json"
    if args.command == "prepare":
        write_or_validate_selection(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        audit = audit_selection(selection)
        base._write_json(audit_path, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["passed"] else 2
    if args.command == "fit":
        manifest = fit_jury(selection_path, DEFAULT_ROOT / "router")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command in {"smoke", "test"}:
        is_smoke = args.command == "smoke"
        execute_test(
            selection_path,
            manifest_path,
            output_dir=DEFAULT_ROOT / ("smoke" if is_smoke else "test"),
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
            smoke=is_smoke,
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate(
            selection_path,
            DEFAULT_ROOT / "test" / "records.jsonl",
            manifest_path,
            DEFAULT_ROOT / "evaluation",
        )
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
