"""Group-robust dual-head provenance repair on an unseen Hy target model."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as v37
from sp500_forecastability import recovery_v3_10 as v310
from sp500_forecastability.pilot_llm_v1 import (
    MAX_COMPLETION_TOKENS,
    MAX_RESPONSE_BYTES,
    ChatResult,
    _canonical_json,
)

PROTOCOL_VERSION = "recovery-v3.11-group-robust-dual-head-hy-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_11_hy")
SELECTION_PATH = DEFAULT_ROOT / "selection_manifest.json"
ROUTER_DIR = DEFAULT_ROOT / "router"
ROUTER_HEAD = ROUTER_DIR / "relation_head.json"
ROUTER_MANIFEST = ROUTER_DIR / "manifest.json"
ROUTER_INPUTS = DEFAULT_ROOT / "router_inputs.npz"
ROUTER_INPUTS_METADATA = ROUTER_INPUTS.with_suffix(".json")
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/recovery_v3_11_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_11.sh")
EMBED_SCRIPT = Path("scripts/embed_recovery_v3_11_development.py")

SOURCE_SELECTION = Path("results/recovery_v3_7_1/selection_manifest.json")
SOURCE_EMBEDDINGS = Path("results/recovery_v3_11_development/provenance_scores.npz")
SOURCE_EMBEDDING_METADATA = SOURCE_EMBEDDINGS.with_suffix(".json")
SOURCE_ACTIONS = {
    "Qwen3.5-4B": Path("results/recovery_v3_7_1/formal/actions/records.jsonl"),
    "Ling-3.0-tiny": Path("results/recovery_v3_8_3_ling/formal/actions/records.jsonl"),
    "SUFE-Fin-R1": Path("results/recovery_v3_10_finr1/formal/actions/records.jsonl"),
}

TARGET_MODEL = "Hy-MT2-7B"
TARGET_ENDPOINT = "http://127.0.0.1:31519/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/tencent/Hy-MT2-7B-FP8")
TARGET_SMALL_ARTIFACTS = tuple(
    TARGET_MODEL_DIR / name
    for name in (
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
)

SELECTION_SALT = b"grdpr-v3.11-hy-formal-selection-2026-09-03\n"
EXPECTED_FORMAL = 188
EXPECTED_PER_LABEL = 94
TARGET_ROOT_POOL = 600
MAX_AUXILIARY_ROOT_REUSE = 5
SCORE_DIRECTION_QUOTA = EXPECTED_PER_LABEL // 2
BOOTSTRAP_SEED = 20_261_103
BOOTSTRAP_REPLICATES = 10_000
RELATION_CONFIDENCE_MARGIN = 0.10
PROVENANCE_SCORE_MARGIN = 0.30
TARGET_ACTION_CONFIDENCE = 0.80
RELATION_C = 1.0
RELATION_CV_FOLDS = 5
RELATION_CV_SEED = 20_261_103
SMOKE_EXAMPLES = 2


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _source_examples() -> list[dict[str, Any]]:
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    return [dict(row) for row in selection["examples"] if row["split"] == "formal"]


def _source_exposure() -> tuple[set[str], set[str]]:
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    claims = {v34._normalise_claim(row["claim"]) for row in selection["examples"]}
    roots = {
        v37._normalise_root(packet["root"])
        for row in selection["examples"]
        for packet in (row["anchor"], *row["candidates"])
    }
    return claims, roots


@contextmanager
def _selection_configuration() -> Any:
    old = {
        "salt": v37.SELECTION_SALT,
        "n_formal": v37.N_FORMAL,
        "n_per_label": v37.N_PER_LABEL,
        "root_target_pool": v37.ROOT_TARGET_POOL,
    }
    v37.SELECTION_SALT = SELECTION_SALT
    v37.N_FORMAL = EXPECTED_FORMAL
    v37.N_PER_LABEL = EXPECTED_PER_LABEL
    v37.ROOT_TARGET_POOL = TARGET_ROOT_POOL
    try:
        yield
    finally:
        v37.SELECTION_SALT = old["salt"]
        v37.N_FORMAL = old["n_formal"]
        v37.N_PER_LABEL = old["n_per_label"]
        v37.ROOT_TARGET_POOL = old["root_target_pool"]


def _make_fresh_formal_examples(
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Select unique gold roots while boundedly reusing only auxiliary roots."""
    roots = sorted(pool, key=lambda root: v37._hash_key("root-partition", root))
    if len(roots) <= TARGET_ROOT_POOL:
        raise ValueError("not enough fresh roots for V3.11 target and auxiliary pools")
    target_roots = roots[:TARGET_ROOT_POOL]
    auxiliary_roots = roots[TARGET_ROOT_POOL:]
    vectorizer = v37.HashingVectorizer(
        n_features=2**16,
        ngram_range=(1, 2),
        stop_words="english",
        alternate_sign=False,
        norm="l2",
    )
    auxiliary_documents = [v37._root_document(pool[root]) for root in auxiliary_roots]
    auxiliary_matrix = vectorizer.transform(auxiliary_documents)
    search = v37.NearestNeighbors(
        n_neighbors=min(300, len(auxiliary_roots)),
        metric="cosine",
        algorithm="brute",
    ).fit(auxiliary_matrix)
    raw_items: list[tuple[str, str, Mapping[str, Any]]] = []
    for root in target_roots:
        for label in ("SUPPORTS", "REFUTES"):
            candidates = sorted(
                (row for row in pool[root] if row["label"] == label),
                key=lambda row: v37._hash_key("target-item", str(row["id"])),
            )[:3]
            raw_items.extend((root, label, row) for row in candidates)
    claim_matrix = vectorizer.transform([str(item[2]["claim"]) for item in raw_items])
    gold_matrix = vectorizer.transform([" ".join(item[2]["texts"]) for item in raw_items])
    gold_scores = np.asarray(claim_matrix.multiply(gold_matrix).sum(axis=1)).ravel()
    distances, indices = search.kneighbors(claim_matrix)
    selected: list[dict[str, Any]] = []
    used_target_roots: set[str] = set()
    auxiliary_usage: Counter[str] = Counter()
    item_indices_by_label = {
        label: sorted(
            (
                index
                for index, item in enumerate(raw_items)
                if item[1] == label and gold_scores[index] >= v37.MIN_GOLD_RETRIEVAL
            ),
            key=lambda index: v37._hash_key("select-item", str(raw_items[index][2]["id"])),
        )
        for label in ("SUPPORTS", "REFUTES")
    }
    selected_by_label: Counter[str] = Counter()
    phase_specs = (
        ("REFUTES", "above"),
        ("SUPPORTS", "above"),
        ("REFUTES", "below"),
        ("SUPPORTS", "below"),
    )
    for label, direction in phase_specs:
        phase_count = 0
        item_indices = item_indices_by_label[label]
        for item_index in item_indices:
            root, _, row = raw_items[item_index]
            if root in used_target_roots:
                continue
            options = []
            for distance, auxiliary_index in zip(
                distances[item_index], indices[item_index], strict=True
            ):
                auxiliary_root = auxiliary_roots[int(auxiliary_index)]
                score = 1.0 - float(distance)
                gap = abs(score - float(gold_scores[item_index]))
                if (
                    gap <= v37.MAX_RETRIEVAL_GAP
                    and auxiliary_usage[auxiliary_root] < MAX_AUXILIARY_ROOT_REUSE
                ):
                    options.append(
                        (
                            gap,
                            auxiliary_usage[auxiliary_root],
                            v37._hash_key("source-tie", auxiliary_root),
                            auxiliary_root,
                            score,
                        )
                    )
            options.sort()
            if len(options) < 2:
                continue
            directional = [
                option
                for option in options
                if (
                    option[4] >= float(gold_scores[item_index])
                    if direction == "above"
                    else option[4] < float(gold_scores[item_index])
                )
            ]
            if not directional:
                continue
            distractor = directional[0]
            anchor = next((option for option in options if option[3] != distractor[3]), None)
            if anchor is None:
                continue
            if distractor[3] == anchor[3]:
                raise AssertionError("auxiliary packets must have distinct roots")
            used_target_roots.add(root)
            auxiliary_usage.update((distractor[3], anchor[3]))
            annotated = {
                "root": root,
                "annotation_role": "held_out_annotated_root",
                "retrieval_score": float(gold_scores[item_index]),
                "evidence": v37._evidence_rows(row["texts"], "CG"),
            }
            distractor_packet = {
                "root": distractor[3],
                "annotation_role": "unannotated_retrieval_candidate",
                "retrieval_score": distractor[4],
                "evidence": v37._evidence_rows(v37._root_packet_texts(pool[distractor[3]]), "CD"),
            }
            raw_candidates = (
                [annotated, distractor_packet]
                if selected_by_label[label] % 2 == 0
                else [distractor_packet, annotated]
            )
            candidates = []
            for candidate_index, candidate in enumerate(raw_candidates):
                packet = dict(candidate)
                packet["evidence"] = [
                    {
                        "evidence_id": f"C{candidate_index}{evidence_index:02d}",
                        "text": evidence["text"],
                    }
                    for evidence_index, evidence in enumerate(candidate["evidence"])
                ]
                candidates.append(packet)
            selected.append(
                {
                    "example_id": "fever-train-" + str(row["id"]),
                    "source_split": "fever_gold_evidence_train",
                    "source_row_id": str(row["id"]),
                    "split": "formal",
                    "claim": str(row["claim"]),
                    "label": "Supported" if label == "SUPPORTS" else "Refuted",
                    "gold_binary": int(label == "SUPPORTS"),
                    "fact_check_root": "fever",
                    "anchor": {
                        "root": anchor[3],
                        "retrieval_score": anchor[4],
                        "evidence": v37._evidence_rows(
                            v37._root_packet_texts(pool[anchor[3]]), "A"
                        ),
                    },
                    "candidates": candidates,
                }
            )
            selected_by_label[label] += 1
            phase_count += 1
            if phase_count == SCORE_DIRECTION_QUOTA:
                break
        if phase_count != SCORE_DIRECTION_QUOTA:
            raise ValueError(
                f"only selected {phase_count}/{SCORE_DIRECTION_QUOTA} "
                f"{label} examples with {direction}-matched distractors"
            )
    if selected_by_label != {
        "SUPPORTS": EXPECTED_PER_LABEL,
        "REFUTES": EXPECTED_PER_LABEL,
    }:
        raise AssertionError("score-direction phases did not produce balanced labels")
    return sorted(selected, key=lambda row: str(row["example_id"]))


def build_selection() -> dict[str, Any]:
    exposed_claims, exposed_roots = _source_exposure()
    pool = v37.load_fever_root_pool()
    filtered = {
        root: [row for row in rows if v34._normalise_claim(row["claim"]) not in exposed_claims]
        for root, rows in pool.items()
        if v37._normalise_root(root) not in exposed_roots
    }
    filtered = {root: rows for root, rows in filtered.items() if rows}
    with _selection_configuration():
        formal = _make_fresh_formal_examples(filtered)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy_study_root_call",
        "dataset": {
            "path": str(v37.DATASET),
            "sha256": v37.DATASET_SHA256,
        },
        "source_training_selection": {
            "path": str(SOURCE_SELECTION),
            "sha256": base._sha256_path(SOURCE_SELECTION),
        },
        "target_model": TARGET_MODEL,
        "root_definition": "FEVER Wikipedia page title",
        "selection_salt_sha256": sha256(SELECTION_SALT).hexdigest(),
        "selection_boundary": (
            "all claims and all anchor/candidate roots are disjoint from every V3.7.1 "
            "development and formal example used by the V3.11 router"
        ),
        "examples": formal,
    }


def audit_selection(selection: Mapping[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    if rebuild and dict(selection) != build_selection():
        raise ValueError("V3.11 selection or source data drifted")
    rows = list(selection["examples"])
    labels = Counter(str(row["label"]) for row in rows)
    exposed_claims, exposed_roots = _source_exposure()
    target_claims = {v34._normalise_claim(row["claim"]) for row in rows}
    target_roots = [
        v37._normalise_root(packet["root"])
        for row in rows
        for packet in (row["anchor"], *row["candidates"])
    ]
    annotated_roots = [
        v37._normalise_root(candidate["root"])
        for row in rows
        for candidate in row["candidates"]
        if candidate["annotation_role"] == "held_out_annotated_root"
    ]
    auxiliary_roots = [
        v37._normalise_root(packet["root"])
        for row in rows
        for packet in (row["anchor"], *row["candidates"])
        if packet.get("annotation_role") != "held_out_annotated_root"
    ]
    auxiliary_counts = Counter(auxiliary_roots)
    score_direction_counts: Counter[str] = Counter()
    for row in rows:
        annotated = next(
            candidate
            for candidate in row["candidates"]
            if candidate["annotation_role"] == "held_out_annotated_root"
        )
        distractor = next(
            candidate
            for candidate in row["candidates"]
            if candidate["annotation_role"] == "unannotated_retrieval_candidate"
        )
        direction = (
            "distractor_at_least_annotated"
            if float(distractor["retrieval_score"]) >= float(annotated["retrieval_score"])
            else "distractor_below_annotated"
        )
        score_direction_counts[f"{row['label']}:{direction}"] += 1
    roles: list[int] = []
    scores: list[float] = []
    for row in rows:
        for candidate in row["candidates"]:
            roles.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            scores.append(float(candidate["retrieval_score"]))
    auc = float(roc_auc_score(roles, scores))
    position_fraction = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root" for row in rows
    ) / max(1, len(rows))
    gates = {
        "exact_count": len(rows) == EXPECTED_FORMAL,
        "native_labels_balanced": labels
        == {"Supported": EXPECTED_PER_LABEL, "Refuted": EXPECTED_PER_LABEL},
        "zero_source_claim_overlap": not (target_claims & exposed_claims),
        "zero_source_root_overlap": not (set(target_roots) & exposed_roots),
        "all_annotated_roots_unique": len(annotated_roots) == len(set(annotated_roots)),
        "annotated_and_auxiliary_roots_disjoint": not (set(annotated_roots) & set(auxiliary_roots)),
        "auxiliary_root_reuse_at_most_5": max(auxiliary_counts.values(), default=0)
        <= MAX_AUXILIARY_ROOT_REUSE,
        "three_distinct_roots_per_item": all(
            len(
                {
                    row["anchor"]["root"],
                    *(candidate["root"] for candidate in row["candidates"]),
                }
            )
            == 3
            for row in rows
        ),
        "candidate_order_balanced": 0.49 <= position_fraction <= 0.51,
        "score_directions_balanced_within_label": set(score_direction_counts.values())
        == {SCORE_DIRECTION_QUOTA}
        and len(score_direction_counts) == 4,
        "oriented_retrieval_role_auc_at_most_065": max(auc, 1.0 - auc) <= v37.MAX_ORIENTED_ROLE_AUC,
        "retrieval_score_forbidden_from_primary_router": True,
    }
    return {
        "counts": {"formal": len(rows)},
        "labels": dict(labels),
        "distinct_roots": len(set(target_roots)),
        "annotated_distinct_roots": len(set(annotated_roots)),
        "auxiliary_distinct_roots": len(set(auxiliary_roots)),
        "maximum_auxiliary_root_reuse": max(auxiliary_counts.values(), default=0),
        "score_direction_counts": dict(sorted(score_direction_counts.items())),
        "candidate_0_annotated_fraction": position_fraction,
        "retrieval_role_auc": auc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection(path: Path = SELECTION_PATH) -> bool:
    expected = build_selection()
    if not audit_selection(expected, rebuild=False)["passed"]:
        raise ValueError("refusing to write a V3.11 selection that fails its audit")
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("frozen V3.11 selection drifted")
        return False
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot create V3.11 selection after target records exist")
    base._write_json(path, expected)
    return True


def _source_embedding_rows() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.load(SOURCE_EMBEDDINGS)
    required = {
        "example_ids",
        "splits",
        "labels",
        "annotated_indices",
        "scores",
        "relation_vectors",
    }
    if not required.issubset(arrays.files):
        raise ValueError("development embedding artifact lacks required fields")
    mask = arrays["splits"] == "formal"
    ids = [str(value) for value in arrays["example_ids"][mask]]
    labels = np.asarray(
        [int(str(value) == "Supported") for value in arrays["labels"][mask]],
        dtype=int,
    )
    return (
        ids,
        labels,
        arrays["annotated_indices"][mask].astype(int),
        arrays["scores"][mask].astype(np.float64),
        arrays["relation_vectors"][mask].astype(np.float64),
    )


def _new_relation_head() -> LogisticRegression:
    return LogisticRegression(
        C=RELATION_C,
        class_weight="balanced",
        max_iter=3_000,
        random_state=RELATION_CV_SEED,
        solver="lbfgs",
    )


def _head_payload(model: LogisticRegression) -> dict[str, Any]:
    if model.classes_.tolist() != [0, 1] or model.coef_.shape[0] != 1:
        raise ValueError("relation head is not a binary Refuted/Supported model")
    return {
        "format": "plain_logistic_regression_coefficients_v1",
        "negative_class": "Refuted",
        "positive_class": "Supported",
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training": {
            "C": RELATION_C,
            "class_weight": "balanced",
            "max_iter": 3_000,
            "solver": "lbfgs",
            "random_state": RELATION_CV_SEED,
        },
    }


def _probability(head: Mapping[str, Any], vectors: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(head["coefficient"], dtype=np.float64)
    logits = np.asarray(vectors, dtype=np.float64) @ coefficient + float(head["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _safe_action_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    allowed = {
        "example_id",
        "phase",
        "action",
        "agent_index",
        "decision",
        "success",
    }
    return [{key: row[key] for key in allowed if key in row} for row in rows]


def _select_routes(
    example_ids: Sequence[str],
    scores: np.ndarray,
    relation_vectors: np.ndarray,
    action_rows: Sequence[Mapping[str, Any]],
    relation_probabilities: np.ndarray,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    grouped = base._record_groups(_safe_action_rows(action_rows))
    if set(grouped) != set(example_ids):
        raise ValueError("action records and inference features cover different examples")
    selected: dict[str, str] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for row_index, example_id in enumerate(example_ids):
        consensus, agreement, _baseline = base._baseline_state(grouped[example_id])
        candidate_index = int(np.argmax(scores[row_index]))
        action = f"candidate_{candidate_index}"
        probability = float(relation_probabilities[row_index, candidate_index])
        relation_answer = "yes" if probability >= 0.5 else "no"
        path_margin = float(abs(scores[row_index, 0] - scores[row_index, 1]))
        relation_margin = abs(probability - 0.5)
        recovery = next(
            row
            for row in grouped[example_id]
            if row["phase"] == "recovery" and row["action"] == action
        )
        action_answer = str(recovery["decision"]["answer"])
        action_confidence = float(recovery["decision"]["confidence"])
        reasons = {
            "high_consensus": agreement >= base.HIGH_CONSENSUS,
            "provenance_margin": path_margin >= PROVENANCE_SCORE_MARGIN,
            "relation_margin": relation_margin >= RELATION_CONFIDENCE_MARGIN,
            "target_action_confidence": action_confidence >= TARGET_ACTION_CONFIDENCE,
            "changes_consensus": action_answer != consensus,
            "target_relation_agreement": action_answer == relation_answer,
        }
        route = action if all(reasons.values()) else "KEEP"
        selected[example_id] = route
        diagnostics[example_id] = {
            "selected_provenance_path": action,
            "provenance_scores": [float(value) for value in scores[row_index]],
            "provenance_margin": path_margin,
            "p_supported": probability,
            "relation_margin": relation_margin,
            "relation_answer": relation_answer,
            "baseline_consensus": consensus,
            "baseline_agreement": agreement,
            "target_action_answer": action_answer,
            "target_action_confidence": action_confidence,
            "gate_components": reasons,
            "route": route,
        }
    return selected, diagnostics


def _development_metric(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    grouped = base._record_groups(actions)
    gains: list[int] = []
    by_label: dict[str, list[int]] = defaultdict(list)
    fixes = harms = annotation_supported = high_correct = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(example, grouped[example_id])
        action = selected[example_id]
        final = keep if action == "KEEP" else outcomes[action]
        gain = final - keep
        gains.append(gain)
        by_label[str(example["label"])].append(gain)
        fixes += int(gain == 1)
        harms += int(gain == -1)
        high_correct += int(keep == 1 and agreement >= base.HIGH_CONSENSUS)
        if gain == 1:
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
    return {
        "routes": sum(action != "KEEP" for action in selected.values()),
        "fixes": fixes,
        "harms": harms,
        "net_fixes": int(sum(gains)),
        "net_gain": float(np.mean(gains)),
        "macro_label_gain": float(np.mean([np.mean(values) for values in by_label.values()])),
        "by_native_label_net_gain": {
            label: float(np.mean(values)) for label, values in sorted(by_label.items())
        },
        "damage_rate_high_consensus_correct": harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
    }


def _development_diagnostics() -> tuple[dict[str, Any], LogisticRegression]:
    ids, labels, annotated, scores, relation_vectors = _source_embedding_rows()
    examples = _source_examples()
    example_by_id = {str(row["example_id"]): row for row in examples}
    if set(ids) != set(example_by_id) or len(ids) != 400:
        raise ValueError("source embedding IDs do not match the 400 source roots")
    annotated_vectors = relation_vectors[np.arange(len(ids)), annotated]
    oof = np.zeros((len(ids), 2), dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=RELATION_CV_FOLDS,
        shuffle=True,
        random_state=RELATION_CV_SEED,
    )
    for train, held_out in splitter.split(annotated_vectors, labels):
        model = _new_relation_head().fit(annotated_vectors[train], labels[train])
        oof[held_out, 0] = _probability(_head_payload(model), relation_vectors[held_out, 0])
        oof[held_out, 1] = _probability(_head_payload(model), relation_vectors[held_out, 1])
    annotated_oof = oof[np.arange(len(ids)), annotated]
    ordered_examples = [example_by_id[example_id] for example_id in ids]
    model_metrics = {}
    for name, path in SOURCE_ACTIONS.items():
        actions = base._load_jsonl(path)
        selected, _diagnostics = _select_routes(ids, scores, relation_vectors, actions, oof)
        model_metrics[name] = _development_metric(ordered_examples, actions, selected)
    final_head = _new_relation_head().fit(annotated_vectors, labels)
    return (
        {
            "status": "development_only_thresholds_selected_before_hy_study_calls",
            "examples": len(ids),
            "provenance_path_accuracy": float(np.mean(np.argmax(scores, axis=1) == annotated)),
            "relation_head_oof_accuracy_on_annotated_path": float(
                np.mean((annotated_oof >= 0.5) == labels)
            ),
            "relation_head_oof_auroc_on_annotated_path": float(
                roc_auc_score(labels, annotated_oof)
            ),
            "thresholds": {
                "relation_confidence_margin": RELATION_CONFIDENCE_MARGIN,
                "provenance_score_margin": PROVENANCE_SCORE_MARGIN,
                "target_action_confidence": TARGET_ACTION_CONFIDENCE,
            },
            "leave_root_fold_out_target_model_metrics": model_metrics,
        },
        final_head,
    )


def fit_router() -> dict[str, Any]:
    if ROUTER_MANIFEST.exists():
        validate_router_manifest()
        return json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8"))
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot fit V3.11 router after Hy study calls")
    if not PREREGISTRATION.is_file() or not SELECTION_PATH.is_file():
        raise ValueError("preregistration and selection must exist before router fitting")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.11 selection audit failed")
    diagnostics, model = _development_diagnostics()
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    base._write_json(ROUTER_HEAD, _head_payload(model))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy_study_root_call",
        "method": "group_robust_dual_head_provenance_repair",
        "relation_head_path": str(ROUTER_HEAD),
        "relation_head_sha256": base._sha256_path(ROUTER_HEAD),
        "source_embedding_path": str(SOURCE_EMBEDDINGS),
        "source_embedding_sha256": base._sha256_path(SOURCE_EMBEDDINGS),
        "source_embedding_metadata_sha256": base._sha256_path(SOURCE_EMBEDDING_METADATA),
        "source_selection_sha256": base._sha256_path(SOURCE_SELECTION),
        "source_action_sha256": {
            name: base._sha256_path(path) for name, path in SOURCE_ACTIONS.items()
        },
        "training_examples": 400,
        "training_target_models": sorted(SOURCE_ACTIONS),
        "thresholds": diagnostics["thresholds"],
        "development_diagnostics": diagnostics,
        "feature_boundary": {
            "uses_hy_labels_or_action_outcomes_for_fit": False,
            "uses_hy_annotation_role_at_inference": False,
            "uses_retrieval_score_in_primary_router": False,
            "uses_target_action_answer_and_self_reported_confidence": True,
            "target_model_only_executes_its_own_candidate_action": True,
        },
    }
    base._write_json(ROUTER_MANIFEST, manifest)
    return manifest


def validate_router_manifest() -> None:
    manifest = json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V3.11 router protocol drifted")
    if manifest.get("status") != "frozen_before_any_hy_study_root_call":
        raise ValueError("V3.11 router was not frozen before Hy study calls")
    if manifest.get("relation_head_sha256") != base._sha256_path(ROUTER_HEAD):
        raise ValueError("V3.11 relation head drifted")
    if manifest.get("source_embedding_sha256") != base._sha256_path(SOURCE_EMBEDDINGS):
        raise ValueError("V3.11 source embeddings drifted")
    if manifest.get("source_selection_sha256") != base._sha256_path(SOURCE_SELECTION):
        raise ValueError("V3.11 source selection drifted")
    expected_actions = {name: base._sha256_path(path) for name, path in SOURCE_ACTIONS.items()}
    if manifest.get("source_action_sha256") != expected_actions:
        raise ValueError("V3.11 source action records drifted")
    boundary = manifest.get("feature_boundary", {})
    if boundary.get("uses_hy_labels_or_action_outcomes_for_fit") is not False:
        raise ValueError("Hy outcomes entered V3.11 fitting")
    if boundary.get("uses_hy_annotation_role_at_inference") is not False:
        raise ValueError("Hy annotation roles entered V3.11 inference")


def _target_fingerprint() -> dict[str, Any]:
    if any(not path.is_file() for path in TARGET_SMALL_ARTIFACTS):
        raise ValueError("Hy model fingerprint files are incomplete")
    index = json.loads(
        (TARGET_MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(set(index["weight_map"].values()))
    shard_sizes = {name: (TARGET_MODEL_DIR / name).stat().st_size for name in shard_names}
    if any(size <= 0 for size in shard_sizes.values()):
        raise ValueError("Hy weight shard is empty")
    return {
        "small_file_sha256": {
            path.name: base._sha256_path(path) for path in TARGET_SMALL_ARTIFACTS
        },
        "weight_shard_sizes_bytes": shard_sizes,
    }


def _validate_router_inputs() -> None:
    metadata = json.loads(ROUTER_INPUTS_METADATA.read_text(encoding="utf-8"))
    if metadata.get("inference_only") is not True:
        raise ValueError("formal router inputs were not generated in inference-only mode")
    if metadata.get("selection_sha256") != base._sha256_path(SELECTION_PATH):
        raise ValueError("formal router input selection drifted")
    if metadata.get("output_sha256") != base._sha256_path(ROUTER_INPUTS):
        raise ValueError("formal router input artifact drifted")
    arrays = np.load(ROUTER_INPUTS)
    if set(arrays.files) != {
        "example_ids",
        "splits",
        "scores",
        "relation_vectors",
    }:
        raise ValueError("formal router inputs contain forbidden or missing fields")
    if arrays["scores"].shape != (EXPECTED_FORMAL, 2):
        raise ValueError("formal provenance score shape drifted")
    if arrays["relation_vectors"].shape[:2] != (EXPECTED_FORMAL, 2):
        raise ValueError("formal relation vector shape drifted")


def build_protocol_manifest() -> dict[str, Any]:
    validate_router_manifest()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    audit = audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("cannot freeze a structurally invalid V3.11 protocol")
    _validate_router_inputs()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy_study_root_call",
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "implementation_path": str(_implementation_path()),
        "implementation_sha256": base._sha256_path(_implementation_path()),
        "run_script_path": str(RUN_SCRIPT),
        "run_script_sha256": base._sha256_path(RUN_SCRIPT),
        "embedding_script_path": str(EMBED_SCRIPT),
        "embedding_script_sha256": base._sha256_path(EMBED_SCRIPT),
        "selection_path": str(SELECTION_PATH),
        "selection_sha256": base._sha256_path(SELECTION_PATH),
        "selection_audit": audit,
        "router_manifest_path": str(ROUTER_MANIFEST),
        "router_manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
        "router_inputs_path": str(ROUTER_INPUTS),
        "router_inputs_sha256": base._sha256_path(ROUTER_INPUTS),
        "router_inputs_metadata_sha256": base._sha256_path(ROUTER_INPUTS_METADATA),
        "target": {
            "model": TARGET_MODEL,
            "endpoint": TARGET_ENDPOINT,
            "model_dir": str(TARGET_MODEL_DIR),
            "artifact_fingerprint": _target_fingerprint(),
            "temperature": 0.0,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "response_format": v310._response_format("action"),
            "response_format_sha256": sha256(
                _canonical_json(v310._response_format("action")).encode()
            ).hexdigest(),
            "schema_repair_attempts": 1,
        },
        "evaluation": {
            "formal_examples": EXPECTED_FORMAL,
            "per_native_label": EXPECTED_PER_LABEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "preoutcome_routes_required_before_gold_evaluation": True,
            "primary_gates": [
                "macro_gain_ci_lower_above_zero",
                "damage_rate_at_most_005",
                "both_label_groups_nonnegative",
                "annotation_supported_repairs_at_least_10",
                "net_fixes_above_keep_and_all_matched_baselines",
                "provenance_path_accuracy_at_least_090",
            ],
        },
        "claim_boundary": {
            "hy_fit_or_threshold_calibration": False,
            "fresh_claims_and_all_packet_roots": True,
            "source_models_used_only_for_router_development": sorted(SOURCE_ACTIONS),
            "target_answer_is_always_a_hy_generated_action": True,
            "static_wikipedia_page_roots": True,
            "publisher_independence": False,
            "universal_cross_model_transfer": False,
        },
    }


def freeze_protocol() -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or not RUN_SCRIPT.is_file():
        raise ValueError("V3.11 preregistration and run script must exist")
    target_records = list(DEFAULT_ROOT.glob("**/records*.jsonl"))
    if not PROTOCOL_MANIFEST.exists() and target_records:
        raise ValueError("cannot freeze V3.11 after Hy study calls")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        actual = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("frozen V3.11 protocol or dependency drifted")
        return actual
    base._write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol_manifest() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != (build_protocol_manifest()):
        raise ValueError("frozen V3.11 protocol or dependency drifted")


class HySchemaChatClient:
    """Content-addressed Hy client using the frozen action JSON schema."""

    def __init__(
        self,
        endpoint: str = TARGET_ENDPOINT,
        model: str = TARGET_MODEL,
        cache_dir: Path = DEFAULT_ROOT / "cache",
        timeout: float = 90.0,
    ) -> None:
        if endpoint != TARGET_ENDPOINT or model != TARGET_MODEL:
            raise ValueError("V3.11 Hy endpoint and model are frozen")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_completion_tokens = MAX_COMPLETION_TOKENS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def call(self, messages: Sequence[Mapping[str, str]], *, seed: int) -> ChatResult:
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": int(self.max_completion_tokens),
            "seed": seed,
            "response_format": v310._response_format("action"),
        }
        material = {"endpoint": self.endpoint, "request": request_payload}
        cache_key = sha256(_canonical_json(material).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChatResult(
                content=str(cached["content"]),
                model=str(cached["model"]),
                usage=dict(cached["usage"]),
                http_status=int(cached["http_status"]),
                request_bytes=0,
                response_bytes=int(cached["response_bytes"]),
                latency_seconds=0.0,
                cache_hit=True,
                cache_key=cache_key,
            )
        body = _canonical_json(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib_request.Request(self.endpoint, data=body, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except urllib_error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib_error.URLError, TimeoutError) as error:
            raise RuntimeError(f"chat request failed: {error}") from error
        latency = time.monotonic() - started
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("chat response exceeded the one-megabyte safety limit")
        try:
            response_payload = json.loads(response_body)
            content = response_payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("chat endpoint returned an unexpected schema") from error
        if not isinstance(content, str) or not content.strip():
            raise TypeError("chat response content must be nonempty text")
        response_model = str(response_payload.get("model", ""))
        if response_model != self.model:
            raise ValueError(
                f"chat endpoint returned model {response_model!r}, expected {self.model!r}"
            )
        usage_payload = response_payload.get("usage", {})
        result = ChatResult(
            content=content,
            model=response_model,
            usage={
                "prompt_tokens": usage_payload.get("prompt_tokens"),
                "completion_tokens": usage_payload.get("completion_tokens"),
                "total_tokens": usage_payload.get("total_tokens"),
            },
            http_status=status,
            request_bytes=len(body),
            response_bytes=len(response_body),
            latency_seconds=latency,
            cache_hit=False,
            cache_key=cache_key,
        )
        base._write_json(
            cache_path,
            {
                "content": result.content,
                "model": result.model,
                "usage": result.usage,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
                "artifact_kind": "action",
                "response_format_sha256": sha256(
                    _canonical_json(v310._response_format("action")).encode()
                ).hexdigest(),
            },
        )
        return result


def endpoint_model_ids() -> set[str]:
    url = TARGET_ENDPOINT.removesuffix("/chat/completions") + "/models"
    try:
        with urllib_request.urlopen(url, timeout=10.0) as response:
            payload = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Hy model inventory failed: {error}") from error
    ids = {
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, Mapping) and "id" in item
    }
    if TARGET_MODEL not in ids:
        raise ValueError(f"expected {TARGET_MODEL!r} in endpoint inventory: {sorted(ids)}")
    return ids


def _tag_rows(rows: Sequence[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    tagged = []
    for source in rows:
        row = dict(source)
        row["protocol_version"] = PROTOCOL_VERSION
        row["split"] = split
        row["runtime_endpoint"] = TARGET_ENDPOINT
        row["runtime_model"] = TARGET_MODEL
        tagged.append(row)
    return tagged


def _validate_actions(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    outcomes_allowed: bool,
) -> None:
    grouped = base._record_groups(records)
    expected = {str(row["example_id"]): row for row in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} Hy action coverage mismatch")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        if len(baseline) != 5 or len(recovery) != 3:
            raise ValueError(f"invalid Hy action bundle for {example_id}")
        if {row.get("agent_index") for row in baseline} != set(range(5)):
            raise ValueError(f"invalid Hy baseline agents for {example_id}")
        if {row.get("action") for row in recovery} != set(base.RECOVERY_ACTIONS):
            raise ValueError(f"invalid Hy recovery actions for {example_id}")
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("runtime_model") != TARGET_MODEL
            or row.get("split") != split
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"invalid Hy action metadata for {example_id}")
        if outcomes_allowed and any(
            row.get("gold_binary") != expected[example_id]["gold_binary"] for row in rows
        ):
            raise ValueError(f"Hy action outcome metadata drifted for {example_id}")


def execute_actions(
    examples: Sequence[Mapping[str, Any]],
    *,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_example[str(row["example_id"])].append(row)
    if any(len(rows) != 8 for rows in by_example.values()):
        raise ValueError("partial Hy file contains an incomplete action bundle")
    existing = {
        example_id: rows
        for example_id, rows in by_example.items()
        if all(row.get("success") for row in rows)
    }
    allowed = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed:
        raise ValueError("partial Hy file contains examples outside this run")
    records = [row for rows in existing.values() for row in rows]
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = HySchemaChatClient(cache_dir=cache_dir)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(v362._run_action_example, client, example): example
            for example in pending
        }
        for future in as_completed(futures):
            bundle = _tag_rows(future.result(), split=split)
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
                f"[{len(records) // 8}/{len(examples)}] {bundle[0]['example_id']} "
                f"success={all(row['success'] for row in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if len(records) != len(examples) * 8:
        raise ValueError("Hy action run is incomplete")
    _validate_actions(examples, records, split=split, outcomes_allowed=False)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _formal_examples() -> list[dict[str, Any]]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return [dict(row) for row in selection["examples"]]


def _inference_features() -> tuple[list[str], np.ndarray, np.ndarray]:
    _validate_router_inputs()
    arrays = np.load(ROUTER_INPUTS)
    ids = [str(value) for value in arrays["example_ids"]]
    if any(str(value) != "formal" for value in arrays["splits"]):
        raise ValueError("formal router inputs contain a non-formal split")
    return (
        ids,
        arrays["scores"].astype(np.float64),
        arrays["relation_vectors"].astype(np.float64),
    )


def _public_examples() -> list[dict[str, Any]]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return [
        {
            "example_id": str(row["example_id"]),
            "candidates": [
                {"retrieval_score": float(candidate["retrieval_score"])}
                for candidate in row["candidates"]
            ],
        }
        for row in selection["examples"]
    ]


def _preoutcome_payload() -> dict[str, Any]:
    validate_protocol_manifest()
    ids, scores, relation_vectors = _inference_features()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    public_examples = _public_examples()
    _validate_actions(
        public_examples,
        actions,
        split="formal",
        outcomes_allowed=False,
    )
    head = json.loads(ROUTER_HEAD.read_text(encoding="utf-8"))
    relation_probabilities = np.column_stack(
        [_probability(head, relation_vectors[:, candidate]) for candidate in range(2)]
    )
    primary, diagnostics = _select_routes(
        ids, scores, relation_vectors, actions, relation_probabilities
    )
    if set(ids) != {str(row["example_id"]) for row in public_examples}:
        raise ValueError("formal selection and router inputs cover different examples")
    root_budget = sum(action != "KEEP" for action in primary.values())
    action_groups = base._record_groups(_safe_action_rows(actions))
    policies: dict[str, dict[str, str]] = {
        "dual_head_provenance_repair": primary,
        "keep": {example_id: "KEEP" for example_id in ids},
    }
    for name, proposed in v34._comparison_proposals(public_examples, action_groups).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            public_examples, proposed, root_budget, name=name
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "router_manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
        "relation_head_sha256": base._sha256_path(ROUTER_HEAD),
        "router_inputs_sha256": base._sha256_path(ROUTER_INPUTS),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "outcomes_accessed_by_route_selection": False,
        "annotation_roles_accessed_by_route_selection": False,
        "retrieval_scores_accessed_by_primary_router": False,
        "hy_fit_or_calibration": False,
        "root_budget": root_budget,
        "policies": policies,
        "diagnostics": diagnostics,
    }


def freeze_routes() -> dict[str, Any]:
    path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    payload = _preoutcome_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen V3.11 preoutcome routes drifted")
        return payload
    base._write_json(path, payload)
    return payload


def _policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps: list[int] = []
    finals: list[int] = []
    labels: list[str] = []
    high_flags: list[bool] = []
    action_counts: Counter[str] = Counter()
    annotation_supported = 0
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
    by_label: dict[str, dict[str, Any]] = {}
    label_indices: dict[str, np.ndarray] = {}
    for label in ("Supported", "Refuted"):
        indices = np.asarray(
            [index for index, value in enumerate(labels) if value == label], dtype=int
        )
        label_indices[label] = indices
        by_label[label] = {
            "n": len(indices),
            "baseline_accuracy": float(keep_array[indices].mean()),
            "final_accuracy": float(final_array[indices].mean()),
            "net_gain": float(gains[indices].mean()),
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        bootstrap[replicate] = float(
            np.mean(
                [
                    gains[rng.choice(indices, size=len(indices), replace=True)].mean()
                    for indices in label_indices.values()
                ]
            )
        )
    high_correct = sum(bool(keep and high) for keep, high in zip(keeps, high_flags, strict=True))
    high_harms = sum(
        bool(keep and high and not final)
        for keep, high, final in zip(keeps, high_flags, finals, strict=True)
    )
    fixes = (keep_array == 0) & (final_array == 1)
    harms = (keep_array == 1) & (final_array == 0)
    return {
        "n": len(examples),
        "baseline_accuracy": float(keep_array.mean()),
        "final_accuracy": float(final_array.mean()),
        "fixes": int(fixes.sum()),
        "harms": int(harms.sum()),
        "net_fixes": int(gains.sum()),
        "net_gain": float(gains.mean()),
        "macro_label_gain": float(np.mean([group["net_gain"] for group in by_label.values()])),
        "macro_gain_ci": np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist(),
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": sum(action != "KEEP" for action in selected.values()),
        "selected_actions": dict(action_counts),
        "by_native_label": by_label,
    }


def evaluate() -> dict[str, Any]:
    validate_protocol_manifest()
    preoutcome_path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    if not preoutcome_path.is_file():
        raise ValueError("freeze preoutcome routes before accessing formal outcomes")
    preoutcome = json.loads(preoutcome_path.read_text(encoding="utf-8"))
    if preoutcome != _preoutcome_payload():
        raise ValueError("preoutcome route artifact drifted")
    examples = _formal_examples()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    _validate_actions(examples, actions, split="formal", outcomes_allowed=True)
    grouped = base._record_groups(actions)
    policies = dict(preoutcome["policies"])
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
        name: _policy_metrics(examples, grouped, selected) for name, selected in policies.items()
    }
    primary_name = "dual_head_provenance_repair"
    primary = metrics[primary_name]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    diagnostics = preoutcome["diagnostics"]
    annotated_by_id = {
        str(row["example_id"]): next(
            index
            for index, candidate in enumerate(row["candidates"])
            if candidate["annotation_role"] == "held_out_annotated_root"
        )
        for row in examples
    }
    path_correct = {
        example_id: int(value["selected_provenance_path"][-1]) == annotated_by_id[example_id]
        for example_id, value in diagnostics.items()
    }
    routed_ids = [
        example_id
        for example_id, action in preoutcome["policies"][primary_name].items()
        if action != "KEEP"
    ]
    provenance = {
        "all_examples_accuracy": sum(path_correct.values()) / len(path_correct),
        "routed_examples_accuracy": sum(path_correct[key] for key in routed_ids)
        / max(1, len(routed_ids)),
        "all_examples_correct": sum(path_correct.values()),
        "routed_examples_correct": sum(path_correct[key] for key in routed_ids),
        "routed_examples": len(routed_ids),
    }
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": primary["damage_rate_high_consensus_correct"] <= 0.005,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": primary["annotation_supported_repairs"] >= 10,
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
        "provenance_path_accuracy_at_least_090": provenance["all_examples_accuracy"] >= 0.90,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "one_shot_unseen_hy_fresh_root_formal",
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "router_manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_formal": len(examples),
        "root_budget": preoutcome["root_budget"],
        "transport_and_schema": {
            "rows": len(actions),
            "successful": sum(bool(row.get("success")) for row in actions),
            "first_pass_valid": sum(bool(row.get("first_pass_valid")) for row in actions),
            "cache_hits": sum(
                bool(attempt.get("cache_hit"))
                for row in actions
                for attempt in row.get("attempts", [])
            ),
        },
        "policies": metrics,
        "provenance_head": provenance,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": (
            "PASS_UNSEEN_MODEL_DUAL_HEAD_PROVENANCE_REPAIR_V3_11"
            if all(gates.values())
            else "NO_VERIFIED_UNSEEN_MODEL_DUAL_HEAD_TRANSFER_V3_11"
        ),
        "claim_boundary": build_protocol_manifest()["claim_boundary"],
    }
    output_dir = DEFAULT_ROOT / "evaluation"
    base._write_json(output_dir / "summary.json", summary)
    _write_report(summary, output_dir / "report.md")
    return summary


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    primary = summary["policies"]["dual_head_provenance_repair"]
    keep = summary["policies"]["keep"]
    lines = [
        "# Recovery V3.11 result: unseen-model dual-head provenance repair",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "| metric | KEEP | V3.11 router |",
        "| --- | ---: | ---: |",
        f"| accuracy | {keep['final_accuracy']:.2%} | {primary['final_accuracy']:.2%} |",
        f"| macro gain | 0.00pp | {100 * primary['macro_label_gain']:+.2f}pp |",
        f"| fixes / harms | 0 / 0 | {primary['fixes']} / {primary['harms']} |",
        f"| routed roots | 0 | {summary['root_budget']} |",
        "",
        (
            "Macro-gain 95% CI: "
            f"[{100 * primary['macro_gain_ci'][0]:+.2f}, "
            f"{100 * primary['macro_gain_ci'][1]:+.2f}]pp."
        ),
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["primary_gates"].items()
    )
    lines.extend(
        [
            "",
            (
                "The relation head, thresholds, fresh roots, embedding model, prompts, "
                "seeds, and gates were frozen before Hy study-root calls. Pre-outcome "
                "routes were content-addressed before gold labels or annotation roles "
                "were used for evaluation."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def smoke(workers: int) -> None:
    validate_protocol_manifest()
    endpoint_model_ids()
    source_selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    examples = [dict(row) for row in source_selection["examples"] if row["split"] == "development"][
        :SMOKE_EXAMPLES
    ]
    execute_actions(
        examples,
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "actions",
        cache_dir=DEFAULT_ROOT / "cache",
        workers=workers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-selection")
    subparsers.add_parser("fit-router")
    subparsers.add_parser("freeze-protocol")
    subparsers.add_parser("endpoint-check")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--workers", type=int, default=2)
    formal_parser = subparsers.add_parser("formal-actions")
    formal_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("freeze-routes")
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "prepare-selection":
        created = write_or_validate_selection()
        print(json.dumps({"created": created, **audit_selection(build_selection())}, indent=2))
        return 0
    if args.command == "fit-router":
        print(json.dumps(fit_router(), indent=2))
        return 0
    if args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
        return 0
    if args.command == "endpoint-check":
        validate_protocol_manifest()
        print(json.dumps(sorted(endpoint_model_ids())))
        return 0
    if args.command == "smoke":
        smoke(args.workers)
        return 0
    validate_protocol_manifest()
    if args.command == "formal-actions":
        execute_actions(
            _formal_examples(),
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / "actions",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "freeze-routes":
        payload = freeze_routes()
        print(json.dumps({"root_budget": payload["root_budget"]}, indent=2))
        return 0
    if args.command == "evaluate":
        summary = evaluate()
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
