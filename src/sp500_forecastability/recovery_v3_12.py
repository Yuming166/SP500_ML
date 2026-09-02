"""Selective Qwen co-sign routing on an unseen Hy-1.8B target."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as v37
from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability.recovery_v2 import RecoveryChatClient

PROTOCOL_VERSION = "recovery-v3.12-selective-cosign-hy18-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_12_hy18")
DEVELOPMENT_ROOT = Path("results/recovery_v3_12_development")
SELECTION_PATH = DEFAULT_ROOT / "selection_manifest.json"
ROUTER_INPUTS = DEFAULT_ROOT / "router_inputs.npz"
ROUTER_INPUTS_METADATA = ROUTER_INPUTS.with_suffix(".json")
ROUTER_MANIFEST = DEFAULT_ROOT / "router" / "manifest.json"
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/recovery_v3_12_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_12.sh")
SERVER_SCRIPT = Path("scripts/start_hy18_v3_12.sh")
EMBED_SCRIPT = Path("scripts/embed_recovery_v3_11_development.py")
DATASET = Path("data/fever/fever-validation.jsonl")
DATASET_SHA256 = "5da0ccc0ccf77f974611de13f8aac6f78c6bba6293912835099eb6029baa85d9"
V311_SELECTION = v311.SELECTION_PATH
V311_ACTIONS = v311.DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
V311_SUMMARY = v311.DEFAULT_ROOT / "evaluation" / "summary.json"
V311_ROUTER = v311.ROUTER_MANIFEST
V311_HEAD = v311.ROUTER_HEAD
DEVELOPMENT_TEACHER = DEVELOPMENT_ROOT / "hy7_qwen_cosign_records.jsonl"

TARGET_MODEL = "Hy-MT2-1.8B"
TARGET_ENDPOINT = "http://127.0.0.1:31520/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/tencent/Hy-MT2-1.8B-FP8")
TARGET_SMALL_ARTIFACTS = tuple(
    TARGET_MODEL_DIR / name
    for name in (
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
)
TEACHER_MODEL = "Qwen3.5-4B"
TEACHER_ENDPOINT = "http://10.63.0.82:31518/v1/chat/completions"
TEACHER_CONFIDENCE = 0.80
RELATION_CONFIDENCE_MARGIN = 0.13
PROVENANCE_SCORE_MARGIN = 0.30
TARGET_ACTION_CONFIDENCE = 0.80

EXPECTED_FORMAL = 112
EXPECTED_PER_LABEL = 56
TARGET_ROOT_POOL = 350
MAX_AUXILIARY_ROOT_REUSE = 7
SCORE_DIRECTION_QUOTA = 28
SELECTION_SALT = b"selective-cosign-v3.12-validation-selection-2026-09-03\n"
BOOTSTRAP_SEED = 20_261_104
BOOTSTRAP_REPLICATES = 10_000
SMOKE_EXAMPLES = 2


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _all_exposure() -> tuple[set[str], set[str]]:
    claims: set[str] = set()
    roots: set[str] = set()
    for path in (v311.SOURCE_SELECTION, V311_SELECTION):
        selection = json.loads(path.read_text(encoding="utf-8"))
        for row in selection["examples"]:
            claims.add(v34._normalise_claim(row["claim"]))
            for packet in (row["anchor"], *row["candidates"]):
                roots.add(v37._normalise_root(packet["root"]))
    return claims, roots


def load_validation_pool() -> dict[str, list[dict[str, Any]]]:
    if base._sha256_path(DATASET) != DATASET_SHA256:
        raise ValueError("FEVER validation checksum drifted")
    exposed_claims, exposed_roots = _all_exposure()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base._load_jsonl(DATASET):
        if row.get("verifiable") != "VERIFIABLE" or row.get("label") not in {
            "SUPPORTS",
            "REFUTES",
        }:
            continue
        evidence_roots = {
            str(raw[0]).strip()
            for raw in row.get("evidence", [])
            if isinstance(raw, list)
            and len(raw) >= 3
            and str(raw[0]).strip()
            and str(raw[2]).strip()
        }
        if len(evidence_roots) != 1:
            continue
        root = next(iter(evidence_roots))
        if (
            v37._normalise_root(root) in exposed_roots
            or v34._normalise_claim(row["claim"]) in exposed_claims
        ):
            continue
        texts = v37._evidence_texts(row, root)
        if texts:
            grouped[root].append(
                {
                    "id": str(row["id"]),
                    "claim": v37._clean(row["claim"]),
                    "label": str(row["label"]),
                    "texts": texts,
                }
            )
    return dict(grouped)


@contextmanager
def _selection_configuration() -> Iterator[None]:
    old = {
        "n": v311.EXPECTED_FORMAL,
        "per_label": v311.EXPECTED_PER_LABEL,
        "target_pool": v311.TARGET_ROOT_POOL,
        "reuse": v311.MAX_AUXILIARY_ROOT_REUSE,
        "direction": v311.SCORE_DIRECTION_QUOTA,
        "salt": v311.SELECTION_SALT,
        "v37_salt": v37.SELECTION_SALT,
    }
    v311.EXPECTED_FORMAL = EXPECTED_FORMAL
    v311.EXPECTED_PER_LABEL = EXPECTED_PER_LABEL
    v311.TARGET_ROOT_POOL = TARGET_ROOT_POOL
    v311.MAX_AUXILIARY_ROOT_REUSE = MAX_AUXILIARY_ROOT_REUSE
    v311.SCORE_DIRECTION_QUOTA = SCORE_DIRECTION_QUOTA
    v311.SELECTION_SALT = SELECTION_SALT
    v37.SELECTION_SALT = SELECTION_SALT
    try:
        yield
    finally:
        v311.EXPECTED_FORMAL = old["n"]
        v311.EXPECTED_PER_LABEL = old["per_label"]
        v311.TARGET_ROOT_POOL = old["target_pool"]
        v311.MAX_AUXILIARY_ROOT_REUSE = old["reuse"]
        v311.SCORE_DIRECTION_QUOTA = old["direction"]
        v311.SELECTION_SALT = old["salt"]
        v37.SELECTION_SALT = old["v37_salt"]


def build_selection() -> dict[str, Any]:
    with _selection_configuration():
        rows = v311._make_fresh_formal_examples(load_validation_pool())
    for row in rows:
        row["example_id"] = str(row["example_id"]).replace("fever-train-", "fever-validation-", 1)
        row["source_split"] = "fever_gold_evidence_validation"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy18_study_root_call",
        "dataset": {"path": str(DATASET), "sha256": DATASET_SHA256},
        "excluded_selection_sha256": {
            str(v311.SOURCE_SELECTION): base._sha256_path(v311.SOURCE_SELECTION),
            str(V311_SELECTION): base._sha256_path(V311_SELECTION),
        },
        "target_model": TARGET_MODEL,
        "selection_salt_sha256": sha256(SELECTION_SALT).hexdigest(),
        "selection_boundary": (
            "claims and all packet roots are disjoint from V3.7.1 and V3.11; "
            "formal data shifts from FEVER train to FEVER validation"
        ),
        "examples": rows,
    }


def audit_selection(selection: Mapping[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    if rebuild and dict(selection) != build_selection():
        raise ValueError("V3.12 selection or source data drifted")
    rows = list(selection["examples"])
    labels = Counter(str(row["label"]) for row in rows)
    exposed_claims, exposed_roots = _all_exposure()
    claims = {v34._normalise_claim(row["claim"]) for row in rows}
    all_roots = [
        v37._normalise_root(packet["root"])
        for row in rows
        for packet in (row["anchor"], *row["candidates"])
    ]
    annotated = [
        v37._normalise_root(candidate["root"])
        for row in rows
        for candidate in row["candidates"]
        if candidate["annotation_role"] == "held_out_annotated_root"
    ]
    auxiliary = [
        v37._normalise_root(packet["root"])
        for row in rows
        for packet in (row["anchor"], *row["candidates"])
        if packet.get("annotation_role") != "held_out_annotated_root"
    ]
    auxiliary_counts = Counter(auxiliary)
    role, score = [], []
    directions: Counter[str] = Counter()
    for row in rows:
        for candidate in row["candidates"]:
            role.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            score.append(float(candidate["retrieval_score"]))
        gold = next(
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
            if float(distractor["retrieval_score"]) >= float(gold["retrieval_score"])
            else "distractor_below_annotated"
        )
        directions[f"{row['label']}:{direction}"] += 1
    auc = float(roc_auc_score(role, score))
    position = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root" for row in rows
    ) / max(1, len(rows))
    gates = {
        "exact_count": len(rows) == EXPECTED_FORMAL,
        "labels_balanced": labels
        == {"Supported": EXPECTED_PER_LABEL, "Refuted": EXPECTED_PER_LABEL},
        "zero_exposed_claim_overlap": not (claims & exposed_claims),
        "zero_exposed_root_overlap": not (set(all_roots) & exposed_roots),
        "annotated_roots_unique": len(annotated) == len(set(annotated)),
        "annotated_auxiliary_disjoint": not (set(annotated) & set(auxiliary)),
        "auxiliary_reuse_at_most_7": max(auxiliary_counts.values(), default=0)
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
        "candidate_order_balanced": position == 0.5,
        "score_directions_balanced": set(directions.values()) == {SCORE_DIRECTION_QUOTA}
        and len(directions) == 4,
        "oriented_retrieval_role_auc_at_most_065": max(auc, 1.0 - auc) <= 0.65,
        "retrieval_score_forbidden_from_primary_router": True,
    }
    return {
        "counts": {"formal": len(rows)},
        "labels": dict(labels),
        "distinct_roots": len(set(all_roots)),
        "annotated_distinct_roots": len(set(annotated)),
        "auxiliary_distinct_roots": len(set(auxiliary)),
        "maximum_auxiliary_root_reuse": max(auxiliary_counts.values(), default=0),
        "score_direction_counts": dict(sorted(directions.items())),
        "candidate_0_annotated_fraction": position,
        "retrieval_role_auc": auc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection() -> bool:
    expected = build_selection()
    if not audit_selection(expected, rebuild=False)["passed"]:
        raise ValueError("refusing to write a V3.12 selection that fails audit")
    if SELECTION_PATH.exists():
        if json.loads(SELECTION_PATH.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.12 selection drifted")
        return False
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot write V3.12 selection after target records")
    base._write_json(SELECTION_PATH, expected)
    return True


def _examples(path: Path = SELECTION_PATH) -> list[dict[str, Any]]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in selection["examples"]]


def _teacher_example_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in _examples(path)}


def _run_teacher_one(
    client: RecoveryChatClient,
    example: Mapping[str, Any],
    action: str,
    consensus: str,
    *,
    split: str,
) -> dict[str, Any]:
    candidate = example["candidates"][int(action[-1])]
    allowed_ids = [
        *(str(item["evidence_id"]) for item in example["anchor"]["evidence"]),
        *(str(item["evidence_id"]) for item in candidate["evidence"]),
    ]
    decision, attempts, final_error = v362._call_action_with_retry(
        client,
        lambda repair: base.build_recovery_messages(example, action, consensus, repair=repair),
        allowed_ids,
        seed=v362.CALL_SEEDS[action],
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runtime_endpoint": TEACHER_ENDPOINT,
        "runtime_model": TEACHER_MODEL,
        "example_id": str(example["example_id"]),
        "split": split,
        "phase": "teacher_cosign",
        "action": action,
        "target_baseline_consensus": consensus,
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision,
        "final_error": final_error,
        "allowed_selected_packet_evidence_ids": allowed_ids[len(example["anchor"]["evidence"]) :],
    }


def _execute_teacher(
    examples: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, str],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    output_path: Path,
    cache_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    expected = {example_id: action for example_id, action in routes.items() if action != "KEEP"}
    existing_rows = base._load_jsonl(output_path) if output_path.exists() else []
    existing = {str(row["example_id"]): row for row in existing_rows}
    if set(existing) - set(expected):
        raise ValueError("teacher artifact contains rows outside frozen provisional routes")
    pending = [example_id for example_id in expected if example_id not in existing]
    client = RecoveryChatClient(TEACHER_ENDPOINT, TEACHER_MODEL, cache_dir)
    records = list(existing.values())
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _run_teacher_one,
                client,
                examples[example_id],
                expected[example_id],
                str(diagnostics[example_id]["baseline_consensus"]),
                split=split,
            ): example_id
            for example_id in pending
        }
        for future in as_completed(futures):
            row = future.result()
            if not row["success"]:
                raise ValueError(f"teacher call failed closed: {row['example_id']}")
            records.append(row)
            records.sort(key=lambda item: str(item["example_id"]))
            base._write_jsonl(output_path, records)
            print(
                f"[{len(records)}/{len(expected)}] {row['example_id']} "
                f"success={row['success']} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if {str(row["example_id"]) for row in records} != set(expected):
        raise ValueError("teacher call coverage is incomplete")
    if any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("runtime_model") != TEACHER_MODEL
        or row.get("split") != split
        or row.get("action") != expected[str(row["example_id"])]
        or not row.get("success")
        for row in records
    ):
        raise ValueError("teacher record metadata drifted")
    return records


def materialize_development_teacher(workers: int) -> list[dict[str, Any]]:
    v311.validate_protocol_manifest()
    ids, scores, relation_vectors = v311._inference_features()
    head = json.loads(V311_HEAD.read_text(encoding="utf-8"))
    probabilities = np.column_stack(
        [v311._probability(head, relation_vectors[:, candidate]) for candidate in range(2)]
    )
    actions = base._load_jsonl(V311_ACTIONS)
    with _relation_threshold():
        routes, diagnostics = v311._select_routes(
            ids, scores, relation_vectors, actions, probabilities
        )
    examples = _teacher_example_map(V311_SELECTION)
    return _execute_teacher(
        examples,
        routes,
        diagnostics,
        split="development_hy7",
        output_path=DEVELOPMENT_TEACHER,
        cache_dir=DEVELOPMENT_ROOT / "qwen_cache",
        workers=workers,
    )


def _cosign_routes(
    provisional: Mapping[str, str],
    diagnostics: Mapping[str, Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    teacher = {str(row["example_id"]): row for row in teacher_rows}
    expected = {key for key, action in provisional.items() if action != "KEEP"}
    if set(teacher) != expected:
        raise ValueError("teacher rows do not match provisional route budget")
    selected = {}
    cosign_diagnostics = {}
    for example_id, action in provisional.items():
        if action == "KEEP":
            selected[example_id] = "KEEP"
            continue
        row = teacher[example_id]
        decision = row["decision"]
        cited = set(decision["cited_evidence_ids"])
        selected_ids = set(row["allowed_selected_packet_evidence_ids"])
        gates = {
            "teacher_same_answer": str(decision["answer"])
            == str(diagnostics[example_id]["target_action_answer"]),
            "teacher_confidence": float(decision["confidence"]) >= TEACHER_CONFIDENCE,
            "teacher_cites_selected_root": bool(cited & selected_ids),
        }
        selected[example_id] = action if all(gates.values()) else "KEEP"
        cosign_diagnostics[example_id] = {
            "provisional_action": action,
            "teacher_answer": str(decision["answer"]),
            "teacher_confidence": float(decision["confidence"]),
            "teacher_cited_evidence_ids": sorted(cited),
            "gate_components": gates,
            "final_action": selected[example_id],
        }
    return selected, cosign_diagnostics


@contextmanager
def _relation_threshold() -> Iterator[None]:
    old = v311.RELATION_CONFIDENCE_MARGIN
    v311.RELATION_CONFIDENCE_MARGIN = RELATION_CONFIDENCE_MARGIN
    try:
        yield
    finally:
        v311.RELATION_CONFIDENCE_MARGIN = old


def _source_oof() -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    ids, labels, annotated, scores, relation_vectors = v311._source_embedding_rows()
    annotated_vectors = relation_vectors[np.arange(len(ids)), annotated]
    probabilities = np.zeros((len(ids), 2), dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=v311.RELATION_CV_FOLDS,
        shuffle=True,
        random_state=v311.RELATION_CV_SEED,
    )
    for train, held_out in splitter.split(annotated_vectors, labels):
        model = v311._new_relation_head().fit(annotated_vectors[train], labels[train])
        head = v311._head_payload(model)
        for candidate in range(2):
            probabilities[held_out, candidate] = v311._probability(
                head, relation_vectors[held_out, candidate]
            )
    by_id = {str(row["example_id"]): row for row in v311._source_examples()}
    return (
        ids,
        scores,
        relation_vectors,
        probabilities,
        [by_id[example_id] for example_id in ids],
    )


def _existing_qwen_teacher_rows(
    provisional: Mapping[str, str],
    diagnostics: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups = base._record_groups(base._load_jsonl(v311.SOURCE_ACTIONS[TEACHER_MODEL]))
    example_map = {str(row["example_id"]): row for row in examples}
    rows = []
    for example_id, action in provisional.items():
        if action == "KEEP":
            continue
        source = next(
            row
            for row in groups[example_id]
            if row["phase"] == "recovery" and row["action"] == action
        )
        candidate = example_map[example_id]["candidates"][int(action[-1])]
        copied = dict(source)
        copied["allowed_selected_packet_evidence_ids"] = [
            str(item["evidence_id"]) for item in candidate["evidence"]
        ]
        copied["target_baseline_consensus"] = diagnostics[example_id]["baseline_consensus"]
        rows.append(copied)
    return rows


def development_diagnostics() -> dict[str, Any]:
    ids, scores, relation_vectors, probabilities, examples = _source_oof()
    metrics = {}
    with _relation_threshold():
        for name in ("Ling-3.0-tiny", "SUFE-Fin-R1"):
            actions = base._load_jsonl(v311.SOURCE_ACTIONS[name])
            provisional, diagnostics = v311._select_routes(
                ids, scores, relation_vectors, actions, probabilities
            )
            teacher_rows = _existing_qwen_teacher_rows(provisional, diagnostics, examples)
            selected, _ = _cosign_routes(provisional, diagnostics, teacher_rows)
            metrics[name] = v311._development_metric(examples, actions, selected)

        ids_hy, scores_hy, relation_hy = v311._inference_features()
        head = json.loads(V311_HEAD.read_text(encoding="utf-8"))
        probabilities_hy = np.column_stack(
            [v311._probability(head, relation_hy[:, candidate]) for candidate in range(2)]
        )
        actions_hy = base._load_jsonl(V311_ACTIONS)
        provisional_hy, diagnostics_hy = v311._select_routes(
            ids_hy,
            scores_hy,
            relation_hy,
            actions_hy,
            probabilities_hy,
        )
    teacher_hy = base._load_jsonl(DEVELOPMENT_TEACHER)
    selected_hy, _ = _cosign_routes(provisional_hy, diagnostics_hy, teacher_hy)
    examples_hy_by_id = {str(row["example_id"]): row for row in v311._formal_examples()}
    metrics["Hy-MT2-7B"] = v311._development_metric(
        [examples_hy_by_id[example_id] for example_id in ids_hy],
        actions_hy,
        selected_hy,
    )
    if any(
        result["harms"] != 0
        or min(result["by_native_label_net_gain"].values()) < 0
        or result["fixes"] < 5
        for result in metrics.values()
    ):
        raise ValueError("V3.12 co-sign rule is not group-safe across development models")
    return {
        "status": "outcome_known_development_only_before_hy18_calls",
        "relation_confidence_margin": RELATION_CONFIDENCE_MARGIN,
        "teacher_confidence": TEACHER_CONFIDENCE,
        "model_metrics": metrics,
    }


def freeze_router() -> dict[str, Any]:
    if ROUTER_MANIFEST.exists():
        validate_router_manifest()
        return json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8"))
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.12 router after Hy-1.8B calls")
    if not DEVELOPMENT_TEACHER.is_file():
        raise ValueError("V3.12 development teacher records are missing")
    diagnostics = development_diagnostics()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy18_study_root_call",
        "method": "dual_head_provenance_with_selective_cross_model_cosign",
        "v311_router_sha256": base._sha256_path(V311_ROUTER),
        "v311_relation_head_sha256": base._sha256_path(V311_HEAD),
        "v311_summary_sha256": base._sha256_path(V311_SUMMARY),
        "development_teacher_sha256": base._sha256_path(DEVELOPMENT_TEACHER),
        "thresholds": {
            "provenance_score_margin": PROVENANCE_SCORE_MARGIN,
            "relation_confidence_margin": RELATION_CONFIDENCE_MARGIN,
            "target_action_confidence": TARGET_ACTION_CONFIDENCE,
            "teacher_confidence": TEACHER_CONFIDENCE,
            "teacher_must_cite_selected_root": True,
        },
        "development_diagnostics": diagnostics,
        "feature_boundary": {
            "uses_hy18_labels_or_outcomes": False,
            "uses_hy18_annotation_roles_at_inference": False,
            "teacher_queried_only_after_provisional_routes_frozen": True,
            "target_answer_replaced_by_teacher": False,
        },
    }
    base._write_json(ROUTER_MANIFEST, manifest)
    return manifest


def validate_router_manifest() -> None:
    manifest = json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V3.12 router protocol drifted")
    if manifest.get("status") != "frozen_before_any_hy18_study_root_call":
        raise ValueError("V3.12 router was not frozen before target calls")
    expected = {
        "v311_router_sha256": base._sha256_path(V311_ROUTER),
        "v311_relation_head_sha256": base._sha256_path(V311_HEAD),
        "v311_summary_sha256": base._sha256_path(V311_SUMMARY),
        "development_teacher_sha256": base._sha256_path(DEVELOPMENT_TEACHER),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("V3.12 router dependency drifted")
    boundary = manifest["feature_boundary"]
    if boundary.get("uses_hy18_labels_or_outcomes") is not False:
        raise ValueError("Hy-1.8B outcomes entered router development")


def _validate_router_inputs() -> None:
    metadata = json.loads(ROUTER_INPUTS_METADATA.read_text(encoding="utf-8"))
    if metadata.get("inference_only") is not True:
        raise ValueError("V3.12 formal embeddings are not inference-only")
    if metadata.get("selection_sha256") != base._sha256_path(SELECTION_PATH):
        raise ValueError("V3.12 embedding selection drifted")
    if metadata.get("output_sha256") != base._sha256_path(ROUTER_INPUTS):
        raise ValueError("V3.12 embedding artifact drifted")
    arrays = np.load(ROUTER_INPUTS)
    if set(arrays.files) != {
        "example_ids",
        "splits",
        "scores",
        "relation_vectors",
    }:
        raise ValueError("V3.12 router inputs contain forbidden or missing fields")
    if arrays["scores"].shape != (EXPECTED_FORMAL, 2):
        raise ValueError("V3.12 provenance score shape drifted")


def _target_fingerprint() -> dict[str, Any]:
    if any(not path.is_file() for path in TARGET_SMALL_ARTIFACTS):
        raise ValueError("Hy-1.8B fingerprint files are incomplete")
    index_path = TARGET_MODEL_DIR / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    sizes = {name: (TARGET_MODEL_DIR / name).stat().st_size for name in shards}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("Hy-1.8B weight shard is empty")
    return {
        "small_file_sha256": {
            path.name: base._sha256_path(path) for path in TARGET_SMALL_ARTIFACTS
        },
        "weight_shard_sizes_bytes": sizes,
    }


def build_protocol_manifest() -> dict[str, Any]:
    validate_router_manifest()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    audit = audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("V3.12 selection audit failed")
    _validate_router_inputs()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_hy18_study_root_call",
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "implementation_path": str(_implementation_path()),
        "implementation_sha256": base._sha256_path(_implementation_path()),
        "run_script_sha256": base._sha256_path(RUN_SCRIPT),
        "server_script_sha256": base._sha256_path(SERVER_SCRIPT),
        "embedding_script_sha256": base._sha256_path(EMBED_SCRIPT),
        "selection_sha256": base._sha256_path(SELECTION_PATH),
        "selection_audit": audit,
        "router_manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
        "router_inputs_sha256": base._sha256_path(ROUTER_INPUTS),
        "router_inputs_metadata_sha256": base._sha256_path(ROUTER_INPUTS_METADATA),
        "target": {
            "model": TARGET_MODEL,
            "endpoint": TARGET_ENDPOINT,
            "model_dir": str(TARGET_MODEL_DIR),
            "artifact_fingerprint": _target_fingerprint(),
            "temperature": 0.0,
            "response_format": v311.v310._response_format("action"),
            "schema_repair_attempts": 1,
        },
        "teacher": {
            "model": TEACHER_MODEL,
            "endpoint": TEACHER_ENDPOINT,
            "queried_only_for_frozen_provisional_routes": True,
            "temperature": 0.0,
        },
        "evaluation": {
            "formal_examples": EXPECTED_FORMAL,
            "per_native_label": EXPECTED_PER_LABEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "primary_gates": [
                "macro_gain_ci_lower_above_zero",
                "zero_observed_harms",
                "both_label_groups_nonnegative",
                "annotation_supported_repairs_at_least_5",
                "net_fixes_above_keep_and_all_matched_baselines",
                "provenance_path_accuracy_at_least_090",
                "teacher_calls_fewer_than_formal_examples",
            ],
        },
        "claim_boundary": {
            "target_fit_or_calibration": False,
            "unseen_target_checkpoint": True,
            "fever_train_to_validation_shift": True,
            "target_answer_is_always_target_generated": True,
            "selective_teacher_compute": True,
            "universal_transfer": False,
        },
    }


def freeze_protocol() -> dict[str, Any]:
    if any(not path.is_file() for path in (PREREGISTRATION, RUN_SCRIPT, SERVER_SCRIPT)):
        raise ValueError("V3.12 protocol documents or scripts are missing")
    if not PROTOCOL_MANIFEST.exists() and any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.12 after Hy-1.8B records")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.12 protocol drifted")
        return expected
    base._write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol_manifest() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != (build_protocol_manifest()):
        raise ValueError("frozen V3.12 protocol drifted")


class Hy18SchemaChatClient(v311.HySchemaChatClient):
    def __init__(
        self,
        endpoint: str = TARGET_ENDPOINT,
        model: str = TARGET_MODEL,
        cache_dir: Path = DEFAULT_ROOT / "cache",
        timeout: float = 90.0,
    ) -> None:
        if endpoint != TARGET_ENDPOINT or model != TARGET_MODEL:
            raise ValueError("V3.12 target endpoint and model are frozen")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_completion_tokens = v311.MAX_COMPLETION_TOKENS
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def _configured_target() -> Iterator[None]:
    old = {
        "protocol": v311.PROTOCOL_VERSION,
        "model": v311.TARGET_MODEL,
        "endpoint": v311.TARGET_ENDPOINT,
        "client": v311.HySchemaChatClient,
    }
    v311.PROTOCOL_VERSION = PROTOCOL_VERSION
    v311.TARGET_MODEL = TARGET_MODEL
    v311.TARGET_ENDPOINT = TARGET_ENDPOINT
    v311.HySchemaChatClient = Hy18SchemaChatClient
    try:
        yield
    finally:
        v311.PROTOCOL_VERSION = old["protocol"]
        v311.TARGET_MODEL = old["model"]
        v311.TARGET_ENDPOINT = old["endpoint"]
        v311.HySchemaChatClient = old["client"]


def _endpoint_ids(endpoint: str) -> set[str]:
    url = endpoint.removesuffix("/chat/completions") + "/models"
    try:
        with urllib_request.urlopen(url, timeout=10.0) as response:
            payload = json.loads(response.read(v311.MAX_RESPONSE_BYTES + 1))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"model inventory failed for {url}: {error}") from error
    return {
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, Mapping) and "id" in item
    }


def endpoint_check() -> dict[str, list[str]]:
    target = _endpoint_ids(TARGET_ENDPOINT)
    teacher = _endpoint_ids(TEACHER_ENDPOINT)
    if TARGET_MODEL not in target or TEACHER_MODEL not in teacher:
        raise ValueError("V3.12 target or teacher model ID is absent")
    return {"target": sorted(target), "teacher": sorted(teacher)}


def execute_target_actions(
    examples: Sequence[Mapping[str, Any]],
    *,
    split: str,
    output_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    with _configured_target():
        return v311.execute_actions(
            examples,
            split=split,
            output_dir=output_dir,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )


def _inference_features() -> tuple[list[str], np.ndarray, np.ndarray]:
    _validate_router_inputs()
    arrays = np.load(ROUTER_INPUTS)
    ids = [str(value) for value in arrays["example_ids"]]
    if any(str(value) != "formal" for value in arrays["splits"]):
        raise ValueError("V3.12 router inputs contain a non-formal split")
    return (
        ids,
        arrays["scores"].astype(np.float64),
        arrays["relation_vectors"].astype(np.float64),
    )


def _public_examples() -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "anchor": {
                "evidence": [
                    {"evidence_id": str(item["evidence_id"])} for item in row["anchor"]["evidence"]
                ]
            },
            "candidates": [
                {
                    "retrieval_score": float(candidate["retrieval_score"]),
                    "evidence": [
                        {"evidence_id": str(item["evidence_id"])} for item in candidate["evidence"]
                    ],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in _examples()
    ]


def _provisional_payload() -> dict[str, Any]:
    validate_protocol_manifest()
    ids, scores, relation_vectors = _inference_features()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    public = _public_examples()
    with _configured_target():
        v311._validate_actions(public, actions, split="formal", outcomes_allowed=False)
    head = json.loads(V311_HEAD.read_text(encoding="utf-8"))
    probabilities = np.column_stack(
        [v311._probability(head, relation_vectors[:, candidate]) for candidate in range(2)]
    )
    with _relation_threshold():
        routes, diagnostics = v311._select_routes(
            ids, scores, relation_vectors, actions, probabilities
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "router_inputs_sha256": base._sha256_path(ROUTER_INPUTS),
        "outcomes_accessed": False,
        "annotation_roles_accessed": False,
        "provisional_budget": sum(action != "KEEP" for action in routes.values()),
        "routes": routes,
        "diagnostics": diagnostics,
    }


def freeze_provisional() -> dict[str, Any]:
    path = DEFAULT_ROOT / "evaluation" / "provisional_routes.json"
    payload = _provisional_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("V3.12 provisional routes drifted")
        return payload
    base._write_json(path, payload)
    return payload


def execute_formal_teacher(workers: int) -> list[dict[str, Any]]:
    provisional_path = DEFAULT_ROOT / "evaluation" / "provisional_routes.json"
    if not provisional_path.is_file():
        raise ValueError("freeze provisional routes before teacher calls")
    provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
    return _execute_teacher(
        {str(row["example_id"]): row for row in _examples()},
        provisional["routes"],
        provisional["diagnostics"],
        split="formal",
        output_path=DEFAULT_ROOT / "formal" / "teacher" / "records.jsonl",
        cache_dir=DEFAULT_ROOT / "teacher_cache",
        workers=workers,
    )


def _final_preoutcome_payload() -> dict[str, Any]:
    provisional_path = DEFAULT_ROOT / "evaluation" / "provisional_routes.json"
    provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
    if provisional != _provisional_payload():
        raise ValueError("V3.12 provisional route artifact drifted")
    teacher_path = DEFAULT_ROOT / "formal" / "teacher" / "records.jsonl"
    teacher = base._load_jsonl(teacher_path)
    selected, cosign = _cosign_routes(provisional["routes"], provisional["diagnostics"], teacher)
    public = _public_examples()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    groups = base._record_groups(v311._safe_action_rows(actions))
    budget = sum(action != "KEEP" for action in selected.values())
    policies = {
        "selective_cosign_repair": selected,
        "keep": {str(row["example_id"]): "KEEP" for row in public},
    }
    for name, proposed in v34._comparison_proposals(public, groups).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(public, proposed, budget, name=name)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "provisional_routes_sha256": base._sha256_path(provisional_path),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "teacher_records_sha256": base._sha256_path(teacher_path),
        "outcomes_accessed": False,
        "annotation_roles_accessed": False,
        "provisional_budget": provisional["provisional_budget"],
        "final_budget": budget,
        "policies": policies,
        "provisional_diagnostics": provisional["diagnostics"],
        "cosign_diagnostics": cosign,
    }


def freeze_routes() -> dict[str, Any]:
    path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    payload = _final_preoutcome_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("V3.12 final preoutcome routes drifted")
        return payload
    base._write_json(path, payload)
    return payload


@contextmanager
def _metric_seed() -> Iterator[None]:
    old_seed = v311.BOOTSTRAP_SEED
    old_replicates = v311.BOOTSTRAP_REPLICATES
    v311.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    v311.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    try:
        yield
    finally:
        v311.BOOTSTRAP_SEED = old_seed
        v311.BOOTSTRAP_REPLICATES = old_replicates


def evaluate() -> dict[str, Any]:
    validate_protocol_manifest()
    preoutcome_path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    if not preoutcome_path.is_file():
        raise ValueError("freeze V3.12 final routes before outcome evaluation")
    preoutcome = json.loads(preoutcome_path.read_text(encoding="utf-8"))
    if preoutcome != _final_preoutcome_payload():
        raise ValueError("V3.12 final route artifact drifted")
    examples = _examples()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    with _configured_target():
        v311._validate_actions(examples, actions, split="formal", outcomes_allowed=True)
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
    with _metric_seed():
        metrics = {
            name: v311._policy_metrics(examples, grouped, selected)
            for name, selected in policies.items()
        }
    primary = metrics["selective_cosign_repair"]
    matched = [name for name in metrics if name.startswith("matched_")]
    annotated = {
        str(row["example_id"]): next(
            index
            for index, candidate in enumerate(row["candidates"])
            if candidate["annotation_role"] == "held_out_annotated_root"
        )
        for row in examples
    }
    path_correct = {
        example_id: int(value["selected_provenance_path"][-1]) == annotated[example_id]
        for example_id, value in preoutcome["provisional_diagnostics"].items()
    }
    provenance_accuracy = sum(path_correct.values()) / len(path_correct)
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "zero_observed_harms": primary["harms"] == 0,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_5": primary["annotation_supported_repairs"] >= 5,
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched)),
        "provenance_path_accuracy_at_least_090": provenance_accuracy >= 0.90,
        "teacher_calls_fewer_than_formal_examples": preoutcome["provisional_budget"]
        < len(examples),
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "one_shot_unseen_hy18_fever_validation_formal",
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "teacher_records_sha256": preoutcome["teacher_records_sha256"],
        "provisional_routes_sha256": preoutcome["provisional_routes_sha256"],
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_formal": len(examples),
        "provisional_teacher_calls": preoutcome["provisional_budget"],
        "final_root_budget": preoutcome["final_budget"],
        "policies": metrics,
        "provenance_head": {
            "accuracy": provenance_accuracy,
            "correct": sum(path_correct.values()),
            "n": len(path_correct),
        },
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": (
            "PASS_UNSEEN_MODEL_SELECTIVE_COSIGN_REPAIR_V3_12"
            if all(gates.values())
            else "NO_VERIFIED_SELECTIVE_COSIGN_TRANSFER_V3_12"
        ),
        "claim_boundary": build_protocol_manifest()["claim_boundary"],
    }
    base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
    _write_report(summary, DEFAULT_ROOT / "evaluation" / "report.md")
    return summary


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    primary = summary["policies"]["selective_cosign_repair"]
    keep = summary["policies"]["keep"]
    lines = [
        "# Recovery V3.12 result: selective cross-model co-sign repair",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "| metric | KEEP | V3.12 |",
        "| --- | ---: | ---: |",
        f"| accuracy | {keep['final_accuracy']:.2%} | {primary['final_accuracy']:.2%} |",
        f"| macro gain | 0.00pp | {100 * primary['macro_label_gain']:+.2f}pp |",
        f"| fixes / harms | 0 / 0 | {primary['fixes']} / {primary['harms']} |",
        (
            f"| teacher calls / final routes | 0 / 0 | "
            f"{summary['provisional_teacher_calls']} / {summary['final_root_budget']} |"
        ),
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
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def smoke(workers: int) -> None:
    validate_protocol_manifest()
    endpoint_check()
    old_selection = json.loads(v311.SOURCE_SELECTION.read_text(encoding="utf-8"))
    examples = [dict(row) for row in old_selection["examples"] if row["split"] == "development"][
        :SMOKE_EXAMPLES
    ]
    execute_target_actions(
        examples,
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "actions",
        workers=workers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-selection")
    development = subparsers.add_parser("development-teacher")
    development.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("freeze-router")
    subparsers.add_parser("freeze-protocol")
    subparsers.add_parser("endpoint-check")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--workers", type=int, default=2)
    formal = subparsers.add_parser("formal-actions")
    formal.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("freeze-provisional")
    teacher = subparsers.add_parser("formal-teacher")
    teacher.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("freeze-routes")
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "prepare-selection":
        created = write_or_validate_selection()
        print(json.dumps({"created": created, **audit_selection(build_selection())}, indent=2))
        return 0
    if args.command == "development-teacher":
        rows = materialize_development_teacher(args.workers)
        print(json.dumps({"rows": len(rows)}))
        return 0
    if args.command == "freeze-router":
        print(json.dumps(freeze_router(), indent=2))
        return 0
    if args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
        return 0
    if args.command == "endpoint-check":
        validate_protocol_manifest()
        print(json.dumps(endpoint_check(), indent=2))
        return 0
    if args.command == "smoke":
        smoke(args.workers)
        return 0
    validate_protocol_manifest()
    if args.command == "formal-actions":
        execute_target_actions(
            _examples(),
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / "actions",
            workers=args.workers,
        )
        return 0
    if args.command == "freeze-provisional":
        print(json.dumps({"provisional_budget": freeze_provisional()["provisional_budget"]}))
        return 0
    if args.command == "formal-teacher":
        print(json.dumps({"rows": len(execute_formal_teacher(args.workers))}))
        return 0
    if args.command == "freeze-routes":
        payload = freeze_routes()
        print(
            json.dumps(
                {
                    "provisional_budget": payload["provisional_budget"],
                    "final_budget": payload["final_budget"],
                }
            )
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate()
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
