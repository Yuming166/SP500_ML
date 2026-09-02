"""Calibrated provenance-gated counter-consensus cascade on fresh FEVER roots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_7 as v37
from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability import recovery_v3_12 as v312

PROTOCOL_VERSION = "recovery-v3.13-counter-consensus-cascade-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_13_hy18")
DEVELOPMENT_ROOT = Path("results/recovery_v3_13_development")
SELECTION_PATH = DEFAULT_ROOT / "selection_manifest.json"
ROUTER_INPUTS = DEFAULT_ROOT / "router_inputs.npz"
ROUTER_INPUTS_METADATA = ROUTER_INPUTS.with_suffix(".json")
ROUTER_MANIFEST = DEFAULT_ROOT / "router" / "manifest.json"
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/recovery_v3_13_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_13.sh")
SERVER_SCRIPT = v312.SERVER_SCRIPT
EMBED_SCRIPT = v312.EMBED_SCRIPT

V312_SELECTION = v312.SELECTION_PATH
V312_ACTIONS = v312.DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
V312_PROVISIONAL = v312.DEFAULT_ROOT / "evaluation" / "provisional_routes.json"
V312_SUMMARY = v312.DEFAULT_ROOT / "evaluation" / "summary.json"
V312_TEACHER_GRID = DEVELOPMENT_ROOT / "v312_qwen_relation_conflicts.jsonl"

TARGET_MODEL = v312.TARGET_MODEL
TARGET_ENDPOINT = v312.TARGET_ENDPOINT
TARGET_MODEL_DIR = v312.TARGET_MODEL_DIR
TEACHER_MODEL = v312.TEACHER_MODEL
TEACHER_ENDPOINT = v312.TEACHER_ENDPOINT

EXPECTED_FORMAL = 80
EXPECTED_PER_LABEL = 40
TARGET_ROOT_POOL = 200
MAX_AUXILIARY_ROOT_REUSE = 7
MAX_RETRIEVAL_GAP = 0.10
SELECTION_SALT = b"v313-4\n"

PROVENANCE_SCORE_MARGIN = 0.30
RELATION_MARGIN_GRID = (0.13, 0.15, 0.18, 0.20)
RELATION_CONFIDENCE_MARGIN = 0.15
TEACHER_CONFIDENCE = 0.80
BOOTSTRAP_SEED = 20_261_305
BOOTSTRAP_REPLICATES = 10_000
SMOKE_EXAMPLES = 2


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _all_exposure() -> tuple[set[str], set[str]]:
    claims: set[str] = set()
    roots: set[str] = set()
    for path in (v311.SOURCE_SELECTION, v311.SELECTION_PATH, V312_SELECTION):
        selection = json.loads(path.read_text(encoding="utf-8"))
        for row in selection["examples"]:
            claims.add(v34._normalise_claim(row["claim"]))
            for packet in (row["anchor"], *row["candidates"]):
                roots.add(v37._normalise_root(packet["root"]))
    return claims, roots


def load_remaining_validation_pool() -> dict[str, list[dict[str, Any]]]:
    pool = v312.load_validation_pool()
    exposed_claims, exposed_roots = _all_exposure()
    filtered = {
        root: [
            row
            for row in rows
            if v34._normalise_claim(row["claim"]) not in exposed_claims
        ]
        for root, rows in pool.items()
        if v37._normalise_root(root) not in exposed_roots
    }
    return {root: rows for root, rows in filtered.items() if rows}


@contextmanager
def _selection_salt() -> Iterator[None]:
    old = v37.SELECTION_SALT
    v37.SELECTION_SALT = SELECTION_SALT
    try:
        yield
    finally:
        v37.SELECTION_SALT = old


def _make_fresh_examples(
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build balanced fresh examples without conditioning on model or router outputs."""
    with _selection_salt():
        roots = sorted(pool, key=lambda root: v37._hash_key("root-partition", root))
        if len(roots) <= TARGET_ROOT_POOL:
            raise ValueError("not enough untouched validation roots")
        target_roots = roots[:TARGET_ROOT_POOL]
        auxiliary_roots = roots[TARGET_ROOT_POOL:]
        vectorizer = HashingVectorizer(
            n_features=2**16,
            ngram_range=(1, 2),
            stop_words="english",
            alternate_sign=False,
            norm="l2",
        )
        auxiliary_matrix = vectorizer.transform(
            [v37._root_document(pool[root]) for root in auxiliary_roots]
        )
        search = NearestNeighbors(
            n_neighbors=min(100, len(auxiliary_roots)),
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
        claims = vectorizer.transform([str(item[2]["claim"]) for item in raw_items])
        gold_documents = vectorizer.transform(
            [" ".join(item[2]["texts"]) for item in raw_items]
        )
        gold_scores = np.asarray(claims.multiply(gold_documents).sum(axis=1)).ravel()
        distances, indices = search.kneighbors(claims)
        selected: list[dict[str, Any]] = []
        selected_by_label: Counter[str] = Counter()
        used_target_roots: set[str] = set()
        auxiliary_usage: Counter[str] = Counter()
        for label in ("SUPPORTS", "REFUTES"):
            item_indices = sorted(
                (
                    index
                    for index, item in enumerate(raw_items)
                    if item[1] == label and gold_scores[index] >= v37.MIN_GOLD_RETRIEVAL
                ),
                key=lambda index: v37._hash_key(
                    "select-item", str(raw_items[index][2]["id"])
                ),
            )
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
                        gap <= MAX_RETRIEVAL_GAP
                        and auxiliary_usage[auxiliary_root] < MAX_AUXILIARY_ROOT_REUSE
                    ):
                        options.append(
                            (
                                gap,
                                auxiliary_usage[auxiliary_root],
                                v37._hash_key("aux", auxiliary_root),
                                auxiliary_root,
                                score,
                            )
                        )
                options.sort()
                if len(options) < 2:
                    continue
                distractor = options[0]
                anchor = next(
                    (option for option in options[1:] if option[3] != distractor[3]),
                    None,
                )
                if anchor is None:
                    continue
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
                    "evidence": v37._evidence_rows(
                        v37._root_packet_texts(pool[distractor[3]]), "CD"
                    ),
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
                        "example_id": "fever-validation-v313-" + str(row["id"]),
                        "source_split": "fever_gold_evidence_validation",
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
                if selected_by_label[label] == EXPECTED_PER_LABEL:
                    break
        if selected_by_label != {
            "SUPPORTS": EXPECTED_PER_LABEL,
            "REFUTES": EXPECTED_PER_LABEL,
        }:
            raise ValueError(f"fresh selection is incomplete: {selected_by_label!r}")
        return sorted(selected, key=lambda row: str(row["example_id"]))


def build_selection() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v313_study_root_call",
        "dataset": {"path": str(v312.DATASET), "sha256": v312.DATASET_SHA256},
        "excluded_selection_sha256": {
            str(path): base._sha256_path(path)
            for path in (v311.SOURCE_SELECTION, v311.SELECTION_PATH, V312_SELECTION)
        },
        "target_model": TARGET_MODEL,
        "selection_salt_sha256": sha256(SELECTION_SALT).hexdigest(),
        "selection_boundary": (
            "claims and all packet roots are disjoint from V3.7.1, V3.11, and V3.12; "
            "selection uses labels only for fixed 40/40 balance and never uses model outputs"
        ),
        "examples": _make_fresh_examples(load_remaining_validation_pool()),
    }


def audit_selection(selection: Mapping[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    if rebuild and dict(selection) != build_selection():
        raise ValueError("V3.13 selection or source data drifted")
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
    roles: list[int] = []
    scores: list[float] = []
    gaps: list[float] = []
    directions: Counter[str] = Counter()
    for row in rows:
        annotated_candidate = next(
            candidate
            for candidate in row["candidates"]
            if candidate["annotation_role"] == "held_out_annotated_root"
        )
        distractor = next(
            candidate
            for candidate in row["candidates"]
            if candidate["annotation_role"] == "unannotated_retrieval_candidate"
        )
        gold_score = float(annotated_candidate["retrieval_score"])
        distractor_score = float(distractor["retrieval_score"])
        gaps.append(abs(gold_score - distractor_score))
        direction = (
            "distractor_at_least_annotated"
            if distractor_score >= gold_score
            else "distractor_below_annotated"
        )
        directions[f"{row['label']}:{direction}"] += 1
        for candidate in row["candidates"]:
            roles.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            scores.append(float(candidate["retrieval_score"]))
    position = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root"
        for row in rows
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
        "candidate_retrieval_gap_at_most_010": max(gaps, default=0.0)
        <= MAX_RETRIEVAL_GAP + 1e-12,
        "retrieval_score_forbidden_from_primary_router": True,
    }
    return {
        "counts": {"formal": len(rows)},
        "labels": dict(labels),
        "distinct_roots": len(set(all_roots)),
        "annotated_distinct_roots": len(set(annotated)),
        "auxiliary_distinct_roots": len(set(auxiliary)),
        "maximum_auxiliary_root_reuse": max(auxiliary_counts.values(), default=0),
        "candidate_0_annotated_fraction": position,
        "maximum_candidate_retrieval_gap": max(gaps, default=0.0),
        "retrieval_role_auc_diagnostic_only": float(roc_auc_score(roles, scores)),
        "score_direction_counts_diagnostic_only": dict(sorted(directions.items())),
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection() -> bool:
    expected = build_selection()
    if not audit_selection(expected, rebuild=False)["passed"]:
        raise ValueError("refusing to write a V3.13 selection that fails audit")
    if SELECTION_PATH.exists():
        if json.loads(SELECTION_PATH.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.13 selection drifted")
        return False
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot create V3.13 selection after target records")
    base._write_json(SELECTION_PATH, expected)
    return True


def _examples(path: Path = SELECTION_PATH) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in json.loads(path.read_text(encoding="utf-8"))["examples"]
    ]


def _teacher_accepts(
    row: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    example: Mapping[str, Any],
) -> bool:
    action = str(diagnostic["selected_provenance_path"])
    candidate = example["candidates"][int(action[-1])]
    selected_ids = {str(item["evidence_id"]) for item in candidate["evidence"]}
    decision = row["decision"]
    cited = {str(value) for value in decision["cited_evidence_ids"]}
    return bool(
        str(decision["answer"]) == str(diagnostic["relation_answer"])
        and float(decision["confidence"]) >= TEACHER_CONFIDENCE
        and cited & selected_ids
    )


def _development_grid() -> dict[str, Any]:
    v312_summary = json.loads(V312_SUMMARY.read_text(encoding="utf-8"))
    if not v312_summary.get("passes") is False:
        raise ValueError("V3.12 must remain a completed negative formal result")
    provisional = json.loads(V312_PROVISIONAL.read_text(encoding="utf-8"))
    examples = {
        str(row["example_id"]): row
        for row in json.loads(V312_SELECTION.read_text(encoding="utf-8"))["examples"]
    }
    actions = base._load_jsonl(V312_ACTIONS)
    grouped = base._record_groups(v311._safe_action_rows(actions))
    teacher_rows = {
        str(row["example_id"]): row for row in base._load_jsonl(V312_TEACHER_GRID)
    }
    table: dict[str, Any] = {}
    for threshold in RELATION_MARGIN_GRID:
        routes = fixes = harms = 0
        for example_id, diagnostic in provisional["diagnostics"].items():
            eligible = bool(
                diagnostic["baseline_agreement"] >= base.HIGH_CONSENSUS
                and diagnostic["provenance_margin"] >= PROVENANCE_SCORE_MARGIN
                and diagnostic["relation_margin"] >= threshold
                and diagnostic["relation_answer"]
                != diagnostic["baseline_consensus"]
            )
            if not eligible:
                continue
            row = teacher_rows.get(example_id)
            if row is None or not _teacher_accepts(row, diagnostic, examples[example_id]):
                continue
            routes += 1
            keep, _, _, _ = base._outcomes(examples[example_id], grouped[example_id])
            final = int(
                str(row["decision"]["answer"])
                == ("yes" if examples[example_id]["gold_binary"] else "no")
            )
            gain = final - keep
            fixes += int(gain == 1)
            harms += int(gain == -1)
        table[f"{threshold:.2f}"] = {
            "accepted_routes": routes,
            "fixes": fixes,
            "harms": harms,
            "net_fixes": fixes - harms,
        }
    if table != {
        "0.13": {"accepted_routes": 16, "fixes": 15, "harms": 1, "net_fixes": 14},
        "0.15": {"accepted_routes": 13, "fixes": 13, "harms": 0, "net_fixes": 13},
        "0.18": {"accepted_routes": 7, "fixes": 7, "harms": 0, "net_fixes": 7},
        "0.20": {"accepted_routes": 3, "fixes": 3, "harms": 0, "net_fixes": 3},
    }:
        raise ValueError(f"V3.13 development grid drifted: {table!r}")
    return table


def build_router_manifest() -> dict[str, Any]:
    table = _development_grid()
    zero_harm = [
        (float(threshold), values)
        for threshold, values in table.items()
        if values["harms"] == 0 and values["fixes"] >= 5
    ]
    chosen = max(zero_harm, key=lambda item: item[1]["accepted_routes"])[0]
    if chosen != RELATION_CONFIDENCE_MARGIN:
        raise ValueError("frozen V3.13 margin is not selected by the declared rule")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v313_study_root_call",
        "method": "provenance_gated_counter_consensus_cascade",
        "source_router_manifest_sha256": base._sha256_path(v311.ROUTER_MANIFEST),
        "source_relation_head_sha256": base._sha256_path(v311.ROUTER_HEAD),
        "development_artifacts": {
            str(V312_SELECTION): base._sha256_path(V312_SELECTION),
            str(V312_ACTIONS): base._sha256_path(V312_ACTIONS),
            str(V312_PROVISIONAL): base._sha256_path(V312_PROVISIONAL),
            str(V312_SUMMARY): base._sha256_path(V312_SUMMARY),
            str(V312_TEACHER_GRID): base._sha256_path(V312_TEACHER_GRID),
        },
        "development_status": "V3.12 reused only after its formal verdict as V3.13 calibration",
        "fixed_margin_grid": list(RELATION_MARGIN_GRID),
        "development_grid": table,
        "selection_rule": (
            "among margins with zero observed harms and at least five fixes, maximize "
            "accepted-route coverage"
        ),
        "thresholds": {
            "provenance_score_margin": PROVENANCE_SCORE_MARGIN,
            "relation_confidence_margin": RELATION_CONFIDENCE_MARGIN,
            "teacher_confidence": TEACHER_CONFIDENCE,
            "high_consensus": base.HIGH_CONSENSUS,
        },
        "execution": (
            "override target consensus only when the relation head and Qwen agree on the "
            "opposite answer and Qwen cites the provenance-selected root"
        ),
    }


def freeze_router() -> dict[str, Any]:
    expected = build_router_manifest()
    if ROUTER_MANIFEST.exists():
        if json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.13 router drifted")
        return expected
    base._write_json(ROUTER_MANIFEST, expected)
    return expected


def validate_router_manifest() -> None:
    if json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8")) != build_router_manifest():
        raise ValueError("frozen V3.13 router drifted")


def _validate_router_inputs() -> None:
    metadata = json.loads(ROUTER_INPUTS_METADATA.read_text(encoding="utf-8"))
    required = {"example_ids", "splits", "scores", "relation_vectors"}
    if metadata.get("inference_only") is not True:
        raise ValueError("V3.13 formal embeddings are not inference-only")
    if set(metadata.get("output_fields", [])) != required:
        raise ValueError("V3.13 formal embedding fields drifted")
    if metadata.get("selection_sha256") != base._sha256_path(SELECTION_PATH):
        raise ValueError("V3.13 embedding selection hash drifted")
    if metadata.get("output_sha256") != base._sha256_path(ROUTER_INPUTS):
        raise ValueError("V3.13 embedding artifact hash drifted")


def build_protocol_manifest() -> dict[str, Any]:
    validate_router_manifest()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    audit = audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("V3.13 selection audit failed")
    _validate_router_inputs()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v313_study_root_call",
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
            "artifact_fingerprint": v312._target_fingerprint(),
            "temperature": 0.0,
            "response_format": v311.v310._response_format("action"),
        },
        "teacher": {
            "model": TEACHER_MODEL,
            "endpoint": TEACHER_ENDPOINT,
            "queried_only_for_frozen_counter_consensus_routes": True,
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
            "target_fit_or_calibration": True,
            "calibration_examples": 112,
            "formal_claim_and_root_overlap_with_calibration": False,
            "cross_model_teacher": True,
            "teacher_is_answer_source_on_accepted_routes": True,
            "selective_teacher_compute": True,
            "universal_transfer": False,
        },
    }


def freeze_protocol() -> dict[str, Any]:
    if any(not path.is_file() for path in (PREREGISTRATION, RUN_SCRIPT, SERVER_SCRIPT)):
        raise ValueError("V3.13 protocol documents or scripts are missing")
    if not PROTOCOL_MANIFEST.exists() and any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.13 after formal records")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.13 protocol drifted")
        return expected
    base._write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol_manifest() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != build_protocol_manifest():
        raise ValueError("frozen V3.13 protocol drifted")


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
    v311.HySchemaChatClient = v312.Hy18SchemaChatClient
    try:
        yield
    finally:
        v311.PROTOCOL_VERSION = old["protocol"]
        v311.TARGET_MODEL = old["model"]
        v311.TARGET_ENDPOINT = old["endpoint"]
        v311.HySchemaChatClient = old["client"]


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


def endpoint_check() -> dict[str, list[str]]:
    return v312.endpoint_check()


def _inference_features() -> tuple[list[str], np.ndarray, np.ndarray]:
    _validate_router_inputs()
    arrays = np.load(ROUTER_INPUTS)
    ids = [str(value) for value in arrays["example_ids"]]
    if any(str(value) != "formal" for value in arrays["splits"]):
        raise ValueError("V3.13 router inputs contain a non-formal split")
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
                    {"evidence_id": str(item["evidence_id"])}
                    for item in row["anchor"]["evidence"]
                ]
            },
            "candidates": [
                {
                    "retrieval_score": float(candidate["retrieval_score"]),
                    "evidence": [
                        {"evidence_id": str(item["evidence_id"])}
                        for item in candidate["evidence"]
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
    with _configured_target():
        v311._validate_actions(_public_examples(), actions, split="formal", outcomes_allowed=False)
    grouped = base._record_groups(v311._safe_action_rows(actions))
    head = json.loads(v311.ROUTER_HEAD.read_text(encoding="utf-8"))
    probabilities = np.column_stack(
        [v311._probability(head, relation_vectors[:, index]) for index in range(2)]
    )
    routes: dict[str, str] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for row_index, example_id in enumerate(ids):
        consensus, agreement, _ = base._baseline_state(grouped[example_id])
        candidate_index = int(np.argmax(scores[row_index]))
        action = f"candidate_{candidate_index}"
        probability = float(probabilities[row_index, candidate_index])
        relation_answer = "yes" if probability >= 0.5 else "no"
        path_margin = float(abs(scores[row_index, 0] - scores[row_index, 1]))
        relation_margin = abs(probability - 0.5)
        gates = {
            "high_consensus": agreement >= base.HIGH_CONSENSUS,
            "provenance_margin": path_margin >= PROVENANCE_SCORE_MARGIN,
            "relation_margin": relation_margin >= RELATION_CONFIDENCE_MARGIN,
            "counter_consensus": relation_answer != consensus,
        }
        routes[example_id] = action if all(gates.values()) else "KEEP"
        diagnostics[example_id] = {
            "selected_provenance_path": action,
            "provenance_scores": [float(value) for value in scores[row_index]],
            "provenance_margin": path_margin,
            "p_supported": probability,
            "relation_margin": relation_margin,
            "relation_answer": relation_answer,
            "baseline_consensus": consensus,
            "baseline_agreement": agreement,
            "gate_components": gates,
            "provisional_route": routes[example_id],
        }
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
            raise ValueError("V3.13 provisional routes drifted")
        return payload
    base._write_json(path, payload)
    return payload


@contextmanager
def _configured_teacher_records() -> Iterator[None]:
    old = v312.PROTOCOL_VERSION
    v312.PROTOCOL_VERSION = PROTOCOL_VERSION
    try:
        yield
    finally:
        v312.PROTOCOL_VERSION = old


def execute_formal_teacher(workers: int) -> list[dict[str, Any]]:
    provisional_path = DEFAULT_ROOT / "evaluation" / "provisional_routes.json"
    if not provisional_path.is_file():
        raise ValueError("freeze V3.13 provisional routes before teacher calls")
    provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
    with _configured_teacher_records():
        return v312._execute_teacher(
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
        raise ValueError("V3.13 provisional route artifact drifted")
    teacher_path = DEFAULT_ROOT / "formal" / "teacher" / "records.jsonl"
    teacher_rows = base._load_jsonl(teacher_path)
    teacher = {str(row["example_id"]): row for row in teacher_rows}
    expected = {
        example_id for example_id, action in provisional["routes"].items() if action != "KEEP"
    }
    if set(teacher) != expected:
        raise ValueError("V3.13 teacher rows do not match provisional routes")
    examples = {str(row["example_id"]): row for row in _examples()}
    routes: dict[str, str] = {}
    override_answers: dict[str, str] = {}
    cosign_diagnostics: dict[str, Any] = {}
    for example_id, action in provisional["routes"].items():
        if action == "KEEP":
            routes[example_id] = "KEEP"
            continue
        row = teacher[example_id]
        diagnostic = provisional["diagnostics"][example_id]
        accepted = _teacher_accepts(row, diagnostic, examples[example_id])
        routes[example_id] = action if accepted else "KEEP"
        if accepted:
            override_answers[example_id] = str(row["decision"]["answer"])
        cosign_diagnostics[example_id] = {
            "provisional_action": action,
            "relation_answer": str(diagnostic["relation_answer"]),
            "teacher_answer": str(row["decision"]["answer"]),
            "teacher_confidence": float(row["decision"]["confidence"]),
            "teacher_cited_evidence_ids": sorted(row["decision"]["cited_evidence_ids"]),
            "accepted": accepted,
            "final_action": routes[example_id],
        }
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    groups = base._record_groups(v311._safe_action_rows(actions))
    budget = sum(action != "KEEP" for action in routes.values())
    matched = {
        f"matched_{name}": base._truncate_to_budget(
            _public_examples(), proposed, budget, name=name
        )
        for name, proposed in v34._comparison_proposals(_public_examples(), groups).items()
    }
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
        "primary_routes": routes,
        "primary_override_answers": override_answers,
        "matched_action_policies": matched,
        "provisional_diagnostics": provisional["diagnostics"],
        "cosign_diagnostics": cosign_diagnostics,
    }


def freeze_routes() -> dict[str, Any]:
    path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    payload = _final_preoutcome_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("V3.13 final preoutcome routes drifted")
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


def _answer_policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    routes: Mapping[str, str],
    override_answers: Mapping[str, str],
    cosign: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped = base._record_groups(actions)
    keeps: list[int] = []
    finals: list[int] = []
    labels: list[str] = []
    high_flags: list[bool] = []
    selected_actions: Counter[str] = Counter()
    annotation_supported = 0
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, _, _ = base._outcomes(example, grouped[example_id])
        route = str(routes[example_id])
        if route == "KEEP":
            final = keep
        else:
            final = int(
                override_answers[example_id]
                == ("yes" if example["gold_binary"] else "no")
            )
            candidate = example["candidates"][int(route[-1])]
            cited = set(cosign[example_id]["teacher_cited_evidence_ids"])
            annotation_supported += int(
                keep == 0
                and final == 1
                and candidate["annotation_role"] == "held_out_annotated_root"
                and bool(cited & {str(item["evidence_id"]) for item in candidate["evidence"]})
            )
        keeps.append(keep)
        finals.append(final)
        labels.append(str(example["label"]))
        high_flags.append(agreement >= base.HIGH_CONSENSUS)
        selected_actions[route] += 1
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
    high_correct = sum(
        bool(keep and high) for keep, high in zip(keeps, high_flags, strict=True)
    )
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
        "macro_label_gain": float(
            np.mean([group["net_gain"] for group in by_label.values()])
        ),
        "macro_gain_ci": np.quantile(
            bootstrap, [0.025, 0.975], method="linear"
        ).tolist(),
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": sum(action != "KEEP" for action in routes.values()),
        "selected_actions": dict(selected_actions),
        "by_native_label": by_label,
    }


def evaluate() -> dict[str, Any]:
    validate_protocol_manifest()
    preoutcome_path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    if not preoutcome_path.is_file():
        raise ValueError("freeze V3.13 final routes before outcome evaluation")
    preoutcome = json.loads(preoutcome_path.read_text(encoding="utf-8"))
    if preoutcome != _final_preoutcome_payload():
        raise ValueError("V3.13 final route artifact drifted")
    examples = _examples()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    with _configured_target():
        v311._validate_actions(examples, actions, split="formal", outcomes_allowed=True)
    with _metric_seed():
        primary = _answer_policy_metrics(
            examples,
            actions,
            preoutcome["primary_routes"],
            preoutcome["primary_override_answers"],
            preoutcome["cosign_diagnostics"],
        )
        keep = v311._policy_metrics(
            examples,
            actions,
            {str(row["example_id"]): "KEEP" for row in examples},
        )
        matched = {
            name: v311._policy_metrics(examples, actions, policy)
            for name, policy in preoutcome["matched_action_policies"].items()
        }
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
    metrics = {"counter_consensus_cascade": primary, "keep": keep, **matched}
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "zero_observed_harms": primary["harms"] == 0,
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0 for group in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_5": primary[
            "annotation_supported_repairs"
        ]
        >= 5,
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(value["net_fixes"] for value in matched.values())),
        "provenance_path_accuracy_at_least_090": provenance_accuracy >= 0.90,
        "teacher_calls_fewer_than_formal_examples": preoutcome["provisional_budget"]
        < len(examples),
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "one_shot_calibrated_hy18_fresh_root_formal",
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
            "PASS_CALIBRATED_COUNTER_CONSENSUS_CASCADE_V3_13"
            if all(gates.values())
            else "NO_VERIFIED_COUNTER_CONSENSUS_CASCADE_V3_13"
        ),
        "claim_boundary": build_protocol_manifest()["claim_boundary"],
    }
    base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
    _write_report(summary, DEFAULT_ROOT / "evaluation" / "report.md")
    return summary


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    primary = summary["policies"]["counter_consensus_cascade"]
    lines = [
        "# Recovery V3.13 result: provenance-gated counter-consensus cascade",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        f"Formal examples: {summary['n_formal']}",
        (
            f"Provisional / final routes: {summary['provisional_teacher_calls']} / "
            f"{summary['final_root_budget']}"
        ),
        f"Accuracy: {primary['baseline_accuracy']:.4f} -> {primary['final_accuracy']:.4f}",
        f"Macro-label gain: {primary['macro_label_gain']:.4f}",
        (
            f"95% CI: [{primary['macro_gain_ci'][0]:.4f}, "
            f"{primary['macro_gain_ci'][1]:.4f}]"
        ),
        f"Fixes / harms: {primary['fixes']} / {primary['harms']}",
        f"Annotation-supported repairs: {primary['annotation_supported_repairs']}",
        "",
        "## Gates",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'}: `{name}`"
            for name, value in summary["primary_gates"].items()
        ],
        "",
        (
            "V3.12 is development/calibration data for this result. This is not a zero-shot "
            "Hy-MT2-1.8B claim, and Qwen is the answer source on accepted routes."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-selection")
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
        old_examples = _examples(V312_SELECTION)[:SMOKE_EXAMPLES]
        execute_target_actions(
            old_examples,
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "actions",
            workers=args.workers,
        )
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
