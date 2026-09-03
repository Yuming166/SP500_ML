"""Zero-shot model- and dataset-holdout provenance repair on HoVer.

V3.14 reuses the frozen V3.11 dual-head router without fitting or calibrating
on the Qwen3.6 target.  Formal examples are selected from HoVer and obtain
their sentence contexts from the official HotpotQA distractor-development
artifact.  See ``docs/recovery_v3_14_preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as v37
from sp500_forecastability import recovery_v3_10 as v310
from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability.pilot_llm_v1 import MAX_COMPLETION_TOKENS

PROTOCOL_VERSION = "recovery-v3.14-zero-shot-qwen36-hover-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_14_qwen36")
SELECTION_PATH = DEFAULT_ROOT / "selection_manifest.json"
SELECTION_AUDIT_PATH = DEFAULT_ROOT / "selection_audit.json"
ROUTER_MANIFEST = DEFAULT_ROOT / "router" / "manifest.json"
ROUTER_INPUTS = DEFAULT_ROOT / "router_inputs.npz"
ROUTER_INPUTS_METADATA = ROUTER_INPUTS.with_suffix(".json")
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/recovery_v3_14_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_14.sh")
SERVE_SCRIPT = Path("scripts/serve_recovery_v3_14_qwen36.sh")
EMBED_SCRIPT = Path("scripts/embed_recovery_v3_11_development.py")

HOVER_DATASET = Path("data/hover/hover_dev_release_v1.1.json")
HOTPOT_CONTEXT = Path("data/hover/hotpot_dev_distractor_v1.parquet")
HOTPOT_CONTEXT_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"

FROZEN_ROUTER_HEAD = v311.ROUTER_HEAD
FROZEN_ROUTER_MANIFEST = v311.ROUTER_MANIFEST
PRIOR_SELECTIONS = (
    Path("results/recovery_v3_4_1/selection_manifest.json"),
    Path("results/recovery_v3_6_2/selection_manifest.json"),
    Path("results/recovery_v3_7_1/selection_manifest.json"),
    Path("results/recovery_v3_11_hy/selection_manifest.json"),
    Path("results/recovery_v3_12_hy18/selection_manifest.json"),
    Path("results/recovery_v3_13_hy18/selection_manifest.json"),
    Path("results/pilot_llm_v10_4/selection_manifest.json"),
    Path("results/pilot_llm_v11_1/selection_manifest.json"),
    Path("results/pilot_llm_v12_1/selection_manifest.json"),
)

TARGET_MODEL = "Qwen3.6-35B-A3B"
TARGET_ENDPOINT = "http://127.0.0.1:31521/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/Qwen/Qwen3.6-35B-A3B-FP8")
TARGET_SMALL_ARTIFACTS = tuple(
    TARGET_MODEL_DIR / name
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
)

SELECTION_SALT = b"recovery-v3.14-qwen36-hover-zero-shot-2026-09-03\n"
EXPECTED_FORMAL = 300
EXPECTED_PER_LABEL = 150
DIRECTION_QUOTA = 75
LABELS = ("Supported", "NotSupported")
MAX_AUXILIARY_ROOT_REUSE = 3
MAX_RETRIEVAL_GAP = 0.20
MAX_ORIENTED_ROLE_AUC = 0.65
BOOTSTRAP_SEED = 20_261_403
BOOTSTRAP_REPLICATES = 10_000
SMOKE_EXAMPLES = 3
SMOKE_EXPECTED_ROWS = SMOKE_EXAMPLES * 8
SMOKE_MIN_FIRST_PASS = 23


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _hash_key(namespace: str, value: str) -> str:
    return sha256(SELECTION_SALT + namespace.encode() + b"\0" + value.encode()).hexdigest()


def _normalise_root(value: str) -> str:
    return v37._normalise_root(value)


def _packet_roots(packet: Mapping[str, Any]) -> list[str]:
    components = packet.get("component_roots")
    if isinstance(components, list):
        return [str(root) for root in components]
    return [str(packet["root"])]


def _prior_exposure() -> tuple[set[str], set[str]]:
    claims: set[str] = set()
    roots: set[str] = set()
    for path in PRIOR_SELECTIONS:
        if not path.is_file():
            raise FileNotFoundError(f"required prior selection is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("examples", []):
            claim = row.get("claim")
            if claim:
                claims.add(v34._normalise_claim(str(claim)))
            packets = [row.get("anchor"), *row.get("candidates", [])]
            for packet in packets:
                if not isinstance(packet, Mapping) or "root" not in packet:
                    continue
                roots.update(_normalise_root(root) for root in _packet_roots(packet))
    return claims, roots


def _hotpot_rows() -> dict[str, dict[str, list[str]]]:
    frame = pd.read_parquet(HOTPOT_CONTEXT)
    output: dict[str, dict[str, list[str]]] = {}
    for row in frame.itertuples(index=False):
        output[str(row.id)] = {
            str(title): [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
            for title, sentences in zip(row.context["title"], row.context["sentences"], strict=True)
        }
    return output


def _joined_hover_rows() -> list[dict[str, Any]]:
    hover = json.loads(HOVER_DATASET.read_text(encoding="utf-8"))
    contexts = _hotpot_rows()
    joined: list[dict[str, Any]] = []
    for source in hover:
        context = contexts.get(str(source["hpqa_id"]))
        if context is None:
            continue
        facts = [(str(title), int(index)) for title, index in source["supporting_facts"]]
        if not all(title in context and index < len(context[title]) for title, index in facts):
            continue
        joined.append({**source, "_context": context, "_facts": facts})
    return joined


def _bundle_root(roots: Sequence[str]) -> str:
    canonical = "\n".join(sorted(roots))
    return "hover-bundle-" + sha256(canonical.encode()).hexdigest()[:20]


def _bundle_evidence(
    context: Mapping[str, Sequence[str]], roots: Sequence[str], prefix: str
) -> list[dict[str, str]]:
    texts = [sentence for root in roots for sentence in context[root][:2]]
    return [
        {"evidence_id": f"{prefix}{index:02d}", "text": text} for index, text in enumerate(texts)
    ]


def _annotated_evidence(
    context: Mapping[str, Sequence[str]], facts: Sequence[tuple[str, int]], prefix: str
) -> list[dict[str, str]]:
    texts = list(dict.fromkeys(context[root][index] for root, index in facts))
    return [
        {"evidence_id": f"{prefix}{index:02d}", "text": text} for index, text in enumerate(texts)
    ]


def _document(evidence: Sequence[Mapping[str, str]]) -> str:
    return " ".join(str(row["text"]) for row in evidence)


def _raw_candidates() -> list[dict[str, Any]]:
    exposed_claims, exposed_roots = _prior_exposure()
    joined = _joined_hover_rows()
    all_supporting_roots = {
        _normalise_root(root) for row in joined for root, _index in row["_facts"]
    }
    vectorizer = HashingVectorizer(
        n_features=2**16,
        ngram_range=(1, 2),
        stop_words="english",
        alternate_sign=False,
        norm="l2",
    )
    raw: list[dict[str, Any]] = []
    for row in joined:
        source_label = str(row["label"])
        if source_label not in {"SUPPORTED", "NOT_SUPPORTED"}:
            continue
        claim = str(row["claim"]).strip()
        if not claim or v34._normalise_claim(claim) in exposed_claims:
            continue
        facts = list(row["_facts"])
        supporting_roots = sorted({root for root, _index in facts})
        if len(supporting_roots) not in {2, 3}:
            continue
        if any(_normalise_root(root) in exposed_roots for root in supporting_roots):
            continue
        context = dict(row["_context"])
        auxiliary_roots = sorted(
            root
            for root in context
            if root not in supporting_roots
            and _normalise_root(root) not in exposed_roots
            and _normalise_root(root) not in all_supporting_roots
        )
        width = len(supporting_roots)
        if len(auxiliary_roots) < 2 * width:
            continue
        annotated = _annotated_evidence(context, facts, "G")
        query = vectorizer.transform([claim])
        annotated_score = float(query.multiply(vectorizer.transform([_document(annotated)])).sum())
        options: list[dict[str, Any]] = []
        for roots in combinations(auxiliary_roots, width):
            evidence = _bundle_evidence(context, roots, "X")
            score = float(query.multiply(vectorizer.transform([_document(evidence)])).sum())
            gap = abs(score - annotated_score)
            if gap <= MAX_RETRIEVAL_GAP:
                options.append(
                    {
                        "roots": list(roots),
                        "score": score,
                        "gap": gap,
                        "evidence": evidence,
                    }
                )
        if not options:
            continue
        options.sort(
            key=lambda item: (
                float(item["gap"]),
                _hash_key("bundle-option", "\n".join(item["roots"])),
            )
        )
        raw.append(
            {
                "uid": str(row["uid"]),
                "source_claim_id": str(row["uid"]),
                "hpqa_id": str(row["hpqa_id"]),
                "claim": claim,
                "label": "Supported" if source_label == "SUPPORTED" else "NotSupported",
                "gold_binary": int(source_label == "SUPPORTED"),
                "supporting_roots": supporting_roots,
                "annotated_score": annotated_score,
                "annotated_evidence": annotated,
                "options": options,
            }
        )
    return raw


def _available(roots: Sequence[str], auxiliary_usage: Counter[str]) -> bool:
    return all(auxiliary_usage[_normalise_root(root)] < MAX_AUXILIARY_ROOT_REUSE for root in roots)


def _select_option_pair(
    item: Mapping[str, Any],
    direction: str,
    auxiliary_usage: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    annotated_score = float(item["annotated_score"])
    options = list(item["options"])
    directional = [
        option
        for option in options
        if (float(option["score"]) >= annotated_score) == (direction == "above")
    ]
    for distractor in directional:
        if not _available(distractor["roots"], auxiliary_usage):
            continue
        used = set(distractor["roots"])
        for anchor in options:
            if used & set(anchor["roots"]):
                continue
            if not _available(anchor["roots"], auxiliary_usage):
                continue
            return dict(distractor), dict(anchor)
    return None


def _packet(option: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    evidence = [
        {"evidence_id": f"{prefix}{index:02d}", "text": str(row["text"])}
        for index, row in enumerate(option["evidence"])
    ]
    roots = [str(root) for root in option["roots"]]
    return {
        "root": _bundle_root(roots),
        "component_roots": roots,
        "retrieval_score": float(option["score"]),
        "evidence": evidence,
    }


def build_selection() -> dict[str, Any]:
    raw = _raw_candidates()
    selected: list[dict[str, Any]] = []
    selected_by_label: Counter[str] = Counter()
    used_claims: set[str] = set()
    used_annotated_roots: set[str] = set()
    auxiliary_usage: Counter[str] = Counter()
    phase_specs = (
        ("NotSupported", "above"),
        ("Supported", "above"),
        ("NotSupported", "below"),
        ("Supported", "below"),
    )
    for label, direction in phase_specs:
        phase_count = 0
        ordered = sorted(
            (item for item in raw if item["label"] == label),
            key=lambda item: _hash_key(f"select-{label}-{direction}", str(item["uid"])),
        )
        for item in ordered:
            claim_key = v34._normalise_claim(str(item["claim"]))
            annotated_roots = {_normalise_root(root) for root in item["supporting_roots"]}
            if claim_key in used_claims or annotated_roots & used_annotated_roots:
                continue
            pair = _select_option_pair(item, direction, auxiliary_usage)
            if pair is None:
                continue
            distractor, anchor = pair
            annotated_index = selected_by_label[label] % 2
            annotated_packet = {
                "root": _bundle_root(item["supporting_roots"]),
                "component_roots": list(item["supporting_roots"]),
                "annotation_role": "held_out_annotated_root",
                "retrieval_score": float(item["annotated_score"]),
                "evidence": [],
            }
            distractor_packet = {
                **_packet(distractor, prefix="D"),
                "annotation_role": "unannotated_retrieval_candidate",
            }
            raw_candidates = (
                [annotated_packet, distractor_packet]
                if annotated_index == 0
                else [distractor_packet, annotated_packet]
            )
            candidates = []
            for candidate_index, packet in enumerate(raw_candidates):
                output = dict(packet)
                source_evidence = (
                    item["annotated_evidence"]
                    if packet["annotation_role"] == "held_out_annotated_root"
                    else packet["evidence"]
                )
                output["evidence"] = [
                    {
                        "evidence_id": f"C{candidate_index}{evidence_index:02d}",
                        "text": str(evidence["text"]),
                    }
                    for evidence_index, evidence in enumerate(source_evidence)
                ]
                candidates.append(output)
            selected.append(
                {
                    "example_id": "hover-dev-" + str(item["uid"]),
                    "source_split": "hover_dev_joined_to_hotpotqa_distractor_dev",
                    "source_row_id": str(item["uid"]),
                    "hpqa_id": str(item["hpqa_id"]),
                    "split": "formal",
                    "claim": str(item["claim"]),
                    "label": label,
                    "source_native_label": (
                        "SUPPORTED" if label == "Supported" else "NOT_SUPPORTED"
                    ),
                    "gold_binary": int(item["gold_binary"]),
                    "fact_check_root": "hover",
                    "anchor": _packet(anchor, prefix="A"),
                    "candidates": candidates,
                }
            )
            used_claims.add(claim_key)
            used_annotated_roots.update(annotated_roots)
            auxiliary_usage.update(_normalise_root(root) for root in distractor["roots"])
            auxiliary_usage.update(_normalise_root(root) for root in anchor["roots"])
            selected_by_label[label] += 1
            phase_count += 1
            if phase_count == DIRECTION_QUOTA:
                break
        if phase_count != DIRECTION_QUOTA:
            raise ValueError(
                f"only selected {phase_count}/{DIRECTION_QUOTA} {label} {direction} items"
            )
    selected.sort(key=lambda row: str(row["example_id"]))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_qwen36_study_root_call",
        "datasets": {
            "hover": {
                "path": str(HOVER_DATASET),
                "sha256": base._sha256_path(HOVER_DATASET),
            },
            "hotpotqa_context": {
                "path": str(HOTPOT_CONTEXT),
                "sha256": base._sha256_path(HOTPOT_CONTEXT),
                "expected_sha256": HOTPOT_CONTEXT_SHA256,
            },
        },
        "target_model": TARGET_MODEL,
        "root_definition": "HoVer multi-hop bundle of HotpotQA Wikipedia page titles",
        "selection_salt_sha256": sha256(SELECTION_SALT).hexdigest(),
        "selection_boundary": (
            "labels enforce fixed balance only; target outputs are unavailable; all packet "
            "component roots are disjoint from prior formal/development selection manifests"
        ),
        "examples": selected,
    }


def audit_selection(selection: Mapping[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    if rebuild and dict(selection) != build_selection():
        raise ValueError("V3.14 selection or source data drifted")
    rows = list(selection["examples"])
    exposed_claims, exposed_roots = _prior_exposure()
    labels = Counter(str(row["label"]) for row in rows)
    claims = [v34._normalise_claim(str(row["claim"])) for row in rows]
    annotated_components: list[str] = []
    auxiliary_components: list[str] = []
    role_labels: list[int] = []
    role_scores: list[float] = []
    direction_counts: Counter[str] = Counter()
    widths_match = True
    for row in rows:
        annotated = next(
            packet
            for packet in row["candidates"]
            if packet["annotation_role"] == "held_out_annotated_root"
        )
        distractor = next(
            packet
            for packet in row["candidates"]
            if packet["annotation_role"] == "unannotated_retrieval_candidate"
        )
        annotated_components.extend(_normalise_root(root) for root in _packet_roots(annotated))
        for packet in (row["anchor"], distractor):
            auxiliary_components.extend(_normalise_root(root) for root in _packet_roots(packet))
        widths_match &= len(_packet_roots(annotated)) == len(_packet_roots(distractor))
        widths_match &= len(_packet_roots(annotated)) == len(_packet_roots(row["anchor"]))
        direction = (
            "distractor_at_least_annotated"
            if float(distractor["retrieval_score"]) >= float(annotated["retrieval_score"])
            else "distractor_below_annotated"
        )
        direction_counts[f"{row['label']}:{direction}"] += 1
        for packet in row["candidates"]:
            role_labels.append(int(packet["annotation_role"] == "held_out_annotated_root"))
            role_scores.append(float(packet["retrieval_score"]))
    auc = float(roc_auc_score(role_labels, role_scores))
    auxiliary_counts = Counter(auxiliary_components)
    position = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root" for row in rows
    ) / max(1, len(rows))
    all_components = set(annotated_components) | set(auxiliary_components)
    gates = {
        "exact_count": len(rows) == EXPECTED_FORMAL,
        "native_labels_balanced": labels
        == {"Supported": EXPECTED_PER_LABEL, "NotSupported": EXPECTED_PER_LABEL},
        "unique_claims": len(claims) == len(set(claims)),
        "zero_prior_claim_overlap": not (set(claims) & exposed_claims),
        "zero_prior_root_overlap": not (all_components & exposed_roots),
        "annotated_components_unique": len(annotated_components) == len(set(annotated_components)),
        "annotated_and_auxiliary_disjoint": not (
            set(annotated_components) & set(auxiliary_components)
        ),
        "auxiliary_component_reuse_at_most_3": max(auxiliary_counts.values(), default=0)
        <= MAX_AUXILIARY_ROOT_REUSE,
        "bundle_widths_match": bool(widths_match),
        "two_or_three_hop_bundles": all(
            len(_packet_roots(row["anchor"])) in {2, 3} for row in rows
        ),
        "candidate_order_balanced": position == 0.5,
        "retrieval_directions_balanced_within_label": set(direction_counts.values())
        == {DIRECTION_QUOTA}
        and len(direction_counts) == 4,
        "oriented_retrieval_role_auc_at_most_065": max(auc, 1.0 - auc) <= MAX_ORIENTED_ROLE_AUC,
        "retrieval_score_forbidden_from_primary_router": True,
    }
    return {
        "counts": {"formal": len(rows)},
        "labels": dict(labels),
        "distinct_annotated_components": len(set(annotated_components)),
        "distinct_auxiliary_components": len(set(auxiliary_components)),
        "maximum_auxiliary_component_reuse": max(auxiliary_counts.values(), default=0),
        "direction_counts": dict(sorted(direction_counts.items())),
        "candidate_0_annotated_fraction": position,
        "retrieval_role_auc": auc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection() -> bool:
    expected = build_selection()
    audit = audit_selection(expected, rebuild=False)
    if not audit["passed"]:
        raise ValueError("refusing to write a V3.14 selection that fails audit")
    if SELECTION_PATH.exists():
        if json.loads(SELECTION_PATH.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen V3.14 selection drifted")
        return False
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot create V3.14 selection after target records")
    base._write_json(SELECTION_PATH, expected)
    base._write_json(SELECTION_AUDIT_PATH, audit)
    return True


def build_router_manifest() -> dict[str, Any]:
    v311.validate_router_manifest()
    source = json.loads(FROZEN_ROUTER_MANIFEST.read_text(encoding="utf-8"))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_without_qwen36_fit_or_calibration",
        "method": "v3.11_group_robust_dual_head_exact_transfer",
        "source_router_manifest": str(FROZEN_ROUTER_MANIFEST),
        "source_router_manifest_sha256": base._sha256_path(FROZEN_ROUTER_MANIFEST),
        "relation_head": str(FROZEN_ROUTER_HEAD),
        "relation_head_sha256": base._sha256_path(FROZEN_ROUTER_HEAD),
        "training_examples": source["training_examples"],
        "training_target_models": source["training_target_models"],
        "thresholds": source["thresholds"],
        "target_model_absent_from_training_models": TARGET_MODEL
        not in source["training_target_models"],
        "no_refit": True,
        "no_target_calibration": True,
        "retrieval_score_used_by_primary_router": False,
    }


def freeze_router() -> dict[str, Any]:
    if any(DEFAULT_ROOT.glob("**/records*.jsonl")) and not ROUTER_MANIFEST.exists():
        raise ValueError("cannot freeze V3.14 router after target records")
    payload = build_router_manifest()
    if ROUTER_MANIFEST.exists():
        if json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen V3.14 router drifted")
    else:
        base._write_json(ROUTER_MANIFEST, payload)
    return payload


def validate_router_manifest() -> None:
    if json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8")) != build_router_manifest():
        raise ValueError("frozen V3.14 router drifted")


def _target_fingerprint() -> dict[str, Any]:
    if any(not path.is_file() for path in TARGET_SMALL_ARTIFACTS):
        raise ValueError("Qwen3.6 model fingerprint files are incomplete")
    index = json.loads(
        (TARGET_MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    sizes = {name: (TARGET_MODEL_DIR / name).stat().st_size for name in shards}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("Qwen3.6 weight shard is empty")
    return {
        "small_file_sha256": {
            path.name: base._sha256_path(path) for path in TARGET_SMALL_ARTIFACTS
        },
        "weight_shard_sizes_bytes": sizes,
    }


def _validate_router_inputs() -> None:
    metadata = json.loads(ROUTER_INPUTS_METADATA.read_text(encoding="utf-8"))
    if metadata.get("inference_only") is not True:
        raise ValueError("V3.14 router inputs were not encoded inference-only")
    if metadata.get("selection_sha256") != base._sha256_path(SELECTION_PATH):
        raise ValueError("V3.14 router input selection drifted")
    if metadata.get("output_sha256") != base._sha256_path(ROUTER_INPUTS):
        raise ValueError("V3.14 router input artifact drifted")
    arrays = np.load(ROUTER_INPUTS)
    if set(arrays.files) != {"example_ids", "splits", "scores", "relation_vectors"}:
        raise ValueError("V3.14 formal router inputs contain forbidden fields")
    if arrays["scores"].shape != (EXPECTED_FORMAL, 2):
        raise ValueError("V3.14 provenance score shape drifted")
    if arrays["relation_vectors"].shape[:2] != (EXPECTED_FORMAL, 2):
        raise ValueError("V3.14 relation-vector shape drifted")


def build_protocol_manifest() -> dict[str, Any]:
    validate_router_manifest()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    audit = audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("cannot freeze a structurally invalid V3.14 protocol")
    _validate_router_inputs()
    if base._sha256_path(HOTPOT_CONTEXT) != HOTPOT_CONTEXT_SHA256:
        raise ValueError("official HotpotQA context checksum drifted")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_qwen36_study_root_call",
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": base._sha256_path(PREREGISTRATION),
        },
        "implementation": {
            "path": str(_implementation_path()),
            "sha256": base._sha256_path(_implementation_path()),
        },
        "run_script": {"path": str(RUN_SCRIPT), "sha256": base._sha256_path(RUN_SCRIPT)},
        "serve_script": {
            "path": str(SERVE_SCRIPT),
            "sha256": base._sha256_path(SERVE_SCRIPT),
            "runtime": {
                "python_environment": "/DATA/lianjh/miniconda3/envs/casevo",
                "vllm": "0.28.0",
                "transformers": "5.16.1",
                "torch": "2.13.0+cu130",
            },
        },
        "embedding_script": {
            "path": str(EMBED_SCRIPT),
            "sha256": base._sha256_path(EMBED_SCRIPT),
        },
        "datasets": {
            "hover": {"path": str(HOVER_DATASET), "sha256": base._sha256_path(HOVER_DATASET)},
            "hotpotqa_context": {
                "path": str(HOTPOT_CONTEXT),
                "sha256": base._sha256_path(HOTPOT_CONTEXT),
            },
        },
        "prior_selection_sha256": {str(path): base._sha256_path(path) for path in PRIOR_SELECTIONS},
        "selection": {
            "path": str(SELECTION_PATH),
            "sha256": base._sha256_path(SELECTION_PATH),
            "audit": audit,
        },
        "router": {
            "manifest": str(ROUTER_MANIFEST),
            "manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
            "source_manifest_sha256": base._sha256_path(FROZEN_ROUTER_MANIFEST),
            "relation_head_sha256": base._sha256_path(FROZEN_ROUTER_HEAD),
            "inputs_sha256": base._sha256_path(ROUTER_INPUTS),
            "inputs_metadata_sha256": base._sha256_path(ROUTER_INPUTS_METADATA),
            "thresholds": {
                "provenance_score_margin": v311.PROVENANCE_SCORE_MARGIN,
                "relation_confidence_margin": v311.RELATION_CONFIDENCE_MARGIN,
                "target_action_confidence": v311.TARGET_ACTION_CONFIDENCE,
            },
        },
        "target": {
            "model": TARGET_MODEL,
            "endpoint": TARGET_ENDPOINT,
            "model_dir": str(TARGET_MODEL_DIR),
            "artifact_fingerprint": _target_fingerprint(),
            "temperature": 0.0,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "response_format": v310._response_format("action"),
            "schema_repair_attempts": 1,
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
                "annotation_supported_repairs_at_least_10",
                "net_fixes_above_keep_and_all_matched_baselines",
                "provenance_path_accuracy_at_least_090",
                "final_schema_yield_is_one",
                "first_pass_schema_yield_at_least_095",
            ],
        },
        "claim_boundary": {
            "target_fit_or_calibration": False,
            "target_model_used_for_method_selection": False,
            "fresh_claims_and_all_packet_components": True,
            "target_answer_is_always_target_generated": True,
            "cross_dataset_transfer": True,
            "same_qwen_model_family": True,
            "cross_family_transfer": False,
            "publisher_independence": False,
            "universal_transfer": False,
        },
    }


def freeze_protocol() -> dict[str, Any]:
    if not all(
        path.is_file() for path in (PREREGISTRATION, RUN_SCRIPT, SERVE_SCRIPT, EMBED_SCRIPT)
    ):
        raise ValueError("V3.14 protocol documents or scripts are missing")
    target_records = list(DEFAULT_ROOT.glob("**/records*.jsonl"))
    if not PROTOCOL_MANIFEST.exists() and target_records:
        raise ValueError("cannot freeze V3.14 after Qwen3.6 study calls")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        actual = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("frozen V3.14 protocol or dependency drifted")
        return actual
    base._write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol_manifest() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != build_protocol_manifest():
        raise ValueError("frozen V3.14 protocol or dependency drifted")


class TargetSchemaChatClient(v311.HySchemaChatClient):
    """V3.11-compatible structured client bound to the held-out target."""

    def __init__(self, cache_dir: Path, timeout: float = 180.0) -> None:
        self.endpoint = TARGET_ENDPOINT
        self.model = TARGET_MODEL
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_completion_tokens = MAX_COMPLETION_TOKENS
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def endpoint_model_ids() -> set[str]:
    url = TARGET_ENDPOINT.removesuffix("/chat/completions") + "/models"
    with urllib_request.urlopen(url, timeout=10.0) as response:
        payload = json.loads(response.read(1_000_001))
    ids = {
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, Mapping) and "id" in item
    }
    if TARGET_MODEL not in ids:
        raise ValueError(f"expected {TARGET_MODEL!r} in endpoint inventory: {sorted(ids)}")
    return ids


def _validate_actions(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    grouped = base._record_groups(records)
    expected = {str(row["example_id"]): row for row in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} Qwen3.6 action coverage mismatch")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        if len(baseline) != 5 or len(recovery) != 3:
            raise ValueError(f"invalid action bundle for {example_id}")
        if {row.get("agent_index") for row in baseline} != set(range(5)):
            raise ValueError(f"invalid baseline agents for {example_id}")
        if {row.get("action") for row in recovery} != set(base.RECOVERY_ACTIONS):
            raise ValueError(f"invalid recovery actions for {example_id}")
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("runtime_model") != TARGET_MODEL
            or row.get("runtime_endpoint") != TARGET_ENDPOINT
            or row.get("split") != split
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"invalid Qwen3.6 action metadata for {example_id}")


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
        raise ValueError("partial V3.14 file contains an incomplete action bundle")
    existing = {
        example_id: rows
        for example_id, rows in by_example.items()
        if all(row.get("success") for row in rows)
    }
    allowed = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed:
        raise ValueError("partial V3.14 file contains examples outside this run")
    records = [row for rows in existing.values() for row in rows]
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = TargetSchemaChatClient(cache_dir)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(v362._run_action_example, client, example): example
            for example in pending
        }
        for future in as_completed(futures):
            bundle = []
            for source in future.result():
                row = dict(source)
                # Formal route selection receives a deliberately reduced view, but
                # the raw target record also does not need to carry the gold label.
                row.pop("gold_binary", None)
                row.update(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "split": split,
                        "runtime_endpoint": TARGET_ENDPOINT,
                        "runtime_model": TARGET_MODEL,
                    }
                )
                bundle.append(row)
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
        raise ValueError("V3.14 target action run is incomplete")
    _validate_actions(examples, records, split=split)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _formal_examples() -> list[dict[str, Any]]:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["examples"]]


def _inference_features() -> tuple[list[str], np.ndarray, np.ndarray]:
    _validate_router_inputs()
    arrays = np.load(ROUTER_INPUTS)
    ids = [str(value) for value in arrays["example_ids"]]
    if any(str(value) != "formal" for value in arrays["splits"]):
        raise ValueError("V3.14 router inputs contain a non-formal split")
    return ids, arrays["scores"].astype(float), arrays["relation_vectors"].astype(float)


def _public_examples() -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "candidates": [
                {"retrieval_score": float(candidate["retrieval_score"])}
                for candidate in row["candidates"]
            ],
        }
        for row in _formal_examples()
    ]


def _preoutcome_payload() -> dict[str, Any]:
    validate_protocol_manifest()
    ids, scores, relation_vectors = _inference_features()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    public_examples = _public_examples()
    _validate_actions(public_examples, actions, split="formal")
    head = json.loads(FROZEN_ROUTER_HEAD.read_text(encoding="utf-8"))
    relation_probabilities = np.column_stack(
        [v311._probability(head, relation_vectors[:, index]) for index in range(2)]
    )
    primary, diagnostics = v311._select_routes(
        ids, scores, relation_vectors, actions, relation_probabilities
    )
    if set(ids) != {str(row["example_id"]) for row in public_examples}:
        raise ValueError("V3.14 selection and router inputs cover different examples")
    root_budget = sum(action != "KEEP" for action in primary.values())
    action_groups = base._record_groups(v311._safe_action_rows(actions))
    policies: dict[str, dict[str, str]] = {
        "zero_shot_dual_head_repair": primary,
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
        "relation_head_sha256": base._sha256_path(FROZEN_ROUTER_HEAD),
        "router_inputs_sha256": base._sha256_path(ROUTER_INPUTS),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "outcomes_accessed_by_route_selection": False,
        "annotation_roles_accessed_by_route_selection": False,
        "retrieval_scores_accessed_by_primary_router": False,
        "target_fit_or_calibration": False,
        "root_budget": root_budget,
        "policies": policies,
        "diagnostics": diagnostics,
    }


def freeze_routes() -> dict[str, Any]:
    path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    payload = _preoutcome_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen V3.14 preoutcome routes drifted")
    else:
        base._write_json(path, payload)
    return payload


def _policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps: list[int] = []
    finals: list[int] = []
    native_labels: list[str] = []
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
        native_labels.append(str(example["label"]))
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
    for label in LABELS:
        indices = np.asarray(
            [index for index, value in enumerate(native_labels) if value == label], dtype=int
        )
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
        "macro_label_gain": float(np.mean([row["net_gain"] for row in by_label.values()])),
        "macro_gain_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "high_consensus_correct_denominator": high_correct,
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
        raise ValueError("freeze V3.14 routes before accessing formal outcomes")
    preoutcome = json.loads(preoutcome_path.read_text(encoding="utf-8"))
    if preoutcome != _preoutcome_payload():
        raise ValueError("V3.14 preoutcome routes drifted")
    examples = _formal_examples()
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    _validate_actions(examples, actions, split="formal")
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
    primary = metrics["zero_shot_dual_head_repair"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
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
        for example_id, value in preoutcome["diagnostics"].items()
    }
    routed_ids = [
        example_id
        for example_id, action in preoutcome["policies"]["zero_shot_dual_head_repair"].items()
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
    first_pass = sum(bool(row.get("first_pass_valid")) for row in actions)
    transport = {
        "rows": len(actions),
        "successful": sum(bool(row.get("success")) for row in actions),
        "first_pass_valid": first_pass,
        "final_schema_yield": sum(bool(row.get("success")) for row in actions) / len(actions),
        "first_pass_schema_yield": first_pass / len(actions),
        "cache_hits": sum(
            bool(attempt.get("cache_hit")) for row in actions for attempt in row.get("attempts", [])
        ),
    }
    gates = {
        "macro_gain_ci_lower_above_zero": primary["macro_gain_ci"][0] > 0,
        "zero_observed_harms": primary["harms"] == 0,
        "both_label_groups_nonnegative": all(
            row["net_gain"] >= 0 for row in primary["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": primary["annotation_supported_repairs"] >= 10,
        "net_fixes_above_keep_and_all_matched_baselines": primary["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
        "provenance_path_accuracy_at_least_090": provenance["all_examples_accuracy"] >= 0.90,
        "final_schema_yield_is_one": transport["final_schema_yield"] == 1.0,
        "first_pass_schema_yield_at_least_095": transport["first_pass_schema_yield"] >= 0.95,
    }
    passed = all(gates.values())
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "one_shot_zero_target_fit_qwen36_hover_formal",
        "protocol_manifest_sha256": base._sha256_path(PROTOCOL_MANIFEST),
        "router_manifest_sha256": base._sha256_path(ROUTER_MANIFEST),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_formal": len(examples),
        "root_budget": preoutcome["root_budget"],
        "transport_and_schema": transport,
        "policies": metrics,
        "provenance_head": provenance,
        "primary_gates": gates,
        "passes": passed,
        "verdict": (
            "PASS_ZERO_SHOT_QWEN36_HOVER_DUAL_HEAD_TRANSFER_V3_14"
            if passed
            else "NO_VERIFIED_ZERO_SHOT_QWEN36_HOVER_TRANSFER_V3_14"
        ),
        "claim_boundary": build_protocol_manifest()["claim_boundary"],
    }
    output = DEFAULT_ROOT / "evaluation"
    base._write_json(output / "summary.json", summary)
    _write_report(summary, output / "report.md")
    return summary


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    primary = summary["policies"]["zero_shot_dual_head_repair"]
    keep = summary["policies"]["keep"]
    lines = [
        "# Recovery V3.14 result: zero-shot Qwen3.6 model holdout on HoVer",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "| metric | KEEP | V3.14 router |",
        "| --- | ---: | ---: |",
        f"| accuracy | {keep['final_accuracy']:.2%} | {primary['final_accuracy']:.2%} |",
        f"| macro gain | 0.00pp | {100 * primary['macro_label_gain']:+.2f}pp |",
        f"| fixes / harms | 0 / 0 | {primary['fixes']} / {primary['harms']} |",
        f"| routed bundles | 0 | {summary['root_budget']} |",
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
                "The V3.11 relation head and all three thresholds were reused byte-for-byte. "
                "Qwen3.6 supplied only target actions; no Qwen3.6 outcome was available before "
                "the pre-outcome routes were frozen. HoVer is a new task domain, while the "
                "target remains in the Qwen family, so this is not cross-family evidence."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _smoke_examples() -> list[dict[str, Any]]:
    payload = json.loads(v311.SOURCE_SELECTION.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload["examples"] if row["split"] == "development"]
    return rows[:SMOKE_EXAMPLES]


def run_smoke(workers: int) -> dict[str, Any]:
    validate_protocol_manifest()
    endpoint_model_ids()
    rows = execute_actions(
        _smoke_examples(),
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "actions",
        cache_dir=DEFAULT_ROOT / "cache",
        workers=workers,
    )
    result = {
        "rows": len(rows),
        "successful": sum(bool(row["success"]) for row in rows),
        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in rows),
    }
    result["qualified"] = (
        result["rows"] == SMOKE_EXPECTED_ROWS
        and result["successful"] == SMOKE_EXPECTED_ROWS
        and result["first_pass_valid"] >= SMOKE_MIN_FIRST_PASS
    )
    base._write_json(DEFAULT_ROOT / "smoke" / "qualification.json", result)
    if not result["qualified"]:
        raise RuntimeError(f"V3.14 smoke qualification failed: {result}")
    return result


def _validate_smoke_qualification() -> None:
    path = DEFAULT_ROOT / "smoke" / "qualification.json"
    if not path.is_file():
        raise ValueError("V3.14 formal run requires smoke qualification")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("qualified"):
        raise ValueError("V3.14 smoke was not qualified")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-selection")
    subparsers.add_parser("freeze-router")
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
        payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        print(json.dumps({"created": created, **audit_selection(payload)}, indent=2))
        return 0
    if args.command == "freeze-router":
        print(json.dumps(freeze_router(), indent=2))
        return 0
    if args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
        return 0
    validate_protocol_manifest()
    if args.command == "endpoint-check":
        print(json.dumps({"models": sorted(endpoint_model_ids())}, indent=2))
        return 0
    if args.command == "smoke":
        print(json.dumps(run_smoke(args.workers), indent=2))
        return 0
    if args.command == "formal-actions":
        _validate_smoke_qualification()
        records = execute_actions(
            _formal_examples(),
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / "actions",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        print(json.dumps({"rows": len(records)}, indent=2))
        return 0
    if args.command == "freeze-routes":
        print(json.dumps(freeze_routes(), indent=2))
        return 0
    if args.command == "evaluate":
        print(json.dumps(evaluate(), indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
