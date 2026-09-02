"""ELAR: entailment-ledger action routing on an untouched FEVER split."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability.pilot_llm_v1 import _attempt_payload, _extract_json_object
from sp500_forecastability.recovery_v2 import RecoveryChatClient

PROTOCOL_VERSION = "recovery-v3.7.1-elar-fever-train-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_7_1")
DATASET = Path("data/fever/train.jsonl")
DATASET_SHA256 = "bfa7b19109af675e5ce914dd96779b6382ca5c8c60d32c38e26f0072dd32f1c1"
PREREGISTRATION = Path("docs/recovery_v3_7_1_preregistration.md")
V362_SELECTION = Path("results/recovery_v3_6_2/selection_manifest.json")
V362_DEV_ACTIONS = Path("results/recovery_v3_6_2/development/actions/records.jsonl")
V362_DEV_CERTIFICATES = Path("results/recovery_v3_6_2/development/certificates/records.jsonl")
V362_TEST_ACTIONS = Path("results/recovery_v3_6_2/test/actions/records.jsonl")
V362_TEST_CERTIFICATES = Path("results/recovery_v3_6_2/test/certificates/records.jsonl")
V362_ROUTER = Path("results/recovery_v3_6_2/router/manifest.json")
V2_SELECTION = Path("results/recovery_v2_2/selection_manifest.json")

N_FORMAL = 400
N_PER_LABEL = N_FORMAL // 2
ROOT_TARGET_POOL = 800
MAX_RETRIEVAL_GAP = 0.08
MIN_GOLD_RETRIEVAL = 0.08
MAX_ORIENTED_ROLE_AUC = 0.65
SELECTION_SALT = b"elar-v3.7-fever-train-selection-2026-09-02\n"
BOOTSTRAP_SEED = 20_261_102
BOOTSTRAP_REPLICATES = 10_000
LEDGER_SEEDS = {"candidate_0": 20_261_111, "candidate_1": 20_261_121}
LEDGER_MAX_COMPLETION_TOKENS = 768
LEDGER_FIELDS = {"entries", "challenge"}
ENTRY_FIELDS = {
    "atom_index",
    "evidence_id",
    "evidence_quote",
    "semantic_verdict",
    "confidence",
    "unsupported_terms",
}
CHALLENGE_FIELDS = {
    "status",
    "reason_code",
    "claim_span",
    "evidence_id",
    "evidence_quote",
}
SEMANTIC_VERDICTS = {"entailed", "contradicted", "insufficient"}
CHALLENGE_CODES = {
    "none",
    "unsupported_attribute",
    "entity_mismatch",
    "numeric_mismatch",
    "negation_mismatch",
    "relation_mismatch",
    "insufficient_context",
}
CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
LEXICAL_THRESHOLDS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
UNSUPPORTED_TERM_CAPS = (0, 1, 2)
DEPENDENCY_PATHS = (
    Path(__file__).with_name("pilot_llm_v1.py"),
    Path(__file__).with_name("recovery_v2.py"),
    Path(__file__).with_name("recovery_v3.py"),
    Path(__file__).with_name("recovery_v3_4.py"),
    Path(__file__).with_name("recovery_v3_6_2.py"),
)


def _hash_key(namespace: str, value: str) -> str:
    return sha256(SELECTION_SALT + namespace.encode() + b"\0" + value.encode()).hexdigest()


def _normalise_root(value: object) -> str:
    return " ".join(str(value).replace("_", " ").casefold().split())


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _evidence_texts(row: Mapping[str, Any], root: str) -> list[str]:
    texts = []
    for raw in row.get("evidence", []):
        if not isinstance(raw, list) or len(raw) < 3 or str(raw[0]) != root:
            continue
        text = _clean(raw[2])
        if text and text not in texts:
            texts.append(text)
    return texts[:3]


def _historically_exposed() -> tuple[set[str], set[str]]:
    claims: set[str] = set()
    roots: set[str] = set()
    for path in (V2_SELECTION, V362_SELECTION):
        selection = json.loads(path.read_text(encoding="utf-8"))
        for example in selection["examples"]:
            claims.add(v34._normalise_claim(example["claim"]))
            for packet in (example["anchor"], *example["candidates"]):
                roots.add(_normalise_root(packet["root"]))
    for path in sorted(Path("results").glob("pilot_llm_v*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "fever-validation" not in json.dumps(manifest):
            continue
        for example in manifest.get("examples", []):
            for item in example.get("items", []):
                claims.add(v34._normalise_claim(item.get("claim", "")))
                roots.add(_normalise_root(item.get("entity", "")))
    return claims, roots


def load_fever_root_pool(path: Path = DATASET) -> dict[str, list[dict[str, Any]]]:
    if base._sha256_path(path) != DATASET_SHA256:
        raise ValueError("FEVER train checksum drifted")
    exposed_claims, exposed_roots = _historically_exposed()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base._load_jsonl(path):
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
            _normalise_root(root) in exposed_roots
            or v34._normalise_claim(row["claim"]) in exposed_claims
        ):
            continue
        texts = _evidence_texts(row, root)
        if not texts:
            continue
        grouped[root].append(
            {
                "id": str(row["id"]),
                "claim": _clean(row["claim"]),
                "label": str(row["label"]),
                "texts": texts,
            }
        )
    return dict(grouped)


def _root_document(rows: Sequence[Mapping[str, Any]]) -> str:
    texts: list[str] = []
    for row in sorted(rows, key=lambda item: _hash_key("root-doc", str(item["id"])))[:8]:
        for text in row["texts"]:
            if text not in texts:
                texts.append(str(text))
    return " ".join(texts[:8])


def _root_packet_texts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in sorted(rows, key=lambda item: _hash_key("root-packet", str(item["id"]))):
        for text in row["texts"]:
            if text not in texts:
                texts.append(str(text)[:1_200])
            if len(texts) == 3:
                return texts
    return texts


def _evidence_rows(texts: Sequence[str], prefix: str) -> list[dict[str, str]]:
    return [
        {"evidence_id": f"{prefix}{index:02d}", "text": str(text)}
        for index, text in enumerate(texts)
    ]


def _make_formal_examples(pool: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    roots = sorted(pool, key=lambda root: _hash_key("root-partition", root))
    if len(roots) < 3 * N_FORMAL:
        raise ValueError("not enough untouched FEVER roots for globally disjoint packets")
    target_roots = roots[:ROOT_TARGET_POOL]
    source_roots = roots[ROOT_TARGET_POOL:]
    vectorizer = HashingVectorizer(
        n_features=2**16,
        ngram_range=(1, 2),
        stop_words="english",
        alternate_sign=False,
        norm="l2",
    )
    source_documents = [_root_document(pool[root]) for root in source_roots]
    source_matrix = vectorizer.transform(source_documents)
    search = NearestNeighbors(
        n_neighbors=min(300, len(source_roots)), metric="cosine", algorithm="brute"
    ).fit(source_matrix)
    raw_items: list[tuple[str, str, Mapping[str, Any]]] = []
    for root in target_roots:
        for label in ("SUPPORTS", "REFUTES"):
            candidates = sorted(
                (row for row in pool[root] if row["label"] == label),
                key=lambda row: _hash_key("target-item", str(row["id"])),
            )[:3]
            raw_items.extend((root, label, row) for row in candidates)
    claim_matrix = vectorizer.transform([str(item[2]["claim"]) for item in raw_items])
    gold_matrix = vectorizer.transform([" ".join(item[2]["texts"]) for item in raw_items])
    gold_scores = np.asarray(claim_matrix.multiply(gold_matrix).sum(axis=1)).ravel()
    distances, indices = search.kneighbors(claim_matrix)
    selected: list[dict[str, Any]] = []
    used_target_roots: set[str] = set()
    used_source_roots: set[str] = set()
    for label in ("SUPPORTS", "REFUTES"):
        item_indices = [
            index
            for index, item in enumerate(raw_items)
            if item[1] == label and gold_scores[index] >= MIN_GOLD_RETRIEVAL
        ]
        item_indices.sort(key=lambda index: _hash_key("select-item", str(raw_items[index][2]["id"])))
        label_count = 0
        for item_index in item_indices:
            root, _, row = raw_items[item_index]
            if root in used_target_roots:
                continue
            options = []
            for distance, source_index in zip(
                distances[item_index], indices[item_index], strict=True
            ):
                source_root = source_roots[int(source_index)]
                score = 1.0 - float(distance)
                gap = abs(score - float(gold_scores[item_index]))
                if source_root not in used_source_roots and gap <= MAX_RETRIEVAL_GAP:
                    options.append(
                        (gap, _hash_key("source-tie", source_root), source_root, score)
                    )
            options.sort()
            if len(options) < 2:
                continue
            distractor = options[0]
            anchor = next((option for option in options[1:] if option[2] != distractor[2]), None)
            if anchor is None:
                continue
            used_target_roots.add(root)
            used_source_roots.update({distractor[2], anchor[2]})
            annotated = {
                "root": root,
                "annotation_role": "held_out_annotated_root",
                "retrieval_score": float(gold_scores[item_index]),
                "evidence": _evidence_rows(row["texts"], "CG"),
            }
            distractor_packet = {
                "root": distractor[2],
                "annotation_role": "unannotated_retrieval_candidate",
                "retrieval_score": distractor[3],
                "evidence": _evidence_rows(_root_packet_texts(pool[distractor[2]]), "CD"),
            }
            annotated_first = label_count % 2 == 0
            raw_candidates = (
                [annotated, distractor_packet]
                if annotated_first
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
                        "root": anchor[2],
                        "retrieval_score": anchor[3],
                        "evidence": _evidence_rows(_root_packet_texts(pool[anchor[2]]), "A"),
                    },
                    "candidates": candidates,
                }
            )
            label_count += 1
            if label_count == N_PER_LABEL:
                break
        if label_count != N_PER_LABEL:
            raise ValueError(f"only selected {label_count}/{N_PER_LABEL} {label} examples")
    return sorted(selected, key=lambda row: str(row["example_id"]))


def _development_examples() -> list[dict[str, Any]]:
    selection = json.loads(V362_SELECTION.read_text(encoding="utf-8"))
    v362.validate_selection(selection)
    rows = []
    for example in selection["examples"]:
        copied = json.loads(json.dumps(example))
        copied["source_development_fold"] = str(example["split"])
        copied["split"] = "development"
        rows.append(copied)
    return rows


def build_selection() -> dict[str, Any]:
    formal = _make_formal_examples(load_fever_root_pool())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "selection_frozen_before_v3_7_formal_calls",
        "datasets": {
            "formal_fever_train": {"path": str(DATASET), "sha256": DATASET_SHA256},
            "development_v3_6_2_selection": {
                "path": str(V362_SELECTION),
                "sha256": base._sha256_path(V362_SELECTION),
            },
        },
        "model": "Qwen3.5-4B",
        "endpoint": base.RELOCATED_RUNTIME_ENDPOINT,
        "root_definition": "FEVER Wikipedia page title",
        "selection_boundary": (
            "all prior EX-FEVER outcomes are development; formal claims and every packet root "
            "come from untouched FEVER train rows and are globally root-disjoint"
        ),
        "examples": [*_development_examples(), *formal],
    }


def validate_selection(selection: Mapping[str, Any]) -> None:
    if dict(selection) != build_selection():
        raise ValueError("V3.7 selection or source data drifted")


def audit_selection(
    selection: Mapping[str, Any], *, rebuild: bool = True
) -> dict[str, Any]:
    if rebuild:
        validate_selection(selection)
    formal = [row for row in selection["examples"] if row["split"] == "formal"]
    development = [row for row in selection["examples"] if row["split"] == "development"]
    labels = Counter(str(row["label"]) for row in formal)
    formal_claims = {v34._normalise_claim(row["claim"]) for row in formal}
    development_claims = {v34._normalise_claim(row["claim"]) for row in development}
    formal_roots = [
        _normalise_root(packet["root"])
        for row in formal
        for packet in (row["anchor"], *row["candidates"])
    ]
    development_roots = {
        _normalise_root(packet["root"])
        for row in development
        for packet in (row["anchor"], *row["candidates"])
    }
    role, score = [], []
    for row in formal:
        for candidate in row["candidates"]:
            role.append(int(candidate["annotation_role"] == "held_out_annotated_root"))
            score.append(float(candidate["retrieval_score"]))
    auc = float(roc_auc_score(role, score))
    oriented_auc = max(auc, 1.0 - auc)
    position_fraction = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root"
        for row in formal
    ) / len(formal)
    gates = {
        "exact_counts": len(development) == 1_000 and len(formal) == N_FORMAL,
        "formal_labels_balanced": labels == {"Supported": N_PER_LABEL, "Refuted": N_PER_LABEL},
        "zero_development_claim_overlap": not (formal_claims & development_claims),
        "zero_development_root_overlap": not (set(formal_roots) & development_roots),
        "all_formal_packet_roots_globally_unique": len(formal_roots) == len(set(formal_roots)),
        "three_distinct_roots_per_item": all(
            len({row["anchor"]["root"], *(item["root"] for item in row["candidates"])})
            == 3
            for row in formal
        ),
        "candidate_order_balanced": 0.49 <= position_fraction <= 0.51,
        "oriented_retrieval_role_auc_at_most_065": oriented_auc <= MAX_ORIENTED_ROLE_AUC,
        "retrieval_score_forbidden_from_router": True,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "counts": {"development": len(development), "formal": len(formal)},
        "formal_labels": dict(labels),
        "formal_distinct_roots": len(set(formal_roots)),
        "candidate_0_annotated_fraction": position_fraction,
        "retrieval_role_auc": auc,
        "oriented_retrieval_role_auc": oriented_auc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_or_validate_selection(path: Path) -> bool:
    expected = build_selection()
    if path.exists():
        validate_selection(json.loads(path.read_text(encoding="utf-8")))
        return False
    base._write_json(path, expected)
    return True


def materialize_development(selection_path: Path, output_dir: Path) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("cannot materialize from invalid selection")
    action_rows = [
        *base._load_jsonl(V362_DEV_ACTIONS),
        *base._load_jsonl(V362_TEST_ACTIONS),
    ]
    certificate_rows = [
        *base._load_jsonl(V362_DEV_CERTIFICATES),
        *base._load_jsonl(V362_TEST_CERTIFICATES),
    ]
    for row in (*action_rows, *certificate_rows):
        row["source_split"] = row["split"]
        row["split"] = "development"
    expected_ids = {
        str(row["example_id"])
        for row in selection["examples"]
        if row["split"] == "development"
    }
    if {str(row["example_id"]) for row in action_rows} != expected_ids:
        raise ValueError("development action ID mismatch")
    if {str(row["example_id"]) for row in certificate_rows} != expected_ids:
        raise ValueError("development certificate ID mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    base._write_jsonl(output_dir / "actions.jsonl", action_rows)
    base._write_jsonl(output_dir / "certificates.jsonl", certificate_rows)


def _normalised_quote(value: object) -> str:
    return " ".join(str(value).split())


def _token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _required_atom_indices(certificate: Mapping[str, Any], consensus: str) -> list[int]:
    status = "supported" if consensus == "no" else "contradicted"
    return [
        index
        for index, check in enumerate(certificate["atomic_checks"])
        if check["status"] == status
    ]


def build_ledger_messages(
    example: Mapping[str, Any],
    action: str,
    certificate: Mapping[str, Any],
    consensus: str,
    *,
    repair: bool = False,
) -> list[dict[str, str]]:
    candidate = example["candidates"][int(action[-1])]
    evidence = [*example["anchor"]["evidence"], *candidate["evidence"]]
    target = "entailed" if consensus == "no" else "contradicted"
    indices = _required_atom_indices(certificate, consensus)
    atoms = [
        {
            "atom_index": index,
            "claim_span": certificate["atomic_checks"][index]["claim_span"],
            "claimed_status": certificate["atomic_checks"][index]["status"],
            "allowed_evidence_ids": certificate["atomic_checks"][index]["evidence_ids"],
        }
        for index in indices
    ]
    system = (
        "You audit a proposed evidence-path transition. Work only from the packet. For every "
        "listed atom, copy the shortest exact quote that actually establishes the requested "
        f"semantic relation ({target}). Then act as a hostile reviewer: look for an unsupported "
        "attribute, wrong entity, number, negation, or relation direction. Return exactly one "
        "JSON object with entries and challenge. Each entry has exactly atom_index, evidence_id, "
        "evidence_quote, semantic_verdict, confidence, unsupported_terms. semantic_verdict is "
        "entailed, contradicted, or insufficient. unsupported_terms is a list copied from the "
        "atom. unsupported_terms means terms for which the requested relation cannot be "
        "established; a term explicitly contradicted by a mismatch is established, not "
        "unsupported. challenge audits whether the quoted evidence fails to establish the "
        "requested relation. For a requested contradiction, the entity, number, negation, or "
        "relation mismatch that proves the contradiction is not itself a challenge. challenge "
        "has exactly status, reason_code, claim_span, evidence_id, evidence_quote. "
        "status is none or found; reason_code is none, unsupported_attribute, entity_mismatch, "
        "numeric_mismatch, negation_mismatch, relation_mismatch, or insufficient_context. For a "
        "none challenge use empty strings for the last three fields. No outside facts or prose."
    )
    user = (
        f"Claim: {example['claim']}\n"
        f"Previous consensus: {consensus}\n"
        f"Required relation: {target}\n"
        f"Atoms: {json.dumps(atoms, ensure_ascii=False)}\n\n"
        f"PACKET:\n{base._packet(evidence)}"
    )
    if repair:
        user += "\nThe previous output was invalid. Return only the exact required JSON schema."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_ledger(
    content: str,
    example: Mapping[str, Any],
    action: str,
    certificate: Mapping[str, Any],
    consensus: str,
) -> dict[str, Any]:
    payload = _extract_json_object(content)
    if set(payload) != LEDGER_FIELDS:
        raise ValueError("ledger top-level fields mismatch")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("ledger entries must be a list")
    candidate = example["candidates"][int(action[-1])]
    evidence = {
        str(item["evidence_id"]): _normalised_quote(item["text"])
        for item in (*example["anchor"]["evidence"], *candidate["evidence"])
    }
    required = _required_atom_indices(certificate, consensus)
    expected_verdict = "entailed" if consensus == "no" else "contradicted"
    entries = []
    seen: set[int] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != ENTRY_FIELDS:
            raise ValueError("ledger entry fields mismatch")
        atom_index = raw["atom_index"]
        if isinstance(atom_index, bool) or not isinstance(atom_index, int):
            raise TypeError("atom_index must be an integer")
        if atom_index not in required or atom_index in seen:
            raise ValueError("ledger atom index is missing, duplicate, or not decisive")
        check = certificate["atomic_checks"][atom_index]
        evidence_id = str(raw["evidence_id"])
        quote = _normalised_quote(raw["evidence_quote"])
        if evidence_id not in check["evidence_ids"] or evidence_id not in evidence:
            raise ValueError("ledger evidence ID is not certificate-local")
        if not quote or quote.casefold() not in evidence[evidence_id].casefold():
            raise ValueError("ledger quote is not an exact evidence substring")
        verdict = str(raw["semantic_verdict"]).casefold()
        if verdict not in SEMANTIC_VERDICTS:
            raise ValueError("invalid semantic verdict")
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("ledger confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("ledger confidence must be in [0,1]")
        terms = raw["unsupported_terms"]
        if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
            raise TypeError("unsupported_terms must be a list of strings")
        span = str(check["claim_span"])
        span_tokens = _token_set(span)
        quote_tokens = _token_set(quote)
        lexical_coverage = len(span_tokens & quote_tokens) / max(1, len(span_tokens))
        entries.append(
            {
                "atom_index": atom_index,
                "claim_span": span,
                "evidence_id": evidence_id,
                "evidence_quote": quote,
                "semantic_verdict": verdict,
                "expected_verdict": expected_verdict,
                "confidence": confidence,
                "unsupported_terms": [_clean(term) for term in terms if _clean(term)],
                "lexical_coverage": lexical_coverage,
            }
        )
        seen.add(atom_index)
    if seen != set(required):
        raise ValueError("ledger does not cover every decisive certificate atom")
    challenge = payload["challenge"]
    if not isinstance(challenge, Mapping) or set(challenge) != CHALLENGE_FIELDS:
        raise ValueError("challenge fields mismatch")
    status = str(challenge["status"]).casefold()
    reason = str(challenge["reason_code"]).casefold()
    if status not in {"none", "found"} or reason not in CHALLENGE_CODES:
        raise ValueError("invalid challenge status or reason")
    if status == "none" and reason != "none":
        raise ValueError("challenge none must use reason none")
    if status == "found" and reason == "none":
        raise ValueError("found challenge requires a reason")
    return {
        "entries": sorted(entries, key=lambda row: int(row["atom_index"])),
        "challenge": {
            "status": status,
            "reason_code": reason,
            "claim_span": _clean(challenge["claim_span"]),
            "evidence_id": _clean(challenge["evidence_id"]),
            "evidence_quote": _normalised_quote(challenge["evidence_quote"]),
        },
        "all_expected_verdict": all(
            entry["semantic_verdict"] == expected_verdict for entry in entries
        ),
        "min_confidence": min((entry["confidence"] for entry in entries), default=0.0),
        "min_lexical_coverage": min(
            (entry["lexical_coverage"] for entry in entries), default=0.0
        ),
        "unsupported_term_count": sum(len(entry["unsupported_terms"]) for entry in entries),
    }


def _call_ledger_with_retry(
    client: RecoveryChatClient,
    example: Mapping[str, Any],
    action: str,
    certificate: Mapping[str, Any],
    consensus: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    attempts = []
    final_error = None
    for attempt_index in range(2):
        attempt = None
        try:
            result = client.call(
                build_ledger_messages(
                    example,
                    action,
                    certificate,
                    consensus,
                    repair=attempt_index > 0,
                ),
                seed=LEDGER_SEEDS[action],
            )
            attempt = _attempt_payload(result)
            ledger = parse_ledger(result.content, example, action, certificate, consensus)
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
        return ledger, attempts, None
    return None, attempts, final_error


def _proof_candidates(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str, Mapping[str, Any], str]]:
    action_groups = base._record_groups(actions)
    certificate_groups = v362._certificate_groups(certificates)
    candidates = []
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(action_groups[example_id])
        if agreement < base.HIGH_CONSENSUS:
            continue
        for action in v362.CERTIFICATE_ACTIONS:
            row = certificate_groups[(example_id, action)]
            if v362._certificate_gate(row, consensus):
                candidates.append((example, action, row["certificate"], consensus))
    return candidates


def _validate_ledgers(
    candidates: Sequence[tuple[Mapping[str, Any], str, Mapping[str, Any], str]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    expected = {
        (str(example["example_id"]), action)
        for example, action, _certificate, _consensus in candidates
    }
    actual = {(str(row["example_id"]), str(row["action"])) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(f"{split} ledger coverage mismatch")
    for row in records:
        if (
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("split") != split
            or row.get("action") not in v362.CERTIFICATE_ACTIONS
        ):
            raise ValueError("invalid ledger metadata")
        if row.get("success"):
            if row.get("ledger") is None:
                raise ValueError("successful ledger is empty")
        elif (
            row.get("ledger") is not None
            or not row.get("final_error")
            or len(row.get("attempts", [])) != 2
        ):
            raise ValueError("invalid fail-closed ledger row")


def execute_ledgers(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    *,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    candidates = _proof_candidates(examples, actions, certificates)
    if limit is not None:
        candidates = candidates[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    existing = {
        (str(row["example_id"]), str(row["action"])): row for row in loaded
    }
    expected_keys = {
        (str(example["example_id"]), action)
        for example, action, _certificate, _consensus in candidates
    }
    if set(existing) - expected_keys:
        raise ValueError("ledger partial file has candidates outside this run")
    pending = [
        item
        for item in candidates
        if (str(item[0]["example_id"]), item[1]) not in existing
    ]
    records = list(existing.values())
    client = RecoveryChatClient(base.RELOCATED_RUNTIME_ENDPOINT, "Qwen3.5-4B", cache_dir)
    client.max_completion_tokens = LEDGER_MAX_COMPLETION_TOKENS

    def run_one(
        item: tuple[Mapping[str, Any], str, Mapping[str, Any], str]
    ) -> dict[str, Any]:
        example, action, certificate, consensus = item
        ledger, attempts, final_error = _call_ledger_with_retry(
            client, example, action, certificate, consensus
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "runtime_endpoint": client.endpoint,
            "example_id": str(example["example_id"]),
            "split": split,
            "action": action,
            "success": ledger is not None,
            "first_pass_valid": ledger is not None and len(attempts) == 1,
            "attempts": attempts,
            "ledger": ledger,
            "final_error": final_error,
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, item): item for item in pending}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            records.sort(key=lambda item: (str(item["example_id"]), str(item["action"])))
            base._write_jsonl(partial_path, records)
            print(
                f"[{len(records)}/{len(candidates)}] {row['example_id']}:{row['action']} "
                f"success={row['success']} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    _validate_ledgers(candidates, records, split=split)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _wrapped_action_example(
    client: RecoveryChatClient, example: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = v362._run_action_example(client, example)
    for row in rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["split"] = "formal"
    return rows


def _wrapped_certificate_example(
    client: RecoveryChatClient, example: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = v362._run_certificate_example(client, example)
    for row in rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["split"] = "formal"
    return rows


def _validate_formal_actions(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> None:
    grouped = base._record_groups(records)
    expected = {str(example["example_id"]): example for example in examples}
    if set(grouped) != set(expected):
        raise ValueError("formal action coverage mismatch")
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
            or row.get("split") != "formal"
            or row.get("gold_binary") != expected[example_id]["gold_binary"]
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"invalid formal action metadata for {example_id}")


def _validate_formal_certificates(
    examples: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> None:
    grouped = base._record_groups(records)
    expected = {str(example["example_id"]) for example in examples}
    if set(grouped) != expected:
        raise ValueError("formal certificate coverage mismatch")
    for example_id, rows in grouped.items():
        if len(rows) != 2 or {row.get("action") for row in rows} != set(
            v362.CERTIFICATE_ACTIONS
        ):
            raise ValueError(f"invalid certificate bundle for {example_id}")
        for row in rows:
            if row.get("protocol_version") != PROTOCOL_VERSION or row.get("split") != "formal":
                raise ValueError("invalid formal certificate metadata")
            if row.get("success") and row.get("certificate") is None:
                raise ValueError("successful certificate is empty")
            if not row.get("success") and (
                row.get("certificate") is not None
                or not row.get("final_error")
                or len(row.get("attempts", [])) != 2
            ):
                raise ValueError("invalid fail-closed certificate")


def execute_formal_bundles(
    examples: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None:
        examples = list(examples)[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    expected_per_example = 2 if kind == "certificates" else 8
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_example[str(row["example_id"])].append(row)
    if any(len(rows) != expected_per_example for rows in by_example.values()):
        raise ValueError("partial file contains an incomplete formal bundle")
    if kind == "certificates":
        terminal = lambda rows: all(
            row.get("success")
            or (
                row.get("certificate") is None
                and row.get("final_error")
                and len(row.get("attempts", [])) == 2
            )
            for row in rows
        )
        run_one = _wrapped_certificate_example
    elif kind == "actions":
        terminal = lambda rows: all(row.get("success") for row in rows)
        run_one = _wrapped_action_example
    else:
        raise ValueError("kind must be actions or certificates")
    existing = {key: rows for key, rows in by_example.items() if terminal(rows)}
    records = [row for rows in existing.values() for row in rows]
    allowed = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed:
        raise ValueError("partial file contains examples outside formal run")
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = RecoveryChatClient(base.RELOCATED_RUNTIME_ENDPOINT, "Qwen3.5-4B", cache_dir)
    if kind == "certificates":
        client.max_completion_tokens = v362.ATOMIC_MAX_COMPLETION_TOKENS
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, client, row): row for row in pending}
        for future in as_completed(futures):
            bundle = future.result()
            records.extend(bundle)
            records.sort(
                key=lambda row: (
                    str(row["example_id"]),
                    str(row.get("phase", "certificate")),
                    -1 if row.get("agent_index") is None else int(row["agent_index"]),
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
    if len(records) != len(examples) * expected_per_example:
        raise ValueError("formal run incomplete")
    if kind == "certificates":
        _validate_formal_certificates(examples, records)
    else:
        _validate_formal_actions(examples, records)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _prior_predictions(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[float, float]]:
    manifest = json.loads(V362_ROUTER.read_text(encoding="utf-8"))
    names, matrix, keys = v362._feature_matrix(examples, actions, certificates)
    if names != manifest["feature_names"]:
        raise ValueError("prior PACE feature schema drifted")
    bundle = joblib.load(manifest["router_joblib"])
    fix, harm = v362._predict_models(bundle["models"], matrix)
    return v362._prediction_map(keys, fix, harm)


def _ledger_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["example_id"]), str(row["action"])): row for row in records}


def _ledger_gate(
    row: Mapping[str, Any] | None,
    *,
    confidence_threshold: float,
    lexical_threshold: float,
    unsupported_term_cap: int,
) -> bool:
    if row is None or not row.get("success") or row.get("ledger") is None:
        return False
    ledger = row["ledger"]
    return bool(
        ledger["all_expected_verdict"]
        and ledger["challenge"]["status"] == "none"
        and float(ledger["min_confidence"]) >= confidence_threshold
        and float(ledger["min_lexical_coverage"]) >= lexical_threshold
        and int(ledger["unsupported_term_count"]) <= unsupported_term_cap
    )


def _select_elar(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    certificate_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    ledger_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    *,
    confidence_threshold: float,
    lexical_threshold: float,
    unsupported_term_cap: int,
    require_ledger: bool = True,
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(action_groups[example_id])
        allowed = []
        if agreement >= base.HIGH_CONSENSUS:
            for action in v362.CERTIFICATE_ACTIONS:
                certificate_row = certificate_groups[(example_id, action)]
                if not v362._certificate_gate(certificate_row, consensus):
                    continue
                ledger_row = ledger_groups.get((example_id, action))
                if require_ledger and not _ledger_gate(
                    ledger_row,
                    confidence_threshold=confidence_threshold,
                    lexical_threshold=lexical_threshold,
                    unsupported_term_cap=unsupported_term_cap,
                ):
                    continue
                ledger = ledger_row.get("ledger", {}) if ledger_row else {}
                p_fix, p_harm = predictions[(example_id, action)]
                # Existing PACE is retained as a conflict arbiter, not as a hard veto.
                allowed.append(
                    (
                        float(ledger.get("min_confidence", 0.0)),
                        float(ledger.get("min_lexical_coverage", 0.0)),
                        p_fix - p_harm,
                        p_fix,
                        -p_harm,
                        action,
                    )
                )
        selected[example_id] = max(allowed)[-1] if allowed else "KEEP"
    return selected


def _development_metrics(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    return v362._basic_metrics(examples, action_groups, selected)


def tune_ledger_policy(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    ledgers: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[float, float]]]:
    action_groups = base._record_groups(actions)
    certificate_groups = v362._certificate_groups(certificates)
    ledger_groups = _ledger_groups(ledgers)
    predictions = _prior_predictions(examples, actions, certificates)
    source_folds = {
        name: [row for row in examples if row["source_development_fold"] == name]
        for name in ("development", "test")
    }
    feasible = []
    for confidence in CONFIDENCE_THRESHOLDS:
        for lexical in LEXICAL_THRESHOLDS:
            for term_cap in UNSUPPORTED_TERM_CAPS:
                fold_metrics = {}
                for name, fold_examples in source_folds.items():
                    selected = _select_elar(
                        fold_examples,
                        action_groups,
                        certificate_groups,
                        ledger_groups,
                        predictions,
                        confidence_threshold=confidence,
                        lexical_threshold=lexical,
                        unsupported_term_cap=term_cap,
                    )
                    fold_metrics[name] = _development_metrics(
                        fold_examples, action_groups, selected
                    )
                if all(
                    metric["damage_rate"] <= 0.05
                    and min(metric["by_label_gain"].values()) >= 0.0
                    and metric["net_fixes"] >= 5
                    and metric["routes"] >= 5
                    and metric["annotation_supported_repairs"] >= 5
                    for metric in fold_metrics.values()
                ):
                    feasible.append(
                        (
                            min(metric["macro_gain"] for metric in fold_metrics.values()),
                            min(metric["net_fixes"] for metric in fold_metrics.values()),
                            sum(metric["net_fixes"] for metric in fold_metrics.values()),
                            -sum(metric["harms"] for metric in fold_metrics.values()),
                            -sum(metric["routes"] for metric in fold_metrics.values()),
                            confidence,
                            lexical,
                            -term_cap,
                            fold_metrics,
                        )
                    )
    if not feasible:
        raise ValueError("no two-source-fold-safe ELAR policy; formal calls remain locked")
    best = max(feasible, key=lambda item: item[:8])
    return (
        {
            "confidence_threshold": best[5],
            "lexical_threshold": best[6],
            "unsupported_term_cap": -best[7],
            "development_fold_metrics": best[8],
            "feasible_configurations": len(feasible),
        },
        predictions,
    )


def fit_router(selection_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_router_manifest(manifest_path, selection_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if any((DEFAULT_ROOT / "formal" / name / "records.jsonl").exists() for name in (
        "actions",
        "certificates",
        "ledgers",
    )):
        raise ValueError("cannot fit or overwrite ELAR after formal calls")
    if not PREREGISTRATION.exists():
        raise ValueError("V3.7 preregistration must exist before fitting")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.7 structural audit failed")
    examples = [row for row in selection["examples"] if row["split"] == "development"]
    actions_path = DEFAULT_ROOT / "development" / "actions.jsonl"
    certificates_path = DEFAULT_ROOT / "development" / "certificates.jsonl"
    ledgers_path = DEFAULT_ROOT / "development" / "ledgers" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    certificates = base._load_jsonl(certificates_path)
    ledgers = base._load_jsonl(ledgers_path)
    candidates = _proof_candidates(examples, actions, certificates)
    _validate_ledgers(candidates, ledgers, split="development")
    policy, _predictions = tune_ledger_policy(examples, actions, certificates, ledgers)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_v3_7_formal_model_call",
        "selection_sha256": base._sha256_path(selection_path),
        "implementation_path": str(Path(__file__)),
        "implementation_sha256": base._sha256_path(Path(__file__)),
        "dependency_sha256": {
            str(path): base._sha256_path(path) for path in DEPENDENCY_PATHS
        },
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "development_action_records_sha256": base._sha256_path(actions_path),
        "development_certificate_records_sha256": base._sha256_path(certificates_path),
        "development_ledger_records_sha256": base._sha256_path(ledgers_path),
        "prior_pace_manifest": str(V362_ROUTER),
        "prior_pace_manifest_sha256": base._sha256_path(V362_ROUTER),
        "policy": policy,
        "feature_boundary": {
            "uses_formal_gold_or_action_outcomes_at_inference": False,
            "uses_formal_annotation_role_at_inference": False,
            "uses_source_identity_or_retrieval_score_at_inference": False,
            "uses_atomic_certificate_exact_quote_ledger_and_baseline_consensus": True,
            "prior_pace_predictions_used_only_for_multi_candidate_tie_break": True,
            "any_certificate_or_ledger_failure_is_keep": True,
        },
        "claim_boundary": {
            "formal_outcomes_seen": False,
            "single_qwen_deployment": True,
            "static_wikipedia_page_roots": True,
            "publisher_independence": False,
            "cross_model": False,
        },
    }
    base._write_json(manifest_path, manifest)
    return manifest


def validate_router_manifest(manifest_path: Path, selection_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("ELAR protocol mismatch")
    if manifest.get("status") != "frozen_before_any_v3_7_formal_model_call":
        raise ValueError("ELAR router was not frozen before formal calls")
    if manifest.get("selection_sha256") != base._sha256_path(selection_path):
        raise ValueError("ELAR selection drifted")
    implementation = Path(str(manifest["implementation_path"]))
    if manifest.get("implementation_sha256") != base._sha256_path(implementation):
        raise ValueError("ELAR implementation drifted")
    dependencies = {str(path): base._sha256_path(path) for path in DEPENDENCY_PATHS}
    if manifest.get("dependency_sha256") != dependencies:
        raise ValueError("ELAR dependencies drifted")
    if manifest.get("preregistration_sha256") != base._sha256_path(PREREGISTRATION):
        raise ValueError("ELAR preregistration drifted")
    if manifest.get("prior_pace_manifest_sha256") != base._sha256_path(V362_ROUTER):
        raise ValueError("prior PACE manifest drifted")
    boundary = manifest.get("feature_boundary", {})
    if boundary.get("uses_formal_gold_or_action_outcomes_at_inference") is not False:
        raise ValueError("formal outcomes entered ELAR inference")
    if boundary.get("uses_formal_annotation_role_at_inference") is not False:
        raise ValueError("formal annotation role entered ELAR inference")


def _policy_metrics(
    examples: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    keeps, finals, labels, high_flags = [], [], [], []
    actions = Counter()
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
        actions[action] += 1
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
    by_label, label_indices = {}, {}
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
    return {
        "n": len(examples),
        "baseline_accuracy": float(keep_array.mean()),
        "final_accuracy": float(final_array.mean()),
        "fixes": int(((keep_array == 0) & (final_array == 1)).sum()),
        "harms": int(((keep_array == 1) & (final_array == 0)).sum()),
        "net_fixes": int(gains.sum()),
        "net_gain": float(gains.mean()),
        "macro_label_gain": float(np.mean([group["net_gain"] for group in by_label.values()])),
        "macro_gain_ci": np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist(),
        "damage_rate_high_consensus_correct": high_harms / max(1, high_correct),
        "annotation_supported_repairs": annotation_supported,
        "total_added_roots": sum(action != "KEEP" for action in selected.values()),
        "selected_actions": dict(actions),
        "by_native_label": by_label,
    }


def evaluate(
    selection_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not audit_selection(selection)["passed"]:
        raise ValueError("V3.7 selection failed evaluation audit")
    validate_router_manifest(manifest_path, selection_path)
    examples = [row for row in selection["examples"] if row["split"] == "formal"]
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    certificates_path = DEFAULT_ROOT / "formal" / "certificates" / "records.jsonl"
    ledgers_path = DEFAULT_ROOT / "formal" / "ledgers" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    certificates = base._load_jsonl(certificates_path)
    ledgers = base._load_jsonl(ledgers_path)
    _validate_formal_actions(examples, actions)
    _validate_formal_certificates(examples, certificates)
    candidates = _proof_candidates(examples, actions, certificates)
    _validate_ledgers(candidates, ledgers, split="formal")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    action_groups = base._record_groups(actions)
    certificate_groups = v362._certificate_groups(certificates)
    ledger_groups = _ledger_groups(ledgers)
    predictions = _prior_predictions(examples, actions, certificates)
    parameters = manifest["policy"]
    primary = _select_elar(
        examples,
        action_groups,
        certificate_groups,
        ledger_groups,
        predictions,
        confidence_threshold=float(parameters["confidence_threshold"]),
        lexical_threshold=float(parameters["lexical_threshold"]),
        unsupported_term_cap=int(parameters["unsupported_term_cap"]),
    )
    proof_only = _select_elar(
        examples,
        action_groups,
        certificate_groups,
        ledger_groups,
        predictions,
        confidence_threshold=0.0,
        lexical_threshold=0.0,
        unsupported_term_cap=10**9,
        require_ledger=False,
    )
    policies = {
        "elar": primary,
        "atomic_proof_only": proof_only,
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    root_budget = sum(action != "KEEP" for action in primary.values())
    for name, proposed in v34._comparison_proposals(examples, action_groups).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            examples, proposed, root_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    preoutcome = {
        "protocol_version": PROTOCOL_VERSION,
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "formal_action_records_sha256": base._sha256_path(actions_path),
        "formal_certificate_records_sha256": base._sha256_path(certificates_path),
        "formal_ledger_records_sha256": base._sha256_path(ledgers_path),
        "outcomes_accessed_by_route_selection": False,
        "policies": policies,
        "predictions": {
            f"{example_id}:{action}": {"p_fix": values[0], "p_harm": values[1]}
            for (example_id, action), values in predictions.items()
        },
    }
    v34._write_or_validate_preoutcome(preoutcome_path, preoutcome)
    oracle = {}
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(
            example, action_groups[example_id]
        )
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
        name: _policy_metrics(examples, action_groups, selected)
        for name, selected in policies.items()
    }
    primary_metrics = metrics["elar"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    gates = {
        "macro_gain_ci_lower_above_zero": primary_metrics["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": (
            primary_metrics["damage_rate_high_consensus_correct"] <= 0.05
        ),
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0
            for group in primary_metrics["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": (
            primary_metrics["annotation_supported_repairs"] >= 10
        ),
        "net_fixes_above_keep_and_all_matched_baselines": primary_metrics["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
    }
    routed = [row for row in examples if primary[str(row["example_id"])] != "KEEP"]
    annotated_choices = sum(
        row["candidates"][int(primary[str(row["example_id"])][-1])]["annotation_role"]
        == "held_out_annotated_root"
        for row in routed
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "prospective_root_disjoint_fever_train_formal",
        "selection_sha256": base._sha256_path(selection_path),
        "router_manifest_sha256": base._sha256_path(manifest_path),
        "formal_action_records_sha256": base._sha256_path(actions_path),
        "formal_certificate_records_sha256": base._sha256_path(certificates_path),
        "formal_ledger_records_sha256": base._sha256_path(ledgers_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_formal": len(examples),
        "root_budget": root_budget,
        "ledger_validity": {
            "proof_candidates": len(candidates),
            "valid_rows": sum(bool(row.get("success")) for row in ledgers),
            "fail_closed_rows": sum(not bool(row.get("success")) for row in ledgers),
        },
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": "PASS_ENTAILMENT_LEDGER_ACTION_ROUTING_V3_7"
        if all(gates.values())
        else "NO_VERIFIED_ELAR_DOMINANCE",
        "annotation_role_selection": {
            "routed": len(routed),
            "annotated_root_selected": annotated_choices,
            "accuracy": annotated_choices / max(1, len(routed)),
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    base._write_json(output_dir / "summary.json", summary)
    return summary


def _selection_examples(selection_path: Path, split: str) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return [row for row in selection["examples"] if row["split"] == split]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("materialize-development")
    dev_ledger = subparsers.add_parser("dev-ledgers")
    dev_ledger.add_argument("--workers", type=int, default=8)
    dev_ledger.add_argument("--limit", type=int)
    subparsers.add_parser("fit")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--workers", type=int, default=2)
    for name in ("formal-certificates", "formal-actions", "formal-ledgers"):
        item = subparsers.add_parser(name)
        item.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    selection_path = DEFAULT_ROOT / "selection_manifest.json"
    manifest_path = DEFAULT_ROOT / "router" / "manifest.json"
    if args.command == "prepare":
        write_or_validate_selection(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        audit = audit_selection(selection, rebuild=False)
        base._write_json(DEFAULT_ROOT / "selection_audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["passed"] else 2
    if args.command == "materialize-development":
        materialize_development(selection_path, DEFAULT_ROOT / "development")
        return 0
    if args.command == "dev-ledgers":
        examples = _selection_examples(selection_path, "development")
        execute_ledgers(
            examples,
            base._load_jsonl(DEFAULT_ROOT / "development" / "actions.jsonl"),
            base._load_jsonl(DEFAULT_ROOT / "development" / "certificates.jsonl"),
            split="development",
            output_dir=DEFAULT_ROOT / "development" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
            limit=args.limit,
        )
        return 0
    if args.command == "fit":
        manifest = fit_router(selection_path, DEFAULT_ROOT / "router")
        print(json.dumps(manifest["policy"], indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        validate_router_manifest(manifest_path, selection_path)
        examples = _selection_examples(selection_path, "formal")[:2]
        actions = execute_formal_bundles(
            examples,
            kind="actions",
            output_dir=DEFAULT_ROOT / "smoke" / "actions",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        certificates = execute_formal_bundles(
            examples,
            kind="certificates",
            output_dir=DEFAULT_ROOT / "smoke" / "certificates",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        execute_ledgers(
            examples,
            actions,
            certificates,
            split="formal",
            output_dir=DEFAULT_ROOT / "smoke" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command in {"formal-actions", "formal-certificates"}:
        validate_router_manifest(manifest_path, selection_path)
        kind = args.command.removeprefix("formal-")
        execute_formal_bundles(
            _selection_examples(selection_path, "formal"),
            kind=kind,
            output_dir=DEFAULT_ROOT / "formal" / kind,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "formal-ledgers":
        validate_router_manifest(manifest_path, selection_path)
        examples = _selection_examples(selection_path, "formal")
        execute_ledgers(
            examples,
            base._load_jsonl(DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"),
            base._load_jsonl(DEFAULT_ROOT / "formal" / "certificates" / "records.jsonl"),
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate(selection_path, manifest_path, DEFAULT_ROOT / "evaluation")
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
