"""Frozen Pilot-LLM V1 paired-intervention runner.

The runner uses only the Python standard library for the OpenAI-compatible HTTP
request so it does not download an SDK or model.  Formal settings are fixed in
``docs/pilot_llm_v1_preregistration.md``; smoke mode is deliberately capped at
two questions, one agent, and six calls.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time
import tokenize
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from math import ceil, isfinite
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from sklearn.metrics import average_precision_score, roc_auc_score

from sp500_forecastability.metrics import brier_score, expected_calibration_error

PROTOCOL_VERSION = "pilot-llm-v1-2026-08-31"
EXPECTED_DATASET_SHA256 = "be345e0a08ea87fb5d1642c076bfb8bb186efd1cc39f9c07a812a26eef606760"
DEFAULT_DATASET = Path(
    "/storage/gaoym/argumentative-llms/Datasets/StrategyQA/Prompt/data.jsonl"
)
DEFAULT_ENDPOINT = "http://10.63.0.88:31519/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.5-4B"
DEFAULT_ROOT = Path("results/pilot_llm_v1")
FORMAL_EXAMPLES = 50
FORMAL_PER_LABEL = 25
CONDITIONS = ("original", "remove", "reverse")
MAX_COMPLETION_TOKENS = 160
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_260_831
MAX_RESPONSE_BYTES = 1_000_000
REPAIR_SUFFIX = (
    "\n\nYour previous response violated the required JSON contract. "
    "Return only the exact JSON object, with no explanation or extra fields."
)

AGENT_PERSONAS: tuple[tuple[str, str], ...] = (
    (
        "literal_evidence",
        "Apply the task-local evidence literally and abstain when it cannot support a judgment.",
    ),
    (
        "skeptical_auditor",
        "Audit whether the available evidence is sufficient before making a cautious judgment.",
    ),
    (
        "consistency_checker",
        "Check the claim and all available evidence for logical consistency before deciding.",
    ),
    (
        "counterfactual_reasoner",
        "Honor task-local counterfactual evidence even when it conflicts with ordinary knowledge.",
    ),
    (
        "minimal_judge",
        "Make the shortest defensible binary judgment supported by the available evidence.",
    ),
)

_DECISION_FIELDS = {
    "agent_id",
    "decision",
    "answer",
    "confidence",
    "cited_evidence_ids",
}


@dataclass(frozen=True)
class StrategyQAExample:
    """One local StrategyQA item with facts parsed from its serialized array."""

    qid: str
    question: str
    claim: str
    label: bool
    facts: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceView:
    """Condition-specific environment evidence shown to an agent."""

    condition: str
    items: tuple[tuple[str, str], ...]

    @property
    def allowed_evidence_ids(self) -> tuple[str, ...]:
        return tuple(evidence_id for evidence_id, _ in self.items)


@dataclass(frozen=True)
class QADecision:
    """Strict observable response contract; no rationale or chain-of-thought."""

    agent_id: str
    decision: str
    answer: str | None
    confidence: float
    cited_evidence_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "decision": self.decision,
            "answer": self.answer,
            "confidence": self.confidence,
            "cited_evidence_ids": list(self.cited_evidence_ids),
        }


@dataclass(frozen=True)
class ChatResult:
    """Minimal cached response and transfer accounting."""

    content: str
    model: str
    usage: Mapping[str, int | None]
    http_status: int
    request_bytes: int
    response_bytes: int
    latency_seconds: float
    cache_hit: bool
    cache_key: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_serialized_string_array(value: object) -> tuple[str, ...]:
    """Parse numpy-style string-array text without executing it.

    StrategyQA's JSONL stores facts as a string such as
    ``['first fact'\n 'second fact']``.  ``ast.literal_eval`` alone would join
    adjacent literals; tokenizing lets us recover each fact separately.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("serialized string array must be non-empty text")
    try:
        tokens = tokenize.generate_tokens(StringIO(value).readline)
        strings = [
            ast.literal_eval(token.string)
            for token in tokens
            if token.type == tokenize.STRING
        ]
    except (SyntaxError, tokenize.TokenError, ValueError) as error:
        raise ValueError("could not parse serialized string array") from error
    normalized = tuple(str(item).strip() for item in strings if str(item).strip())
    if not normalized:
        raise ValueError("serialized string array contains no string items")
    return normalized


def load_strategyqa(path: Path, *, verify_digest: bool = True) -> list[StrategyQAExample]:
    """Load the frozen local JSONL and reject ambiguous identifiers or labels."""

    if verify_digest and file_sha256(path) != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "StrategyQA source digest does not match Pilot-LLM V1; create a new protocol version"
        )
    examples: list[StrategyQAExample] = []
    seen_qids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on dataset line {line_number}") from error
            if not isinstance(payload, Mapping):
                raise TypeError(f"dataset line {line_number} must be an object")
            qid = str(payload.get("qid", "")).strip()
            if not qid or qid in seen_qids:
                raise ValueError(f"empty or duplicate qid on dataset line {line_number}")
            label = payload.get("valid")
            if not isinstance(label, bool):
                raise TypeError(f"valid must be boolean on dataset line {line_number}")
            question = str(payload.get("question", "")).strip()
            claim = str(payload.get("claim", "")).strip()
            if not question or not claim:
                raise ValueError(f"missing question or claim on dataset line {line_number}")
            examples.append(
                StrategyQAExample(
                    qid=qid,
                    question=question,
                    claim=claim,
                    label=label,
                    facts=parse_serialized_string_array(payload.get("facts")),
                )
            )
            seen_qids.add(qid)
    return examples


def _selection_digest(qid: str) -> str:
    return sha256(f"{PROTOCOL_VERSION}\n{qid}".encode()).hexdigest()


def select_frozen_examples(
    examples: Sequence[StrategyQAExample], *, per_label: int = FORMAL_PER_LABEL
) -> list[StrategyQAExample]:
    """Select a deterministic, label-balanced sample before model calls."""

    selected: list[StrategyQAExample] = []
    for label in (False, True):
        stratum = sorted(
            (example for example in examples if example.label is label),
            key=lambda example: (_selection_digest(example.qid), example.qid),
        )
        if len(stratum) < per_label:
            raise ValueError(f"not enough examples for label={label}")
        selected.extend(stratum[:per_label])
    return sorted(selected, key=lambda example: (_selection_digest(example.qid), example.qid))


def build_manifest(dataset_path: Path) -> dict[str, object]:
    examples = select_frozen_examples(load_strategyqa(dataset_path))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "selection": {
            "per_label": FORMAL_PER_LABEL,
            "total": FORMAL_EXAMPLES,
            "salt": PROTOCOL_VERSION,
        },
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "conditions": list(CONDITIONS),
        "agents": [agent_id for agent_id, _ in AGENT_PERSONAS],
        "examples": [
            {
                "qid": example.qid,
                "label": example.label,
                "fact_count": len(example.facts),
                "selection_sha256": _selection_digest(example.qid),
            }
            for example in examples
        ],
    }


def validate_manifest(
    manifest: Mapping[str, object], dataset_path: Path
) -> list[StrategyQAExample]:
    """Recompute the frozen sample and reject manifest or source drift."""

    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manifest protocol_version does not match Pilot-LLM V1")
    if manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("manifest dataset digest does not match Pilot-LLM V1")
    if manifest.get("model") != DEFAULT_MODEL or manifest.get("endpoint") != DEFAULT_ENDPOINT:
        raise ValueError("manifest model or endpoint does not match Pilot-LLM V1")
    if manifest.get("conditions") != list(CONDITIONS):
        raise ValueError("manifest conditions do not match Pilot-LLM V1")
    if manifest.get("agents") != [agent_id for agent_id, _ in AGENT_PERSONAS]:
        raise ValueError("manifest agents do not match Pilot-LLM V1")

    expected = select_frozen_examples(load_strategyqa(dataset_path))
    expected_rows = [
        {
            "qid": example.qid,
            "label": example.label,
            "fact_count": len(example.facts),
            "selection_sha256": _selection_digest(example.qid),
        }
        for example in expected
    ]
    if manifest.get("examples") != expected_rows:
        raise ValueError("manifest examples do not match the recomputed frozen sample")
    if Counter(example.label for example in expected) != Counter({False: 25, True: 25}):
        raise ValueError("manifest must contain 25 examples per label")
    return expected


def evidence_view(example: StrategyQAExample, condition: str) -> EvidenceView:
    """Create one frozen task-local intervention without looking at the label."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if condition == "remove":
        return EvidenceView(condition=condition, items=())
    items = []
    for index, fact in enumerate(example.facts, start=1):
        text = fact
        if condition == "reverse":
            text = f"Task-local counterfactual: it is false that: {fact}"
        items.append((f"E{index:02d}", text))
    return EvidenceView(condition=condition, items=tuple(items))


def build_messages(
    example: StrategyQAExample,
    view: EvidenceView,
    *,
    agent_id: str,
    persona: str,
    repair: bool = False,
) -> list[dict[str, str]]:
    """Build a stateless prompt that exposes neither label nor condition name."""

    system = (
        "You are a binary evidence judge in a provenance audit. "
        f"Your fixed decision style is: {persona} "
        "Treat packet statements as task-local evidence, including counterfactual statements. "
        "You may use general knowledge when evidence is insufficient, but cite only packet IDs. "
        "Do not provide reasoning, analysis, or chain-of-thought. Return only one JSON object."
    )
    evidence_packet = [
        {"evidence_id": evidence_id, "text": text} for evidence_id, text in view.items
    ]
    user_payload = {
        "question": example.question,
        "claim": example.claim,
        "evidence_packet": evidence_packet,
    }
    user = (
        "Judge whether the claim is true under the task-local evidence. "
        "Use decision='answer' with answer='yes' or 'no', or decision='abstain' with "
        "answer=null and confidence=0.0. Confidence is the probability that your yes/no answer "
        "is correct. cited_evidence_ids must be a unique list drawn only from the packet.\n\n"
        f"agent_id must be exactly: {agent_id}\n"
        "Required keys, with no others: agent_id, decision, answer, confidence, "
        "cited_evidence_ids.\n\n"
        f"Task payload:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json_object(content: str) -> Mapping[str, object]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("unterminated Markdown JSON fence")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("response must be exactly one JSON object") from error
    if not isinstance(payload, Mapping):
        raise TypeError("response JSON must be an object")
    return payload


def parse_qa_decision(
    content: str, *, expected_agent_id: str, allowed_evidence_ids: Sequence[str]
) -> QADecision:
    """Strictly validate observable output against the current evidence packet."""

    payload = _extract_json_object(content)
    unknown = set(payload) - _DECISION_FIELDS
    missing = _DECISION_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"unexpected decision fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing decision fields: {sorted(missing)}")

    agent_id = payload["agent_id"]
    if not isinstance(agent_id, str) or agent_id != expected_agent_id:
        raise ValueError("agent_id does not match the expected agent")
    decision = payload["decision"]
    if decision not in {"answer", "abstain"}:
        raise ValueError("decision must be answer or abstain")
    answer = payload["answer"]
    if answer is not None and answer not in {"yes", "no"}:
        raise ValueError("answer must be yes, no, or null")
    confidence_value = payload["confidence"]
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence_value)
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    if decision == "answer" and answer not in {"yes", "no"}:
        raise ValueError("answer decisions require yes or no")
    if decision == "abstain" and (answer is not None or confidence != 0.0):
        raise ValueError("abstain requires answer=null and confidence=0.0")

    citations_value = payload["cited_evidence_ids"]
    if not isinstance(citations_value, Sequence) or isinstance(citations_value, (str, bytes)):
        raise TypeError("cited_evidence_ids must be a list of strings")
    if any(not isinstance(item, str) or not item for item in citations_value):
        raise TypeError("cited_evidence_ids must contain non-empty strings")
    citations = tuple(citations_value)
    if len(set(citations)) != len(citations):
        raise ValueError("cited_evidence_ids must be unique")
    outside = set(citations) - set(allowed_evidence_ids)
    if outside:
        raise ValueError(f"citations reference evidence outside the packet: {sorted(outside)}")
    return QADecision(
        agent_id=agent_id,
        decision=decision,
        answer=answer,
        confidence=confidence,
        cited_evidence_ids=citations,
    )


class CachedChatClient:
    """Small content-addressed client for one OpenAI-compatible endpoint."""

    def __init__(self, endpoint: str, model: str, cache_dir: Path, timeout: float = 60.0):
        if endpoint != DEFAULT_ENDPOINT or model != DEFAULT_MODEL:
            raise ValueError("Pilot-LLM V1 endpoint and model are frozen")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout

    def call(self, messages: Sequence[Mapping[str, str]], *, seed: int) -> ChatResult:
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "seed": seed,
        }
        cache_material = {"endpoint": self.endpoint, "request": request_payload}
        cache_key = sha256(_canonical_json(cache_material).encode()).hexdigest()
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
            message = response_payload["choices"][0]["message"]
            content = message["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("chat endpoint returned an unexpected response schema") from error
        if not isinstance(content, str):
            raise TypeError("chat response content must be text")
        usage_payload = response_payload.get("usage", {})
        usage = {
            "prompt_tokens": _optional_int(usage_payload.get("prompt_tokens")),
            "completion_tokens": _optional_int(usage_payload.get("completion_tokens")),
            "total_tokens": _optional_int(usage_payload.get("total_tokens")),
        }
        result_payload = {
            "content": content,
            "model": str(response_payload.get("model", self.model)),
            "usage": usage,
            "http_status": status,
            "response_bytes": len(response_body),
        }
        _write_json(cache_path, result_payload)
        return ChatResult(
            content=content,
            model=result_payload["model"],
            usage=usage,
            http_status=status,
            request_bytes=len(body),
            response_bytes=len(response_body),
            latency_seconds=latency,
            cache_hit=False,
            cache_key=cache_key,
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _agent_seed(agent_index: int) -> int:
    return (7_101, 7_207, 7_313, 7_421, 7_529)[agent_index]


def _attempt_payload(result: ChatResult) -> dict[str, object]:
    return {
        "cache_hit": result.cache_hit,
        "cache_key": result.cache_key,
        "http_status": result.http_status,
        "request_bytes": result.request_bytes,
        "response_bytes": result.response_bytes,
        "latency_seconds": result.latency_seconds,
        "usage": dict(result.usage),
        "parse_error": None,
        "transport_error": None,
    }


def run_one_call(
    client: CachedChatClient,
    example: StrategyQAExample,
    view: EvidenceView,
    *,
    agent_index: int,
) -> dict[str, Any]:
    """Run one condition with at most one fixed repair/transport retry."""

    agent_id, persona = AGENT_PERSONAS[agent_index]
    attempts: list[dict[str, object]] = []
    repair = False
    decision: QADecision | None = None
    final_error: str | None = None
    for _ in range(2):
        messages = build_messages(
            example,
            view,
            agent_id=agent_id,
            persona=persona,
            repair=repair,
        )
        try:
            result = client.call(messages, seed=_agent_seed(agent_index))
        except (RuntimeError, TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempts.append(
                {
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
            )
            continue
        attempt = _attempt_payload(result)
        try:
            decision = parse_qa_decision(
                result.content,
                expected_agent_id=agent_id,
                allowed_evidence_ids=view.allowed_evidence_ids,
            )
        except (TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempt["parse_error"] = final_error
            attempts.append(attempt)
            repair = True
            continue
        attempts.append(attempt)
        final_error = None
        break

    return {
        "protocol_version": PROTOCOL_VERSION,
        "qid": example.qid,
        "label": example.label,
        "agent_id": agent_id,
        "condition": view.condition,
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision.to_payload() if decision is not None else None,
        "final_error": final_error,
    }


def _observable_decision(decision: Mapping[str, object]) -> str:
    return "abstain" if decision["decision"] == "abstain" else str(decision["answer"])


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _risk_at_coverage(rows: Sequence[Mapping[str, Any]], coverage: float) -> float | None:
    if not rows:
        return None
    selected_count = max(1, ceil(len(rows) * coverage))
    ordered = sorted(rows, key=lambda row: (float(row["risk"]), str(row["qid"])))
    return _mean([float(row["consensus_error"]) for row in ordered[:selected_count]])


def _aurc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (float(row["risk"]), str(row["qid"])))
    cumulative_errors = 0
    risks = []
    for index, row in enumerate(ordered, start=1):
        cumulative_errors += int(row["consensus_error"])
        risks.append(cumulative_errors / index)
    return _mean(risks)


def _bootstrap_clustered_mean(
    values_by_qid: Mapping[str, Sequence[float]], *, seed_offset: int
) -> dict[str, object]:
    usable = {qid: list(values) for qid, values in values_by_qid.items() if values}
    if not usable:
        return {"estimate": None, "ci95": [None, None], "clusters": 0}
    estimate = _mean([value for values in usable.values() for value in values])
    qids = sorted(usable)
    random_generator = random.Random(BOOTSTRAP_SEED + seed_offset)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [random_generator.choice(qids) for _ in qids]
        values = [value for qid in sampled for value in usable[qid]]
        bootstrap.append(float(_mean(values)))
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    return {"estimate": estimate, "ci95": [lower, upper], "clusters": len(qids)}


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    expected_examples: int,
    agent_count: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Compute instrumentation, paired-effect, and original-answer metrics."""

    expected_calls = expected_examples * agent_count * len(CONDITIONS)
    valid_records = [record for record in records if record["success"]]
    original = [record for record in valid_records if record["condition"] == "original"]
    total_attempts = [attempt for record in records for attempt in record["attempts"]]
    transferred_request_bytes = sum(
        int(attempt["request_bytes"] or 0)
        for attempt in total_attempts
        if not attempt["cache_hit"]
    )
    transferred_response_bytes = sum(
        int(attempt["response_bytes"] or 0)
        for attempt in total_attempts
        if not attempt["cache_hit"]
    )
    original_answered = [
        record for record in original if record["decision"]["decision"] == "answer"
    ]
    original_correct_by_qid: dict[str, list[float]] = defaultdict(list)
    original_probabilities: list[float] = []
    original_labels: list[int] = []
    citation_counts = []
    for record in original:
        citation_counts.append(len(record["decision"]["cited_evidence_ids"]))
    for record in original_answered:
        decision = record["decision"]
        correct = (decision["answer"] == "yes") == bool(record["label"])
        original_correct_by_qid[str(record["qid"])].append(float(correct))
        probability_yes = (
            float(decision["confidence"])
            if decision["answer"] == "yes"
            else 1.0 - float(decision["confidence"])
        )
        original_probabilities.append(probability_yes)
        original_labels.append(int(record["label"]))

    indexed = {
        (str(record["qid"]), str(record["agent_id"]), str(record["condition"])): record
        for record in valid_records
    }
    paired_rows = []
    responsiveness_by_qid: dict[str, list[float]] = defaultdict(list)
    remove_changes_by_qid: dict[str, list[float]] = defaultdict(list)
    reverse_changes_by_qid: dict[str, list[float]] = defaultdict(list)
    confidence_drop_remove = []
    confidence_drop_reverse = []
    for qid, agent_id in sorted({(key[0], key[1]) for key in indexed}):
        triple = [indexed.get((qid, agent_id, condition)) for condition in CONDITIONS]
        if any(record is None for record in triple):
            continue
        original_record, remove_record, reverse_record = triple
        original_decision = original_record["decision"]
        remove_decision = remove_record["decision"]
        reverse_decision = reverse_record["decision"]
        remove_changed = _observable_decision(remove_decision) != _observable_decision(
            original_decision
        )
        reverse_changed = _observable_decision(reverse_decision) != _observable_decision(
            original_decision
        )
        responsive = remove_changed or reverse_changed
        paired_rows.append(
            {
                "qid": qid,
                "agent_id": agent_id,
                "remove_changed": remove_changed,
                "reverse_changed": reverse_changed,
                "responsive": responsive,
            }
        )
        responsiveness_by_qid[qid].append(float(responsive))
        remove_changes_by_qid[qid].append(float(remove_changed))
        reverse_changes_by_qid[qid].append(float(reverse_changed))
        confidence_drop_remove.append(
            float(original_decision["confidence"]) - float(remove_decision["confidence"])
        )
        confidence_drop_reverse.append(
            float(original_decision["confidence"]) - float(reverse_decision["confidence"])
        )

    majority_rows = []
    risk_rows = []
    majority_accuracy_by_qid: dict[str, list[float]] = defaultdict(list)
    harmful_by_qid: dict[str, list[float]] = defaultdict(list)
    for qid in sorted({str(record["qid"]) for record in original}):
        qid_original = [record for record in original if str(record["qid"]) == qid]
        if len(qid_original) != agent_count:
            continue
        answers = [
            record["decision"]["answer"]
            for record in qid_original
            if record["decision"]["decision"] == "answer"
        ]
        counts = Counter(answers)
        if not counts or counts["yes"] == counts["no"]:
            continue
        consensus = "yes" if counts["yes"] > counts["no"] else "no"
        agreement = counts[consensus] / agent_count
        label = bool(qid_original[0]["label"])
        consensus_error = (consensus == "yes") != label
        harmful = consensus_error and agreement >= 0.8
        row = {
            "qid": qid,
            "consensus": consensus,
            "agreement": agreement,
            "consensus_error": int(consensus_error),
            "harmful_false_consensus": int(harmful),
        }
        majority_rows.append(row)
        majority_accuracy_by_qid[qid].append(float(not consensus_error))
        harmful_by_qid[qid].append(float(harmful))
        if len(responsiveness_by_qid[qid]) == agent_count:
            causal_risk = 1.0 - _mean(responsiveness_by_qid[qid])
            risk_rows.append(
                {
                    **row,
                    "risk": causal_risk,
                    "fixed_v3_intervention_contribution": min(1.0, 0.60 * causal_risk),
                }
            )

    risk_labels = [int(row["consensus_error"]) for row in risk_rows]
    risk_scores = [float(row["risk"]) for row in risk_rows]
    auroc = None
    auprc = None
    if len(set(risk_labels)) == 2:
        auroc = float(roc_auc_score(risk_labels, risk_scores))
        auprc = float(average_precision_score(risk_labels, risk_scores))

    condition_answer_coverage = {}
    for condition in CONDITIONS:
        condition_rows = [record for record in valid_records if record["condition"] == condition]
        answered = sum(record["decision"]["decision"] == "answer" for record in condition_rows)
        condition_answer_coverage[condition] = answered / len(condition_rows) if condition_rows else None

    summary: dict[str, Any] = {
        "protocol_version": protocol_version,
        "mode": mode,
        "expected_examples": expected_examples,
        "agent_count": agent_count,
        "expected_calls": expected_calls,
        "records": len(records),
        "instrumentation": {
            "valid_decisions": len(valid_records),
            "valid_decision_rate": len(valid_records) / expected_calls,
            "first_pass_valid_rate": sum(record["first_pass_valid"] for record in records)
            / expected_calls,
            "calls_with_http_or_cache": sum(
                any(
                    attempt["cache_hit"] or attempt["http_status"] is not None
                    for attempt in record["attempts"]
                )
                for record in records
            ),
            "cache_hits": sum(attempt["cache_hit"] for attempt in total_attempts),
            "attempts": len(total_attempts),
            "transferred_request_bytes": transferred_request_bytes,
            "transferred_response_bytes": transferred_response_bytes,
            "transferred_total_bytes": transferred_request_bytes + transferred_response_bytes,
            "prompt_tokens": sum(
                int(attempt["usage"].get("prompt_tokens") or 0) for attempt in total_attempts
            ),
            "completion_tokens": sum(
                int(attempt["usage"].get("completion_tokens") or 0)
                for attempt in total_attempts
            ),
        },
        "original": {
            "valid": len(original),
            "answered": len(original_answered),
            "answer_coverage": len(original_answered) / len(original) if original else None,
            "accuracy": _bootstrap_clustered_mean(original_correct_by_qid, seed_offset=1),
            "brier": brier_score(original_labels, original_probabilities)
            if original_probabilities
            else None,
            "ece": expected_calibration_error(
                original_labels, original_probabilities, n_bins=10
            )
            if original_probabilities
            else None,
            "citation_rate": _mean([float(value > 0) for value in citation_counts]),
            "mean_citations": _mean([float(value) for value in citation_counts]),
        },
        "interventions": {
            "complete_triplets": len(paired_rows),
            "expected_triplets": expected_examples * agent_count,
            "condition_answer_coverage": condition_answer_coverage,
            "paired_responsiveness": _bootstrap_clustered_mean(
                responsiveness_by_qid, seed_offset=2
            ),
            "remove_decision_change": _bootstrap_clustered_mean(
                remove_changes_by_qid, seed_offset=3
            ),
            "reverse_decision_change": _bootstrap_clustered_mean(
                reverse_changes_by_qid, seed_offset=4
            ),
            "mean_confidence_drop_remove": _mean(confidence_drop_remove),
            "mean_confidence_drop_reverse": _mean(confidence_drop_reverse),
        },
        "majority": {
            "questions_with_consensus": len(majority_rows),
            "accuracy": _bootstrap_clustered_mean(majority_accuracy_by_qid, seed_offset=5),
            "mean_agreement": _mean([float(row["agreement"]) for row in majority_rows]),
            "harmful_false_consensus": _bootstrap_clustered_mean(
                harmful_by_qid, seed_offset=6
            ),
        },
        "causal_effect_risk": {
            "complete_questions": len(risk_rows),
            "auroc_for_consensus_error": auroc,
            "auprc_for_consensus_error": auprc,
            "aurc": _aurc(risk_rows),
            "risk_at_80": _risk_at_coverage(risk_rows, 0.8),
        },
    }

    if mode == "formal":
        http_or_cache = summary["instrumentation"]["calls_with_http_or_cache"]
        summary["instrumentation_gates"] = {
            "balanced_manifest": expected_examples == 50,
            "http_or_cache_rate_at_least_0_98": http_or_cache / expected_calls >= 0.98,
            "valid_decision_rate_at_least_0_95": len(valid_records) / expected_calls >= 0.95,
            "complete_triplet_rate_at_least_0_95": len(paired_rows)
            / (expected_examples * agent_count)
            >= 0.95,
            "accepted_citations_packet_validated": True,
        }
    else:
        summary["instrumentation_gates"] = "not evaluated in smoke mode"
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    """Render a compact audit report without promoting smoke to formal evidence."""

    instrumentation = summary["instrumentation"]
    original = summary["original"]
    interventions = summary["interventions"]
    majority = summary["majority"]
    risk = summary["causal_effect_risk"]

    def fmt(value: object) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        f"# Pilot-LLM V1 {summary['mode']} report",
        "",
        "This report follows the frozen paired-intervention protocol. Smoke output is",
        "instrumentation-only and must not be used as formal research evidence.",
        "",
        "## Transfer and schema audit",
        "",
        "| Expected calls | Valid decisions | First-pass valid rate | Cache hits | Transfer bytes |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['expected_calls']} | {instrumentation['valid_decisions']} | "
            f"{fmt(instrumentation['first_pass_valid_rate'])} | "
            f"{instrumentation['cache_hits']} | "
            f"{instrumentation['transferred_total_bytes']} |"
        ),
        "",
        "## Observable behavior",
        "",
        "| Metric | Estimate |",
        "| --- | ---: |",
        f"| Original answer coverage | {fmt(original['answer_coverage'])} |",
        f"| Original accuracy | {fmt(original['accuracy']['estimate'])} |",
        f"| Original Brier | {fmt(original['brier'])} |",
        f"| Original ECE | {fmt(original['ece'])} |",
        f"| Citation rate | {fmt(original['citation_rate'])} |",
        f"| Complete paired triplets | {interventions['complete_triplets']} |",
        f"| Paired responsiveness | {fmt(interventions['paired_responsiveness']['estimate'])} |",
        f"| Remove decision change | {fmt(interventions['remove_decision_change']['estimate'])} |",
        f"| Reverse decision change | {fmt(interventions['reverse_decision_change']['estimate'])} |",
        f"| Majority questions | {majority['questions_with_consensus']} |",
        f"| Majority accuracy | {fmt(majority['accuracy']['estimate'])} |",
        f"| Harmful false consensus | {fmt(majority['harmful_false_consensus']['estimate'])} |",
        f"| Causal-risk AUROC | {fmt(risk['auroc_for_consensus_error'])} |",
        f"| Causal-risk AURC | {fmt(risk['aurc'])} |",
        "",
        "## Interpretation boundary",
        "",
        "Generic fact negation is a strong, mechanically generated intervention and is not",
        "guaranteed to be a logically minimal counterfactual. Undefined one-class metrics",
        "remain NA. These outputs do not establish LLM faithfulness, S&P 500 predictability,",
        "investment performance, or router superiority.",
        "",
    ]
    return "\n".join(lines)


def execute_run(
    *,
    mode: str,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    smoke_examples: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = validate_manifest(manifest, dataset_path)
    if mode == "smoke":
        if not 1 <= smoke_examples <= 2:
            raise ValueError("smoke mode permits only one or two examples")
        examples = examples[:smoke_examples]
        agents = AGENT_PERSONAS[:1]
    elif mode == "formal":
        if len(examples) != FORMAL_EXAMPLES:
            raise ValueError("formal mode requires the exact 50-question manifest")
        agents = AGENT_PERSONAS
    else:
        raise ValueError("mode must be smoke or formal")

    client = CachedChatClient(DEFAULT_ENDPOINT, DEFAULT_MODEL, cache_dir)
    records: list[dict[str, Any]] = []
    total = len(examples) * len(agents) * len(CONDITIONS)
    completed = 0
    for example in examples:
        for agent_index, _ in enumerate(agents):
            for condition in CONDITIONS:
                record = run_one_call(
                    client,
                    example,
                    evidence_view(example, condition),
                    agent_index=agent_index,
                )
                records.append(record)
                completed += 1
                print(
                    f"[{completed}/{total}] {example.qid} {record['agent_id']} "
                    f"{condition} success={record['success']}",
                    flush=True,
                )
                _write_jsonl(output_dir / "records.partial.jsonl", records)

    summary = summarize_records(
        records,
        mode=mode,
        expected_examples=len(examples),
        agent_count=len(agents),
    )
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    partial = output_dir / "records.partial.jsonl"
    if partial.exists():
        partial.unlink()
    return records, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write and validate the frozen manifest")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")

    smoke = subparsers.add_parser("smoke", help="run at most six instrumentation calls")
    smoke.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    smoke.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "smoke")
    smoke.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    smoke.add_argument("--examples", type=int, default=2)

    formal = subparsers.add_parser("run", help="run the frozen 750-call formal pilot")
    formal.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    formal.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    formal.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "formal")
    formal.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = build_manifest(args.dataset)
        _write_json(args.output, manifest)
        validate_manifest(manifest, args.dataset)
        print(f"Wrote frozen manifest: {args.output}")
        return 0
    if args.command == "smoke":
        execute_run(
            mode="smoke",
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            smoke_examples=args.examples,
        )
        return 0
    if args.command == "run":
        execute_run(
            mode="formal",
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
