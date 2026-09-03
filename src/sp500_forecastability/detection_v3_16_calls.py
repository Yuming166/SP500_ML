"""Cross-model smoke and development calls for V3.16.

The CLI intentionally supports only the registered smoke and development
partitions. Formal candidate calls require a later, separately frozen module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from sp500_forecastability import detection_v3_16 as protocol
from sp500_forecastability import detection_v3_16_prompts as prompts

CALL_PROTOCOL = "detection-v3.16.5-vitaminc-symmetric-calls-2026-09-03"
SEED_PROTOCOL = "detection-v3.16-common-seed-2026-09-03"
AMENDMENT = Path("docs/detection_v3_16_1_preregistration.md")
TOKEN_AMENDMENT = Path("docs/detection_v3_16_3_ling_token_budget.md")
TOKEN_AMENDMENT_2 = Path("docs/detection_v3_16_4_ling_token_budget.md")
INTERFACE_AMENDMENT = Path("docs/detection_v3_16_5_common_interface.md")
CONDITIONS = ("original", "remove", "reverse", "substitute")
MAX_COMPLETION_TOKENS = 256
MAX_RESPONSE_BYTES = 1_000_000
MAX_ATTEMPTS = 2
DEFAULT_WORKERS = 16


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model: str
    endpoint: str


MODELS = {
    "qwen": ModelSpec("qwen", "Qwen3.5-4B", "http://127.0.0.1:31518/v1/chat/completions"),
    "ling": ModelSpec("ling", "Ling-3.0-tiny", "http://127.0.0.1:31520/v1/chat/completions"),
}


@dataclass(frozen=True)
class Task:
    split: str
    pair_id: str
    item_id: str
    agent_id: str
    agent_index: int
    persona: str
    condition: str
    claim: str
    evidence: tuple[dict[str, str], ...]
    allowed_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    usage: dict[str, int]
    status: int
    latency_seconds: float
    cache_hit: bool
    cache_key: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_root(model: ModelSpec) -> Path:
    return protocol.DEFAULT_ROOT / "calls_4" / model.key


def _protocol_manifest_path(model: ModelSpec) -> Path:
    return _model_root(model) / "protocol_manifest.json"


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "symmetric_fact_decision",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer": {"type": "string", "enum": ["SUPPORTS", "REFUTES"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "cited_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["answer", "confidence", "cited_evidence_ids"],
            },
            "strict": True,
        },
    }


class ChatClient:
    def __init__(self, model: ModelSpec, cache_dir: Path, timeout: float = 90.0) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        cache_dir.mkdir(parents=True, exist_ok=True)

    def call(self, messages: Sequence[Mapping[str, str]], seed: int) -> ChatResult:
        payload = {
            "model": self.model.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        key_material = {"endpoint": self.model.endpoint, "request": payload}
        cache_key = hashlib.sha256(_canonical_json(key_material).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChatResult(
                content=str(cached["content"]),
                model=str(cached["model"]),
                usage=dict(cached["usage"]),
                status=int(cached["status"]),
                latency_seconds=0.0,
                cache_hit=True,
                cache_key=cache_key,
            )
        body = _canonical_json(payload).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib_request.Request(
            self.model.endpoint, data=body, headers=headers, method="POST"
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except urllib_error.HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib_error.URLError, TimeoutError) as error:
            raise RuntimeError(f"chat request failed: {error}") from error
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("chat response exceeded safety limit")
        response_payload = json.loads(response_body)
        content = response_payload["choices"][0]["message"]["content"]
        observed_model = str(response_payload.get("model", ""))
        if observed_model != self.model.model:
            raise ValueError(
                f"endpoint returned model {observed_model!r}, expected {self.model.model!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise TypeError("response content must be nonempty text")
        usage_raw = response_payload.get("usage", {})
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }
        result = ChatResult(
            content=content,
            model=observed_model,
            usage=usage,
            status=status,
            latency_seconds=time.monotonic() - started,
            cache_hit=False,
            cache_key=cache_key,
        )
        _write_json(
            cache_path,
            {
                "content": content,
                "model": observed_model,
                "usage": usage,
                "status": status,
            },
        )
        return result


def endpoint_models(model: ModelSpec) -> set[str]:
    url = model.endpoint.removesuffix("/chat/completions") + "/models"
    with urllib_request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    observed = {str(row["id"]) for row in payload.get("data", [])}
    if model.model not in observed:
        raise ValueError(f"expected model {model.model!r}, observed {sorted(observed)}")
    return observed


def parse_decision(
    content: str, allowed_ids: Sequence[str], *, allow_empty_nonempty: bool = False
) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(content.strip())
    if content.strip()[end:].strip():
        raise ValueError("response must be exactly one JSON object")
    if not isinstance(payload, dict) or set(payload) != {
        "answer",
        "confidence",
        "cited_evidence_ids",
    }:
        raise ValueError("response keys do not match the contract")
    answer = payload["answer"]
    if answer not in {"SUPPORTS", "REFUTES"}:
        raise ValueError("answer must be SUPPORTS or REFUTES")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    citations = payload["cited_evidence_ids"]
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise TypeError("cited_evidence_ids must be a string list")
    if allowed_ids:
        if not citations and allow_empty_nonempty:
            pass
        elif len(citations) != 1 or citations[0] != allowed_ids[0]:
            raise ValueError("nonempty packet requires its exact visible evidence ID")
    elif citations:
        raise ValueError("empty packet requires empty citations")
    return {"answer": answer, "confidence": confidence, "cited_evidence_ids": citations}


def _source_id(page: str, unique_id: str) -> str:
    root = hashlib.sha256(page.encode()).hexdigest()[:12]
    return f"root_{root}::evidence_{unique_id}"


def load_tasks(split: str) -> list[Task]:
    if split not in {"smoke", "development"}:
        raise ValueError("V3.16 development caller refuses non-development partitions")
    protocol.audit()
    manifest = json.loads(protocol.SELECTION_MANIFEST.read_text(encoding="utf-8"))
    rows = protocol.load_rows()
    by_id = {str(row["unique_id"]): row for row in rows}
    tasks: list[Task] = []
    for pair in manifest["pairs"]:
        if pair["split"] != split:
            continue
        for item in pair["items"]:
            original = by_id[item["original_id"]]
            claim = str(original["claim"])
            for agent_index, (agent_id, persona) in enumerate(prompts.AGENT_PERSONAS):
                for condition in CONDITIONS:
                    evidence_row = None
                    if condition == "original":
                        evidence_row = original
                    elif condition == "reverse":
                        evidence_row = by_id[item["reverse_id"]]
                    elif condition == "substitute":
                        evidence_row = by_id[pair["distractor_id"]]
                    evidence: tuple[dict[str, str], ...] = ()
                    allowed: tuple[str, ...] = ()
                    if evidence_row is not None:
                        evidence_id = _source_id(
                            str(evidence_row["page"]), str(evidence_row["unique_id"])
                        )
                        evidence = (
                            {"evidence_id": evidence_id, "text": str(evidence_row["evidence"])},
                        )
                        allowed = (evidence_id,)
                    tasks.append(
                        Task(
                            split=split,
                            pair_id=str(pair["pair_id"]),
                            item_id=str(item["item_id"]),
                            agent_id=agent_id,
                            agent_index=agent_index,
                            persona=persona,
                            condition=condition,
                            claim=claim,
                            evidence=evidence,
                            allowed_evidence_ids=allowed,
                        )
                    )
    return sorted(tasks, key=lambda task: task_key(task.__dict__))


def task_key(task: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(task["item_id"]), int(task["agent_index"]), str(task["condition"])


def _seed(model: ModelSpec, task: Task) -> int:
    value = f"{SEED_PROTOCOL}|{model.model}|{task.item_id}|{task.agent_id}|{task.condition}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _build_protocol_manifest(model: ModelSpec) -> dict[str, Any]:
    return {
        "protocol_version": CALL_PROTOCOL,
        "status": "frozen_before_model_specific_v3_16_calls",
        "model": {"key": model.key, "id": model.model, "endpoint": model.endpoint},
        "selection_manifest": {
            "path": str(protocol.SELECTION_MANIFEST),
            "sha256": protocol.file_sha256(protocol.SELECTION_MANIFEST),
        },
        "preregistration_sha256": protocol.file_sha256(protocol.PREREGISTRATION),
        "amendment_sha256": protocol.file_sha256(AMENDMENT),
        "token_amendment_sha256": protocol.file_sha256(TOKEN_AMENDMENT),
        "token_amendment_2_sha256": protocol.file_sha256(TOKEN_AMENDMENT_2),
        "interface_amendment_sha256": protocol.file_sha256(INTERFACE_AMENDMENT),
        "implementation_sha256": protocol.file_sha256(Path(__file__)),
        "prompts_sha256": protocol.file_sha256(Path(prompts.__file__)),
        "conditions": list(CONDITIONS),
        "agents": [agent_id for agent_id, _ in prompts.AGENT_PERSONAS],
        "server_response_format": None,
        "prompt_schema": _response_format()["json_schema"]["schema"],
        "temperature": 0.0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_attempts": MAX_ATTEMPTS,
        "seed_protocol": SEED_PROTOCOL,
        "citation_contract": {
            "original_and_reverse": "exact_visible_id_required",
            "remove": "empty_required",
            "substitute": "empty_or_exact_visible_id",
        },
        "expected_calls": {
            "smoke": SMOKE_CALLS,
            "development": DEVELOPMENT_CALLS,
        },
        "claim_boundary": {
            "development_only": True,
            "formal_calls_authorized": False,
            "formal_outcomes_accessed": False,
        },
    }


SMOKE_CALLS = protocol.SMOKE_PAIRS * 2 * len(prompts.AGENT_PERSONAS) * len(CONDITIONS)
DEVELOPMENT_CALLS = protocol.DEVELOPMENT_PAIRS * 2 * len(prompts.AGENT_PERSONAS) * len(CONDITIONS)


def freeze_protocol(model: ModelSpec) -> dict[str, Any]:
    expected = _build_protocol_manifest(model)
    path = _protocol_manifest_path(model)
    records = list(_model_root(model).glob("**/records*.jsonl"))
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{model.key} V3.16 protocol drifted")
        return actual
    if records:
        raise ValueError("cannot freeze a model protocol after calls")
    _write_json(path, expected)
    return expected


def validate_protocol(model: ModelSpec) -> None:
    actual = json.loads(_protocol_manifest_path(model).read_text(encoding="utf-8"))
    if actual != _build_protocol_manifest(model):
        raise ValueError(f"{model.key} V3.16 protocol drifted")


def invoke(client: ChatClient, model: ModelSpec, task: Task) -> dict[str, Any]:
    attempts = []
    decision = None
    final_error = None
    for attempt_index in range(MAX_ATTEMPTS):
        messages = [
            {"role": "system", "content": prompts.system_prompt(task.persona)},
            {
                "role": "user",
                "content": prompts.user_prompt(
                    claim=task.claim,
                    evidence=task.evidence,
                    allowed_evidence_ids=task.allowed_evidence_ids,
                    allow_empty_citation=task.condition == "substitute",
                    repair=attempt_index > 0,
                ),
            },
        ]
        attempt: dict[str, Any] = {"attempt_index": attempt_index}
        try:
            result = client.call(messages, _seed(model, task))
            attempt.update(
                {
                    "cache_hit": result.cache_hit,
                    "cache_key": result.cache_key,
                    "http_status": result.status,
                    "latency_seconds": result.latency_seconds,
                    "usage": result.usage,
                    "content": result.content,
                }
            )
            decision = parse_decision(
                result.content,
                task.allowed_evidence_ids,
                allow_empty_nonempty=task.condition == "substitute",
            )
            final_error = None
        except (RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempt["parse_error"] = final_error
        attempts.append(attempt)
        if decision is not None:
            break
    return {
        "protocol_version": CALL_PROTOCOL,
        "runtime_model": model.model,
        "split": task.split,
        "pair_id": task.pair_id,
        "item_id": task.item_id,
        "agent_id": task.agent_id,
        "agent_index": task.agent_index,
        "condition": task.condition,
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "decision": decision,
        "attempts": attempts,
        "final_error": final_error,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(model: ModelSpec, split: str, workers: int, resume: bool = True) -> dict[str, Any]:
    validate_protocol(model)
    endpoint_models(model)
    tasks = load_tasks(split)
    expected = {task_key(task.__dict__) for task in tasks}
    output_dir = _model_root(model) / split
    partial_path = output_dir / "records.partial.jsonl"
    final_path = output_dir / "records.jsonl"
    if not resume:
        partial_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
    records = _load_jsonl(partial_path) if resume else []
    done = {task_key(record) for record in records}
    if len(done) != len(records) or done - expected:
        raise ValueError("partial records contain duplicate or foreign keys")
    client = ChatClient(model, _model_root(model) / "cache" / split)
    started = time.monotonic()
    pending = [task for task in tasks if task_key(task.__dict__) not in done]
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(invoke, client, model, task): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)
            done.add(task_key(record))
            if completed % 25 == 0 or completed == len(pending):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                _write_json(
                    output_dir / "progress.json",
                    {
                        "model": model.model,
                        "split": split,
                        "completed": len(done),
                        "total": len(expected),
                        "successful": sum(bool(row["success"]) for row in records),
                        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in records),
                        "eta_seconds": (len(expected) - len(done)) / rate if rate else None,
                    },
                )
    if done != expected:
        raise ValueError(f"run incomplete: {len(done)}/{len(expected)}")
    records = sorted(records, key=task_key)
    final_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    result = {
        "model": model.model,
        "split": split,
        "rows": len(records),
        "successful": sum(bool(row["success"]) for row in records),
        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in records),
    }
    result["valid_rate"] = result["successful"] / result["rows"]
    result["first_pass_valid_rate"] = result["first_pass_valid"] / result["rows"]
    result["qualified"] = result["valid_rate"] >= 0.98 and result["first_pass_valid_rate"] >= 0.95
    _write_json(output_dir / "qualification.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "smoke", "development"))
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = MODELS[args.model]
    if args.command == "freeze":
        print(json.dumps(freeze_protocol(model), indent=2))
        return 0
    result = run(model, args.command, args.workers, resume=not args.no_resume)
    print(json.dumps(result, indent=2))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
