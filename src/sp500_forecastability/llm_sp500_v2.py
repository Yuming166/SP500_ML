"""LLM-S&P500 V2 — prompt-hardening re-run on the frozen V1 design.

V2 preregisters exactly two changes relative to V1 (see
``docs/llm_sp500_v2_preregistration.md``):

1.  **Abstention-must-cite prompts** (fixes the 585/795 = 73.6% of V1
    first-pass failures that were empty ``claims`` arrays from the
    ``consistency`` / ``skeptic`` / occasional ``literal`` roles).
2.  **Verbatim-evidence-first catalog + hardened retry reminder**
    (fixes most of the remaining ~200 evidence_id-hallucination and
    truncation failures).

Everything else is inherited from V1 unchanged: the manifest
(salt ``llm-sp500-v1-2026-09-03``, sha256 ``6fd3c3ed…``), the packet
builder, the five routers, the AMIR fit, the AURC paired moving-block
CI, and the report structure.  Because the manifest is inherited, V2
is **paired** with V1 at the decision-date level: the primary endpoint
comparison and any V1-vs-V2 yield comparison use identical dates.

The backend (endpoint + model) is injected via CLI (D1_v2) so the same
frozen code runs against the local vLLM fallback or the
openapi.center relay without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from sp500_forecastability import agent_contracts
from sp500_forecastability import historical_router_v5 as v5
from sp500_forecastability.llm_sp500_v1 import (
    MANIFEST_SIZE,
    SMOKE_DATES,
    TEST_SIZE,
    TRAIN_SIZE,
    Manifest,
    OpenAIChatClient,
    Record,
    _extract_decision_json,
    _fit_amir_on,
    _label_frame,
    _load_frame,
    _score_amir,
    _summarize_router,
    build_packet,
    load_manifest,
    router_majority,
    router_mean_confidence_long,
    router_single_min,
    router_v5_provenance,
)
from sp500_forecastability.llm_sp500_v2_prompts import (
    HORIZON_DAYS,
    MODEL_NAME,
    ROLES,
    SCHEMA_REMINDER,
    render_user_message,
    system_prompt,
)

# --------------------------------------------------------------------------- #
# Frozen constants (V2 prereg §3)
# --------------------------------------------------------------------------- #

PROTOCOL_VERSION = "llm-sp500-v2-2026-09-03"
#: V2 does NOT re-sample the manifest.  The decision dates, the train /
#: test split, and min_base_rate are inherited verbatim from V1 so the
#: two versions are paired at the date level (prereg §4.1).
SALT = "llm-sp500-v1-2026-09-03"
INHERITED_MANIFEST_SHA256 = (
    "6fd3c3edbfa8253246252f658fbc220367d4f596568096bd53ac885c788d477f"
)

#: Default backend: the local vLLM fallback used by V1 (D7_v1).  The
#: relay (openapi.center, gpt-5.4-mini) is selected with
#: ``--endpoint/--model`` or the driver's ``--relay`` flag (D1_v2).
DEFAULT_ENDPOINT = "http://localhost:31519/v1/chat/completions"

V1_ROOT = Path("results/llm_sp500_v1")
DEFAULT_ROOT_V2 = Path("results/llm_sp500_v2")
DEFAULT_CACHE_DIR_V2 = DEFAULT_ROOT_V2 / "cache"
PREREG_PATH = Path("docs/llm_sp500_v2_preregistration.md")

EXPECTED_FORMAL = MANIFEST_SIZE * len(ROLES)  # 2,500


def _client(
    endpoint: str,
    model: str,
    cache_dir: Path = DEFAULT_CACHE_DIR_V2,
) -> OpenAIChatClient:
    """Build a V1 client bound to the V2 cache dir and backend."""

    return OpenAIChatClient(
        endpoint=endpoint, model=model, cache_dir=cache_dir
    )


# --------------------------------------------------------------------------- #
# Manifest: inherit V1's, verify the sha (prereg §4.1)
# --------------------------------------------------------------------------- #


def _load_manifest_v2() -> Manifest:
    """Load V1's frozen manifest and pin the V2 results dir to it."""

    manifest = load_manifest(V1_ROOT)
    if manifest.manifest_sha256 != INHERITED_MANIFEST_SHA256:
        raise RuntimeError(
            "V1 manifest sha mismatch: "
            f"expected {INHERITED_MANIFEST_SHA256}, "
            f"got {manifest.manifest_sha256}"
        )
    if manifest.salt != SALT:
        raise RuntimeError(
            f"V1 manifest salt changed: {manifest.salt!r} != {SALT!r}"
        )
    return manifest


# --------------------------------------------------------------------------- #
# _invoke_one: V1 logic with V2 prompts and D2_v2 empty-claims rescue
# --------------------------------------------------------------------------- #


def _invoke_one(
    client: OpenAIChatClient,
    *,
    frame: pd.DataFrame,
    decision_date: str,
    role: str,
    min_base_rate: float,
    seed: int,
) -> Record:
    """Issue one call with the V2 prompts and parse it into a Record."""

    items, graph = build_packet(frame, decision_date)
    decision_iso = pd.Timestamp(decision_date).strftime(
        "%Y-%m-%dT16:00:00+00:00"
    )
    horizon_end = (
        pd.Timestamp(decision_date) + pd.tseries.offsets.BDay(HORIZON_DAYS)
    ).strftime("%Y-%m-%d")
    packet_json = json.dumps(
        [asdict(item) for item in items], separators=(",", ":"), sort_keys=True
    )
    user_message = render_user_message(
        role=role,
        decision_time=decision_iso,
        horizon_end_date=horizon_end,
        packet_json=packet_json,
        min_base_rate=min_base_rate,
        valid_evidence_ids=[item.evidence_id for item in items],
    )
    messages = [
        {"role": "system", "content": system_prompt(role)},
        {"role": "user", "content": user_message},
    ]
    last_error: str | None = None
    parsed_payload: dict[str, Any] | None = None
    response_content = ""
    response_model = client.model
    http_status = 0
    cache_hit = False
    usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for attempt in range(2):  # initial + 1 retry
        result = client.call(messages, seed=seed)
        response_content = result.content
        response_model = result.model
        http_status = result.http_status
        cache_hit = result.cache_hit
        usage = result.usage
        try:
            decision_payload = _extract_decision_json(result.content)
            decision_payload.setdefault("agent_id", role)
            decision_payload.setdefault("decision_time", decision_iso)
            decision_payload.setdefault("horizon_days", HORIZON_DAYS)
            claims = decision_payload.get("claims") or []
            if not claims:
                claim = {
                    "claim_id": decision_payload.get("claim_id", "c1"),
                    "text": decision_payload.get("text", ""),
                    "stance": decision_payload.get("stance", "supports"),
                    "evidence_ids": decision_payload.get("evidence_ids", []),
                }
                decision_payload["claims"] = [claim]
            # D9_v1 carry-over: drop claims whose evidence_ids fall
            # outside the packet catalog instead of rejecting the whole
            # decision; a decision with no surviving claim is a parse
            # failure and is retried once.
            valid_ids = {item.evidence_id for item in items}
            kept_claims: list[dict[str, object]] = []
            for claim in decision_payload["claims"]:
                if not isinstance(claim, dict):
                    continue
                eids = claim.get("evidence_ids") or []
                if all(
                    isinstance(eid, str) and eid in valid_ids for eid in eids
                ):
                    kept_claims.append(claim)
            if not kept_claims:
                raise ValueError(
                    "no claim has fully-valid evidence_ids; "
                    "treating as parse failure"
                )
            decision_payload["claims"] = kept_claims
            decision = agent_contracts.parse_agent_decision(
                decision_payload,
                expected_agent_id=role,
                provenance_graph=graph,
                allowed_evidence_ids=[item.evidence_id for item in items],
            )
            parsed_payload = decision.to_payload()
            last_error = None
            break
        except (ValueError, TypeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            messages = [
                {"role": "system", "content": system_prompt(role)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": SCHEMA_REMINDER},
            ]
    cqid = f"{decision_date}__{role}"
    return Record(
        cqid=cqid,
        decision_date=decision_date,
        role=role,
        decision_iso=decision_iso,
        request_payload={
            "system": system_prompt(role),
            "user": user_message,
            "packet_n_items": len(items),
        },
        response_content=response_content,
        response_model=response_model,
        http_status=http_status,
        cache_hit=cache_hit,
        usage=usage,
        parsed=parsed_payload,
        parse_error=last_error,
    )


def _run_calls_v2(
    client: OpenAIChatClient,
    manifest: Manifest,
    frame: pd.DataFrame,
    dates: Sequence[str],
    output_path: Path,
    *,
    progress_path: Path | None = None,
    progress_phase: str = "formal",
    workers: int = 1,
) -> list[Record]:
    """V2 _run_calls: same concurrency/progress schema, V2 _invoke_one."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    total = len(dates) * len(ROLES)
    started = time.monotonic()

    def _do_one(date: str, role: str) -> Record:
        seed = int(
            hashlib.sha256(f"{date}__{role}__{SALT}".encode()).hexdigest()[:8],
            16,
        )
        return _invoke_one(
            client,
            frame=frame,
            decision_date=date,
            role=role,
            min_base_rate=manifest.min_base_rate,
            seed=seed,
        )

    def _append(record: Record, done: int) -> None:
        with output_path.open("a", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "cqid": record.cqid,
                        "decision_date": record.decision_date,
                        "role": record.role,
                        "decision_iso": record.decision_iso,
                        "request_packet_n_items": record.request_payload[
                            "packet_n_items"
                        ],
                        "response_content": record.response_content,
                        "response_model": record.response_model,
                        "http_status": record.http_status,
                        "cache_hit": record.cache_hit,
                        "usage": record.usage,
                        "parsed": record.parsed,
                        "parse_error": record.parse_error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if progress_path is not None:
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            progress_path.write_text(
                json.dumps(
                    {
                        "completed": done,
                        "total": total,
                        "rate": rate,
                        "eta_seconds": eta,
                        "last_cqid": record.cqid,
                        "last_agent": record.role,
                        "last_success": record.parsed is not None,
                        "phase": progress_phase,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    if workers <= 1:
        done = 0
        for date in dates:
            for role in ROLES:
                done += 1
                _append(_do_one(date, role), done)
        return records

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_do_one, date, role): (date, role)
            for date in dates
            for role in ROLES
        }
        for done, future in enumerate(as_completed(futures), start=1):
            _append(future.result(), done)
    return records


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def _cmd_prepare(args: argparse.Namespace) -> int:
    manifest = _load_manifest_v2()
    frame = _load_frame()
    DEFAULT_ROOT_V2.mkdir(parents=True, exist_ok=True)
    (DEFAULT_ROOT_V2 / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "inherited_from": "results/llm_sp500_v1/manifest.json",
                "inherited_sha256": manifest.manifest_sha256,
                "salt": manifest.salt,
                "window_start": manifest.window_start,
                "window_end": manifest.window_end,
                "manifest_size": manifest.manifest_size,
                "train_size": manifest.train_size,
                "test_size": manifest.test_size,
                "decision_dates": manifest.decision_dates,
                "train_dates": manifest.train_dates,
                "test_dates": manifest.test_dates,
                "min_base_rate": manifest.min_base_rate,
                "endpoint": getattr(args, "endpoint", DEFAULT_ENDPOINT),
                "model": getattr(args, "model", MODEL_NAME),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    eligible = len(frame)
    print(
        f"[prepare] inherited V1 manifest sha={manifest.manifest_sha256[:12]} "
        f"({manifest.manifest_size} dates; {manifest.train_size}/"
        f"{manifest.test_size}); frame rows in window: {eligible}"
    )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    manifest = _load_manifest_v2()
    frame = _load_frame()
    # Gate 1: manifest shape.
    if manifest.manifest_size != MANIFEST_SIZE:
        print(f"[audit] FAIL manifest_size={manifest.manifest_size}")
        return 1
    if manifest.train_size != TRAIN_SIZE or manifest.test_size != TEST_SIZE:
        print("[audit] FAIL split sizes")
        return 1
    # Gate 2: packet construction on 3 spot dates (one per third).
    spot = [
        manifest.decision_dates[0],
        manifest.decision_dates[len(manifest.decision_dates) // 2],
        manifest.decision_dates[-1],
    ]
    for date in spot:
        items, _graph = build_packet(frame, date)
        if not items:
            print(f"[audit] FAIL empty packet at {date}")
            return 1
        horizon_end = (
            pd.Timestamp(date) + pd.tseries.offsets.BDay(HORIZON_DAYS)
        ).strftime("%Y-%m-%d")
        user_message = render_user_message(
            role="literal",
            decision_time=pd.Timestamp(date).strftime(
                "%Y-%m-%dT16:00:00+00:00"
            ),
            horizon_end_date=horizon_end,
            packet_json=json.dumps(
                [asdict(i) for i in items],
                separators=(",", ":"),
                sort_keys=True,
            ),
            min_base_rate=manifest.min_base_rate,
            valid_evidence_ids=[i.evidence_id for i in items],
        )
        for item in items:
            if item.available_at != item.publication_time:
                print(f"[audit] FAIL leak guard at {date}")
                return 1
            if item.evidence_id not in user_message:
                print(
                    f"[audit] FAIL catalog missing {item.evidence_id} at {date}"
                )
                return 1
    print(
        f"[audit] OK: manifest {manifest.manifest_size} dates "
        f"(sha {manifest.manifest_sha256[:12]}); packets OK at {spot}; "
        f"catalog complete in rendered prompt"
    )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    manifest = _load_manifest_v2()
    frame = _load_frame()
    client = _client(args.endpoint, args.model)
    smoke_dir = DEFAULT_ROOT_V2 / "smoke"
    records_path = smoke_dir / "records.jsonl"
    progress_path = smoke_dir / "progress.json"
    if records_path.exists() and not args.no_resume:
        existing = [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line
        ]
        valid = sum(1 for r in existing if r.get("parsed") is not None)
        print(
            f"[smoke] resuming from {len(existing)} existing records "
            f"({valid} valid)"
        )
    else:
        records_path.unlink(missing_ok=True)
    dates = manifest.decision_dates[:SMOKE_DATES]
    _run_calls_v2(
        client,
        manifest,
        frame,
        dates,
        records_path,
        progress_path=progress_path,
        progress_phase="smoke",
        workers=getattr(args, "workers", 1),
    )
    existing_lines = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line
    ]
    valid = sum(1 for r in existing_lines if r.get("parsed") is not None)
    print(
        f"[smoke] {valid}/{len(existing_lines)} valid; "
        f"first-pass yield = {valid / max(1, len(existing_lines)):.2%}"
    )
    return 0 if valid == len(existing_lines) else 1


def _cmd_formal(args: argparse.Namespace) -> int:
    manifest = _load_manifest_v2()
    frame = _load_frame()
    client = _client(args.endpoint, args.model)
    formal_dir = DEFAULT_ROOT_V2 / "formal"
    records_path = formal_dir / "records.jsonl"
    progress_path = formal_dir / "progress.json"
    if records_path.exists() and not args.no_resume:
        print(f"[formal] resuming from existing {records_path}")
    else:
        records_path.unlink(missing_ok=True)
    _run_calls_v2(
        client,
        manifest,
        frame,
        manifest.decision_dates,
        records_path,
        progress_path=progress_path,
        progress_phase="formal",
        workers=getattr(args, "workers", 8),
    )
    print(f"[formal] done; records at {records_path}")
    return 0


def _load_records(records_path: Path) -> pd.DataFrame:
    rows = []
    for line in records_path.read_text().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("parsed") is None:
            continue
        rows.append(
            {
                "decision_date": obj["decision_date"],
                "role": obj["role"],
                "action": obj["parsed"]["action"],
                "confidence": float(obj["parsed"]["confidence"]),
            }
        )
    return pd.DataFrame(rows)


def _cmd_report(args: argparse.Namespace) -> int:
    manifest = _load_manifest_v2()
    formal_records_path = DEFAULT_ROOT_V2 / "formal" / "records.jsonl"
    smoke_records_path = DEFAULT_ROOT_V2 / "smoke" / "records.jsonl"
    if not formal_records_path.exists():
        print(f"[report] missing {formal_records_path}; run smoke + formal first")
        return 1
    raw_formal_records = [
        json.loads(line)
        for line in formal_records_path.read_text().splitlines()
        if line
    ]
    records = _load_records(formal_records_path)
    train_records = records[records["decision_date"].isin(manifest.train_dates)]
    test_records = records[records["decision_date"].isin(manifest.test_dates)]

    routers: dict[str, pd.DataFrame] = {}
    routers["majority"] = router_majority(test_records, manifest.test_dates)
    routers["mean_confidence_long"] = router_mean_confidence_long(
        test_records, manifest.test_dates
    )
    routers["v5_provenance_baseline"] = router_v5_provenance(
        test_records, manifest.test_dates
    )
    routers["single_min_agent"] = router_single_min(
        test_records, manifest.test_dates
    )
    model = _fit_amir_on(train_records, manifest.train_dates)
    routers["amir_router_v5"] = _score_amir(
        model, test_records, manifest.test_dates
    )

    label_df = _label_frame(manifest)
    summaries = {
        name: _summarize_router(router_df, label_df)
        for name, router_df in routers.items()
    }

    def _prep_for_aurc(router_df: pd.DataFrame) -> pd.DataFrame:
        merged = router_df.merge(
            label_df[["decision_date", "label"]], on="decision_date", how="inner"
        )
        merged["timestamp"] = pd.to_datetime(merged["decision_date"])
        if "risk" not in merged.columns:
            merged["risk"] = (
                1.0 - merged["confidence"].fillna(0.5).astype(float)
            ).clip(0.0, 1.0)
        if "error" not in merged.columns:
            merged["error"] = (
                (merged["action"] == "long").astype(int) != merged["label"]
            ).astype(int)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged

    amir_risk = _prep_for_aurc(routers["amir_router_v5"])
    majority_risk = _prep_for_aurc(routers["majority"])
    observed, (ci_low, ci_high) = v5._aurc_difference_ci(
        amir_risk, majority_risk
    )

    prompt = completion = 0
    for obj in raw_formal_records:
        prompt += int(obj.get("usage", {}).get("prompt_tokens", 0))
        completion += int(obj.get("usage", {}).get("completion_tokens", 0))
    token_totals = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    raw_smoke_records = []
    if smoke_records_path.exists():
        raw_smoke_records = [
            json.loads(line)
            for line in smoke_records_path.read_text().splitlines()
            if line
        ]
    response_models = sorted(
        {
            str(obj["response_model"])
            for obj in raw_formal_records
            if obj.get("response_model")
        }
    )
    # D3_v2: paired V1-vs-V2 per-role yield comparison (informational).
    v1_formal_path = V1_ROOT / "formal" / "records.jsonl"
    yield_comparison: list[dict[str, Any]] = []
    if v1_formal_path.exists():
        v1_raw = [
            json.loads(line)
            for line in v1_formal_path.read_text().splitlines()
            if line
        ]
        for role in ROLES:
            v1_role = [r for r in v1_raw if r["role"] == role]
            v2_role = [r for r in raw_formal_records if r["role"] == role]
            yield_comparison.append(
                {
                    "role": role,
                    "v1_valid": sum(r.get("parsed") is not None for r in v1_role),
                    "v1_total": len(v1_role),
                    "v2_valid": sum(r.get("parsed") is not None for r in v2_role),
                    "v2_total": len(v2_role),
                }
            )
    primary = {
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass": ci_high < 0.0,
    }
    report_path = _write_report_v2(
        manifest,
        routers,
        summaries,
        primary,
        token_totals,
        response_models=response_models,
        formal_calls=len(raw_formal_records),
        formal_valid=sum(
            obj.get("parsed") is not None for obj in raw_formal_records
        ),
        smoke_calls=len(raw_smoke_records),
        smoke_valid=sum(
            obj.get("parsed") is not None for obj in raw_smoke_records
        ),
        yield_comparison=yield_comparison,
    )
    print(f"[report] wrote {report_path}")
    print(
        f"[report] primary: AURC(amir) - AURC(majority) = "
        f"{observed:.4f} CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"{'PASS' if primary['pass'] else 'FAIL'}"
    )
    return 0


def _write_report_v2(
    manifest: Manifest,
    routers: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, float]],
    primary: dict[str, Any],
    token_totals: dict[str, int],
    *,
    response_models: Sequence[str],
    formal_calls: int,
    formal_valid: int,
    smoke_calls: int,
    smoke_valid: int,
    yield_comparison: Sequence[Mapping[str, Any]],
) -> Path:
    report_path = DEFAULT_ROOT_V2 / "report.md"
    model_label = ", ".join(response_models) if response_models else MODEL_NAME
    formal_yield = formal_valid / formal_calls if formal_calls else float("nan")
    smoke_yield = smoke_valid / smoke_calls if smoke_calls else float("nan")
    lines = [
        "# LLM-S&P500 V2 report: prompt-hardening re-run (paired with V1)",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        (
            f"- Manifest: inherited from V1 "
            f"(salt `{SALT}`, sha `{manifest.manifest_sha256[:12]}…`) — "
            f"decision dates identical to V1"
        ),
        (
            f"- Window: {manifest.window_start} → {manifest.window_end} "
            f"({manifest.manifest_size} decision dates; "
            f"{manifest.train_size} train / {manifest.test_size} test)"
        ),
        f"- Observed response model: `{model_label}`",
        f"- Roles: {', '.join(ROLES)}",
        (
            f"- Smoke: {smoke_valid}/{smoke_calls} parsed "
            f"({smoke_yield:.1%})"
        ),
        (
            f"- Formal valid-response yield: {formal_valid}/{formal_calls} "
            f"({formal_yield:.1%}); {formal_calls - formal_valid} calls "
            f"failed closed"
        ),
        "",
        "## Recorded final-response token usage",
        "",
        f"- Prompt tokens: {token_totals['prompt_tokens']:,}",
        f"- Completion tokens: {token_totals['completion_tokens']:,}",
        f"- Total tokens: {token_totals['total_tokens']:,}",
        "",
        "## Per-router metrics (test window)",
        "",
        "| Router | Coverage | Routed error | Risk Brier | Risk ECE | n_rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['coverage']:.3f} | "
            f"{summary['routed_error']:.3f} | "
            f"{summary['risk_brier']:.3f} | "
            f"{summary['risk_ece_10']:.3f} | "
            f"{summary['n_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Primary endpoint",
            "",
            (
                "**H1_v2** (frozen identical to H1_v1): "
                "AURC(amir_router_v5) − AURC(majority) < 0"
            ),
            "",
            f"- Observed AURC difference: **{primary['observed']:.4f}**",
            (
                "- 95% paired moving-block CI: "
                f"[{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]"
            ),
            (
                f"- PASS criterion (upper CI < 0): "
                f"**{'PASS' if primary['pass'] else 'FAIL'}**"
            ),
        ]
    )
    if yield_comparison:
        lines.extend(
            [
                "",
                "## D3_v2: paired V1 → V2 per-role yield (informational)",
                "",
                "| Role | V1 valid | V2 valid | Δ |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in yield_comparison:
            delta = row["v2_valid"] - row["v1_valid"]
            lines.append(
                f"| {row['role']} | {row['v1_valid']}/{row['v1_total']} | "
                f"{row['v2_valid']}/{row['v2_total']} | {delta:+d} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "V2 is a confirmatory re-run of V1 with prompts hardened "
                "against the two dominant V1 failure modes "
                "(empty-claims abstention, evidence_id hallucination). "
                "The decision dates are identical to V1, so the V1→V2 "
                "yield comparison is paired. A PASS closes the V5 "
                "signpost question under the V2 prompts but does not "
                "establish S&P 500 predictability, investment "
                "performance, or cross-model generalization. All five "
                "routers are reported."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-S&P500 V2 preregistered pipeline "
        "(prompt hardening on the frozen V1 manifest)"
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, help_text in (
        ("prepare", "verify + pin the inherited V1 manifest (no API calls)"),
        ("audit", "re-run the offline gates"),
        ("smoke", "5 API calls (1 date × 5 roles)"),
        ("formal", "2,500 API calls (500 dates × 5 roles)"),
        ("report", "routers + AURC CI + report.md"),
    ):
        p = sub.add_parser(name, help=help_text)
        # D1_v2: backend injection.  Accepted on every subcommand so the
        # driver can pass it uniformly.
        p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
        p.add_argument("--model", default=MODEL_NAME)
    sub.choices["smoke"].add_argument("--no-resume", action="store_true")
    sub.choices["smoke"].add_argument("--workers", type=int, default=1)
    sub.choices["formal"].add_argument("--no-resume", action="store_true")
    sub.choices["formal"].add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)
    handler = {
        "prepare": _cmd_prepare,
        "audit": _cmd_audit,
        "smoke": _cmd_smoke,
        "formal": _cmd_formal,
        "report": _cmd_report,
    }[args.subcommand]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
