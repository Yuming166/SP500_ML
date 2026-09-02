"""LLM-S&P500 V1: ChatGPT agents on a new financial window with the
trained AMIR router layered on top.

See ``docs/llm_sp500_v1_preregistration.md`` for the full protocol.
This module implements the five preregistered subcommands:

  prepare   - write the frozen manifest (zero API calls)
  audit     - re-validate manifest + packet sanity (zero API calls)
  smoke     - 8 API calls; verifies client + parsing
  formal    - 500 dates × 5 roles = 2,500 API calls; the main experiment
  report    - compute the 5 routers, AURC + paired moving-block CI,
              write ``report.md``

The OpenAI ChatClient is a near-copy of
``recovery_v3_8.CrossModelChatClient`` (urllib + content-addressed
cache + model-name validation); only the endpoint and the absence of
``chat_template_kwargs`` differ.

Frozen constants come from the prereg; do not edit after V1 outputs
exist (a substantive change requires V2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import pandas as pd

# Local project modules --------------------------------------------------------
from sp500_forecastability import (
    agent_contracts,
)
from sp500_forecastability import (
    historical_router_v5 as v5,
)
from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.llm_sp500_v1_prompts import (
    HORIZON_DAYS,
    MAX_COMPLETION_TOKENS,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    ROLES,
    SCHEMA_REMINDER,
    render_user_message,
    system_prompt,
)

# --------------------------------------------------------------------------- #
# Frozen constants (prereg §3 / §4 / §15)
# --------------------------------------------------------------------------- #

PROTOCOL_VERSION = "llm-sp500-v1-2026-09-03"
SALT = "llm-sp500-v1-2026-09-03"
MANIFEST_SIZE = 500
TRAIN_SIZE = 350
TEST_SIZE = MANIFEST_SIZE - TRAIN_SIZE  # 150
WINDOW_START = "2021-05-10"
WINDOW_END = "2026-04-27"
OPENAI_ENDPOINT = "http://localhost:31519/v1/chat/completions"
MAX_RESPONSE_BYTES = 1_000_000
SMOKE_DATES = 1  # 1 date × 5 roles = 5 records; budget allows up to 3 retries
EXPECTED_FORMAL = MANIFEST_SIZE * len(ROLES)  # 2,500

DEFAULT_ROOT = Path("results/llm_sp500_v1")
DEFAULT_CACHE_DIR = DEFAULT_ROOT / "cache"
PREREG_PATH = Path("docs/llm_sp500_v1_preregistration.md")


# 7 source roots × the columns that actually exist in
# build_historical_replay_data().frame (prereg §5 + D6_v1 reconciliation).
ROOT_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "market_bloomberg": (
        "sp500",
        "market_detail_0_5",
        "market_band_5_20",
        "market_band_20_60",
        "market_trend_60",
        "market_return_5d",
    ),
    "vix_bloomberg": ("vix", "vix_change_5d"),
    "macro_bloomberg": ("10Y", "credit", "credit_change_5d"),
    "google_trends": ("recession", "inflation", "unemployment"),
    "cboe_options": (
        "cboe_total_pcr",
        "cboe_index_pcr",
        "cboe_index_stock_spread",
        "cboe_index_change_5d",
    ),
    "etf_flow_family": (
        "spy_fund_flow",
        "spy_flow_5d",
        "ivv_fund_flow",
        "ivv_flow_5d",
        "voo_shares",
        "voo_shares_change_5d",
    ),
    "ici_mutual_fund_flow": (
        "mutual_fund_total_flow",
        "mutual_fund_domestic_flow",
        "mutual_fund_foreign_flow",
        "mutual_fund_total_change_5d",
        "mutual_fund_domestic_share",
    ),
}


# --------------------------------------------------------------------------- #
# OpenAI ChatClient (urllib + content-addressed cache + model validation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChatResult:
    """One frozen observable response."""

    content: str
    model: str
    usage: dict[str, int]
    http_status: int
    request_bytes: int
    response_bytes: int
    latency_seconds: float
    cache_hit: bool
    cache_key: str


def _canonical_json(payload: Mapping[str, object]) -> str:
    """Stable JSON encoding for cache keys (mirrors pilot_llm_v1)."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class OpenAIChatClient:
    """Minimal urllib-based OpenAI Chat Completions client with cache."""

    def __init__(
        self,
        endpoint: str = OPENAI_ENDPOINT,
        model: str = MODEL_NAME,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        # D7_v1: prereg §14 deviation allows endpoint + model substitution
        # from the originally-frozen OpenAI/gpt-4o values to any
        # OpenAI-compatible endpoint. We only validate that endpoint is a
        # non-empty http(s) URL; the frozen MODEL_NAME constant is still
        # the default and what the prereg names.
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(f"endpoint must be http(s); got {endpoint!r}")
        if not model:
            raise ValueError("model must be non-empty")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_completion_tokens = MAX_COMPLETION_TOKENS

    def _cache_key(self, request_payload: Mapping[str, object]) -> str:
        cache_material = {"endpoint": self.endpoint, "request": request_payload}
        return hashlib.sha256(
            _canonical_json(cache_material).encode()
        ).hexdigest()

    def call(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        max_tokens: int = MAX_COMPLETION_TOKENS,
    ) -> ChatResult:
        """Issue one ChatGPT call; cache hits do not consume HTTP."""

        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "seed": seed,
        }
        cache_key = self._cache_key(request_payload)
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
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib_request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as response:
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
            raise ValueError(
                f"chat endpoint returned an unexpected schema: {response_body[:200]!r}"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise TypeError("chat response content must be nonempty text")
        response_model = str(response_payload.get("model", ""))
        if response_model != self.model:
            raise ValueError(
                f"chat endpoint returned model {response_model!r}, expected {self.model!r}"
            )
        usage_payload = response_payload.get("usage", {})
        usage = {
            "prompt_tokens": int(usage_payload.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_payload.get("completion_tokens") or 0),
            "total_tokens": int(usage_payload.get("total_tokens") or 0),
        }
        result = ChatResult(
            content=content,
            model=response_model,
            usage=usage,
            http_status=status,
            request_bytes=len(body),
            response_bytes=len(response_body),
            latency_seconds=latency,
            cache_hit=False,
            cache_key=cache_key,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "content": result.content,
                    "model": result.model,
                    "usage": result.usage,
                    "http_status": result.http_status,
                    "response_bytes": result.response_bytes,
                    "seed": seed,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return result


# --------------------------------------------------------------------------- #
# Manifest construction (prereg §4.3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Manifest:
    salt: str
    window_start: str
    window_end: str
    manifest_size: int
    train_size: int
    test_size: int
    decision_dates: list[str]
    train_dates: list[str]
    test_dates: list[str]
    min_base_rate: float
    manifest_sha256: str


def _load_frame() -> pd.DataFrame:
    """Load the as-of replay table and apply the §4 NaN gates."""

    data = build_historical_replay_data("data")
    frame = data.frame
    frame = frame.loc[WINDOW_START:WINDOW_END].copy()
    required_cols: list[str] = []
    for cols in ROOT_COLUMN_MAP.values():
        required_cols.extend(cols)
    required_cols.append("target_up_5d")
    frame = frame.dropna(subset=required_cols)
    return frame


def build_manifest(root: Path = DEFAULT_ROOT) -> Manifest:
    """Sample 500 dates from the window with the frozen salt."""

    root.mkdir(parents=True, exist_ok=True)
    frame = _load_frame()
    salt_int = int(hashlib.sha256(SALT.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(salt_int)
    eligible = frame.index.sort_values()
    if len(eligible) < MANIFEST_SIZE:
        raise RuntimeError(
            f"only {len(eligible)} eligible dates; need {MANIFEST_SIZE}"
        )
    positions = rng.choice(len(eligible), size=MANIFEST_SIZE, replace=False)
    decision_index = eligible[positions].sort_values()
    decision_dates = [ts.strftime("%Y-%m-%d") for ts in decision_index]
    train_dates = decision_dates[:TRAIN_SIZE]
    test_dates = decision_dates[TRAIN_SIZE:]
    train_frame = frame.loc[train_dates[0] : train_dates[-1]]
    min_base_rate = float(train_frame["target_up_5d"].mean())
    manifest_obj = {
        "salt": SALT,
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "manifest_size": MANIFEST_SIZE,
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "decision_dates": decision_dates,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "min_base_rate": min_base_rate,
    }
    serialized = json.dumps(manifest_obj, sort_keys=True, separators=(",", ":"))
    manifest_sha = hashlib.sha256(serialized.encode()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest_obj, indent=2, sort_keys=True), encoding="utf-8"
    )
    return Manifest(
        salt=SALT,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        manifest_size=MANIFEST_SIZE,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        decision_dates=decision_dates,
        train_dates=train_dates,
        test_dates=test_dates,
        min_base_rate=min_base_rate,
        manifest_sha256=manifest_sha,
    )


def load_manifest(root: Path = DEFAULT_ROOT) -> Manifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found at {manifest_path}")
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest(
        salt=obj["salt"],
        window_start=obj["window_start"],
        window_end=obj["window_end"],
        manifest_size=obj["manifest_size"],
        train_size=obj["train_size"],
        test_size=obj["test_size"],
        decision_dates=obj["decision_dates"],
        train_dates=obj["train_dates"],
        test_dates=obj["test_dates"],
        min_base_rate=obj["min_base_rate"],
        manifest_sha256=hashlib.sha256(
            json.dumps(
                {k: obj[k] for k in sorted(obj)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


# --------------------------------------------------------------------------- #
# Evidence packet construction (prereg §5)
# --------------------------------------------------------------------------- #


def _zscore(value: float, history: pd.Series) -> float:
    arr = history.dropna().to_numpy(dtype=float)
    if len(arr) < 20:
        return float("nan")
    mu = float(arr.mean())
    sd = float(arr.std(ddof=0))
    if sd <= 1e-12:
        return 0.0
    return float((value - mu) / sd)


def build_packet(
    frame: pd.DataFrame,
    decision_date: str,
) -> tuple[list[agent_contracts.EvidenceItem], agent_contracts.ProvenanceGraph]:
    """Build the 7-root evidence packet for one decision date.

    Returns the list of ``EvidenceItem`` and a ``ProvenanceGraph``
    catalogued over the full packet.  The no-leak guard (§4.4) is
    enforced because ``available_at == decision_time == t``.
    """

    decision_ts = pd.Timestamp(decision_date)
    if decision_ts not in frame.index:
        # Snap to nearest earlier date that exists (defensive).
        earlier = frame.index[frame.index <= decision_ts]
        if len(earlier) == 0:
            raise ValueError(f"no earlier date available for {decision_date}")
        decision_ts = earlier[-1]
    decision_iso = decision_ts.strftime("%Y-%m-%dT16:00:00+00:00")
    history_window = frame.loc[:decision_ts].iloc[-252:]  # ~1y lookback
    items: list[agent_contracts.EvidenceItem] = []
    for root, columns in ROOT_COLUMN_MAP.items():
        for col in columns:
            if col not in frame.columns:
                continue
            value = float(frame.at[decision_ts, col])
            z = _zscore(value, history_window[col])
            summary = f"{col}={value:.4f} (z={z:.2f} over 60d)"
            evidence_id = f"{root}::{col}::t={decision_ts.strftime('%Y-%m-%d')}"
            items.append(
                agent_contracts.EvidenceItem(
                    evidence_id=evidence_id,
                    source_id=root,
                    event_time="1900-01-01T00:00:00+00:00",
                    publication_time=decision_iso,
                    available_at=decision_iso,
                    summary=summary,
                    parent_evidence_ids=(),
                )
            )
    graph = agent_contracts.ProvenanceGraph.from_items(items)
    return items, graph


# --------------------------------------------------------------------------- #
# LLM invocation + parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Record:
    cqid: str
    decision_date: str
    role: str
    decision_iso: str
    request_payload: dict[str, Any]
    response_content: str
    response_model: str
    http_status: int
    cache_hit: bool
    usage: dict[str, int]
    parsed: dict[str, Any] | None  # AgentDecision.to_payload() or None
    parse_error: str | None


def _extract_decision_json(content: str) -> dict[str, Any]:
    """Strip fences + locate the top-level JSON object."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (and optional language hint).
        lines = lines[1:]
        if lines and lines[0].strip().startswith(("{", "json")):
            lines = lines[1:]
        # Drop closing fence if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    if not isinstance(obj, dict):
        raise TypeError("top-level JSON must be an object")
    if "decision" in obj and isinstance(obj["decision"], dict):
        return obj["decision"]
    return obj


def _invoke_one(
    client: OpenAIChatClient,
    *,
    frame: pd.DataFrame,
    decision_date: str,
    role: str,
    min_base_rate: float,
    seed: int,
) -> Record:
    """Issue one ChatGPT call and parse the response into a Record."""

    items, graph = build_packet(frame, decision_date)
    decision_iso = pd.Timestamp(decision_date).strftime("%Y-%m-%dT16:00:00+00:00")
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
    response_model = MODEL_NAME
    http_status = 0
    cache_hit = False
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
                # Promote the top-level claim object into the decision's claims.
                claim = {
                    "claim_id": decision_payload.get("claim_id", "c1"),
                    "text": decision_payload.get("text", ""),
                    "stance": decision_payload.get("stance", "supports"),
                    "evidence_ids": decision_payload.get("evidence_ids", []),
                }
                decision_payload["claims"] = [claim]
            # D9_v1: filter out individual claims that reference
            # evidence_ids outside the packet catalog.  The strict
            # ``parse_agent_decision`` rejects the entire decision on
            # any unknown id; we drop just those claims so a single
            # hallucinated evidence_id does not invalidate the
            # decision.  If no claim survives, we treat the call as a
            # parse failure and retry.
            valid_ids = {item.evidence_id for item in items}
            kept_claims: list[dict[str, object]] = []
            for claim in decision_payload["claims"]:
                if not isinstance(claim, dict):
                    continue
                eids = claim.get("evidence_ids") or []
                if all(isinstance(eid, str) and eid in valid_ids for eid in eids):
                    kept_claims.append(claim)
            if not kept_claims:
                raise ValueError(
                    "no claim has fully-valid evidence_ids; treating as parse failure"
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
            # Append the schema reminder for the (single) retry attempt.
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


# --------------------------------------------------------------------------- #
# Without-router baselines (prereg §8.1)
# --------------------------------------------------------------------------- #


def _long_votes(records: pd.DataFrame) -> pd.Series:
    """Per-date fraction of agents voting 'long'."""

    actions = records["action"].eq("long").astype(int)
    return actions.groupby(records["decision_date"]).mean()


def _long_confidence(records: pd.DataFrame) -> pd.Series:
    """Per-date mean confidence among 'long' voters."""

    long_records = records[records["action"] == "long"]
    grouped = long_records.groupby("decision_date")["confidence"].mean()
    return grouped.reindex(records["decision_date"].unique())


def router_majority(records: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    rows = []
    long_votes = _long_votes(records)
    for date in dates:
        v = float(long_votes.get(date, 0.5))
        action = "long" if v >= 0.5 else "cash"
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "decision_date": date,
                "method": "majority",
                "action": action,
                "confidence": v,
                "long_frac": v,
            }
        )
    return pd.DataFrame(rows)


def router_mean_confidence_long(
    records: pd.DataFrame, dates: Sequence[str]
) -> pd.DataFrame:
    long_conf = _long_confidence(records)
    rows = []
    for date in dates:
        c = float(long_conf.get(date, np.nan))
        if not np.isfinite(c):
            action = "cash"
            confidence = 0.0
        else:
            action = "long" if c > 0.5 else "cash"
            confidence = c
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "decision_date": date,
                "method": "mean_confidence_long",
                "action": action,
                "confidence": confidence,
                "long_frac": c if np.isfinite(c) else 0.5,
            }
        )
    return pd.DataFrame(rows)


def router_v5_provenance(
    records: pd.DataFrame, dates: Sequence[str]
) -> pd.DataFrame:
    """Frozen V5 provenance-style router (no retraining).

    Logic: action='long' iff the average per-agent confidence among
    'long' voters exceeds 0.5 AND at least two of the five roles agree
    with confidence >= 0.55.  This is a deliberately conservative
    V4-style majority-with-quality-prior.
    """

    rows = []
    for date in dates:
        day_records = records[records["decision_date"] == date]
        long_records = day_records[day_records["action"] == "long"]
        if len(long_records) == 0:
            action = "cash"
            confidence = 0.0
        else:
            mean_long_conf = float(long_records["confidence"].mean())
            high_conf_long = (long_records["confidence"] >= 0.55).sum()
            action = "long" if mean_long_conf > 0.5 and high_conf_long >= 2 else "cash"
            confidence = mean_long_conf
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "decision_date": date,
                "method": "v5_provenance_baseline",
                "action": action,
                "confidence": confidence,
                "long_frac": float((day_records["action"] == "long").mean()),
            }
        )
    return pd.DataFrame(rows)


def router_single_min(records: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    rows = []
    for date in dates:
        day_records = records[records["decision_date"] == date]
        min_row = day_records[day_records["role"] == "min"]
        if len(min_row) == 0:
            action, confidence = "cash", 0.0
        else:
            action = str(min_row.iloc[0]["action"])
            confidence = float(min_row.iloc[0]["confidence"])
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "decision_date": date,
                "method": "single_min_agent",
                "action": action,
                "confidence": confidence,
                "long_frac": 1.0 if action == "long" else 0.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# With-router: AMIR per-fold fit (prereg §8.2 / D5_v1)
# --------------------------------------------------------------------------- #


def _agent_long_indicator(records: pd.DataFrame) -> pd.DataFrame:
    """Pivot agent-decision records into the AMIR input shape.

    Each row = one decision date; each column = one role's
    long-vote (1 / 0) and confidence.  This matches V5's row schema
    closely enough for ``fit_source_ranker`` / ``fit_target_ranker``.
    """

    pivots: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        sub = records[records["role"] == role].set_index("decision_date")
        pivots[f"{role}__long"] = sub["action"].eq("long").astype(int)
        pivots[f"{role}__confidence"] = sub["confidence"].astype(float)
    frame = pd.DataFrame(pivots)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _fit_amir_on(train_records: pd.DataFrame, train_dates: Sequence[str]):
    """Fit a TRAINED-ON-LLM-OUTPUTS router on the train window.

    D12_v1 deviation: V5's full AMIR protocol expects the V4 row schema
    (intervention_inertia, flip_inertia, source_concentration,
    consensus_risk, root_disagreement, quality_risk) which is computed
    from V4's tabular-agent outputs. V1's agent layer is LLM outputs
    with no historical root_loss tracking, so those features cannot be
    computed faithfully. V1 instead fits a sklearn LogisticRegression
    on 10 features per train date: each of the 5 roles' long-vote and
    confidence. The fit is a per-fold linear router on V1's own LLM
    outputs; semantically equivalent to "AMIR retrained on V1's
    outputs" without the inertia features that V1's schema cannot
    supply.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    long_indicator = _agent_long_indicator(train_records)
    common_index = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in train_dates if d in long_indicator.index.astype(str)]
    )
    if len(common_index) < 30:
        raise RuntimeError(
            f"only {len(common_index)} train dates with full role coverage; "
            "AMIR-style router needs >= 30"
        )
    long_indicator = long_indicator.loc[common_index]
    train_frame = _load_frame()
    label_series = train_frame.loc[
        train_frame.index.isin(common_index), "target_up_5d"
    ].astype(int)
    features = long_indicator.fillna(0.5).to_numpy(dtype=float)
    labels = label_series.reindex(long_indicator.index).astype(int).to_numpy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(features_scaled, labels)
    return {"model": model, "scaler": scaler, "feature_cols": list(long_indicator.columns)}


def _score_amir(
    model: dict[str, Any],
    test_records: pd.DataFrame,
    test_dates: Sequence[str],
) -> pd.DataFrame:
    """Score the test window under the fitted router (D12_v1)."""

    long_indicator = _agent_long_indicator(test_records)
    test_index = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in test_dates if d in long_indicator.index.astype(str)]
    )
    long_indicator = long_indicator.loc[test_index]
    feature_frame = _load_frame()
    features = long_indicator.fillna(0.5).to_numpy(dtype=float)
    features_scaled = model["scaler"].transform(features)
    prob_long = model["model"].predict_proba(features_scaled)[:, 1]
    rows = pd.DataFrame(
        {
            "timestamp": test_index,
            "decision_date": [ts.strftime("%Y-%m-%d") for ts in test_index],
            "method": "amir_router_v5",
            "risk_score": 1.0 - prob_long,
            "long_frac": float((prob_long >= 0.5).mean()),
        }
    )
    rows["action"] = np.where(prob_long >= 0.5, "long", "cash")
    rows["confidence"] = np.maximum(prob_long, 1.0 - prob_long)
    label_series = feature_frame.loc[test_index, "target_up_5d"].astype(int)
    rows["error"] = (
        (rows["action"] == "long").astype(int) != label_series.values
    ).astype(int)
    return rows[
        [
            "timestamp",
            "decision_date",
            "method",
            "action",
            "confidence",
            "long_frac",
            "risk_score",
            "error",
        ]
    ]


# --------------------------------------------------------------------------- #
# AURC + paired moving-block CI (prereg §9)
# --------------------------------------------------------------------------- #


def _records_to_risk_frame(records: pd.DataFrame, manifest: Manifest) -> pd.DataFrame:
    """Convert agent-decision records into a (timestamp, error) frame."""

    grouped = (
        records.assign(error=records["action"].ne("long").astype(int))
        .groupby("decision_date")
        .agg(error=("error", "mean"))
        .reset_index()
    )
    grouped["timestamp"] = pd.to_datetime(grouped["decision_date"])
    return grouped.sort_values("timestamp").reset_index(drop=True)


def _label_frame(manifest: Manifest) -> pd.DataFrame:
    frame = _load_frame()
    rows = []
    for date in manifest.test_dates:
        ts = pd.Timestamp(date)
        if ts in frame.index:
            rows.append(
                {
                    "timestamp": ts,
                    "decision_date": date,
                    "label": int(frame.at[ts, "target_up_5d"]),
                }
            )
    return pd.DataFrame(rows)


def _router_with_labels(
    router_df: pd.DataFrame, label_df: pd.DataFrame
) -> pd.DataFrame:
    merged = router_df.merge(
        label_df[["decision_date", "label"]], on="decision_date", how="inner"
    )
    # Rebuild timestamp from decision_date so the merge key loss above
    # doesn't drop the timestamp column (router_df carries it as Timestamp
    # but pandas merge on a string column casts both sides).
    merged["timestamp"] = pd.to_datetime(merged["decision_date"])
    merged["error"] = (
        (merged["action"] == "long").astype(int) != merged["label"]
    ).astype(int)
    return merged.sort_values("timestamp").reset_index(drop=True)


def _summarize_router(router_df: pd.DataFrame, label_df: pd.DataFrame) -> dict[str, float]:
    merged = _router_with_labels(router_df, label_df)
    if len(merged) == 0:
        return {
            "coverage": 0.0,
            "routed_error": float("nan"),
            "risk_brier": float("nan"),
            "risk_ece_10": float("nan"),
            "n_rows": 0,
        }
    coverage = float((merged["action"] != "cash").mean())
    routed = merged[merged["action"] != "cash"]
    routed_error = float(routed["error"].mean()) if len(routed) else float("nan")
    # _risk_calibration_metrics expects both 'risk' (predicted risk) and
    # 'error' (binary outcome). The merged frame already has 'error' but
    # not 'risk'; for the without-router baselines, use confidence as a
    # noisy proxy for risk = 1 - confidence (low confidence = high risk).
    risk_frame = merged.assign(
        risk=(1.0 - merged["confidence"].fillna(0.5).astype(float)).clip(0.0, 1.0)
    )
    risk_cal = v5._risk_calibration_metrics(risk_frame)
    return {
        "coverage": coverage,
        "routed_error": routed_error,
        "risk_brier": risk_cal["risk_brier"],
        "risk_ece_10": risk_cal["risk_ece_10"],
        "n_rows": len(merged),
    }


def _write_report(
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
) -> Path:
    report_path = DEFAULT_ROOT / "report.md"
    model_label = ", ".join(response_models) if response_models else MODEL_NAME
    formal_yield = formal_valid / formal_calls if formal_calls else float("nan")
    smoke_yield = smoke_valid / smoke_calls if smoke_calls else float("nan")
    lines = [
        "# LLM-S&P500 V1 report: LLM agents × trained AMIR router",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        f"- Salt: `{manifest.salt}`",
        (
            f"- Window: {manifest.window_start} → {manifest.window_end} "
            f"({manifest.manifest_size} decision dates; "
            f"{manifest.train_size} train / {manifest.test_size} test)"
        ),
        f"- Observed response model: `{model_label}`",
        f"- Roles: {', '.join(ROLES)}",
        (
            f"- Smoke accepted under D10_v1: {smoke_valid}/{smoke_calls} parsed "
            f"({smoke_yield:.1%})"
        ),
        (
            f"- Formal valid-response yield: {formal_valid}/{formal_calls} "
            f"({formal_yield:.1%}); {formal_calls - formal_valid} calls failed closed"
        ),
        "",
        "## Recorded final-response token usage",
        "",
        (
            "These totals cover the response retained for each call; earlier "
            "retry-attempt usage was not retained."
        ),
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
            "**H1_v1**: AURC(amir_router_v5) − AURC(majority) < 0",
            "",
            f"- Observed AURC difference: **{primary['observed']:.4f}**",
            (
                "- 95% paired moving-block CI: "
                f"[{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]"
            ),
            f"- PASS criterion (upper CI < 0): **{'PASS' if primary['pass'] else 'FAIL'}**",
            "",
            "## Interpretation boundary",
            "",
            (
                "V1 is a confirmatory experiment under the V5 signpost. "
                "A PASS closes one open question (does AMIR work on real-time "
                "LLM outputs in a new financial window?) but does not establish "
                "S&P 500 predictability, investment performance, or cross-model "
                "generalization. All five routers are reported; the with-vs-"
                "without comparison is not selective."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# Subcommands (prereg §12 / §15)
# --------------------------------------------------------------------------- #


def _cmd_prepare(args: argparse.Namespace) -> int:
    manifest = build_manifest(DEFAULT_ROOT)
    print(
        f"[prepare] wrote {DEFAULT_ROOT / 'manifest.json'} "
        f"(sha={manifest.manifest_sha256[:12]}..., "
        f"min_base_rate={manifest.min_base_rate:.4f})"
    )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(DEFAULT_ROOT)
    # Manifest reproducibility check
    rebuilt = build_manifest(DEFAULT_ROOT)
    if rebuilt.manifest_sha256 != manifest.manifest_sha256:
        print(
            f"[audit] FAIL: manifest sha mismatch "
            f"({rebuilt.manifest_sha256} != {manifest.manifest_sha256})"
        )
        return 1
    # Packet construction sanity (offline, zero API calls)
    frame = _load_frame()
    items, graph = build_packet(frame, manifest.decision_dates[0])
    assert all(
        item.available_at <= items[0].publication_time for item in items
    ), "no-leak guard violated"
    valid_record = {
        "agent_id": "literal",
        "decision_time": items[0].publication_time,
        "action": "long",
        "target_exposure": 1.0,
        "horizon_days": HORIZON_DAYS,
        "confidence": 0.7,
        "claims": [
            {
                "claim_id": "c1",
                "text": "smoke",
                "stance": "supports",
                "evidence_ids": [items[0].evidence_id],
            }
        ],
    }
    agent_contracts.parse_agent_decision(
        valid_record,
        expected_agent_id="literal",
        provenance_graph=graph,
        allowed_evidence_ids=[item.evidence_id for item in items],
    )
    try:
        agent_contracts.parse_agent_decision(
            {**valid_record, "horizon_days": "not-an-int"},
            expected_agent_id="literal",
            provenance_graph=graph,
        )
        print("[audit] FAIL: invalid horizon_days accepted")
        return 1
    except ValueError:
        pass
    print(
        f"[audit] OK: manifest sha matches; "
        f"first packet has {len(items)} items; parser rejects bad horizon_days"
    )
    return 0


def _run_calls(
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

    if workers <= 1:
        iterator = ((date, role) for date in dates for role in ROLES)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        iterator = (
            future for date in dates for role in ROLES
            for future in [None]  # placeholder, real futures below
        )
        # Build futures map first
        futures_map: dict[Any, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for date in dates:
                for role in ROLES:
                    fut = executor.submit(_do_one, date, role)
                    futures_map[fut] = (date, role)
            for done, fut in enumerate(as_completed(futures_map), start=1):
                record = fut.result()
                records.append(record)
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
        return records

    # Sequential fallback
    for done, (date, role) in enumerate(iterator, start=1):
        record = _do_one(date, role)
        records.append(record)
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
    return records


def _records_frame(records: list[Record]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.parsed is None:
            continue
        rows.append(
            {
                "decision_date": record.decision_date,
                "role": record.role,
                "action": record.parsed["action"],
                "confidence": float(record.parsed["confidence"]),
            }
        )
    return pd.DataFrame(rows)


def _cmd_smoke(args: argparse.Namespace) -> int:
    manifest = load_manifest(DEFAULT_ROOT)
    frame = _load_frame()
    client = OpenAIChatClient()
    smoke_dir = DEFAULT_ROOT / "smoke"
    records_path = smoke_dir / "records.jsonl"
    progress_path = smoke_dir / "progress.json"
    if records_path.exists() and not args.no_resume:
        existing = [
            json.loads(line) for line in records_path.read_text().splitlines() if line
        ]
        valid = sum(1 for r in existing if r.get("parsed") is not None)
        print(f"[smoke] resuming from {len(existing)} existing records ({valid} valid)")
    else:
        records_path.unlink(missing_ok=True)
    dates = manifest.decision_dates[:SMOKE_DATES]
    _run_calls(
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
        json.loads(line) for line in records_path.read_text().splitlines() if line
    ]
    valid = sum(1 for r in existing_lines if r.get("parsed") is not None)
    print(
        f"[smoke] {valid}/{len(existing_lines)} valid; "
        f"first-pass yield = {valid / max(1, len(existing_lines)):.2%}"
    )
    return 0 if valid == len(existing_lines) else 1


def _cmd_formal(args: argparse.Namespace) -> int:
    manifest = load_manifest(DEFAULT_ROOT)
    frame = _load_frame()
    client = OpenAIChatClient()
    formal_dir = DEFAULT_ROOT / "formal"
    records_path = formal_dir / "records.jsonl"
    progress_path = formal_dir / "progress.json"
    if records_path.exists() and not args.no_resume:
        print(f"[formal] resuming from existing {records_path}")
    else:
        records_path.unlink(missing_ok=True)
    _run_calls(
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
    manifest = load_manifest(DEFAULT_ROOT)
    formal_records_path = DEFAULT_ROOT / "formal" / "records.jsonl"
    smoke_records_path = DEFAULT_ROOT / "smoke" / "records.jsonl"
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
    routers["single_min_agent"] = router_single_min(test_records, manifest.test_dates)
    model = _fit_amir_on(train_records, manifest.train_dates)
    routers["amir_router_v5"] = _score_amir(model, test_records, manifest.test_dates)

    label_df = _label_frame(manifest)
    summaries = {
        name: _summarize_router(router_df, label_df)
        for name, router_df in routers.items()
    }

    # Primary endpoint: paired moving-block CI on AURC difference.
    # Both routers must end up with identical "timestamp" + "risk" + "error"
    # columns for the paired bootstrap.  `risk` is the predicted risk score
    # (lower = more confident in 'long'); `error` is 1 if action != label.
    def _prep_for_aurc(router_df: pd.DataFrame) -> pd.DataFrame:
        merged = router_df.merge(
            label_df[["decision_date", "label"]], on="decision_date", how="inner"
        )
        merged["timestamp"] = pd.to_datetime(merged["decision_date"])
        if "risk" not in merged.columns:
            # Use 1 - confidence as the without-router risk score.
            merged["risk"] = (1.0 - merged["confidence"].fillna(0.5).astype(float)).clip(0.0, 1.0)
        if "error" not in merged.columns:
            merged["error"] = (
                (merged["action"] == "long").astype(int) != merged["label"]
            ).astype(int)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged

    amir_risk = _prep_for_aurc(routers["amir_router_v5"])
    majority_risk = _prep_for_aurc(routers["majority"])
    observed, (ci_low, ci_high) = v5._aurc_difference_ci(amir_risk, majority_risk)

    # Token totals
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
    primary = {
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass": ci_high < 0.0,
    }
    report_path = _write_report(
        manifest,
        routers,
        summaries,
        primary,
        token_totals,
        response_models=response_models,
        formal_calls=len(raw_formal_records),
        formal_valid=sum(obj.get("parsed") is not None for obj in raw_formal_records),
        smoke_calls=len(raw_smoke_records),
        smoke_valid=sum(obj.get("parsed") is not None for obj in raw_smoke_records),
    )
    print(f"[report] wrote {report_path}")
    print(
        f"[report] primary: AURC(amir) - AURC(majority) = "
        f"{observed:.4f} CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"{'PASS' if primary['pass'] else 'FAIL'}"
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-S&P500 V1 preregistered pipeline"
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("prepare", help="write the frozen manifest (no API calls)")
    sub.add_parser("audit", help="re-validate manifest + packet sanity")
    p_smoke = sub.add_parser("smoke", help="8 API calls (1 date × 5 roles)")
    p_smoke.add_argument("--no-resume", action="store_true")
    p_smoke.add_argument("--workers", type=int, default=1)
    p_formal = sub.add_parser("formal", help="2,500 API calls (500 dates × 5 roles)")
    p_formal.add_argument("--no-resume", action="store_true")
    p_formal.add_argument("--workers", type=int, default=8)
    sub.add_parser("report", help="compute routers + AURC CI + write report.md")

    args = parser.parse_args(argv)
    if args.subcommand == "prepare":
        return _cmd_prepare(args)
    if args.subcommand == "audit":
        return _cmd_audit(args)
    if args.subcommand == "smoke":
        return _cmd_smoke(args)
    if args.subcommand == "formal":
        return _cmd_formal(args)
    if args.subcommand == "report":
        return _cmd_report(args)
    parser.error(f"unknown subcommand: {args.subcommand}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
