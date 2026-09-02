"""Frozen prompt templates for LLM-S&P500 V1.

See ``docs/llm_sp500_v1_preregistration.md`` §6 for the canonical
contract.  This module is imported by ``llm_sp500_v1`` and **must not be
edited after the formal run starts**; any change after V1 outputs
requires a new version (V2).

The five roles (``literal`` / ``skeptic`` / ``consistency`` / ``cf`` /
``min``) inherit verbatim from ``Pilot-LLM V5 §7``.  The user message
is identical across roles — only the system prompt differs.

The response schema is enforced server-side by
``agent_contracts.parse_agent_decision`` (one ``AgentDecision`` object
per call, with at least one ``Claim`` referencing evidence from the
shared packet).
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent

#: The five roles used in V1 (V5 §7 inheritance).
ROLES: tuple[str, ...] = ("literal", "skeptic", "consistency", "cf", "min")

#: Decision horizon (trading days).
HORIZON_DAYS: int = 5

#: Default upstream model name.  Frozen at gpt-4o per prereg §3;
#: D7_v1 deviation (§14) allows substitution with any
#: OpenAI-compatible local endpoint; the active value is passed in
#: via the client constructor at call time.
MODEL_NAME: str = "Hy-MT2-7B"

#: Per-call defaults (frozen in prereg §3; D8_v1 deviation bumps
#: max_tokens from 200 → 800 because Fin-R1 verbose JSON claims
#: regularly exceed 200 tokens at temperature=0).
TEMPERATURE: float = 0.0
MAX_COMPLETION_TOKENS: int = 800
REQUEST_TIMEOUT_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# System prompts (one per role).
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[str, str] = {
    "literal": (
        "You are a literal investment analyst. Take the evidence packet "
        "at face value. If the packet contains a strongly bearish signal "
        "(e.g., VIX spike, credit-spread widening, persistent fund "
        "outflow), answer accordingly; if it contains a strongly bullish "
        "signal (e.g., sustained inflow, falling VIX, narrowing spreads), "
        "answer accordingly. Do not look for hidden contradictions — the "
        "packet is what it is."
    ),
    "skeptic": (
        "You are a skeptical investment analyst. Look for inconsistencies "
        "within the packet. If VIX says risk-off but flow says risk-on, "
        "the packet is unreliable; lean toward 'cash' with low "
        "confidence. If two roots contradict each other on direction, "
        "report the contradiction in your claim text and choose 'cash' "
        "with confidence below 0.6."
    ),
    "consistency": (
        "You are a consistency-checking investment analyst. Cross-check "
        "the seven source roots (market, VIX, macro, sentiment, options, "
        "ETF flows, mutual-fund flows). If three or more roots agree on "
        "direction, that direction wins. If fewer than three agree, "
        "abstain with action='cash' and confidence below 0.6."
    ),
    "cf": (
        "You are a counterfactual investment analyst. Imagine the packet "
        "showed the opposite extreme of the most extreme feature "
        "(e.g., if VIX is currently 30, imagine it is 12). Would your "
        "decision flip? If yes, your confidence in the original "
        "decision is low; report the counterfactual explicitly in your "
        "claim text and reduce confidence by at least 0.2."
    ),
    "min": (
        "You are a minimum-information baseline analyst. Ignore the "
        "packet contents and predict the majority class on the training "
        "window (provided in the user message as ``min_base_rate``). "
        "If ``min_base_rate >= 0.5`` choose action='long'; else "
        "choose action='cash'. Report confidence = 1 - abs(0.5 - "
        "min_base_rate). Cite at least one evidence_id so the schema "
        "validator accepts the response."
    ),
}


def system_prompt(role: str) -> str:
    """Return the frozen system prompt for ``role`` (V1 prereg §6.2)."""

    if role not in _SYSTEM_PROMPTS:
        raise ValueError(f"unknown role: {role!r}; expected one of {ROLES}")
    return _SYSTEM_PROMPTS[role]


# ---------------------------------------------------------------------------
# User message template (shared across roles).
# ---------------------------------------------------------------------------

_USER_TEMPLATE = dedent(
    """\
    You are an investment analyst at decision_time = {decision_time}.
    Given the as-of evidence packet below, decide whether the S&P 500
    will close higher {horizon_days} trading days from now
    ({horizon_end_date}) than it closed at decision_time.

    Evidence packet ({n_items} items across 7 source roots):
    {packet_json}

    VALID_EVIDENCE_IDS (you MUST cite ONLY from this list — copy exactly):
    {valid_ids_block}

    Respond with one JSON object and nothing else. The object MUST
    contain ONLY these top-level keys (no others):

    {{
      "agent_id": "{role}",
      "decision_time": "{decision_time}",
      "action": "long" | "cash",
      "target_exposure": 0.0 | 1.0,
      "horizon_days": {horizon_days},
      "confidence": <float in [0, 1]>,
      "claims": [
        {{
          "claim_id": "c1",
          "text": "<one-sentence rationale>",
          "stance": "supports" | "attacks",
          "evidence_ids": ["<id_from_VALID_EVIDENCE_IDS_list>", ...]
        }}
      ]
    }}

    Rules:
      - top-level keys MUST be exactly the 7 listed above; no claim_id,
        text, stance, evidence_ids, or other keys at the top level.
      - action='cash' requires target_exposure=0.0; action='long' uses 1.0.
      - horizon_days MUST equal {horizon_days}.
      - claims[].evidence_ids MUST be copied verbatim from VALID_EVIDENCE_IDS.
      - Do not invent evidence_ids, source prefixes, or column names.
      - min_base_rate (training-window positive-class rate) = {min_base_rate:.4f}.
    """
)


def render_user_message(
    *,
    role: str,
    decision_time: str,
    horizon_end_date: str,
    packet_json: str,
    min_base_rate: float,
    valid_evidence_ids: Sequence[str],
) -> str:
    """Render the frozen user message for one role at one decision time."""

    if role not in ROLES:
        raise ValueError(f"unknown role: {role!r}; expected one of {ROLES}")
    n_items = packet_json.count('"evidence_id"')
    valid_ids_block = "\n    ".join(valid_evidence_ids)
    return _USER_TEMPLATE.format(
        decision_time=decision_time,
        horizon_end_date=horizon_end_date,
        horizon_days=HORIZON_DAYS,
        packet_json=packet_json,
        n_items=n_items,
        valid_ids_block=valid_ids_block,
        role=role,
        min_base_rate=min_base_rate,
    )


# ---------------------------------------------------------------------------
# Schema reminder (concatenated to the user message when retrying).
# ---------------------------------------------------------------------------

SCHEMA_REMINDER = (
    "Your previous response did not parse. Re-emit ONE JSON object "
    "matching the schema in the user message above. Do not include "
    "any prose, code fences, or commentary outside the JSON object."
)
