"""Frozen prompt templates for LLM-S&P500 V2.

See ``docs/llm_sp500_v2_preregistration.md`` §6 for the canonical
contract.  This module **must not be edited after the V2 formal run
starts**; any change afterwards requires a new version (V3).

V2 changes relative to V1 (prereg §5 / §6):

1.  **Abstention-must-cite rule.**  V1's dominant failure mode was
    ``"claims": []`` — 585 of 795 first-pass failures (73.6%) came from
    the ``consistency`` (458) and ``skeptic`` (115) roles emitting an
    empty claims list when their system prompt told them to abstain,
    while ``parse_agent_decision`` requires >= 1 claim with nonempty
    evidence_ids.  V2 rewrites every role's system prompt to state that
    an abstain decision still carries a claim citing the conflicting
    evidence (``stance='attacks'``), so abstention remains representable
    in the frozen contract.
2.  **Verbatim-evidence-first rule.**  The remaining failures were
    evidence_id hallucination (``literal`` 54, ``cf`` 104, ``min`` 19)
    and rare JSON truncation.  V2 hardens the user template: the
    evidence catalog is presented as a numbered copy table and the model
    must copy the id character-for-character from it.

Roles, horizon, temperature, max_tokens, and the shared-packet layout
are unchanged from V1.
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent

#: The five roles used in V2 (identical taxonomy to V1).
ROLES: tuple[str, ...] = ("literal", "skeptic", "consistency", "cf", "min")

#: Decision horizon (trading days) — frozen identical to V1.
HORIZON_DAYS: int = 5

#: Default upstream model name (D1_v2: backend is injected at call time;
#: this constant is the prereg-named default endpoint model).
MODEL_NAME: str = "Hy-MT2-7B"

#: Per-call defaults — frozen identical to V1 (D8_v1 carry-over).
TEMPERATURE: float = 0.0
MAX_COMPLETION_TOKENS: int = 800
REQUEST_TIMEOUT_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# System prompts (one per role).
# ---------------------------------------------------------------------------

#: Shared rule appended to every role prompt.  This is the V2 fix for
#: the empty-claims abstention conflict (the 585/795 V1 failure modes).
_ABSTAIN_RULE = (
    " IMPORTANT: If you abstain or answer 'cash', you MUST still emit"
    " exactly one claim in the claims array — cite the evidence that"
    " conflicts or that you distrust, with stance='attacks'.  NEVER"
    " return an empty claims array; a decision with zero claims is"
    " invalid and will be rejected."
)

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
        "with confidence below 0.6 — and cite the two conflicting "
        "evidence items in your claim."
    ),
    "consistency": (
        "You are a consistency-checking investment analyst. Cross-check "
        "the seven source roots (market, VIX, macro, sentiment, options, "
        "ETF flows, mutual-fund flows). If three or more roots agree on "
        "direction, that direction wins. If fewer than three agree, "
        "abstain with action='cash' and confidence below 0.6 — and cite "
        "the roots you checked that failed to agree, with "
        "stance='attacks'."
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
        "min_base_rate). Even though you ignore the packet contents, "
        "you MUST still cite evidence: copy the FIRST evidence_id in "
        "the EVIDENCE CATALOG in the user message, character-for-"
        "character. Never cite 'min_base_rate' itself — it is not an "
        "evidence id."
    ),
}

#: Apply the abstention-must-cite rule to every role (it is harmless for
#: roles that never abstain by design, and it protects ``literal`` which
#: occasionally emitted empty claims in V1).
_SYSTEM_PROMPTS = {
    role: text + _ABSTAIN_RULE for role, text in _SYSTEM_PROMPTS.items()
}


def system_prompt(role: str) -> str:
    """Return the frozen system prompt for ``role`` (V2 prereg §6.2)."""

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

    EVIDENCE CATALOG — every claim you make MUST cite its evidence_id
    copied character-for-character from this numbered list:
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
          "evidence_ids": ["<id copied from the EVIDENCE CATALOG>", ...]
        }}
      ]
    }}

    Rules:
      - top-level keys MUST be exactly the 7 listed above; no claim_id,
        text, stance, evidence_ids, or other keys at the top level.
      - claims MUST contain AT LEAST ONE claim — an empty claims array
        is invalid and will be rejected, even when you abstain ('cash').
      - action='cash' requires target_exposure=0.0; action='long' uses 1.0.
      - horizon_days MUST equal {horizon_days}.
      - claims[].evidence_ids MUST be copied verbatim from the
        EVIDENCE CATALOG above. Never invent or shorten ids.
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
    valid_ids_block = "\n    ".join(
        f"{i + 1}. {eid}" for i, eid in enumerate(valid_evidence_ids)
    )
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
# Hardened in V2: names the two dominant V1 failure modes explicitly.
# ---------------------------------------------------------------------------

SCHEMA_REMINDER = (
    "Your previous response did not parse. The two most common causes "
    "are (1) an empty claims array and (2) evidence_ids not copied "
    "verbatim from the EVIDENCE CATALOG. Re-emit ONE JSON object with "
    "AT LEAST ONE claim whose evidence_ids are copied character-for-"
    "character from the catalog, matching the schema in the user "
    "message above. Do not include any prose, code fences, or "
    "commentary outside the JSON object."
)
