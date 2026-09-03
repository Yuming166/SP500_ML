"""Frozen prompt candidates for the V3.16 development protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

AGENT_PERSONAS: tuple[tuple[str, str], ...] = (
    (
        "literal",
        "Judge only the literal relation between the claim and the visible evidence.",
    ),
    (
        "skeptic",
        "Audit numbers, dates, entities, negation, and quantifiers before deciding.",
    ),
    (
        "consistency",
        "Check whether every decisive detail in the claim is consistent with the evidence.",
    ),
    (
        "counterfactual",
        "Consider whether a minimal factual edit would reverse the relation, then judge the text shown.",
    ),
    (
        "minimal",
        "Use the shortest evidence-grounded decision rule and do not add outside knowledge.",
    ),
)


def system_prompt(persona: str) -> str:
    return (
        "You are one member of an evidence-grounded fact-verification team. "
        f"{persona} The evidence packet may be empty or unrelated. You must still choose "
        "SUPPORTS or REFUTES, using confidence to express uncertainty. Do not use outside "
        "knowledge. Return only the requested JSON object."
    )


def user_prompt(
    *,
    claim: str,
    evidence: Sequence[Mapping[str, str]],
    allowed_evidence_ids: Sequence[str],
    repair: bool = False,
) -> str:
    if set(allowed_evidence_ids) != {row["evidence_id"] for row in evidence}:
        raise ValueError("evidence packet and allowed IDs disagree")
    citation_rule = (
        "cited_evidence_ids must be [] because the packet is empty."
        if not evidence
        else "cited_evidence_ids must contain the one visible evidence_id copied exactly."
    )
    message = f"""Decide whether the visible evidence SUPPORTS or REFUTES the claim.

CLAIM:
{claim}

EVIDENCE_PACKET:
{json.dumps(list(evidence), ensure_ascii=False, indent=2, sort_keys=True)}

ALLOWED_EVIDENCE_IDS:
{json.dumps(list(allowed_evidence_ids), ensure_ascii=False)}

Return exactly:
{{
  "answer": "SUPPORTS" | "REFUTES",
  "confidence": <number from 0 to 1>,
  "cited_evidence_ids": ["<exact visible ID>"] | []
}}

Rules:
- Judge the claim against the text currently visible, not against memory.
- Check small changes in numbers, dates, names, comparisons, and negation.
- {citation_rule}
- Do not mention the experimental condition.
- Do not emit prose, markdown, or additional keys."""
    if repair:
        message += (
            "\n\nYour previous output violated the transport contract. Re-emit only one "
            "complete JSON object with the three exact keys above."
        )
    return message
