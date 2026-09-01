"""Pilot-LLM V10 preregistered paired-intervention runner.

BoolQ (Wikipedia yes/no) binary composites from the ``google/boolq``
HF mirror, partitioned evidence packets (5 agents x 2-of-3 subsets of
passages), four conditions (original / remove / reverse / substitute),
three co-registered risk endpoints (D_inert, D_conf, D_OR), and the
co-primary shared-citation detector ``shared_weighted`` (§9 of
docs/pilot_llm_v10_preregistration.md).

V10 is a **domain pivot** from V5/V6/V7 (FEVER, saturated consensus,
89% unanimous) to BoolQ (Wikipedia yes/no, expected diverse consensus
given 62/38 True/False raw label distribution). It tests whether the
R2 (AUROC-weighted vote) router's V4 advantage (+0.43 AUROC vs
BL_majority on TQA) generalizes to other diverse-consensus domains.

V10 inherits V7's protocol structure (partitions, condition set,
agent-persona set, instrumentation, scoring), V7's N=100 design, V7's
co-primary any-passes verdict logic, and V9's router variants. The
dataset change is registered as D1_v10.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from sp500_forecastability.metrics import expected_calibration_error
from sp500_forecastability.pilot_llm_v1 import (
    DEFAULT_MODEL,
    REPAIR_SUFFIX,
    CachedChatClient,
    EvidenceView,
    _agent_seed,
    _attempt_payload,
    _extract_json_object,
    _write_json,
    _write_jsonl,
    file_sha256,
)

# --- V6 constants ---------------------------------------------------------- #

PROTOCOL_VERSION = "pilot-llm-v10-2026-09-01"
SALT = b"pilot-llm-v10-2026-09-01\n"  # V10 D2_v10: fresh salt
# CQID hash uses V10's PROTOCOL_VERSION so V10 cqids DON'T collide with V5/V7/V9
CQID_PROTOCOL_VERSION = "pilot-llm-v10-2026-09-01"
CQID_PREFIX = "v10"  # cqids are v10-prefixed (V10 selection is independent of V5/V7)
BOOTSTRAP_SEED = 20_260_902          # V6 §11.0 (inherited from V5)
BOOTSTRAP_REPLICATES = 1_000
N_AGENTS = 5
FACTS_PER_QUESTION = 3
FACTS_PER_AGENT = 2
FORMAL_EXAMPLES = 100                 # V6 D1_v6 (was 50)
FORMAL_PER_LABEL = 50                 # V6 D1_v6 (was 25)
CONDITIONS: tuple[str, ...] = ("original", "remove", "reverse", "substitute")
CONFIDENCE_BAND = 0.05
PLATT_TARGET_COVERAGE = 0.80
SUBSTITUTE_FAIL_FAST_PCT = 0.10      # V6 §6.3 (inherited from V5)

# V6 endpoint: reuse the V5 Qwen3.5-4B endpoint (D3_v6: same-model).
DEFAULT_ENDPOINT = "http://10.63.0.88:31519/v1/chat/completions"

DEFAULT_DATASET = Path(
    "/storage/gaoym/sp500-forecastability-lab/data/boolq/train.parquet"
)
DEFAULT_ROOT = Path("results/pilot_llm_v10")

# Same partition table as V5: 5 agents x 2-of-3 subsets of {E01, E02, E03}.
PARTITION_TABLE: tuple[frozenset[str], ...] = (
    frozenset({"E01", "E02"}),
    frozenset({"E01", "E03"}),
    frozenset({"E02", "E03"}),
    frozenset({"E01", "E02"}),
    frozenset({"E01", "E03"}),
)

AGENT_PERSONAS: tuple[tuple[str, str], ...] = (
    ("literal_evidence", "Apply the task-local evidence literally to choose yes or no."),
    (
        "skeptical_auditor",
        "Audit evidence sufficiency, then choose the better-supported yes or no answer.",
    ),
    (
        "consistency_checker",
        "Check question-evidence consistency, then choose a definite yes or no answer.",
    ),
    (
        "counterfactual_reasoner",
        "Honor task-local counterfactual evidence when choosing a definite yes or no answer.",
    ),
    ("minimal_judge", "Return the shortest defensible yes or no judgment."),
)

RESPONSE_FIELDS = {"agent_id", "answer", "confidence", "cited_evidence_ids"}


# --- 1. BoolQ loader ------------------------------------------------------- #

import re as _re


@dataclass(frozen=True)
class BoolQItem:
    qid: str
    question: str       # the BoolQ yes/no question
    passage: str       # the Wikipedia passage (used as "evidence")
    label: str         # "yes" or "no" (V5-style binary)
    entity: str        # Wikipedia page title (extracted heuristic; D5_v10)


_SKIP_TITLE_WORDS = frozenset({
    # articles
    "The", "A", "An", "The", "Da", "De", "Le", "La", "El",
    # prepositions / subordinators
    "In", "On", "At", "By", "For", "With", "From", "Of", "To", "As",
    "About", "Between", "During", "Before", "After", "Since", "Until",
    "Through", "Against", "Without", "Within", "Among", "Across",
    "Although", "Though", "However", "While", "When", "Whenever",
    "Where", "Because", "Since", "Unless", "If", "Whether",
    # pronouns
    "It", "Its", "This", "That", "These", "Those", "He", "She",
    "They", "His", "Her", "Their", "Them", "I", "We", "You", "They",
    "There", "Here",
    # common non-title openers
    "Although", "However", "Most", "Some", "Many", "Several",
    "Other", "Such", "Each", "Every", "Any", "All", "Both",
    "First", "Second", "Third", "One", "Two", "Three",
})


def _extract_wiki_title(passage: str) -> str:
    """V10 D5_v10: heuristic Wikipedia title extraction from passage
    lead text. Strips leading articles/pronouns/prepositions, then
    captures up to 5 capitalized words (proper-noun like). Returns "" if
    no proper-noun sequence can be identified (the passage is then
    excluded from V10's clustering).
    """
    if not passage:
        return ""
    head = passage[:200]
    tokens = head.split()
    # Strip leading common words (compare stripped token to skip set)
    def _strip_punct(t: str) -> str:
        return t.strip(".,;:!?'\"()[]{}").strip()
    while tokens and _strip_punct(tokens[0]) in _SKIP_TITLE_WORDS:
        tokens = tokens[1:]
    # Collect up to 5 capitalized words
    caps: list[str] = []
    for t in tokens:
        t_clean = t.strip(".,;:!?'\"()[]{}").strip()
        if not t_clean:
            continue
        if not (t_clean[0].isalpha() and t_clean[0].isupper()):
            # non-capitalized word
            if caps:
                # we already collected something → stop
                break
            else:
                # still in leading skip phase → keep going
                continue
        caps.append(t_clean)
        if len(caps) >= 5:
            break
    return " ".join(caps)


def load_boolq(path: Path) -> list[BoolQItem]:
    """Load BoolQ train.parquet (or validation.parquet) and convert
    boolean answers to V5-style 'yes'/'no' labels. D1_v10: cluster by
    Wikipedia page title (extracted from passage lead, D5_v10).
    """
    import pandas as pd
    df = pd.read_parquet(str(path))
    items: list[BoolQItem] = []
    seen_qids: set[str] = set()
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        passage = str(row.get("passage", "")).strip()
        answer = row.get("answer", None)
        if not question or not passage or answer is None:
            continue
        qid = "boolq-" + sha256(question.encode()).hexdigest()[:16]
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        label = "yes" if bool(answer) else "no"
        entity = _extract_wiki_title(passage)
        items.append(BoolQItem(
            qid=qid, question=question, passage=passage,
            label=label, entity=entity,
        ))
    return items


# --- 2. Composite sampling ------------------------------------------------ #

@dataclass(frozen=True)
class CompositeQuestion:
    cqid: str
    question_text: str
    items: tuple[BoolQItem, ...]
    label: str

    @property
    def gold_binary(self) -> int:
        return 1 if self.label == "no" else 0


def _sha_selection_key(qid: str) -> str:
    return sha256(SALT + qid.encode()).hexdigest()


def _composite_label_majority(items: Sequence[BoolQItem]) -> str | None:
    """V10 §4.3: 2-1 splits are dropped; 3-0 only. Returns label."""
    counts = Counter(it.label for it in items)
    if counts["yes"] == counts["no"]:
        return None
    return "yes" if counts["yes"] > counts["no"] else "no"


def build_composite_questions(
    items: Sequence[BoolQItem],
    substitute_manifest: Mapping[str, Mapping[str, str]],
) -> list[CompositeQuestion]:
    """V10 §4.3: per cluster (Wikipedia page title), salt-sort, group
    every 3 rows as a composite. Returns the top-50 composites per
    label ('yes' / 'no'). Items without a successful substitute are
    excluded upstream.
    """
    eligible = [
        it for it in items
        if substitute_manifest.get(it.qid, {}).get("substitute_sentence")
    ]
    by_cluster: dict[str, dict[str, list[BoolQItem]]] = {}
    for it in eligible:
        by_cluster.setdefault(it.entity, {"yes": [], "no": []})[it.label].append(it)

    composites_per_label: dict[str, list[CompositeQuestion]] = {
        "yes": [], "no": [],
    }
    for ent, d in by_cluster.items():
        for label, members in d.items():
            members_sorted = sorted(members, key=lambda r: _sha_selection_key(r.qid))
            for i in range(0, len(members_sorted) - FACTS_PER_QUESTION + 1,
                           FACTS_PER_QUESTION):
                triple = tuple(members_sorted[i : i + FACTS_PER_QUESTION])
                if len(triple) != FACTS_PER_QUESTION:
                    continue
                maj = _composite_label_majority(triple)
                if maj is None:
                    continue
                cqid_seed = sha256(
                    f"{CQID_PROTOCOL_VERSION}\n{ent}\n{i}".encode()
                ).hexdigest()[:10]
                cqid = f"{CQID_PREFIX}-{maj[:3].lower()}-{cqid_seed}"
                qtext = _build_composite_question_text(triple)
                composites_per_label[maj].append(
                    CompositeQuestion(
                        cqid=cqid, question_text=qtext, items=triple, label=maj,
                    )
                )

    out: list[CompositeQuestion] = []
    for label in ("yes", "no"):
        composites_per_label[label].sort(key=lambda c: c.cqid)
        out.extend(composites_per_label[label][:FORMAL_PER_LABEL])

    n_yes = sum(1 for c in out if c.label == "yes")
    n_no = sum(1 for c in out if c.label == "no")
    if n_yes < FORMAL_PER_LABEL or n_no < FORMAL_PER_LABEL:
        raise ValueError(
            f"V10 §11: insufficient balanced manifest: yes={n_yes}, no={n_no}, "
            f"target={FORMAL_PER_LABEL} per label. Run aborted before any LLM call. "
            f"Total clusters considered: {len(by_cluster)}"
        )
    return sorted(out, key=lambda c: c.cqid)


def _build_composite_question_text(triple: Sequence[BoolQItem]) -> str:
    """V10: 3 BoolQ Q&A pairs, each with its passage as evidence."""
    sections = []
    for i, it in enumerate(triple):
        snippet = it.passage[:500] + ('...' if len(it.passage) > 500 else '')
        sections.append(
            f"Q{i+1}: {it.question}\n"
            f"  Passage: {snippet}\n"
            f"  Gold answer: {it.label}"
        )
    body = "\n\n".join(sections)
    return (
        f"Answer the following three yes/no questions about "
        f"\"{triple[0].entity}\" based on the Wikipedia passages. "
        f"Treat each as a separate yes/no decision, but return ONE final "
        f"yes/no answer that reflects the dominant verdict.\n\n{body}"
    )
    # original legacy function (kept for compat)
    legacy = (
        "Verify the following three claims about "
        f"`{triple[0].entity}`:\n{bullets}\n\n"
        "Treat each as a separate fact-verification yes/no decision, but "
        "return ONE final yes/no answer that reflects the dominant verdict "
        "across the three claims (yes = each claim is supported, no = at "
        "least one is refuted)."
    )


# --- 3. Substitute generation (LLM rewrites) ------------------------------ #

def _substitute_prompt(question: str, passage: str,
                       src_answer: str) -> str:
    """V10 §6.3 substitute prompt: rewrite a Wikipedia passage to support
    the opposite BoolQ answer."""
    opp = "no" if src_answer == "yes" else "yes"
    return (
        f"You will see a Wikipedia passage that was used to answer the "
        f"following yes/no question: \"{question}\". The gold answer "
        f"was \"{src_answer}\".\n\n"
        f"Rewrite the passage so that it would support the OPPOSITE "
        f"answer (\"{opp}\"), while keeping it on the same topic and "
        "entity.\n\n"
        "Constraints:\n"
        "- keep the passage length within ±50% of the original token count\n"
        "- do not introduce new entities not present in the original\n"
        "- do not include meta-language like 'rewritten' or 'opposite'\n"
        "- respond with one paragraph, no bullet points\n\n"
        f"Original passage:\n{passage}"
    )


def _parse_substitute_response(content: str, src_tokens: int) -> str | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    sentences = [s.strip() for s in re.split(r"[\n]+", text) if s.strip()]
    if not sentences:
        return None
    rewrite = sentences[0]
    rewrite = re.sub(r"^[-*]\s+", "", rewrite)
    if not rewrite:
        return None
    rt_tokens = max(1, len(rewrite.split()))
    if not (0.5 <= rt_tokens / src_tokens <= 1.5):
        return None
    return rewrite


def build_substitute_manifest(
    items: Sequence[BoolQItem],
    *,
    client: CachedChatClient | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Generate one LLM rewrite per source evidence sentence.

    V6 §6.3 + D4_v6: budget <=300 calls preregistered; ~13K realized
    (per-sentence cardinality, same mechanism as V5 D5_v5).
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "substitute_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8")), {
                "reused_from_cache": True, "path": str(manifest_path),
            }

    if client is None:
        client = CachedChatClient(
            DEFAULT_ENDPOINT, DEFAULT_MODEL, cache_dir or DEFAULT_ROOT / "cache"
        )

    manifest: dict[str, dict[str, str]] = {}
    stats = {
        "n_items": len(items),
        "n_rewritten": 0,
        "n_unusable": 0,
        "transfer_bytes": 0,
        "in_window": 0,
        "out_of_window": 0,
    }
    for idx, item in enumerate(items):
        prompt = _substitute_prompt(item.claim, item.passage, item.label)
        src_tokens = max(1, len(item.passage.split()))
        try:
            result = client.call(
                [{"role": "user", "content": prompt}],
                seed=20_260_903,
            )
            rewrite = _parse_substitute_response(result.content, src_tokens)
            stats["transfer_bytes"] += len(prompt) + len(result.content)
        except Exception:
            rewrite = None
        if rewrite is None:
            manifest[item.qid] = {
                "substitute_sentence": "", "in_length_window": False,
                "deviation_log": ["rewrite_failed_or_out_of_window"],
                "source_label": item.label, "source_entity": item.entity,
            }
            stats["n_unusable"] += 1
        else:
            rt_tokens = len(rewrite.split())
            in_window = bool(0.5 <= rt_tokens / src_tokens <= 1.5)
            manifest[item.qid] = {
                "substitute_sentence": rewrite,
                "in_length_window": in_window,
                "deviation_log": ["llm_negative_paraphrase_v6"],
                "source_label": item.label,
                "source_entity": item.entity,
            }
            stats["n_rewritten"] += 1
            if in_window:
                stats["in_window"] += 1
            else:
                stats["out_of_window"] += 1
        if (idx + 1) % 50 == 0 or idx == len(items) - 1:
            print(
                f"[substitute-gen] {idx+1}/{len(items)} rewritten={stats['n_rewritten']} "
                f"unusable={stats['n_unusable']} in_window={stats['in_window']}",
                flush=True,
            )

    unusable_pct = stats["n_unusable"] / max(1, stats["n_items"])
    stats["unusable_fraction"] = unusable_pct
    stats["passed_fail_fast"] = unusable_pct < SUBSTITUTE_FAIL_FAST_PCT

    if cache_dir is not None:
        _write_json(cache_dir / "substitute_manifest.json", manifest)
        _write_json(cache_dir / "substitute_generation_stats.json", stats)

    return manifest, stats


# --- 4. Manifest build + validate ----------------------------------------- #

def expected_dataset_sha256(path: Path) -> str:
    return file_sha256(path)


def build_manifest(
    dataset_path: Path, composite_questions: Sequence[CompositeQuestion],
    substitute_manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": expected_dataset_sha256(dataset_path),
        "selection": {
            "per_label": FORMAL_PER_LABEL,
            "total": FORMAL_EXAMPLES,
            "salt": PROTOCOL_VERSION,
        },
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "n_agents": N_AGENTS,
        "facts_per_question": FACTS_PER_QUESTION,
        "facts_per_agent": FACTS_PER_AGENT,
        "partition_table": [sorted(s) for s in PARTITION_TABLE],
        "conditions": list(CONDITIONS),
        "agents": [agent_id for agent_id, _ in AGENT_PERSONAS],
        "co_primary_endpoints": ["D_OR", "shared_weighted"],  # V7 D3_v7: any-passes
        "secondary_endpoints": ["shared_citation_signal"],
        "examples": [
            {
                "cqid": comp.cqid,
                "label": comp.label,
                "gold_binary": comp.gold_binary,
                "items": [
                    {
                        "qid": item.qid, "claim": item.claim,
                        "passage": item.passage,
                        "label": item.label, "entity": item.entity,
                        "evidence_id": f"E0{i + 1}",
                    }
                    for i, item in enumerate(comp.items)
                ],
            }
            for comp in composite_questions
        ],
        "substitute_manifest": {
            item.qid: dict(substitute_manifest[item.qid])
            for comp in composite_questions
            for item in comp.items
        },
    }


def validate_manifest(
    manifest: Mapping[str, object], dataset_path: Path,
) -> list[CompositeQuestion]:
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manifest is not Pilot-LLM V7")
    if manifest.get("dataset_sha256") != expected_dataset_sha256(dataset_path):
        raise ValueError("manifest dataset_sha256 does not match current file")
    if manifest.get("n_agents") != N_AGENTS:
        raise ValueError("manifest n_agents does not match V7 protocol")
    if manifest.get("facts_per_question") != FACTS_PER_QUESTION:
        raise ValueError("manifest facts_per_question does not match V7 protocol")
    if manifest.get("facts_per_agent") != FACTS_PER_AGENT:
        raise ValueError("manifest facts_per_agent does not match V7 protocol")
    if manifest.get("partition_table") != [sorted(s) for s in PARTITION_TABLE]:
        raise ValueError("manifest partition_table does not match V7 protocol")
    rows = manifest.get("examples", [])
    if len(rows) != FORMAL_EXAMPLES:
        raise ValueError(f"manifest must contain {FORMAL_EXAMPLES} composites")
    counter = Counter(row["label"] for row in rows)
    if counter["SUPPORTS"] != FORMAL_PER_LABEL or counter["REFUTES"] != FORMAL_PER_LABEL:
        raise ValueError(
            f"V7 manifest must contain {FORMAL_PER_LABEL} composites per label; "
            f"got {dict(counter)}"
        )
    composites: list[CompositeQuestion] = []
    for row in rows:
        items = tuple(
            BoolQItem(
                qid=item["qid"], question=item["question"],
                passage=item["passage"], label=item["label"],
                entity=item["entity"],
            )
            for item in row["items"]
        )
        lab_counter = Counter(it.label for it in items)
        if lab_counter["SUPPORTS"] == lab_counter["REFUTES"]:
            raise ValueError(
                f"composite {row['cqid']} has a label tie ({dict(lab_counter)}); "
                "V7 §D2_v5 forbids 2-1 splits"
            )
        composites.append(CompositeQuestion(
            cqid=row["cqid"],
            question_text=_build_composite_question_text(items),
            items=items, label=row["label"],
        ))
    return composites


# --- 5. Per-agent evidence view + prompts --------------------------------- #

def agent_packet(comp: CompositeQuestion, agent_index: int) -> tuple[str, ...]:
    partition = PARTITION_TABLE[agent_index]
    out: list[str] = []
    for i, _item in enumerate(comp.items):
        eid = f"E0{i + 1}"
        if eid in partition:
            out.append(eid)
    return tuple(out)


def build_evidence_view(
    composite: CompositeQuestion, agent_index: int, condition: str,
    substitute_manifest: Mapping[str, Mapping[str, str]],
) -> EvidenceView:
    partition_ids = PARTITION_TABLE[agent_index]
    items: list[tuple[str, str]] = []
    for i, item in enumerate(composite.items):
        evidence_id = f"E0{i + 1}"
        if evidence_id not in partition_ids:
            continue
        if condition == "original":
            text = item.passage
        elif condition == "remove":
            continue
        elif condition == "reverse":
            text = f"Task-local counterfactual: it is false that: {item.passage}"
        elif condition == "substitute":
            sub = substitute_manifest.get(item.qid, {}).get("substitute_sentence", "")
            text = sub if sub else item.passage
        else:
            raise ValueError(f"unknown condition: {condition}")
        items.append((evidence_id, text))
    return EvidenceView(condition=condition, items=tuple(items))


def build_messages(
    composite: CompositeQuestion, view: EvidenceView, *,
    agent_id: str, persona: str, repair: bool = False,
) -> list[dict[str, str]]:
    system = (
        "You are a binary evidence judge before a separate selective router. "
        f"Your fixed decision style is: {persona} "
        "Treat packet statements as task-local evidence, including counterfactual "
        "and substituted statements. "
        "When evidence is empty or insufficient, use your best general-knowledge "
        "judgment. You must choose yes or no; you cannot abstain. "
        "Do not provide reasoning, analysis, or chain-of-thought. Return only one "
        "JSON object."
    )
    task_payload = {
        "question": composite.question_text,
        "evidence_packet": [
            {"evidence_id": eid, "text": text} for eid, text in view.items
        ],
    }
    user = (
        "Answer the original yes/no question under the task-local evidence. "
        "answer must be exactly 'yes' or 'no'. Confidence is the probability "
        "that your answer is correct. cited_evidence_ids must be a unique list "
        f"drawn only from the packet.\n\nagent_id must be exactly: {agent_id}\n"
        "Required keys, with no others: agent_id, answer, confidence, "
        f"cited_evidence_ids.\n\nTask payload:\n"
        f"{json.dumps(task_payload, ensure_ascii=False, indent=2)}"
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_forced_qa_decision(
    content: str, *, expected_agent_id: str,
    allowed_evidence_ids: Sequence[str],
) -> dict[str, Any]:
    payload = _extract_json_object(content)
    unknown = set(payload) - RESPONSE_FIELDS
    missing = RESPONSE_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"unexpected decision fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing decision fields: {sorted(missing)}")
    agent_id = payload["agent_id"]
    if not isinstance(agent_id, str) or agent_id != expected_agent_id:
        raise ValueError("agent_id does not match expected agent")
    answer = payload["answer"]
    if answer not in {"yes", "no"}:
        raise ValueError("answer must be yes or no; agent abstention is not allowed")
    conf = payload["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise TypeError("confidence must be numeric")
    conf = float(conf)
    if not isfinite(conf) or not 0.0 <= conf <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    cites = payload["cited_evidence_ids"]
    if not isinstance(cites, Sequence) or isinstance(cites, (str, bytes)):
        raise TypeError("cited_evidence_ids must be a list of strings")
    cites_tuple = tuple(cites)
    if any(not isinstance(c, str) or not c for c in cites_tuple):
        raise TypeError("cited_evidence_ids must contain non-empty strings")
    if len(set(cites_tuple)) != len(cites_tuple):
        raise ValueError("cited_evidence_ids must be unique")
    outside = set(cites_tuple) - set(allowed_evidence_ids)
    if outside:
        raise ValueError(f"citations outside packet: {sorted(outside)}")
    return {
        "agent_id": agent_id, "answer": answer, "confidence": conf,
        "cited_evidence_ids": list(cites_tuple), "decision": "answer",
    }


def run_one_call(
    client: CachedChatClient, composite: CompositeQuestion, view: EvidenceView,
    *, agent_index: int,
    substitute_manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    agent_id, persona = AGENT_PERSONAS[agent_index]
    attempts: list[dict[str, object]] = []
    decision: dict[str, Any] | None = None
    final_error: str | None = None
    repair = False
    for _ in range(2):
        try:
            messages = build_messages(
                composite, view, agent_id=agent_id, persona=persona, repair=repair,
            )
            result = client.call(messages, seed=_agent_seed(agent_index))
        except (RuntimeError, TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempts.append({
                "cache_hit": False, "cache_key": None, "http_status": None,
                "request_bytes": None, "response_bytes": None,
                "latency_seconds": None, "usage": {}, "parse_error": None,
                "transport_error": final_error,
            })
            continue
        attempt = _attempt_payload(result)
        try:
            decision = parse_forced_qa_decision(
                result.content, expected_agent_id=agent_id,
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
        "cqid": composite.cqid,
        "label": composite.label,
        "gold_binary": composite.gold_binary,
        "agent_id": agent_id,
        "agent_index": agent_index,
        "condition": view.condition,
        "partition": sorted(PARTITION_TABLE[agent_index]),
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision,
        "final_error": final_error,
    }


# --- 6. Execute (smoke + formal) ----------------------------------------- #

def _load_partial_records(partial_path: Path) -> list[dict[str, Any]]:
    if not partial_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with partial_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"partial record corrupt on line {line_no}") from exc
    return rows


def _write_progress(progress_path: Path, payload: Mapping[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(progress_path, payload)


def execute_run(
    *, mode: str, dataset_path: Path, manifest_path: Path, output_dir: Path,
    cache_dir: Path, smoke_examples: int = 2, resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[CompositeQuestion]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composites = validate_manifest(manifest, dataset_path)
    substitute_manifest = manifest["substitute_manifest"]
    if mode == "smoke":
        if not 1 <= smoke_examples <= 2:
            raise ValueError("smoke mode permits only one or two composites")
        composites = composites[:smoke_examples]
        agents = list(range(1))
    elif mode == "formal":
        if len(composites) != FORMAL_EXAMPLES:
            raise ValueError(f"formal mode requires {FORMAL_EXAMPLES} composites")
        agents = list(range(N_AGENTS))
    else:
        raise ValueError("mode must be smoke or formal")

    client = CachedChatClient(DEFAULT_ENDPOINT, DEFAULT_MODEL, cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    progress_path = output_dir / "progress.json"
    records: list[dict[str, Any]] = []
    done: set[tuple[str, int, str]] = set()
    if resume:
        records = _load_partial_records(partial_path)
        for record in records:
            done.add((
                record["cqid"],
                int(record["agent_index"]),
                str(record["condition"]),
            ))
        if done:
            print(f"[resume] loaded {len(done)} completed tuples", flush=True)
    total = len(composites) * len(agents) * len(CONDITIONS)
    completed = len(done)
    start_ts = time.time()
    for comp in composites:
        for agent_index in agents:
            for condition in CONDITIONS:
                key = (comp.cqid, agent_index, condition)
                if key in done:
                    continue
                view = build_evidence_view(
                    comp, agent_index, condition, substitute_manifest,
                )
                record = run_one_call(
                    client, comp, view, agent_index=agent_index,
                    substitute_manifest=substitute_manifest,
                )
                records.append(record)
                done.add(key)
                completed += 1
                elapsed = time.time() - start_ts
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"[{completed}/{total}] {comp.cqid} "
                    f"{record['agent_id']} {condition} "
                    f"success={record['success']} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s",
                    flush=True,
                )
                _write_jsonl(partial_path, records)
                _write_progress(progress_path, {
                    "mode": mode,
                    "completed": completed,
                    "total": total,
                    "elapsed_seconds": elapsed,
                    "rate_per_second": rate,
                    "eta_seconds": eta,
                    "last_cqid": comp.cqid,
                    "last_agent": record["agent_id"],
                    "last_condition": condition,
                    "last_success": record["success"],
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
    summary = summarize_records(
        records, mode=mode, expected_examples=len(composites),
        agent_count=len(agents), substitute_manifest=substitute_manifest,
    )
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    return records, summary, composites


# --- 7. Metrics + summarize ----------------------------------------------- #

def _group_by_question(records):
    grouped: dict[str, list] = {}
    for r in records:
        grouped.setdefault(r["cqid"], []).append(r)
    return grouped


def _group_agents(grouped):
    by_agent: dict[int, list] = {}
    for r in grouped:
        by_agent.setdefault(int(r["agent_index"]), []).append(r)
    if not by_agent:
        return []
    return [by_agent.get(i, []) for i in range(N_AGENTS)]


def _agent_signal(agent_records):
    by_cond = {r["condition"]: r for r in agent_records}
    if "original" not in by_cond:
        return {"complete": False}
    original = by_cond["original"]["decision"] or {}
    orig_answer = original.get("answer")
    orig_conf = original.get("confidence", 0.0) or 0.0
    flips, conf_drops = {}, {}
    for cond in ("remove", "reverse", "substitute"):
        if cond not in by_cond:
            return {"complete": False}
        other = by_cond[cond]["decision"] or {}
        flips[cond] = int(orig_answer != other.get("answer"))
        conf_drops[cond] = orig_conf - (other.get("confidence", orig_conf) or orig_conf)
    inert = int(all(flips[c] == 0 for c in ("remove", "reverse", "substitute")))
    conf_stable = int(
        all(abs(conf_drops[c]) < CONFIDENCE_BAND
            for c in ("remove", "reverse", "substitute"))
    )
    return {
        "complete": True,
        "orig_answer": orig_answer, "orig_conf": orig_conf,
        "inert": inert, "conf_stable": conf_stable,
        "flips": flips, "conf_drops": conf_drops,
        "citations": {
            "original": set(original.get("cited_evidence_ids", [])),
            "remove": set((by_cond["remove"]["decision"] or {})
                          .get("cited_evidence_ids", [])),
            "reverse": set((by_cond["reverse"]["decision"] or {})
                           .get("cited_evidence_ids", [])),
            "substitute": set((by_cond["substitute"]["decision"] or {})
                              .get("cited_evidence_ids", [])),
        },
    }


def _per_question_risks(grouped):
    rows = []
    for cqid, recs in sorted(grouped.items()):
        per_agent_orig = [r for r in recs if r["condition"] == "original"]
        if len(per_agent_orig) != N_AGENTS:
            continue
        label = per_agent_orig[0]["label"]
        gold_binary = per_agent_orig[0]["gold_binary"]
        answers = [r["decision"]["answer"] for r in per_agent_orig if r["decision"]]
        if not answers:
            # V7 fix: skip questions whose original-condition records all failed
            # (e.g., vLLM endpoint went down mid-run). Preserved on records.jsonl
            # but excluded from AUROC computation.
            continue
        cnt = Counter(answers)
        cons, n = cnt.most_common(1)[0]
        agreement = n / len(answers)
        correct = int((cons == "yes") == bool(gold_binary))
        harmful_fc = int(correct == 0 and agreement >= 0.8)

        agent_groups = _group_agents(recs)
        if len(agent_groups) != N_AGENTS:
            continue
        agent_signals = []
        shared_count = 0
        for agent_recs in agent_groups:
            sig = _agent_signal(agent_recs)
            if not sig.get("complete"):
                continue
            agent_signals.append(sig)
            orig_cites = sig["citations"]["original"]
            for other_sig in agent_signals[:-1]:
                if orig_cites & other_sig["citations"]["original"]:
                    shared_count += 1
                    break
        if len(agent_signals) != N_AGENTS:
            continue

        d_inert = sum(s["inert"] for s in agent_signals) / N_AGENTS
        d_conf = sum(s["conf_stable"] for s in agent_signals) / N_AGENTS
        d_or = sum(int(s["inert"] or s["conf_stable"])
                   for s in agent_signals) / N_AGENTS
        d_majority = 1.0 - agreement
        frac_shared = shared_count / N_AGENTS
        shared_weighted = frac_shared * (1 - correct) + 0.5 * frac_shared * correct
        shared_signal = frac_shared

        rows.append({
            "cqid": cqid, "label": label, "gold_binary": gold_binary,
            "consensus": cons, "agreement": agreement,
            "correct": correct, "harmful_fc": harmful_fc,
            "any_wrong": int(correct == 0),
            "D_inert": d_inert, "D_conf": d_conf, "D_OR": d_or,
            "D_majority": d_majority,
            "shared_citation_signal": shared_signal,
            "shared_weighted": shared_weighted,
            "frac_shared": frac_shared,
            "_agent_inert": [s["inert"] for s in agent_signals],
            "_agent_conf_stable": [s["conf_stable"] for s in agent_signals],
        })
    return rows


def _auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _risk_at_coverage(scores, labels, coverage):
    if not scores or coverage <= 0 or coverage > 1:
        return None
    n_keep = max(1, round(len(scores) * coverage))
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    kept = paired[:n_keep]
    return sum(1 for _, l in kept if l == 1) / len(kept)


def _per_question_metric_bootstrap(metric_fn, rows, field, target_field,
                                    n_replicates=BOOTSTRAP_REPLICATES,
                                    seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    n = len(rows)
    samples = []
    for _ in range(n_replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [rows[i] for i in idx]
        scores = [r[field] for r in sub]
        labels = [r[target_field] for r in sub]
        m = metric_fn(scores, labels)
        if m is not None:
            samples.append(m)
    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    return (samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))])


def _safe_auprc(scores, labels):
    if not scores or len(set(labels)) < 2:
        return None
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return None


def _safe_brier(scores, labels):
    if not scores:
        return None
    try:
        return float(brier_score_loss(labels, scores))
    except Exception:
        return None


def _safe_ece(scores, labels):
    if not scores or len(set(labels)) < 2:
        return None
    try:
        return float(expected_calibration_error(labels, scores))
    except Exception:
        return None


def _platt_loo_brier_ece(rows, field, target_field="harmful_fc",
                         seed=BOOTSTRAP_SEED):
    if not rows:
        return {"brier_platt": None, "ece_platt": None, "n": 0}
    n = len(rows)
    if len({r[target_field] for r in rows}) < 2:
        return {"brier_platt": None, "ece_platt": None,
                "n": n, "skipped": "single_class"}
    platt_scores, labels_out = [], []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        x_train = [[float(rows[j][field])] for j in train_idx]
        y_train = [int(rows[j][target_field]) for j in train_idx]
        if len(set(y_train)) < 2:
            prior = sum(y_train) / len(y_train)
            platt_scores.append(prior)
        else:
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
            clf.fit(x_train, y_train)
            p = clf.predict_proba([[float(rows[i][field])]])[0, 1]
            platt_scores.append(float(p))
        labels_out.append(int(rows[i][target_field]))
    return {
        "brier_platt": _safe_brier(platt_scores, labels_out),
        "ece_platt": _safe_ece(platt_scores, labels_out),
        "brier_raw": _safe_brier(
            [float(r[field]) for r in rows], [int(r[target_field]) for r in rows]
        ),
        "n": n,
    }


def _intervention_flip_rates(grouped):
    flips = Counter()
    total = 0
    for _cqid, recs in grouped.items():
        for agent_recs in _group_agents(recs):
            sig = _agent_signal(agent_recs)
            if not sig.get("complete"):
                continue
            for cond in ("remove", "reverse", "substitute"):
                flips[cond] += sig["flips"][cond]
                total += 1
    return {c: flips[c] / (total / 3) if total else 0.0
            for c in ("remove", "reverse", "substitute")}


def _loao_aurocs(rows):
    """V7: LOAO is reported for both co-primaries (D_OR + shared_weighted)."""
    out = []
    if not rows:
        return {
            "median_auroc": None, "p05": None, "p95": None,
            "n_agents": 0, "n_variants": 0,
            "deterministic_auroc_D_OR": None,
            "deterministic_auroc_shared_weighted": None,
        }
    det_or = _auroc([r["D_OR"] for r in rows],
                    [r["harmful_fc"] for r in rows])
    det_sw = _auroc([r["shared_weighted"] for r in rows],
                    [r["harmful_fc"] for r in rows])
    for k in range(N_AGENTS):
        scores, labels = [], []
        for r in rows:
            inerts = r.get("_agent_inert", [])
            confs = r.get("_agent_conf_stable", [])
            if len(inerts) != N_AGENTS:
                continue
            keep = [j for j in range(N_AGENTS) if j != k]
            score = sum(int(inerts[j] or confs[j]) for j in keep) / max(1, len(keep))
            scores.append(score)
            labels.append(int(r["harmful_fc"]))
        a = _auroc(scores, labels)
        if a is not None:
            out.append(a)
    if out:
        s = sorted(out)
        median = s[len(s) // 2]
        p05 = s[max(0, int(round(0.05 * (len(s) - 1))))]
        p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    else:
        median = p05 = p95 = None
    return {
        "median_auroc": median, "p05": p05, "p95": p95,
        "n_agents": N_AGENTS, "n_variants": len(out),
        "deterministic_auroc_D_OR": det_or,
        "deterministic_auroc_shared_weighted": det_sw,
        "deviation_note": (
            "loao_substitutes_for_preregistered_partition_permutation: "
            "v6 single-call-per-agent design cannot re-assign subsets without "
            "additional LLM calls; LOAO is the closest honest proxy."
        ),
    }


def summarize_records(
    records, *, mode, expected_examples, agent_count, substitute_manifest,
):
    grouped = _group_by_question(records)
    rows = _per_question_risks(grouped)

    labels_hf = [r["harmful_fc"] for r in rows]
    labels_aw = [int(r["correct"] == 0) for r in rows]
    metric_block = {}
    for field, target in [
        ("D_inert", "harmful_fc"),
        ("D_conf", "harmful_fc"),
        ("D_OR", "harmful_fc"),
        ("D_majority", "harmful_fc"),
        ("shared_weighted", "harmful_fc"),         # V7 co-primary (§9.1)
        ("shared_citation_signal", "harmful_fc"),
        ("D_OR", "any_wrong"),
        ("D_majority", "any_wrong"),
    ]:
        key = f"{field}__{target}"
        scores = [r[field] for r in rows]
        labels = labels_hf if target == "harmful_fc" else labels_aw
        metric_block[key] = {
            "auroc": _auroc(scores, labels),
            "auroc_ci": list(_per_question_metric_bootstrap(
                _auroc, rows, field, target,
            )),
            "auprc": _safe_auprc(scores, labels),
            "auprc_ci": list(_per_question_metric_bootstrap(
                lambda s, l: _safe_auprc(s, l), rows, field, target,
            )),
            "risk_at_80": _risk_at_coverage(scores, labels, PLATT_TARGET_COVERAGE),
            "risk_at_80_ci": list(_per_question_metric_bootstrap(
                lambda s, l: _risk_at_coverage(s, l, PLATT_TARGET_COVERAGE),
                rows, field, target,
            )),
            "n_questions": len(rows),
        }

    if rows:
        cal = _platt_loo_brier_ece(rows, "D_OR", "harmful_fc")
        cal["prevalence"] = sum(labels_hf) / max(1, len(labels_hf))
        metric_block["D_OR__calibration"] = cal

    loao = _loao_aurocs(rows)

    # V7 §9.2 any-passes verdict logic (V5 §9.2 structure).
    # Both D_OR and shared_weighted are co-primary; PASS_BOTH if both clear,
    # PASS_SINGLE if only one clears, FAIL_BOTH if neither clears.
    co_primary_verdict = None
    if rows:
        or_key = "D_OR__harmful_fc"
        sw_key = "shared_weighted__harmful_fc"
        or_ci = metric_block[or_key]["auroc_ci"]
        sw_ci = metric_block[sw_key]["auroc_ci"]
        or_pass = (or_ci[0] is not None
                   and not (isinstance(or_ci[0], float) and or_ci[0] != or_ci[0])
                   and or_ci[0] > 0.5)
        sw_pass = (sw_ci[0] is not None
                   and not (isinstance(sw_ci[0], float) and sw_ci[0] != sw_ci[0])
                   and sw_ci[0] > 0.5)
        if or_pass and sw_pass:
            verdict = "PASS_BOTH"
        elif or_pass:
            verdict = "PASS_SINGLE_D_OR"
        elif sw_pass:
            verdict = "PASS_SINGLE_SHARED_WEIGHTED"
        else:
            verdict = "FAIL_BOTH"
        co_primary_verdict = {
            "D_OR": {
                "auroc": metric_block[or_key]["auroc"],
                "ci_lo": or_ci[0], "ci_hi": or_ci[1],
                "passes_lower_bound_above_0_5": or_pass,
            },
            "shared_weighted": {
                "auroc": metric_block[sw_key]["auroc"],
                "ci_lo": sw_ci[0], "ci_hi": sw_ci[1],
                "passes_lower_bound_above_0_5": sw_pass,
            },
            "verdict": verdict,
            "any_passes": or_pass or sw_pass,
        }

    valid_records = [r for r in records if r.get("success")]
    transfer_bytes = sum(
        attempt.get("request_bytes", 0) + attempt.get("response_bytes", 0)
        for r in records for attempt in r.get("attempts", [])
        if attempt.get("request_bytes") is not None
        and attempt.get("response_bytes") is not None
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "expected_examples": expected_examples,
        "agent_count": agent_count,
        "expected_calls": expected_examples * agent_count * len(CONDITIONS),
        "records": len(records),
        "instrumentation": {
            "valid_records": len(valid_records),
            "valid_rate": (len(valid_records) / len(records)) if records else 0.0,
            "first_pass_valid_rate": (
                sum(1 for r in records if r.get("first_pass_valid")) / len(records)
                if records else 0.0
            ),
            "transfer_bytes": transfer_bytes,
        },
        "outcome_prevalence": {
            "harmful_fc": sum(labels_hf),
            "any_wrong_consensus": sum(labels_aw),
            "n_questions": len(rows),
        },
        "intervention_flip_rates": _intervention_flip_rates(grouped),
        "metrics": metric_block,
        "loao_robustness": loao,
        "co_primary_verdict": co_primary_verdict,
        "substitute_manifest_summary": {
            qid: meta for qid, meta in substitute_manifest.items()
        },
    }


def render_report(summary) -> str:
    lines = [f"# Pilot-LLM V7 {summary['mode']} report", ""]
    inst = summary["instrumentation"]
    lines.append("## Transfer and schema audit")
    lines.append("")
    lines.append("| Expected calls | Valid records | First-pass valid rate | Transfer bytes |")
    lines.append("| ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {summary['expected_calls']} | {inst['valid_records']} | "
        f"{inst['first_pass_valid_rate']:.3f} | {inst['transfer_bytes']} |"
    )
    lines.append("")
    lines.append("## Co-primary verdict (V7 §9.2: any-passes; D_OR + shared_weighted)")
    cpv = summary.get("co_primary_verdict")
    if cpv is None:
        lines.append("- NA")
    else:
        lines.append(f"- **Verdict: {cpv['verdict']}**")
        lines.append(f"- D_OR AUROC = {cpv['D_OR']['auroc']:.3f} "
                     f"[{cpv['D_OR']['ci_lo']:.3f}, {cpv['D_OR']['ci_hi']:.3f}] "
                     f"passes_lo>0.5: {cpv['D_OR']['passes_lower_bound_above_0_5']}")
        lines.append(f"- shared_weighted AUROC = {cpv['shared_weighted']['auroc']:.3f} "
                     f"[{cpv['shared_weighted']['ci_lo']:.3f}, "
                     f"{cpv['shared_weighted']['ci_hi']:.3f}] "
                     f"passes_lo>0.5: {cpv['shared_weighted']['passes_lower_bound_above_0_5']}")
    lines.append("")
    lines.append("## Outcomes")
    prev = summary["outcome_prevalence"]
    lines.append(f"- N: {prev['n_questions']}")
    lines.append(f"- Harmful false consensus: {prev['harmful_fc']} "
                 f"({prev['harmful_fc'] / max(1, prev['n_questions']) * 100:.1f}%)")
    lines.append(f"- Any wrong consensus: {prev['any_wrong_consensus']}")
    lines.append("")
    lines.append("## Per-condition flip rates (per agent)")
    for cond, rate in summary["intervention_flip_rates"].items():
        lines.append(f"- `{cond}`: {rate:.3f}")
    lines.append("")
    lines.append("## Pre-registered metrics (with 95% question-cluster bootstrap CIs)")
    lines.append("")
    for key, metrics in summary["metrics"].items():
        lines.append(f"### {key}")
        for k, v in metrics.items():
            if k.endswith("_ci") and isinstance(v, list) and len(v) == 2:
                lo, hi = v
                if isinstance(lo, float) and isinstance(hi, float):
                    lines.append(f"- {k}: [{lo:.3f}, {hi:.3f}]")
                else:
                    lines.append(f"- {k}: NA")
            elif isinstance(v, float):
                lines.append(f"- {k}: {v:.3f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    loao = summary["loao_robustness"]
    lines.append("## LOAO robustness")
    lines.append(f"- Deterministic AUROC D_OR: "
                 f"{loao.get('deterministic_auroc_D_OR')}")
    lines.append(f"- Deterministic AUROC shared_weighted: "
                 f"{loao.get('deterministic_auroc_shared_weighted')}")
    lines.append(f"- LOAO median AUROC: {loao['median_auroc']}")
    lines.append(f"- LOAO [p05, p95]: [{loao['p05']}, {loao['p95']}]")
    if "deviation_note" in loao:
        lines.append("")
        lines.append(f"> **Deviation:** {loao['deviation_note']}")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "These results test whether V5's signal (D_OR = 0.656 at N = 50) "
        "holds at N = 100 with V5's same questions plus 50 more drawn "
        "under the same selection rule (V5 ⊂ V7 by construction). They "
        "do not establish LLM faithfulness in general, S&P 500 "
        "predictability, investment performance, or cross-model "
        "generalization. Cross-model generalization is the V8 prereg."
    )
    return "\n".join(lines) + "\n"


# --- 8. Pre-formal audit ------------------------------------------------ #

def _pre_formal_audit(dataset_path, manifest_path, *,
                       substitute_yield_min=0.90):
    items = load_fever(dataset_path)
    cache_dir = DEFAULT_ROOT / "cache"
    sub_path = cache_dir / "substitute_manifest.json"
    if not sub_path.exists():
        raise ValueError(
            f"substitute manifest missing at {sub_path}; run `substitute-generation` first"
        )
    substitute_manifest = json.loads(sub_path.read_text(encoding="utf-8"))
    composites = build_composite_questions(items, substitute_manifest)
    manifest = build_manifest(dataset_path, composites, substitute_manifest)
    validate_manifest(manifest, dataset_path)
    n_items = len(items)
    n_subs = sum(1 for v in substitute_manifest.values()
                 if v.get("substitute_sentence"))
    yield_pct = n_subs / max(1, n_items)
    audit = {
        "n_items": n_items,
        "n_substitute_hits": n_subs,
        "substitute_yield": yield_pct,
        "n_composites": len(composites),
        "balance": dict(Counter(c.label for c in composites)),
        "passes_yield_threshold": yield_pct >= substitute_yield_min,
    }
    if not audit["passes_yield_threshold"]:
        raise ValueError(
            f"substitute yield {yield_pct:.3f} < threshold "
            f"{substitute_yield_min:.3f}; refusing formal run"
        )
    return audit


def _confirm(prompt, *, yes):
    if yes:
        return True
    try:
        reply = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return reply in {"y", "yes"}


# --- 9. CLI / main ------------------------------------------------------- #

def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="write the frozen V7 manifest")
    prep.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    prep.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")

    subg = sub.add_parser(
        "substitute-generation",
        help="run the LLM-rewrite substitute-generation pass (V6 §6.3)",
    )
    subg.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    subg.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")

    smoke = sub.add_parser("smoke", help="run at most 8 V6 instrumentation calls")
    smoke.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    smoke.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "smoke")
    smoke.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    smoke.add_argument("--examples", type=int, default=2)
    smoke.add_argument("--no-resume", action="store_true")

    formal = sub.add_parser("run", help="run the frozen 2,000-call V6 formal pilot")
    formal.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    formal.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    formal.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "formal")
    formal.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    formal.add_argument("--no-resume", action="store_true")

    audit = sub.add_parser("audit",
                           help="pre-formal audit only: dataset digest, balance, "
                                "substitute yield, ties")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--yes", "-y", action="store_true")

    chain = sub.add_parser("all",
                           help="prepare -> substitute-generation -> audit -> "
                                "smoke -> formal (non-interactive, resumable)")
    chain.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    chain.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    chain.add_argument("--smoke-dir", type=Path, default=DEFAULT_ROOT / "smoke")
    chain.add_argument("--formal-dir", type=Path, default=DEFAULT_ROOT / "formal")
    chain.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    chain.add_argument("--yes", "-y", action="store_true")
    chain.add_argument("--skip-smoke", action="store_true")
    chain.add_argument("--skip-formal", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.command == "substitute-generation":
        items = load_boolq(args.dataset)
        _manifest, stats = build_substitute_manifest(
            items, cache_dir=args.cache_dir,
        )
        for k, v in stats.items():
            print(f"[substitute-gen] {k}: {v}")
        if not stats["passed_fail_fast"]:
            print(f"[substitute-gen] FAIL_FAST: unusable fraction "
                  f"{stats['unusable_fraction']:.3f} >= "
                  f"{SUBSTITUTE_FAIL_FAST_PCT}", flush=True)
            return 2
        return 0

    if args.command == "prepare":
        items = load_boolq(args.dataset)
        sub_path = args.cache_dir / "substitute_manifest.json"
        if not sub_path.exists():
            raise ValueError(
                f"substitute manifest missing at {sub_path}; run "
                "`substitute-generation` first"
            )
        substitute_manifest = json.loads(sub_path.read_text(encoding="utf-8"))
        composites = build_composite_questions(items, substitute_manifest)
        manifest = build_manifest(args.dataset, composites, substitute_manifest)
        _write_json(args.output, manifest)
        validate_manifest(manifest, args.dataset)
        print(f"Wrote frozen V6 manifest: {args.output}")
        return 0

    if args.command == "smoke":
        execute_run(
            mode="smoke", dataset_path=args.dataset,
            manifest_path=args.manifest, output_dir=args.output_dir,
            cache_dir=args.cache_dir, smoke_examples=args.examples,
            resume=not args.no_resume,
        )
        return 0

    if args.command == "run":
        execute_run(
            mode="formal", dataset_path=args.dataset,
            manifest_path=args.manifest, output_dir=args.output_dir,
            cache_dir=args.cache_dir, resume=not args.no_resume,
        )
        return 0

    if args.command == "audit":
        audit = _pre_formal_audit(args.dataset, args.manifest)
        for k, v in audit.items():
            print(f"{k}: {v}")
        return 0

    if args.command == "all":
        # 1. substitute-generation
        print("[all] step 1/5: substitute-generation (LLM rewrites, <=300 prereg / ~13K realized)", flush=True)
        items = load_boolq(args.dataset)
        _manifest, stats = build_substitute_manifest(items, cache_dir=args.cache_dir)
        for k, v in stats.items():
            print(f"[all] substitute-gen.{k}: {v}", flush=True)
        if not stats["passed_fail_fast"]:
            print("[all] substitute-generation FAIL_FAST; aborting", flush=True)
            return 2

        # 2. prepare
        print("[all] step 2/5: prepare (frozen V6 manifest, N=100)", flush=True)
        substitute_manifest_path = args.cache_dir / "substitute_manifest.json"
        substitute_manifest = json.loads(
            substitute_manifest_path.read_text(encoding="utf-8")
        )
        composites = build_composite_questions(items, substitute_manifest)
        manifest = build_manifest(args.dataset, composites, substitute_manifest)
        _write_json(args.manifest, manifest)
        validate_manifest(manifest, args.dataset)
        print(f"[all] wrote frozen V6 manifest: {args.manifest}", flush=True)

        # 3. audit
        print("[all] step 3/5: pre-formal audit", flush=True)
        audit = _pre_formal_audit(args.dataset, args.manifest)
        for k, v in audit.items():
            print(f"[all] audit.{k}: {v}", flush=True)
        if not audit["passes_yield_threshold"]:
            print("[all] audit failed; aborting", flush=True)
            return 2

        # 4. smoke
        if not args.skip_smoke or not (args.smoke_dir / "records.jsonl").exists():
            print("[all] step 4/5: smoke (8 calls)", flush=True)
            execute_run(
                mode="smoke", dataset_path=args.dataset,
                manifest_path=args.manifest, output_dir=args.smoke_dir,
                cache_dir=args.cache_dir, smoke_examples=2,
            )
        else:
            print("[all] step 4/5: smoke skipped (records present)", flush=True)

        if args.skip_formal:
            print("[all] step 5/5: formal skipped (--skip-formal)", flush=True)
            return 0

        # 5. formal
        print("[all] step 5/5: formal (~2,000 calls + substitute-gen above)",
              flush=True)
        if not _confirm("[all] proceed with formal run?", yes=args.yes):
            print("[all] aborted at user request", flush=True)
            return 1
        execute_run(
            mode="formal", dataset_path=args.dataset,
            manifest_path=args.manifest, output_dir=args.formal_dir,
            cache_dir=args.cache_dir, resume=True,
        )
        return 0

    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())