# Pilot-LLM V12.1 auxiliary amendment

Protocol version: `pilot-llm-v12.1-2026-09-02`

Frozen after V12's auxiliary fail-fast abort and before any V12/V12.1 validation-agent output.

## Inheritance

- Inherit the exact V12 selection of 358 untouched BoolQ validation roots without changing order, labels, `cqid`s, agents, prompts, evidence partitions, four conditions, risk score, high-consensus threshold, primary endpoint, bootstrap rule, or four-worker assignment.
- Inherit exactly the 1,073 usable V12 substitutes.
- The only unusable V12 evidence ID is frozen as `boolq-1592052e5f54e039-e03`.
- Parent substitute manifest SHA-256: `779ad8caee82b029cff504fccf55bca10735f4c4bb7f38b838ff7bdf753f4ffc`.
- No validation-agent output existed before this amendment.

## Bounded second repair

The frozen failed item has a 13-token source and an allowed window of 7 through 19 tokens. Its initial candidate and first repair were both 20 tokens. V12.1 makes exactly one additional model repair call with seed 20260924. The prompt requires one plain-text sentence, preserves the topic and named entities, introduces no new entity, supports the opposite BoolQ answer, and targets exactly 13 whitespace tokens while restating the 7-to-19-token window.

The result is accepted only if the unchanged single-line parser succeeds and its length is within 7 through 19 tokens. A short result may receive the already frozen repeated neutral suffix `in the described local situation.` until first entering the window. An overlong result is not truncated. If this one repair is unusable, V12.1 aborts before smoke or formal calls.

No other substitute receives a new call or content change.

## Confirmatory and execution contract

- Primary endpoint: `AUROC(R_PI, consensus_wrong | original_agreement >= 0.8)`.
- Score: `R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared` using only pre-outcome fields.
- Bootstrap: 1,000 question replicates, seed 20260921.
- Pass rule and count gates are identical to V12.
- Four endpoint-concurrent question-block workers and deterministic merge are identical to V12.
- Smoke: four questions, one per shard, exactly 80 unique successful logical calls.
- Formal: all 358 questions, exactly 7,160 unique successful logical calls.
- No interim metrics, early stopping, sample extension, endpoint promotion, or post-result exclusion.

V12 remains an auxiliary-stage abort. V12.1 is the only version eligible to produce the prospective validation result.
