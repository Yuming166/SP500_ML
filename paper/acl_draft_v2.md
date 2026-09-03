# When Consensus Lies: Intervention-Tested Provenance for Reliable Multi-Agent LLM Decisions

**Anonymous ACL submission draft v2 — September 2026**

> Draft status: second version. Relative to v1 (`paper/acl_draft.md`), this
> draft (i) adds the preregistered **LLM-S&P500 V1→V2** agent-level replay
> (5,000 LLM calls over 500 decision dates), which replaces the statistical
> agent replay as the paper's external stress test; (ii) adds the
> **protocol-failure chain** (abstention–contract conflict, evidence-id
> hallucination, paired prompt repair) as a first-class contribution; and
> (iii) reorganizes the narrative around one arc: *consensus is unreliable →
> intervention-tested provenance ranks bad consensus (confirmatory, BoolQ) →
> the same mechanism, moved to a sequential real-world domain, yields
> calibration gains but no confirmatory routing gain → and the move itself
> surfaces reproducible protocol failure modes that we diagnose and repair
> under preregistration.* The compiled LaTeX version remains in `paper/latex/`
> and must be regenerated.

---

## Abstract

Agreement among language-model agents is often treated as independent
corroboration, even when the agents share a model, prompts, or evidence roots.
This creates *false consensus*: a highly agreed answer can be wrong because the
agents repeat the same unsupported or behaviorally irrelevant evidence. We
introduce an environment-controlled evaluation framework that records the graph
from sources to evidence, agents, and decisions, then applies paired evidence
interventions—removal, reversal, and substitution—without inspecting hidden
chain-of-thought. From observable answer changes and citation overlap we define
an outcome-independent Provenance–Intervention risk score, R_PI, for ranking
erroneous high-consensus decisions before their labels are revealed.

We evaluate five Qwen3.5-4B agents on frozen BoolQ splits. An initial held-out
run (200 questions, 4,000 calls) was directionally positive but inconclusive
(AUROC 0.605, CI [0.495, 0.721]). A preregistered, provenance-disjoint
replication (358 questions, 7,160 calls) passed the unchanged endpoint: among
300 high-consensus questions, 66 were wrong; R_PI ranked wrong consensus with
AUROC 0.705 ([0.620, 0.781]) and reduced retained error at 80% coverage from
0.220 to 0.133 (reduction 0.087, [0.046, 0.098]). Descriptive BoolQ-label
subgroups reverse direction, and citation sharing alone is at chance.

We then test how far the mechanism travels. In a preregistered, as-of-provenance
S&P 500 replay, five LLM agents answer 5-day-ahead direction questions on 500
decision dates (10,000 calls across two paired protocol versions), and a
trained provenance-style router reallocates trust. The router achieves the best
risk calibration in both versions (risk Brier 0.240, risk ECE 0.055 in V1), but
its risk–coverage advantage over the majority baseline is not confirmatory in
either version (V1: −0.0874, CI [−0.1994, +0.0736]; V2: +0.0317, CI [−0.0886,
+0.1805]). The V1→V2 pair itself isolates a reproducible deployment failure
mode: 73.6% of agent failures were *abstentions* that the output contract
rejected; a frozen prompt-only repair lifted paired validity from 68.2% to
75.6% (sign test p ≈ 7×10⁻⁹) without touching the contract or the parser.
These results support provenance-aware intervention as a falsifiable
reliability signal—not consensus, not confidence, not financial alpha, and not
universal cross-domain robustness—and they document, end to end, how
preregistered protocols catch and repair the failure modes that silent
pipelines normalize.

## 1. Introduction

Multi-agent language-model systems often improve a final answer by sampling
several agents, eliciting different roles, and aggregating their votes. Debate
and collaboration can improve reasoning and factuality, but the benefit of a
majority implicitly depends on diversity in the agents' information and errors.
Five agreeing agents are not five independent witnesses when they inherit the
same model bias or cite overlapping evidence. Recent studies find conformity
effects in LLM-agent groups, where apparent role diversity need not produce
independent judgments.

This paper studies a concrete failure mode: **high-consensus answers supported
by correlated evidence**. We call an answer a harmful false consensus when at
least 80% of agents agree and the consensus is wrong. Conventional confidence
and vote agreement describe the answer *as produced*, but not whether it would
respond appropriately if its claimed evidence changed. Citation correctness is
also insufficient: an agent can cite relevant text after committing to an
answer, and multiple citations may descend from one provenance root.

We instead make the evaluation environment—not the model—responsible for
evidence identity and interventions. Each agent receives a fixed subset of
evidence IDs and returns only an answer, confidence, and cited IDs. For the
same question–agent pair, the environment removes evidence, reverses its
direction, or substitutes an opposite-answer passage. We then measure whether
the answer changes, whether it remains inert under every intervention, and
whether agents reuse the same cited roots. No hidden reasoning trace is stored
or evaluated.

### 1.1 The narrative chain

The paper is organized as one deliberate chain of experiments, each link
motivated by the previous link's verdict (Table 1):

- **L1 — Mechanism exists (BoolQ, confirmatory PASS).** On a
  provenance-disjoint replication, the frozen R_PI ranks wrong high-consensus
  answers with AUROC 0.705, lower CI above chance, and cuts retained error at
  80% coverage. This is the paper's positive core: *observable response to
  controlled evidence interventions carries pre-outcome information about
  consensus failure.*

- **L2 — The signal is not label-invariant (BoolQ, descriptive FAIL).** The
  preregistered subgroup analysis reverses direction across native BoolQ
  labels, and the citation-sharing coordinate is at chance. The mechanism
  claim is narrowed to what actually carries signal: behavioral intervention
  response, not citation overlap.

- **L3 — The mechanism moves to a sequential real-world domain, but the
  confirmatory gain does not (S&P 500 statistical replay, FAIL → LLM replay,
  FAIL twice with structure).** V4/V5 statistical replays showed favorable
  selective operating points with negative transfer and unstable calibration.
  The new LLM-S&P500 V1/V2 replays keep the best-calibrated router (V1 risk
  ECE 0.055 vs. 0.174 for the majority baseline) yet cannot confirm an
  AURC advantage on 150 test dates—twice, with a sign flip between versions.
  The honest reading: intervention-tested provenance *transfers as a
  calibration discipline* even where it does not (yet) transfer as a
  confirmatory routing gain.

- **L4 — Protocol failure modes are themselves findings (V1→V2 paired
  repair, PASS).** 585 of 795 V1 agent failures (73.6%) were *correct
  abstentions* that the frozen output contract rejected—an interaction
  between role instructions and schema validation that silently biases any
  router built on fail-closed coverage. A preregistered, prompt-only repair
  on the identical manifest recovered the abstaining roles (consistency
  33→397 of 500) and significantly raised paired validity, while revealing a
  second failure mode (counterfactual-role evidence hallucination) that no
  end-to-end accuracy metric would have surfaced.

The chain is designed to be falsifiable at every link: each stage had a frozen
preregistration, a single primary endpoint, and all unfavorable results
reported.

Our contributions are:

1. **An intervention-based false-consensus protocol** combining an
   environment-maintained evidence graph with remove, reverse, and substitute
   conditions, evaluated on observable actions rather than hidden
   chain-of-thought.
2. **A frozen pre-outcome risk contract** (R_PI) combining complete
   intervention inertia, answer-flip inertia, and shared citations, with an
   explicit label-allowlist/denylist and a poison-test regression.
3. **A preregistered real-LLM result with failures preserved** (BoolQ V11.1
   FAIL, V12.1 PASS), including label-conditional heterogeneity and a
   citation-only null.
4. **A cross-domain stress program with a transfer boundary.** Statistical
   (V4/V5) and LLM-agent (V1/V2) S&P 500 replays under as-of provenance:
   best-in-class calibration, favorable selective operating points, but no
   confirmatory risk–coverage gain—an honest external-validity boundary.
5. **A preregistered protocol-failure chain with paired repair.** The
   abstention–contract conflict, quantified failure decomposition, and
   prompt-only paired fix (V1→V2) as a reusable methodology for multi-agent
   output-contract design.

## 2. Problem Formulation

### 2.1 Decisions, evidence, and provenance

For question q, let E_q = {e_1, e_2, e_3} be evidence units with
environment-assigned identities and source roots. Agent i receives a fixed
view V_qi ⊆ E_q and returns

  o_qi = (a_qi, c_qi, C_qi),

where a_qi ∈ {0,1} is the answer, c_qi ∈ [0,1] is confidence, and C_qi ⊆ V_qi
contains cited evidence IDs. The environment rejects citations not present in
the assigned view. Five fixed agents are used throughout: literal evidence
user, skeptical auditor, consistency checker, counterfactual reasoner, and
minimal judge.

The original-condition consensus is the majority answer ŷ_q with agreement
A_q. The high-consensus subset is fixed before evaluation as A_q ≥ 0.8. Only
after routing do we reveal the label y_q and compute z_q = 1[ŷ_q ≠ y_q]. The
primary task is to rank z_q = 1 above z_q = 0 within the high-consensus
subset using only pre-outcome observations.

### 2.2 Paired evidence interventions

Each original response is paired with three conditions: **remove** (delete an
assigned evidence unit), **reverse** (replace its directional content with an
opposite-answer form), and **substitute** (replace it with a separately
generated opposite-answer sentence constrained to topic, entities, and length
window). The same model, persona, schema, and partition are used across
conditions. The interventions approximate an operational faithfulness
question—does the agent's observable decision depend on the evidence it was
asked to use?

### 2.3 False consensus under dependence

Agreement can arise from independent evidence, shared valid evidence, or
common failure; a vote count cannot distinguish them. Our graph retains the
mapping source root → evidence ID → agent view → cited IDs → answer. Two
agents citing evidence derived from the same root do not constitute two
independent sources; we operationalize overlap with environment IDs, not
semantic similarity of rationales.

## 3. Provenance–Intervention Risk

### 3.1 Observable coordinates

For each question we compute three coordinates in [0,1]: **complete
intervention inertia** D_inert (fraction of agents whose answer never changes
under any intervention), **flip inertia** I_flip (one minus the mean
answer-flip rate over all 5×3 agent–intervention pairs), and **shared-source
fraction** F_shared (fraction of agents whose citations overlap a previously
observed agent under fixed order). High values are declared risk directions.

### 3.2 Frozen score

The V10.4 development split was used once to select among 66 convex
combinations on a 0.1-spaced simplex; the selected formula was frozen before
either validation experiment:

  R_PI(q) = 0.1·D_inert(q) + 0.3·I_flip(q) + 0.6·F_shared(q).

The implementation accepts only these three fields; labels, correctness, and
all outcome aliases are forbidden, enforced by a poison-test regression that
must leave the score unchanged.

### 3.3 Selective router

For target coverage κ = 0.8, the router sorts high-consensus questions by
increasing R_PI and retains the lowest-risk fraction. We report baseline
error, retained error, error reduction, AUROC, AUPRC, and question-cluster
bootstrap intervals. A lower retained error does not imply that abstained
questions have been solved.

### 3.4 Provenance-style routers for sequential decisions (AMIR)

For temporal markets we instantiate an adaptive provenance router (AMIR) and,
in the new LLM replay, its D12_v1 LLM-output analogue: a per-fold logistic
router over the five roles' long-vote indicators and confidences (10
features), fit on the replay's train window with frozen seasonal
standardization. Ranking quality is evaluated by the area under the
risk–coverage curve (AURC); paired 21-step moving-block bootstrap intervals
apply to all with-vs-without comparisons. Calibration (risk Brier, risk ECE)
is reported alongside ranking because §5.5 shows the two dissociate.

## 4. Experimental Setup

### 4.1 Datasets and environments

**BoolQ.** Official training split for V10.4 development; official validation
split for V11.1 and V12.1. A text-only eligibility rule extracts three
8–80-token sentences whose TF–IDF similarity falls inside a frozen
evidence-diversity band. V11.1 selects 100 yes and 100 no roots by salted
hashing; V12.1 uses all 358 remaining eligible validation roots (246 yes,
112 no), with zero root overlap with V11.1.

**FEVER structural audit.** A preregistered anti-herding audit found that
1,602 of 1,766 triples (90.7%) exceeded the maximum evidence-overlap
threshold; the formal run aborted before any LLM call. Evidence-redundant
clusters cannot study independent multi-agent corroboration—reported as a
dataset boundary.

**Financial replay (two generations).** The as-of environment provides 1,247
market rows (2021-05-10 → 2026-04-27 after frozen NaN gates), 29 features
across 7 provenance roots (market, VIX, macro, Google Trends, CBOE options,
ETF flows, mutual-fund flows), a 5-trading-day direction target, and strict
as-of visibility (every evidence item's available_at = publication_time =
decision time). Generation 1 (V4/V5) used 11 source-specialized statistical
agents with six outer folds. Generation 2 (**LLM-S&P500 V1/V2**, new in this
draft) replaces statistical agents with five frozen-prompt LLM roles on 500
salt-sampled decision dates (350 train / 150 test); every one of the 2,500
calls per version is an actual LLM call rendering the as-of packet
(29 z-scored evidence items) into a schema-validated decision with cited
evidence IDs.

### 4.2 Models and agents

All language-model calls use locally deployed instruction models through
OpenAI-compatible endpoints: Qwen3.5-4B (BoolQ), Hy-MT2-7B (LLM-S&P500 V1/V2).
Formal V11.1/V12.1 records achieved 100% first-pass schema validity. The
LLM-S&P500 calls carry a content-addressed response cache, salt-derived
seeds, and a per-claim evidence-id filter (D9_v1) that drops claims citing
ids outside the packet instead of rejecting the whole decision. Protocol
deviations are numbered and frozen in each preregistration (D1–D12 in V1;
D1–D4 carried in V2).

### 4.3 Protocol sequence

Table 1 separates development, auxiliary aborts, validation, and the two
replay generations.

| Stage | Version | Role | Scale | Outcome |
|---|---|---|---:|---|
| Dev | V10.4 | BoolQ development | 100 q / 2,000 calls | leaked co-primary invalidated |
| Gate | V11, V12 | rewrite gates | 0 evaluation calls | preregistered aborts |
| Valid | V11.1 | first held-out validation | 200 q / 4,000 calls | primary FAIL (CI crossed 0.5) |
| Valid | V12.1 | disjoint replication | 358 q / 7,160 calls | **primary PASS** |
| Replay G1 | V4, V5 | statistical-agent market replay | 1,247 rows / 6 folds | selective gains; AURC CI crosses 0 |
| Replay G2 | LLM-S&P500 V1 | LLM agents on market replay | 500 dates / 2,500 calls | yield 68.2%; AURC CI crosses 0 |
| Replay G2 | LLM-S&P500 V2 | paired prompt-only repair | same manifest / 2,500 calls | yield 75.6% (p≈7e−9); AURC CI crosses 0 |

V11.1/V12.1 amendments were frozen after observing only auxiliary rewrite
failures and before any validation-agent outcome. LLM-S&P500 V2 inherited
V1's frozen manifest verbatim (sha-pinned), making every V1→V2 comparison
date-paired.

### 4.4 Metrics and uncertainty

The BoolQ primary endpoint is AUROC(R_PI, z | A ≥ 0.8) with a 1,000-replicate
question-bootstrap 95% CI lower bound strictly above 0.5. Secondary metrics
include AUPRC, individual coordinates, error at 80% coverage, and preregistered
descriptive subgroups. The market experiments use AURC, routed error,
coverage, risk Brier/ECE, and 21-step moving-block bootstrap intervals. All
tests and ablations are reported, including unfavorable ones.

## 5. Results

### 5.1 L1: Real-LLM high-consensus risk (BoolQ)

| Protocol | High-consensus N | Wrong | Wrong rate | AUROC | 95% CI | AUPRC | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| V11.1 | 180 | 30 | 0.167 | 0.605 | [0.495, 0.721] | 0.237 | FAIL |
| V12.1 | 300 | 66 | 0.220 | **0.705** | **[0.620, 0.781]** | **0.414** | PASS |

V11.1 was directionally consistent but failed its frozen criterion. V12.1
used no overlapping provenance root and left score, threshold, endpoint, and
pass rule unchanged; its lower bound clears chance. The pooled secondary
analysis (480 high-consensus questions) gives AUROC 0.669 ([0.602, 0.738]).

### 5.2 L1: Selective routing

In V12.1, retaining the lowest-risk 80% of high-consensus questions reduces
error from 0.220 to 0.133 (reduction 0.087, [0.046, 0.098]); V11.1 gives
0.035 ([0.000, 0.069]). The rank router abstains on high-risk cases but does
not repair them; observable evidence response contains information not
captured by the vote alone.

### 5.3 L2: Which coordinates carry the signal?

| V12.1 coordinate | AUROC | 95% CI | AUPRC |
|---|---:|---:|---:|
| complete intervention inertia | 0.791 | [0.724, 0.850] | 0.489 |
| flip inertia | **0.807** | **[0.744, 0.863]** | **0.494** |
| shared-source fraction | 0.498 | [0.419, 0.571] | 0.224 |
| frozen combined R_PI | 0.705 | [0.620, 0.781] | 0.414 |

The behavioral intervention coordinates clearly outperform citation sharing
alone; the mechanism claim narrows to *provenance as the definition of
controlled interventions*, not citation counting. Observed answer-flip rates:
removal 0.470, reversal 0.327, substitution 0.301. Retuning the frozen 0.6
weight on the null coordinate would invalidate the replication and was not
done.

The preregistered descriptive subgroups reverse direction across native
BoolQ labels (yes: AUROC 0.834, n=210; no: 0.213, n=90). The aggregate PASS
is driven by the yes regime; the result must not be read as label-invariant
reliability.

### 5.4 L3 (G1): Statistical-agent market replay

| Router | Coverage | Routed error | AURC | Main comparison |
|---|---:|---:|---:|---|
| confidence baseline (V4) | 0.767 | 0.382 | 0.418 | — |
| CPR-Router (V4) | 0.824 | 0.387 | **0.366** | error Δ +0.0049 [−0.0295, 0.0377] |
| confidence baseline (V5) | 0.721 | 0.408 | 0.418 | — |
| AMIR (V5) | **0.907** | **0.360** | 0.411 | AURC Δ −0.0078 [−0.0607, 0.0459] |
| provenance baseline (V5) | 0.709 | 0.359 | 0.431 | — |

V4 improves global risk ordering but not error at its operating point;
removing the fixed source anchor lowers routed error (negative transfer). V5
descriptively dominates confidence on coverage and error and nearly matches
provenance error while covering 19.8 points more, but its preregistered AURC
interval crosses zero. AMIR satisfies every hard intervention constraint, yet
the shift coefficient is nonzero in one of six folds, and removing the hard
constraint changes AURC by <0.001: a useful selective operating point, not
isolated cross-domain adaptation.

### 5.5 L3 (G2): LLM agents on the market replay (V1/V2, new)

Five frozen-prompt LLM roles answer 5-day-ahead direction questions on 500
as-of decision dates; routers aggregate their votes (2,500 calls per
version; identical manifest).

**V1 router table (test window, 150 dates).**

| Router | Coverage | Routed error | Risk Brier | Risk ECE |
|---|---:|---:|---:|---:|
| majority | 1.000 | 0.400 | 0.295 | 0.174 |
| mean confidence long | 1.000 | 0.400 | 0.269 | 0.174 |
| provenance baseline | 0.933 | 0.379 | 0.258 | 0.156 |
| single min agent | 0.953 | 0.406 | 0.279 | 0.197 |
| AMIR-style (10-feature logistic) | 0.920 | 0.406 | **0.240** | **0.055** |

**Primary endpoint (both versions).**

| Version | AURC(amir) − AURC(majority) | 95% paired moving-block CI | Verdict |
|---|---:|---:|---|
| V1 | −0.0874 | [−0.1994, +0.0736] | FAIL |
| V2 | +0.0317 | [−0.0886, +0.1805] | FAIL |

The point estimate flips sign between paired versions while both intervals
cross zero: on 150 test dates the routing gain is within noise. What
replicates is **calibration**: the provenance-style router attains the best
risk Brier and a risk ECE (0.055) three times lower than the majority
baseline (0.174) in V1. Consistent with G1 (§5.4), the mechanism transfers
across domains as a *calibration and selective-coverage discipline*, while
confirmatory risk–coverage gains do not (yet) transfer at this sample size
and agent strength.

### 5.6 L4: The V1→V2 paired protocol repair (new)

Failure decomposition of V1's 795 failed calls:

| Failure shape | n | % of failures | Roles |
|---|---:|---:|---|
| empty `claims` on abstention | 585 | 73.6% | consistency 458, skeptic 115, literal 12 |
| evidence id outside packet | 195 | 24.5% | cf 104, literal 54, min 19, skeptic 18 |
| JSON truncation/malformed | 15 | 1.9% | scattered |

The dominant mode is a **contract interaction, not model incapacity**: the
consistency and skeptic role instructions tell the model to abstain when
roots disagree, the model abstains with an empty claims list, and the frozen
output contract rejects decisions without at least one cited claim—so the
pipeline fail-closed precisely when agents behaved as designed, biasing every
router's effective coverage.

The preregistered V2 repair froze two prompt-level changes—(F1) every
abstaining decision must still cite the conflicting evidence with
stance='attacks'; (F2) evidence ids must be copied verbatim from a numbered
catalog—on V1's manifest, sha-pinned, with no change to contracts, parsers,
dates, seeds, or routers.

| Role (of 500) | V1 valid | V2 valid | Δ |
|---|---:|---:|---:|
| consistency | 33 | 397 | **+364** |
| skeptic | 367 | 472 | +105 |
| literal | 434 | 386 | −48 |
| cf | 390 | 166 | **−224** |
| min | 481 | 469 | −12 |
| **total** | **1,705 (68.2%)** | **1,890 (75.6%)** | paired sign test p ≈ 7×10⁻⁹ |

The repair recovered both abstaining roles and overall paired validity rose
significantly, but the catalog format induced a new hallucination mode in the
counterfactual role (citing bare feature names like `market_trend_60::t=…`
as source prefixes), which the per-claim filter then drops. The primary
endpoint remained non-confirmatory under 7.4 points higher yield—so V1's
interval width was not a yield artifact, and the router's advantage is
bounded by signal, not by pipeline coverage. We report both the successful
fix and the induced regression because the pair is the evidence: multi-agent
output contracts and role instructions interact non-trivially, and only a
preregistered paired design can attribute the delta to the prompt change.

## 6. Validity Audits and Failure Analysis

### 6.1 Outcome leakage discovered after V10.4

V10.4 originally emitted a positive-looking shared-weighted AUROC of 0.959.
A post-run audit found the implementation multiplied citation overlap by a
term containing answer correctness; the metric was outcome-contaminated and
is invalidated. This failure motivated R_PI's explicit allowlist/denylist
contract and its poison test.

### 6.2 Auxiliary-stage aborts

V10.1–V10.3, V11, and V12 stopped under registered rewrite-quality gates
before evaluation outcomes existed. These aborts are instrumentation
evidence, not router results.

### 6.3 Evidence redundancy as a dataset property

The FEVER audit (90.7% over-redundant triples) shows false-consensus
experiments require evidence units that are neither unrelated nor
near-duplicates; persona diversity cannot manufacture independent
information.

### 6.4 Ranking versus calibration

Across V4, V5, and LLM-V1, structural scores rank errors and calibrate
probabilities to different degrees: V5's uncalibrated ranking ablation has
good AURC but poor Brier/ECE, while LLM-V1's router shows the best
calibration of its table without a confirmatory AURC gain. Evaluation should
report the two separately; they do not imply each other.

### 6.5 Fail-closed coverage as a hidden bias (new)

Any router evaluated on fail-closed multi-agent output inherits the
abstention pattern of its agents. In V1, one role's effective coverage was
6.6%; after repair, 79.4%. Coverage shifts of this size change which dates
are routable at all and can masquerade as routing improvements or
degradations. The V1→V2 pair is a controlled demonstration; we recommend
validity-fraction decomposition (§5.6 table) as a standard reporting
requirement for LLM multi-agent pipelines.

## 7. Related Work

**Multi-agent reasoning.** Multi-agent debate and evaluator teams improve
generation and judging; our question is whether agreement is trustworthy when
evidence and model errors are dependent. Empirical work on conformity in
LLM-agent groups motivates testing provenance rather than assuming personas
are independent witnesses.

**Truthfulness, factuality, and attribution.** TruthfulQA, FEVER, and
attribution studies evaluate accuracy and citation support; our protocol adds
a behavioral criterion—cited evidence should affect the decision under
controlled changes—complementing citation correctness rather than replacing
it.

**Faithful explanations and interventions.** We avoid asking models to expose
private reasoning and instead manipulate environment-provided evidence while
observing answer changes: a black-box intervention test.

**Selective prediction and calibration.** Selective classification abstains
to reduce retained risk; calibration and conformal methods quantify
uncertainty. We use risk–coverage curves and empirical temporal quantiles;
for dependent financial observations we do not claim exact IID conformal
coverage.

**LLM output-structure reliability.** Schema-constrained decoding and
function-call evaluations study whether models emit parseable outputs. Our
V1→V2 pair adds a preregistered, paired treatment of an upstream cause:
role-conditioned *behavioral* instructions (abstain under conflict) colliding
with validity contracts, and the coverage bias such collisions induce in
downstream aggregation.

**Distribution shift.** Group DRO and domain adaptation study transfer under
changing distributions; AMIR's gate and the LLM replays' temporal splits are
in this spirit, but their present results do not establish successful
cross-domain adaptation.

## 8. Limitations

First, completed language-model experiments use Qwen3.5-4B (BoolQ) and
Hy-MT2-7B (replay G2). Different personas of one base model are not
independent models; the preregistered cross-model matrix (Qwen →
Llama-3.1-8B; replay on a stronger API model) remains the largest gap. In
particular the V1/V2 primary intervals are wide (half-width ≈ 0.13) at 150
test dates; a confirmatory routing claim requires a larger test window or a
stronger agent.

Second, V12.1's label subgroups reverse direction; the score may exploit a
BoolQ-specific prior. The aggregate result must not be generalized to
arbitrary binary QA.

Third, R_PI was selected on one 100-question development split; its 0.6
weight on the null citation coordinate is not supported by the component
analysis, but retuning post hoc would invalidate the replication.

Fourth, generated substitutes may differ in fluency or logical force from
original passages; removal, reversal, and substitution measure different
response behaviors and do not identify one unique causal mechanism.

Fifth, abstention defers rather than solves high-risk cases; operational use
needs a fallback policy or human review.

Sixth, the financial replays are retrospective. Market agents (statistical in
G1, LLM in G2) are not deployed traders; the replays cannot establish alpha,
causal market impact, or prospective performance. The G2 backend was a local
7B model substituted under a registered deviation (D7_v1) after the
originally intended API endpoint became unreachable; the GPT-class replication
is preregistered as the V2 `--relay` arm and not yet run.

Finally, related-work coverage and formatting are preliminary; a submission
version needs a systematic literature audit, released prompts and licenses,
compute accounting, and error examples.

## 9. Ethics and Broader Impact

False consensus is consequential in medicine, law, finance, and content
moderation. A provenance-aware risk score can help systems recognize when
many agreeing agents merely repeat one weak source; it can also create false
reassurance—a low score is not proof of correctness. We recommend the score
for triage and evidence acquisition, not as an autonomous truth certificate.
The fail-closed analysis (§6.5) carries an additional deployment warning:
validity filters silently reshape which inputs a system will answer, and
teams should audit abstention-induced coverage as a first-class metric. The
experiments use existing QA datasets and automated model outputs; no private
human data or hidden chain-of-thought is collected. Financial results are
research diagnostics, not investment advice.

## 10. Reproducibility

Every formal stage has a versioned preregistration, deterministic selection
manifest, fixed salts, a response cache, expected record counts, bootstrap
seeds, and post-run integrity audits. V11.1 contains exactly 4,000 unique
records; V12.1 exactly 7,160; LLM-S&P500 V1 and V2 exactly 2,500 each, with
V2's manifest sha-pinned to V1's (`6fd3c3ed…`) so the pair is date-identical.
All 12 (V1) and 4 (V2) protocol deviations are enumerated in their
preregistrations, including the backend substitution (D7_v1) and the
pre-freeze min-role prompt repair (D2_v2). Token accounting for the two
replay runs is retained per call (≈15.3M and ≈16.2M tokens respectively).
The repository test suite passes at the current checkpoint.

## 11. Conclusion

Consensus is reliable only to the extent that its evidence and errors are
independent. We presented an environment-controlled paired-intervention
framework that tests this assumption, a frozen pre-outcome risk score that
confirmatorily ranks wrong high-consensus decisions on BoolQ, and a
preregistered transfer program into a sequential real-world domain that
locates the boundary: calibration transfers, confirmatory risk–coverage gains
do not (yet), and the transfer itself exposes a reproducible protocol-failure
chain—an abstention–contract conflict responsible for nearly three quarters
of agent failures—that we diagnose and repair under a paired preregistration.
The bounded conclusion stands: provenance-aware interventions are a
falsifiable reliability signal, not a universal truth detector. The next
decisive experiments are the frozen cross-model matrix (BoolQ and replay arms
on a second, stronger model family) and a test window large enough to
resolve the routing effect the calibration results predict.

## References

- Anastasios N. Angelopoulos and Stephen Bates. 2021. [A Gentle Introduction
  to Conformal Prediction and Distribution-Free Uncertainty
  Quantification](https://arxiv.org/abs/2107.07511).
- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael
  Collins, and Kristina Toutanova. 2019. [BoolQ: Exploring the Surprising
  Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/).
  NAACL-HLT.
- Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor
  Mordatch. 2023. [Improving Factuality and Reasoning in Language Models
  through Multi-agent Debate](https://arxiv.org/abs/2305.14325).
  arXiv:2305.14325.
- Yonatan Geifman and Ran El-Yaniv. 2017. [Selective Classification for Deep
  Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
  NeurIPS.
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. [On
  Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
  ICML.
- Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu
  Yang, Shuming Shi, and Zhaopeng Tu. 2023. [Encouraging Divergent Thinking
  in Large Language Models through Multi-Agent
  Debate](https://arxiv.org/abs/2305.19118).
- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. [TruthfulQA: Measuring
  How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/).
  ACL.
- Benjamin Muller, John Wieting, Jonathan H. Clark, Tom Kwiatkowski, Sebastian
  Ruder, Livio Baldini Soares, Roee Aharoni, Jonathan Herzig, and Xinyi Wang.
  2023. [Evaluating and Modeling Attribution for Cross-Lingual Question
  Answering](https://aclanthology.org/2023.emnlp-main.10/). EMNLP.
- Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. 2020.
  [Distributionally Robust Neural Networks for Group
  Shifts](https://arxiv.org/abs/1911.08731). ICLR.
- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit
  Mittal. 2018. [FEVER: a Large-scale Dataset for Fact Extraction and
  VERification](https://aclanthology.org/N18-1074/). NAACL-HLT.
- Min Choi, Keonwoo Kim, Sungwon Chae, and Sangyeop Baek. 2025. [An Empirical
  Study of Group Conformity in Multi-Agent
  Systems](https://aclanthology.org/2025.findings-acl.265/). Findings of ACL.

## Appendix A. Frozen risk contract

```text
Inputs allowed:
    D_inert
    flip_inertia
    frac_shared

Inputs forbidden:
    label, gold_binary, correct, any_wrong, harmful_fc,
    aliases or transformations of the above

Score:
    R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared

Primary subset:
    original agreement >= 0.8

Primary endpoint (BoolQ):
    AUROC(R_PI, consensus_wrong)

Primary endpoint (replay G2):
    AURC(amir_router) - AURC(majority) < 0,
    PASS iff 95% paired moving-block bootstrap upper bound < 0
```

## Appendix B. Evidence chain: experiment → verdict → narrative role

| Experiment | Scale | Verdict | Narrative link |
|---|---|---|---|
| FEVER audit | 1,766 triples | abort (90.7% redundant) | dataset boundary; motivates diversity band |
| V10.4 dev | 100 q / 2,000 calls | leaked metric invalidated | motivates frozen output-free score |
| V11.1 | 200 q / 4,000 calls | FAIL (CI crosses 0.5) | power discipline; motivates disjoint replication |
| V12.1 | 358 q / 7,160 calls | **PASS** (AUROC 0.705) | **L1 core positive claim** |
| subgroup analysis | descriptive | direction reverses | **L2 mechanism narrowing** |
| V4/V5 replay | 1,247 rows / 6 folds | selective gains; AURC crosses 0 | L3 transfer boundary (G1) |
| LLM-S&P500 V1 | 500 dates / 2,500 calls | yield 68.2%; AURC crosses 0; best ECE 0.055 | **L3 transfer boundary (G2)** + L4 diagnosis input |
| LLM-S&P500 V2 | same manifest / 2,500 calls | yield 75.6% (p≈7e−9); AURC crosses 0 | **L4 paired repair**; confirms yield ≠ signal |

## Appendix C. Planned completion matrix

| Source model/data | Target model/data | Status | Role |
|---|---|---|---|
| Qwen3.5-4B / BoolQ dev | Qwen3.5-4B / disjoint BoolQ validation | complete | within-model replication |
| Qwen3.5-4B / frozen TQA-BoolQ | Llama-3.1-8B-Instruct / same items | pending | main cross-model claim |
| remove + reverse | substitute + alias/duplicate holdout | pending | mechanism transfer |
| Hy-MT2-7B / S&P 500 replay | GPT-5.x relay / same manifest (`--relay`) | preregistered, pending relay availability | agent-strength scaling |
| Hy-MT2-7B / 150 test dates | same backend / widened test window | proposed V3 | power for the routing endpoint |
