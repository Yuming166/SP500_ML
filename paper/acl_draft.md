# When Consensus Lies: Intervention-Tested Provenance for Reliable Multi-Agent LLM Decisions

**Anonymous ACL submission draft — September 2026**

> Draft status: companion content manuscript based on the currently frozen
> experiments. The compiled ACL LaTeX version is in `paper/latex/`. Author
> information, a second model family, and final error analysis remain to be
> added. Exploratory financial results are explicitly separated from
> confirmatory language-model evidence.

## Abstract

Agreement among language-model agents is often treated as independent
corroboration, even when the agents share a model, prompts, or evidence roots.
This creates *false consensus*: a highly agreed answer can be wrong because the
agents repeat the same unsupported or behaviorally irrelevant evidence. We
introduce an environment-controlled evaluation framework that records the graph
from sources to evidence, agents, and decisions, then applies paired evidence
interventions—removal, reversal, and substitution—without inspecting hidden
chain-of-thought. From observable answer changes and citation overlap, we define
an outcome-independent Provenance--Intervention risk score, (R_{PI}), for
ranking erroneous high-consensus decisions before their labels are revealed.

We evaluate five Qwen3.5-4B agents on frozen BoolQ splits. An initial held-out
run with 200 questions and 4,000 calls was directionally positive but
inconclusive (AUROC (0.605), 95% CI ([0.495,0.721])). A preregistered,
provenance-disjoint replication used every remaining eligible validation root:
358 questions and 7,160 calls. Among 300 high-consensus questions, 66 were
wrong. The frozen (R_{PI}) ranked wrong consensus with AUROC (0.705)
(([0.620,0.781])) and reduced retained error at 80% coverage from (0.220)
to (0.133) (reduction (0.087), ([0.046,0.098])). The aggregate effect is
not universal: descriptive BoolQ-label subgroups reverse direction, and citation
sharing alone is at chance. A leakage-controlled S&P 500 replay further shows
that intervention signatures can support useful risk ordering while fixed
cross-domain weights and thresholds fail to transfer reliably. These results
support provenance-aware intervention as a falsifiable reliability signal, not
consensus, confidence, financial alpha, or universal cross-model robustness.

## 1. Introduction

Multi-agent language-model systems often improve a final answer by sampling
several agents, eliciting different roles, and aggregating their votes. Debate
and collaboration can improve reasoning and factuality, but the benefit of a
majority implicitly depends on diversity in the agents' information and errors.
Five agreeing agents are not five independent witnesses when they inherit the
same model bias or cite overlapping evidence. Indeed, recent studies find
conformity effects in LLM-agent groups, where apparent role diversity need not
produce independent judgments.

This paper studies a concrete failure mode: **high-consensus answers supported
by correlated evidence**. We call an answer a harmful false consensus when at
least 80% of agents agree and the consensus is wrong. Conventional confidence
and vote agreement describe the answer *as produced*, but not whether it would
respond appropriately if its claimed evidence changed. Citation correctness is
also insufficient: an agent can cite relevant text after committing to an
answer, and multiple citations may descend from one provenance root.

We instead make the evaluation environment—not the model—responsible for
evidence identity and interventions. Each agent receives a fixed subset of
evidence IDs and returns only a binary answer, confidence, and cited IDs. For
the same question-agent pair, the environment removes evidence, reverses its
direction, or substitutes an opposite-answer passage. We then measure whether
the answer changes, whether it remains inert under every intervention, and
whether agents reuse the same cited roots. No hidden reasoning trace is stored
or evaluated.

The central hypothesis is deliberately narrow:

> Observable response to controlled evidence interventions and environment-held
> provenance can rank wrong high-consensus decisions before the outcome is
> revealed.

Our experiments evolved through frozen protocols. A development run exposed an
outcome-leaking metric; we invalidate that metric rather than retain its
positive-looking result. A first untouched BoolQ validation study was
underpowered under its registered confidence-interval rule. A larger,
provenance-disjoint replication then passed the unchanged endpoint. We report
all three stages because the instrumentation failures and negative results are
part of the method's validity evidence.

Our contributions are:

1. **An intervention-based false-consensus protocol.** It combines an
   environment-maintained evidence graph with remove, reverse, and substitute
   conditions, and evaluates observable actions rather than hidden
   chain-of-thought or self-reported explanations.
2. **A frozen pre-outcome risk contract.** The Provenance--Intervention score
   combines complete intervention inertia, answer-flip inertia, and shared
   citations while explicitly forbidding labels or correctness fields.
3. **A preregistered real-LLM result with failures preserved.** On a 358-question
   Qwen3.5-4B BoolQ replication, the score achieves AUROC (0.705) with a lower
   confidence bound above chance and reduces error at 80% coverage. We also
   expose severe label-conditional heterogeneity and a citation-only null.
4. **A cross-domain stress test and transfer boundary.** A causal, temporal
   S&P 500 replay shows favorable selective operating points but also negative
   transfer, unstable calibration, and inconclusive primary intervals. Finance
   is an external reliability test, not the paper's main positive claim.

## 2. Problem Formulation

### 2.1 Decisions, evidence, and provenance

For question (q), let (E_q=\{e_1,e_2,e_3\}) be evidence units with
environment-assigned identities and source roots. Agent (i\) receives a fixed
view (V_{qi}\subset E_q) and returns

\[
o_{qi}=(a_{qi}, c_{qi}, C_{qi}),
\]

where (a_{qi}\in\{0,1\}) is the answer, (c_{qi}\in[0,1]\) is confidence,
and (C_{qi}\subseteq V_{qi}\) contains cited evidence IDs. The environment
rejects citations not present in the assigned view. Five fixed agents are used
throughout this study.

The original-condition consensus is the majority answer

\[
\hat y_q=\mathop{\mathrm{majority}}_i a_{qi}^{(0)},
\]

with agreement (A_q\), the fraction of agents taking (\hat y_q\). The
high-consensus subset is fixed before evaluation as (A_q\ge 0.8\). Only after
routing do we reveal the native label (y_q\) and compute

\[
z_q=\mathbb{1}[\hat y_q\ne y_q].
\]

The primary task is to rank (z_q=1\) above (z_q=0\) within the
high-consensus subset using only pre-outcome observations.

### 2.2 Paired evidence interventions

Each original response is paired with three conditions:

- **remove:** delete an assigned evidence unit;
- **reverse:** replace its directional content by an opposite-answer form;
- **substitute:** replace it with a separately generated opposite-answer
  sentence constrained to the original topic, entities, and length window.

The same model, agent persona, response schema, and evidence partition are used
across conditions. The interventions are not natural perturbation robustness
tests alone: they approximate an operational faithfulness question—does the
agent's observable decision depend on the evidence it was asked to use?

### 2.3 False consensus under dependence

Agreement can arise for at least three distinct reasons: independent evidence,
shared valid evidence, or common failure. A vote count cannot distinguish them.
Our graph therefore retains the mapping

```text
source root -> evidence ID -> agent view -> cited IDs -> answer.
```

Two agents citing evidence derived from the same root do not constitute two
independent sources. We operationalize overlap using environment IDs rather
than semantic similarity between model-generated rationales.

## 3. Provenance--Intervention Risk

### 3.1 Observable coordinates

For each question, we compute three coordinates in ([0,1]):

1. **Complete intervention inertia** (D_{\mathrm{inert}}): the fraction of
   agents whose answer never changes under remove, reverse, or substitute.
2. **Flip inertia** (I_{\mathrm{flip}}): one minus the mean answer-flip rate
   over all (5\times3) agent-intervention pairs.
3. **Shared-source fraction** (F_{\mathrm{shared}}): the fraction of agents
   whose original cited evidence overlaps a previously observed agent under a
   fixed agent order.

High values are declared risk directions: an agent team that remains unchanged
when its evidence is removed or contradicted is behaviorally suspicious; shared
citations reduce the effective independence of agreement.

### 3.2 Frozen score

The V10.4 development split was used once to select among 66 convex
combinations on a 0.1-spaced simplex. The selected formula was frozen before
either validation experiment:

\[
R_{PI}(q)=0.1D_{\mathrm{inert}}(q)
          +0.3I_{\mathrm{flip}}(q)
          +0.6F_{\mathrm{shared}}(q).
\]

The implementation accepts only these three fields. Labels, correctness,
harmful-consensus indicators, and aliases of these outcomes are forbidden. A
regression test poisons outcome fields while holding the three inputs fixed and
requires the score to remain identical.

### 3.3 Selective router

For a target coverage (\kappa=0.8\), the router sorts high-consensus questions
by increasing (R_{PI}\) and retains the lowest-risk fraction. This is a
selective prediction evaluation, not a mechanism for changing the answer. We
report baseline error, retained error, error reduction, AUROC, AUPRC, and
question-cluster bootstrap intervals. A lower retained error does not imply
that abstained questions have been solved.

### 3.4 Cross-domain adaptive router

To test whether intervention structure transfers beyond language questions, we
also instantiate an Adaptive Mechanism-Invariant Router (AMIR) in a temporal
market replay. A source ranker learns the three common coordinates from frozen
V12.1 records. A target ranker uses analogous graph interventions over
provenance-root probabilities. A shift-monotone gate combines standardized
source and target scores:

\[
g(x)=\sigma(\alpha-\beta d(x)),\quad \beta\ge0,
\]
\[
s(x)=g(x)s_{src}(x)+(1-g(x))s_{tgt}(x),
\]

where (d(x)) is standardized distance from the source feature distribution.
The target ranker imposes non-negative risk coefficients and a hard minimum
logit rise under declared graph interventions. Ranking and Platt calibration
are trained in temporally separated slices. This component is exploratory
because it was designed after earlier replay outcomes.

## 4. Experimental Setup

### 4.1 Datasets

**BoolQ.** BoolQ contains naturally occurring yes/no questions paired with
Wikipedia passages. We use the official training split only for V10.4
development and the official validation split for V11.1 and V12.1. A text-only
eligibility rule extracts three 8--80-token sentences whose TF--IDF similarity
falls inside a frozen evidence-diversity band. V11.1 selects 100 yes and 100 no
roots by salted hashing. V12.1 uses all 358 remaining eligible validation roots
(246 yes and 112 no), with zero root overlap with V11.1.

**FEVER structural audit.** Earlier experiments used FEVER evidence clusters.
A preregistered anti-herding audit found that 1,602 of 1,766 triples (90.7%)
exceeded the maximum evidence-overlap threshold; only 29 triples (1.6%)
survived the full band. The formal run therefore aborted before any LLM call.
We treat this as a dataset boundary: evidence-redundant FEVER clusters do not
provide the diversity needed to study independent multi-agent corroboration.

**Financial replay.** The external test uses 1,247 as-of market rows, an
expanding 504-row training window, a five-row label gap, 126-row tests, and six
outer folds. Eleven source-specialized statistical agents map to seven
provenance roots. Features, quality estimates, calibration, and routing use
only information visible at the decision time. The target is five-trading-day
S&P 500 direction; return statistics are secondary diagnostics only.

### 4.2 Model and agents

All reported language-model calls use a locally deployed Qwen3.5-4B
instruction model through an OpenAI-compatible endpoint. The five fixed
personas are literal evidence user, skeptical auditor, consistency checker,
counterfactual reasoner, and minimal judge. Each receives a deterministic
two-of-three evidence subset. Outputs contain a definite yes/no answer,
confidence, and cited evidence IDs. Formal V11.1 and V12.1 records achieved
100% schema validity and 100% first-pass validity.

We do not claim physical four-GPU execution. V12.1 used four concurrent client
workers against one endpoint; server-side device allocation was unobservable.

### 4.3 Protocol sequence

Table 1 separates development, auxiliary aborts, and validation.

| Version | Role | Questions | Logical calls | Outcome |
|---|---|---:|---:|---|
| V10.4 | BoolQ development | 100 | 2,000 | completed; leaked co-primary invalidated |
| V11 | auxiliary rewrite gate | 200 | 0 evaluation calls | preregistered abort at 597/600 rewrites |
| V11.1 | first held-out validation | 200 | 4,000 | primary FAIL; CI crossed 0.5 |
| V12 | auxiliary rewrite gate | 358 | 0 evaluation calls | preregistered abort at 1,073/1,074 |
| V12.1 | disjoint replication | 358 | 7,160 | primary PASS |

V11.1 and V12.1 amendments were frozen after observing only auxiliary rewrite
length/format failures and before any validation-agent outcome. They inherit
the parent selections and change only bounded rewrite repair.

### 4.4 Metrics and uncertainty

The single primary endpoint is

\[
\mathrm{AUROC}(R_{PI},z\mid A\ge0.8).
\]

The pass rule requires at least 80 high-consensus questions, at least ten
positive and ten negative cases, and a 1,000-replicate question-bootstrap 95%
CI lower bound strictly above 0.5. Secondary metrics include AUPRC, individual
coordinates, all-question harmful-consensus ranking, intervention flip rates,
and error at 80% coverage. V12.1 also preregisters descriptive native-label
subgroups and a pooled V11.1+V12.1 analysis; neither can replace the primary
endpoint.

The market experiments use AURC, routed error, coverage, calibration, and
21-row moving-block bootstrap intervals. All tests and ablations are reported,
including unfavorable ones.

## 5. Results

### 5.1 Real-LLM high-consensus risk

Table 2 contains the two untouched validation evaluations.

| Protocol | High-consensus N | Wrong | Wrong rate | AUROC | 95% CI | AUPRC | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| V11.1 | 180 | 30 | 0.167 | 0.605 | [0.495, 0.721] | 0.237 | FAIL |
| V12.1 | 300 | 66 | 0.220 | **0.705** | **[0.620, 0.781]** | **0.414** | PASS |

V11.1 was directionally consistent but failed its frozen confidence-bound
criterion. V12.1 used no overlapping provenance root and left the score,
threshold, endpoint, and pass rule unchanged. Its lower bound clears chance.
The secondary pooled analysis across 480 high-consensus questions gives AUROC
(0.669), 95% CI ([0.602,0.738]).

### 5.2 Selective routing

In V12.1, retaining the lowest-risk 80% of high-consensus questions reduces
error from (0.220) to (0.133). The absolute reduction is (0.087), with
bootstrap interval ([0.046,0.098]). V11.1 gives a smaller reduction from
(0.167) to (0.132), difference (0.035), interval ([0.000,0.069]).

These are rank-router results: the method abstains on high-risk cases but does
not repair their answers. Nevertheless, V12.1 shows that observable evidence
response contains information not captured by the original vote alone.

### 5.3 Which coordinates carry the signal?

| V12.1 coordinate | AUROC | 95% CI | AUPRC |
|---|---:|---:|---:|
| complete intervention inertia | 0.791 | [0.724, 0.850] | 0.489 |
| flip inertia | **0.807** | **[0.744, 0.863]** | **0.494** |
| shared-source fraction | 0.498 | [0.419, 0.571] | 0.224 |
| frozen combined (R_{PI}) | 0.705 | [0.620, 0.781] | 0.414 |

The two behavioral intervention coordinates clearly outperform citation sharing
alone. This result narrows the paper's mechanism claim: provenance is valuable
when it defines controlled interventions and dependence structure; merely
counting shared citations is not sufficient. Because the frozen formula assigns
0.6 weight to the null coordinate, a new score could likely improve, but
retuning on V12.1 would invalidate the replication and is not performed.

The observed answer-flip rates were 0.470 for removal, 0.327 for reversal, and
0.301 for substitution. Different interventions are therefore not equivalent;
simple removal produced the most observable sensitivity in this setup.

### 5.4 Native-label heterogeneity

The preregistered descriptive analysis reveals a major limitation:

| Native BoolQ label | High-consensus N | Wrong | AUROC | 95% CI |
|---|---:|---:|---:|---:|
| yes | 210 | 55 | 0.834 | [0.747, 0.913] |
| no | 90 | 11 | 0.213 | [0.112, 0.339] |

The directions reverse. The aggregate PASS is driven by the yes regime and
cannot be interpreted as label-invariant reliability. Plausible mechanisms
include asymmetric evidence construction, yes/no response priors, and
intervention polarity, but V12.1 was not designed to distinguish them. We make
no corrective subgroup weighting after observing the result.

### 5.5 Financial external validation

Table 4 summarizes the two post-LLM historical routers.

| Router | Coverage | Routed error | AURC | Main comparison |
|---|---:|---:|---:|---|
| confidence baseline (V4) | 0.767 | 0.382 | 0.418 | -- |
| CPR-Router (V4) | 0.824 | 0.387 | **0.366** | error delta +0.0049 [-0.0295, 0.0377] |
| confidence baseline (V5) | 0.721 | 0.408 | 0.418 | -- |
| AMIR (V5) | **0.907** | **0.360** | 0.411 | AURC delta -0.0078 [-0.0607, 0.0459] |
| provenance baseline (V5) | 0.709 | 0.359 | 0.431 | -- |

V4 substantially improves global risk ordering but not error at its calibrated
operating point. Removing the fixed source anchor lowers routed error, exposing
negative transfer. V5 descriptively dominates confidence in both coverage and
error and nearly matches provenance error while covering 19.8 percentage points
more. Its preregistered AURC interval, however, crosses zero; the realized
coverage also drifts above the nominal 80% target.

AMIR satisfies every hard intervention constraint, but the shift coefficient is
nonzero in only one of six folds, and removing the hard constraint changes AURC
by less than 0.001. Thus the market replay supports a useful selective operating
point, not a claim that adaptive cross-domain transfer has been isolated.

## 6. Validity Audits and Failure Analysis

### 6.1 Outcome leakage discovered after V10.4

V10.4 originally emitted a positive-looking `shared_weighted` AUROC of 0.959.
A post-run audit found that the implementation multiplied citation overlap by a
term containing answer correctness. The score therefore used the target outcome
and is invalid for pre-outcome routing. We preserve the records and audit,
invalidate the verdict, and never use that metric as supporting evidence.

This failure motivated the explicit allowlist/denylist contract of (R_{PI}).
It also illustrates why high evaluation scores alone do not establish a valid
reliability method: the data path from labels to scores must be audited.

### 6.2 Auxiliary-stage aborts

V10.1--V10.3, V11, and V12 stopped under registered rewrite-quality gates
before evaluation outcomes existed. No later-ranked question replaced a failed
item. V10.4, V11.1, and V12.1 were separate amendments that changed only
outcome-independent length normalization or bounded repair. These aborts are
instrumentation evidence, not negative or positive router results.

### 6.3 Evidence redundancy as a dataset property

The FEVER audit shows that false-consensus experiments require evidence units
that are neither unrelated nor near-duplicates. When 90.7% of candidate triples
exceed the redundancy ceiling, persona diversity cannot manufacture independent
information. This finding motivates reporting evidence-overlap distributions as
a benchmark property rather than assuming that multiple retrieved passages
constitute multiple sources.

### 6.4 Ranking versus calibration

V4 and V5 expose a recurring distinction. Structural scores may rank errors
well while a fixed threshold or calibration map transfers poorly. In V5, the
uncalibrated ranking ablation has AURC 0.377 but poor Brier score and ECE;
calibration improves Brier from 0.418 to 0.244 and ECE from 0.422 to 0.116 while
degrading global cross-fold AURC. Future work should evaluate within-fold
ranking and cross-fold probability calibration separately.

## 7. Related Work

**Multi-agent reasoning.** Multi-agent debate asks several language-model
instances to propose and revise solutions and has reported gains in reasoning
and factuality. Other work studies divergent debate and LLM-based evaluator
teams. These methods mainly improve answer generation or judging; our question
is whether agreement is trustworthy when evidence and model errors are
dependent. Empirical work on conformity in LLM-agent groups provides direct
motivation for testing provenance rather than treating personas as independent.

**Truthfulness, factuality, and attribution.** TruthfulQA measures imitative
falsehoods across misconception-prone questions. FEVER supplies claims with
supporting or refuting evidence, while attribution work asks whether generated
content is supported by identified sources. Our protocol adds a behavioral
criterion: cited evidence should affect the decision under controlled changes.
It therefore complements citation correctness and factual accuracy rather than
replacing them.

**Faithful explanations and interventions.** Faithfulness work distinguishes
plausible explanations from causal dependence on model inputs. We avoid asking
an LLM to expose private reasoning and instead manipulate environment-provided
evidence while observing answer changes. The method is closer to a black-box
intervention test than to chain-of-thought evaluation.

**Selective prediction and calibration.** Selective classification allows a
model to abstain in exchange for lower retained risk. Calibration work studies
whether predicted probabilities match empirical correctness, and conformal
methods provide distribution-free uncertainty tools under their assumptions.
We use risk--coverage curves and empirical temporal quantiles; for dependent
financial observations we do not claim exact IID conformal coverage.

**Distribution shift.** Group distributionally robust optimization targets
worst-group loss, while domain adaptation studies transfer under changing
distributions. AMIR uses a monotone source gate and temporal group robustness,
but its present market result does not establish successful cross-domain
adaptation.

## 8. Limitations

First, all completed language-model experiments use Qwen3.5-4B. Different
personas of one base model are not independent models, and cross-model transfer
to Llama-3.1-8B-Instruct remains untested. This is the largest gap for an ACL
generalization claim.

Second, V12.1's label subgroups reverse direction. The score may exploit a
BoolQ-specific answer prior or evidence-polarity asymmetry. The aggregate result
must not be generalized to arbitrary binary QA without an independently frozen
mechanism study.

Third, (R_{PI}) was selected on one 100-question development split. Although
V11.1 and V12.1 are provenance-disjoint, all use the same model and task family.
The high weight on citation sharing is not supported by the V12.1 component
analysis, but changing it now would be post-hoc.

Fourth, generated substitutes may differ in fluency or logical force from
original passages despite entity and length constraints. Removal, reversal,
and substitution measure different response behaviors and do not identify one
unique causal mechanism.

Fifth, abstention defers rather than solves high-risk questions. Operational use
would require a fallback policy, human review, retrieval from independent roots,
or a genuinely different model family.

Sixth, the financial replay is retrospective and reuses dates inspected by
earlier versions. Market agents are statistical models, not deployed LLM
traders. The replay cannot establish alpha, causal market impact, or prospective
performance.

Finally, the related-work coverage and manuscript formatting are preliminary.
A submission version needs a systematic literature audit, released prompts and
licenses, compute accounting, error examples, and a frozen cross-model matrix.

## 9. Ethics and Broader Impact

False consensus is consequential in domains such as medicine, law, finance, and
content moderation. A provenance-aware risk score could help systems recognize
when many agreeing agents merely repeat one weak source. It can also create
false reassurance: a low score is not proof of correctness, and an
intervention-sensitive answer can still be wrong. We therefore recommend using
the score for triage and evidence acquisition, not as an autonomous truth
certificate.

The experiments use existing QA datasets and automated model outputs; no
private human data or hidden chain-of-thought is collected. Evidence IDs and
observable answers are sufficient for the reported analyses. Financial results
are research diagnostics and should not be treated as investment advice.

## 10. Reproducibility

Every formal stage has a versioned preregistration, deterministic selection
manifest, fixed salts, response cache, expected record count, bootstrap seed,
and post-run integrity audit. V11.1 contains exactly 4,000 unique records;
V12.1 contains exactly 7,160. V12.1 was produced by four endpoint-concurrent
workers and deterministically merged by logical record key. Artifact hashes are
recorded in the integrity audit.

The repository includes frozen protocols, source code, tests, formal records,
summaries, and market reports. Generated request caches and process logs are not
required to reproduce the metrics and should not be treated as paper artifacts.
At the current checkpoint, the full repository test suite passes 102 tests.

## 11. Conclusion

Consensus is reliable only to the extent that its evidence and errors are
independent. We presented an environment-controlled paired-intervention
framework for testing that assumption in multi-agent language models. On a
provenance-disjoint BoolQ replication, a frozen pre-outcome score ranks wrong
high-consensus decisions and improves retained error at 80% coverage. The
result is carried by behavioral response to evidence changes, not citation
sharing alone, and it reverses across native answer labels. A temporal financial
stress test similarly shows useful selective behavior alongside negative
transfer and calibration failure.

The evidence therefore supports a bounded conclusion: provenance-aware
interventions provide a falsifiable signal for multi-agent reliability, but not
a universal truth detector. The next decisive experiment is a frozen
Qwen-to-Llama transfer on common TQA/BoolQ items with matched coverage and
mechanism-held-out interventions.

## References

- Anastasios N. Angelopoulos and Stephen Bates. 2021. [A Gentle Introduction to
  Conformal Prediction and Distribution-Free Uncertainty
  Quantification](https://arxiv.org/abs/2107.07511).
- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael
  Collins, and Kristina Toutanova. 2019. [BoolQ: Exploring the Surprising
  Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/).
  NAACL-HLT.
- Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch.
  2023. [Improving Factuality and Reasoning in Language Models through
  Multiagent Debate](https://arxiv.org/abs/2305.14325). arXiv:2305.14325.
- Yonatan Geifman and Ran El-Yaniv. 2017. [Selective Classification for Deep
  Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
  NeurIPS.
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. [On
  Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
  ICML.
- Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu
  Yang, Shuming Shi, and Zhaopeng Tu. 2023. [Encouraging Divergent Thinking in
  Large Language Models through Multi-Agent Debate](https://arxiv.org/abs/2305.19118).
- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. [TruthfulQA: Measuring How
  Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/).
  ACL.
- Benjamin Muller, John Wieting, Jonathan H. Clark, Tom Kwiatkowski, Sebastian
  Ruder, Livio Baldini Soares, Roee Aharoni, Jonathan Herzig, and Xinyi Wang.
  2023. [Evaluating and Modeling Attribution for Cross-Lingual Question
  Answering](https://aclanthology.org/2023.emnlp-main.10/). EMNLP.
- Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. 2020.
  [Distributionally Robust Neural Networks for Group
  Shifts](https://arxiv.org/abs/1911.08731). ICLR.
- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
  2018. [FEVER: a Large-scale Dataset for Fact Extraction and
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
    label
    gold_binary
    correct
    any_wrong
    harmful_fc
    aliases or transformations of the above

Score:
    R_PI = 0.1 * D_inert
         + 0.3 * flip_inertia
         + 0.6 * frac_shared

Primary subset:
    original agreement >= 0.8

Primary endpoint:
    AUROC(R_PI, consensus_wrong)
```

## Appendix B. Planned completion matrix

| Source model/data | Target model/data | Status | Role |
|---|---|---|---|
| Qwen3.5-4B / BoolQ development | Qwen3.5-4B / disjoint BoolQ validation | complete | within-model replication |
| Qwen3.5-4B / frozen TQA-BoolQ | Llama-3.1-8B-Instruct / same items | pending | main cross-model claim |
| remove + reverse | substitute + alias/duplicate holdout | pending | mechanism transfer |
| Qwen3.5-4B paired interventions | S&P 500 temporal provenance replay | exploratory complete | external sequential stress test |
