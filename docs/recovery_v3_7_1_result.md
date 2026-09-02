# Recovery V3.7.1 result: entailment-ledger action routing

Protocol: `recovery-v3.7.1-elar-fever-train-2026-09-02`

Formal verdict: **PASS_ENTAILMENT_LEDGER_ACTION_ROUTING_V3_7 (five of five
primary gates passed)**.

## Integrity and execution

All 1,000 previously evaluated EX-FEVER examples were development-only. ELAR
unlocked on both frozen source folds before any V3.7.1 formal model call. The
selected development thresholds were ledger confidence at least 0.8, lexical
coverage at least 0.0, and at most one unsupported term. The two fold results
were:

| development source fold | net fixes | macro gain | damage | Supported gain | Refuted gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3.6.1 development | +37 | +6.17pp | 0.25% | +11.00pp | +1.33pp |
| V3.6.2 former formal | +15 | +3.75pp | 0.40% | +7.50pp | 0.00pp |

The new formal set contains 400 previously uncalled FEVER-train claims,
balanced 200/200, and 1,200 globally unique page roots. It has zero claim or
root overlap with development or earlier FEVER experiments. Candidate position
is exactly balanced. Raw retrieval score has candidate-role AUROC 0.621, below
the frozen 0.65 structural cap, and is not a router feature.

All 3,200 action rows and 800 atomic certificates completed successfully; all
certificates parsed on the first call. The proof gate produced 62 candidate
actions over 58 items. The exact-quote ledger validated 50; 12 failed closed.
The pre-outcome route and prediction artifact was written before any accuracy,
oracle, or gain was constructed.

The original evaluator then exposed a null-handling bug only in the proof-only
ablation and stopped before writing pre-outcome routes or constructing outcome
metrics. The documented evaluation-only amendment normalizes that ablation's
`ledger: null` display values; it does not change primary ELAR eligibility or
ranking. Its code and document hashes are embedded in the pre-outcome artifact.

## Formal primary result

| metric | KEEP | ELAR |
| --- | ---: | ---: |
| accuracy | 51.00% | **58.25%** |
| native-label-macro gain | 0.00pp | **+7.25pp** |
| paired 95% macro-gain CI | [0.00, 0.00]pp | **[+4.75, +9.75]pp** |
| fixes / harms | 0 / 0 | **29 / 0** |
| high-consensus-correct damage | 0.00% | **0.00%** |
| acquired roots | 0 | 32 |
| annotation-supported repairs | 0 | **29** |

Native-label results were:

- Supported: 3.50% to 17.00%, **+13.50pp**;
- Refuted: 98.50% to 99.50%, **+1.00pp**.

ELAR routed 32 items and selected the annotated root 31 times (96.88%). Every
one of its 29 repairs cited evidence from the acquired annotated root.

## Frozen gates

| primary gate | observed result | pass |
| --- | ---: | ---: |
| macro-gain CI lower bound above zero | +4.75pp | Yes |
| high-consensus-correct damage at most 5% | 0.00% | Yes |
| nonnegative gain in both native labels | +13.50pp / +1.00pp | Yes |
| at least ten annotation-supported repairs | 29 | Yes |
| net fixes above KEEP and every matched baseline | +29 versus best +9 | Yes |

## Baselines and ablation

| policy | accuracy | net fixes | fixes / harms | macro-gain CI | acquired roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| ELAR | **58.25%** | **+29** | 29 / 0 | [+4.75, +9.75]pp | 32 |
| Atomic proof-only | 61.75% | +43 | 45 / 2 | [+7.75, +13.75]pp | 58 |
| matched retrieval score | 53.25% | +9 | 9 / 0 | [+1.00, +3.75]pp | 32 |
| matched hash random | 52.00% | +4 | 5 / 1 | [0.00, +2.25]pp | 32 |
| matched fixed candidate 0 | 52.75% | +7 | 7 / 0 | [+0.50, +3.00]pp | 32 |
| matched fixed candidate 1 | 52.25% | +5 | 7 / 2 | [-0.25, +2.75]pp | 32 |
| matched fixed both | 51.50% | +2 | 2 / 0 | [0.00, +1.25]pp | 32 |
| available-action oracle | 84.50% | +134 | 134 / 0 | [+30.00, +37.00]pp | 134 |

Atomic proof-only has higher gain but causes two harms; ELAR trades 14 net
fixes for zero observed harms and a smaller source budget. This experiment does
not establish a statistically significant harm reduction over proof-only from
only two events. It establishes that the preregistered conservative ledger
policy itself passes all five gates and beats every equal-budget nonlearned
comparator.

The evaluator JSON's field named `total_added_roots` counts routed actions. For
the matched `both` baseline only, 16 routed actions consume 32 roots; the
underlying frozen budget matcher correctly charges two roots per `both` action.
Primary ELAR acquires one root per action, so its reported 32 is unaffected.

## Interpretation and paper boundary

The positive result supports the paper's central hypothesis: consensus error
detection can be extended to **pre-outcome corrective path selection** with an
auditable evidence witness. The distinctive contribution is the composition
of paired potential outcomes, provenance-local atomic certificates, exact
claim-to-quote ledgers, adversarial semantic challenges, and fail-closed action
execution. Exact attribution and fact verification alone are not claimed as
novel.

The formal environment is deliberately controlled. An unannotated matched
retrieval root creates the initial evidence condition, so baseline behavior is
strongly asymmetric: it is usually `no`, yielding 3.5% Supported accuracy and
98.5% Refuted accuracy. Balanced macro evaluation, per-label nonnegativity, and
the matched baselines prevent this from being hidden, but the result must not
be presented as the prevalence or magnitude of naturally occurring web-agent
errors. It is evidence-path routing under a controlled false-consensus
intervention.

The result is one Qwen3.5-4B deployment, static Wikipedia-page provenance, and
one constructed formal split. Cross-model replication, live retrieval,
publisher independence, and naturalistic consensus incidence remain required
for a full NAACL-level generality claim.

## Artifact hashes

- selection: `f2a0fa3a6fd9d3529758989537855b5862dbe39de0e42dac153d08d253fa2665`;
- router manifest: `1b75b96948dfe5473146f78ea137ff6cb675155960c7e624a3b62faad0f3f516`;
- formal actions: `dde8708f2c8a2331a5073400d95a50e6eb33ab40d385b258e3567697f8fb0c34`;
- formal certificates: `d5350284abe4bbccc1fc3ce9a84557edbb5811c0e6274dff444633d6f3af3a54`;
- formal ledgers: `784b713d077790826f3a7a7d954e26bcc690d44726ce808b59ad739eee244dd2`;
- pre-outcome routes: `7c2a4fbe14cbae98584a7b08afc3b787f51fa00db01762ec2a369f3d638477b8`;
- evaluation amendment: `6812dde494903abb486d8cff0e50bf01bea42ae096f6165b46f2539548e69eed`;
- final summary: `8516cfe9e88711b27298686a2103910117cb1e0eb8c00236bbc67ad3dc758633`.
