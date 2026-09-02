# Recovery V3.4.1 result: external PBWJ transfer did not pass

Protocol: `recovery-v3.4.1-pbwj-climate-fever-2026-09-02`

Verdict: **NO_VERIFIED_PBWJ_DOMINANCE**.

This is a frozen prospective negative result. The complete 483-item
CLIMATE-FEVER action matrix was collected once after the selection, method,
thresholds, model, and primary gates were frozen. No result below is used to
revise V3.4.1.

## Run integrity

- 483 examples and 3,864 expected records were present.
- All 3,864 records succeeded and parsed on the first attempt; 3,848 were new
  network calls and 16 deterministically reused the two-item smoke cache.
- The endpoint and model were the frozen `Qwen3.5-4B` service on host 82.
- The external set contained 361 Supported and 122 Refuted claims, 363 distinct
  Wikipedia page roots, and no exact normalized claim overlap with the local
  AVeriTeC or FEVER material.
- Candidate position was balanced at 50.10%; retrieval-score candidate-role AUC
  was 0.542.
- The route table and jury diagnostics were serialized before outcome metrics.

Key hashes:

- preregistration: `1bab2a17ddb9ed6028fd80924aa8f455279d04f38ab481209e8ea516ff17edd4`
- selection: `30e21c75c84b1f6c01ce2f5cc49267dd21a67b8e266bc1cb53b08b217141f52e`
- router manifest: `68f9401d970f656f16e093cc70f4661d97ae533ba27e7805e822ce1f1e456f24`
- test records: `7c49ba54bd177803675670d40910c939b24e7f3f7049e17def313915721958a5`
- pre-outcome routes: `cc79cc3b030831f822f320fced443384b7b5ba4eafab2d4a0dfee7e9353e9de9`
- summary: `4bd01973dda959926798a5716aa796d421a42287126fa59399accfad498cbbf2`

## Primary result

PBWJ acquired 71 single roots. Baseline micro accuracy was 55.69%; PBWJ final
accuracy was 54.66%. It made zero repairs and five harms, for net `-5`.

The label-macro gain was **-0.693 percentage points**, with the frozen
10,000-replicate paired-bootstrap 95% interval **[-1.385, -0.139] pp**.

| Native label | n | Baseline | PBWJ | Gain |
|---|---:|---:|---:|---:|
| Supported | 361 | 41.27% | 39.89% | -1.385 pp |
| Refuted | 122 | 98.36% | 98.36% | 0.000 pp |

Damage among initially correct high-consensus examples was 1.91%, below the
5% cap. This safety rate alone is insufficient because the router never fixed a
case and reduced accuracy.

## Frozen gates

| Gate | Result |
|---|---|
| Macro-gain CI lower bound above zero | Fail |
| Damage at most 5% | Pass |
| Both native-label gains nonnegative | Fail |
| At least ten citation-supported repairs | Fail (`0`) |
| Net fixes exceed KEEP and all matched baselines | Fail |

## Prespecified ablations and comparators

Removing only the dispersion penalty routed 88 roots and produced 3 fixes and
5 harms (`net=-2`), with macro gain -0.277 pp and interval
[-1.108, 0.416] pp. Removing both uncertainty vetoes routed 147 roots but had
the same 3-fix/5-harm outcome. Thus the negative result is not explained only by
an overly conservative dispersion threshold.

At PBWJ's 71-root budget, retrieval routing achieved 18 fixes and 6 harms
(`net=+12`) but its label-macro gain was only +0.034 pp with interval
[-2.292, 2.100] pp because it harmed the small Refuted group. The matched fixed-
both comparator spent 70 roots on 35 items and achieved 9 fixes, 0 harms, and
macro gain **+1.247 pp [0.554, 2.078]**. This prespecified comparator result is
valid, but it is not evidence that PBWJ learned successful path selection.

The available-action oracle repaired 129 cases with no harms using 136 roots,
for macro gain **+17.867 pp [15.512, 20.360]**. Hence the action space contains
substantial recoverability; identifying when and where to route remains the
bottleneck.

## Frozen post-result failure analysis

PBWJ's candidate choice selected the annotated root in 38 of 71 routes
(53.5%), close to chance. More importantly, 70 of 71 routes had initial
consensus `yes`; all 71 routed cases were already correct. It routed none of the
203 initially wrong high-consensus cases.

The distribution explains this directional collapse:

- 212 Supported items initially had consensus `no`; 202 were high-consensus and
  therefore represented the main repair opportunity.
- For those 202 cases, candidate-0, candidate-1, and both would have repaired
  80, 86, and 109 cases respectively.
- The transferred jury nevertheless rejected virtually every `no -> supports`
  route because its AVeriTeC-trained support probabilities fell below 0.4 on
  climate evidence.
- It instead opened 70 `yes -> refutes` routes on already-correct Supported
  cases and caused all five harms.

The central failure is therefore **cross-domain stance-score calibration and
direction selection**, not absence of useful recovery actions. Publisher-
blocked source training did not make a sparse TF-IDF stance model invariant to
the AVeriTeC-to-CLIMATE-FEVER evidence shift. Jury unanimity can be confidently
wrong under shared representation bias.

## Paper implication

V3.4.1 should be reported as an honest external-transfer stress test. It
weakens any claim that CEW's lexical stance scores transfer without target-
domain calibration, while strengthening a narrower finding: false-consensus
repair is highly possible in the available action space, but safe selective
routing requires a domain-invariant witness or target-domain calibration that
does not consume formal test outcomes.

The result does not test independent publishers because CLIMATE-FEVER roots are
Wikipedia pages, and it remains single-model evidence.
