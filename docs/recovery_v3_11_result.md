# Recovery V3.11 result: unseen-model dual-head provenance repair

Protocol: `recovery-v3.11-group-robust-dual-head-hy-2026-09-03`

Verdict: **NO_VERIFIED_UNSEEN_MODEL_DUAL_HEAD_TRANSFER_V3_11** under the
conjunctive preregistered gate. The primary effect is nevertheless large,
positive, and statistically separated from zero.

## Frozen formal result

- Target: `Hy-MT2-7B`, absent from router fitting and threshold selection.
- Formal set: 188 fresh FEVER examples, balanced 94/94; zero claim/root
  overlap with all 1,400 V3.7.1 source-selection examples.
- Hy action transport: 1,504/1,504 successful; 1,465 first-pass valid.
- Frozen routes: 27 candidate roots and 161 KEEP decisions.
- Accuracy: 54.26% KEEP to 67.55% V3.11.
- Native-label macro gain: +13.30pp, stratified-bootstrap 95% CI
  [+9.04pp, +18.09pp].
- Repairs / harms / net fixes: 26 / 1 / +25.
- Annotation-supported repairs: 26.
- Provenance selection: 187/188 overall and 27/27 among routed examples.

All registered budget-matched baselines were much weaker. Their net fixes
were +1 (retrieval score), +3 (hash random), +2 (fixed candidate 0), +2
(fixed candidate 1), and +2 (both candidates), versus +25 for V3.11.

## Gate accounting

Four of six frozen gates pass:

- PASS: macro-gain CI lower bound above zero;
- PASS: at least 10 annotation-supported repairs;
- PASS: net fixes above KEEP and every matched baseline;
- PASS: provenance accuracy at least 90%;
- FAIL: both native-label groups nonnegative; and
- FAIL: damage among initially correct high-consensus examples at most 0.5%.

The two failures are the same single routed question. Hy changed one Refuted
example from correct to incorrect, giving Refuted gain -1/94 (-1.06pp) and
damage 1/101 (0.99%). Supported gain was +26/94 (+27.66pp). Because the gate
was conjunctive and frozen, the large aggregate gain cannot be relabeled as a
pass.

## Interpretation

V3.11 establishes that the proposed dual-head mechanism can select the right
fresh provenance path and convert it into target-model repairs: every routed
path was annotation-correct and 26 of 27 routes repaired the consensus. It
also exposes the remaining problem precisely: an embedding relation score of
0.622 and Hy confidence 0.90 were jointly overconfident on one refuted claim.
This motivates a separately preregistered safety-abstention study; it does not
authorize changing the V3.11 threshold or reevaluating these 188 examples as
formal.

## Content-addressed artifacts

- protocol manifest: `97aa4e7968c4b2e1dfaffe04d4866ad141156d36d054b84e0bf74c2cbef4f852`
- selection manifest: `57d156621ae19c8a6b2379fa998fe15a095b80522ee60bf922d8554cad549e73`
- router manifest: `5ed74d04ceb460ea12bec1428bbe062ed12ba49815490e0b6982a8fe1de6035f`
- inference-only router inputs: `325b05c619415e936266766ec4f105a358c5cb141d83158af0bd07d739bf9d0f`
- Hy action records: `cad77784c070a764e2c9573a2b21be7ee3bb0db01c5ffea9c9658349367baa36`
- pre-outcome routes: `e346589aaf03ec6cd326eeb622458e7db944a925c9489b39e7610a78ceff2e55`
- summary: `a07ac51a11f4368e5fa251211d1f9076213b00dd087908d03d31312adeb54320`
