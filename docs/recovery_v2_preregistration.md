# Recovery V2.2 preregistration: provenance-action routing

Protocol version: `recovery-v2.2-page-root-2026-09-02`

Status: **bounded parser amendment frozen before any test call**.

Operational amendment, recorded before resuming dev and before any test call:
the same `Qwen3.5-4B` deployment was moved by its operator from
`10.63.0.88:31519` to `10.63.0.82:31518`. The new `/v1/models` response exposes
only `Qwen3.5-4B` and reports the same named weight directory. The model ID,
prompts, decoding parameters, seeds, actions, parser, datasets, features,
thresholds, and all existing records remain unchanged. New records store their
actual runtime endpoint. This is treated as a documented compute-location
relocation, not a scientific-protocol change.

V2.0 ended in a pre-formal implementation audit before any model call. Its
component allocator produced a 38/31/31 split rather than the documented
60/20/20 target. V2.1 changes only that allocator objective and all split,
anchor, and candidate-order salts. The V2.0 manifests are retained under
`results/recovery_v2_preformal_abort/`; no response or outcome informed this
amendment.

V2.1 then completed the exact 689-example train split (5,512 successful
logical calls) and began dev. It stopped after materializing 114 dev bundles
(113 fully successful) when Qwen repeatedly emitted an otherwise valid JSON
object with one extra quote immediately after the numeric confidence value, for example
`"confidence":1.0"`. V2.2 inherits the exact V2.1 selection and train records.
Its sole change is a response-local parser normalization that removes exactly
one quote only in the pattern `"confidence":NUMBER"` immediately before a
comma. All other malformed outputs still fail; answer, numeric confidence,
citations, prompts, seeds, actions, splits, features, thresholds, and outcomes
are unchanged. No test call or test outcome existed before this amendment.

## 1. Scope and relation to V12.1

Recovery V2.2 is a new mechanism experiment. It does not alter Pilot-LLM V12.1
or Recovery V1. V12.1 supplies the motivating failure mode and the distinction
between detecting a risky consensus and choosing a useful correction. Recovery
V2 uses a separate, page-identified FEVER pool because BoolQ exposes passages
but not upstream page identities and therefore cannot support a source-root
disjoint split.

This experiment tests page-root provenance, not publisher independence: all
annotated roots are Wikipedia pages. A positive result would establish
source-page-disjoint recovery within this benchmark, not independent
organizational corroboration or cross-model generality.

## 2. Frozen dataset universe

Input: `data/fever/fever-validation.jsonl`, SHA-256 recorded in the selection
manifest.

An example is eligible exactly when:

- `verifiable == "VERIFIABLE"`;
- the label is `SUPPORTS` or `REFUTES`;
- its annotated evidence contains exactly two distinct non-empty page IDs;
- each page has at least one non-empty annotated evidence sentence.

No Qwen response, consensus outcome, correctness value, or recovery result is
used for selection. The expected eligible universe from the local frozen file
is 1,149 claims.

## 3. Page-root-disjoint split

Claims are connected when their annotated evidence shares a page root.
Connected components, not individual claims, are assigned to train/dev/test by
a deterministic size- and label-balancing algorithm with targets 60/20/20.
Ties are broken by a protocol-salted SHA-256 component key. The audit requires
zero page-root overlap across all three splits and at least 200 claims in both
dev and test.

The native label may be used to balance this pre-model split, but it is never a
router feature or prompt field.

## 4. Frozen evidence environment

For each claim, one of the two annotated roots is selected as the `anchor` by
a protocol-salted hash. It is the only evidence shown to the initial agents.
The other annotated root is the `held-out annotated root`.

A second candidate is retrieved from another claim in the same split using a
fixed HashingVectorizer cosine score. It must have a distinct page ID and must
not be an annotated root of the current claim. This is called an `unannotated
retrieval candidate`, not a known negative: it may incidentally be useful.

The held-out annotated root and unannotated retrieval candidate are assigned
to `candidate_0` and `candidate_1` by a separate hash. Thus neither the action
name nor candidate position reveals which root is annotated. The router is
forbidden from reading the manifest's annotation-role field.

Structural pre-call gates:

- 1,149 eligible claims;
- no page-root overlap between splits, including retrieved candidate roots;
- no candidate equals the anchor or duplicates the other candidate;
- annotated-root frequency in `candidate_0` lies in [0.45, 0.55];
- both native labels occupy at least 40% of each split;
- the retrieval-score-only AUROC for identifying the annotated candidate is at
  most 0.80, preventing a nearly trivial relevance-score shortcut.

Failure of any gate aborts before Qwen calls and requires a new protocol
version; thresholds may not be changed in place.

## 5. Initial consensus and recovery actions

Five frozen personas independently verify the claim from the anchor packet
only. Each returns exactly:

```json
{"answer":"yes|no","confidence":0.0,"cited_evidence_ids":["A00"]}
```

The initial answer is majority vote; high consensus is agreement at least 0.8.
The recovery policy acts only on high-consensus cases.

Every example receives the complete potential-outcome matrix after the
relevant split policy is frozen:

- `KEEP`: retain initial consensus, zero added roots;
- `candidate_0`: adjudicate from anchor plus candidate 0, cost one root;
- `candidate_1`: adjudicate from anchor plus candidate 1, cost one root;
- `both`: adjudicate from anchor plus both candidates, cost two roots.

Recovery prompts do not receive the native label, correctness, annotation
role, split outcome, or any alias of these fields. Qwen must cite only evidence
IDs in the selected action packet.

## 6. Learning target and policy freeze

For claim `q` and action `a`:

```text
gain(q, a) = correct(q, a) - correct(q, KEEP)
harm(q, a) = 1[correct(q, KEEP)=1 and correct(q, a)=0]
```

The router trains on train-split paired action outcomes and pre-outcome
features only. Candidate 0 and candidate 1 share a symmetric single-root model;
`both` has an explicit two-root action feature. Dev outcomes calibrate frozen
one-sided residual margins. The deployed test policy selects an action only
when its calibrated lower gain score exceeds its acquisition cost and its
calibrated harm upper score is at most 0.10. Acquisition cost is 0.01 per added
root. No hyperparameter or threshold is selected from test outcomes.

The serialized router manifest, feature schema, train/dev record hashes, model
parameters, and calibration margins must be written and hashed before any test
recovery outcome is inspected.

## 7. Pre-test train structural gate

After train calls, execution proceeds to dev/test only if train contains:

- at least 100 high-consensus examples;
- at least 20 high-consensus wrong initial consensuses;
- at least 10 high-consensus wrong examples repaired by some non-KEEP action;
- at least 10 high-consensus correct examples harmed by some non-KEEP action.

These gates ensure that both benefit and harm are learnable. They do not select
examples or change the frozen test set.

## 8. Outcomes and success criterion

Primary evaluation is end-to-end test accuracy under the frozen conservative
policy versus `KEEP`, with a 1,000-replicate paired question bootstrap and seed
20260932.

Recovery V2 passes only if all hold:

1. the 95% paired net-gain interval lower bound is greater than zero;
2. net fixes exceed every fixed non-oracle action policy;
3. the policy's damage rate among initially correct high-consensus examples is
   at most 5%;
4. net gain is non-negative in both native-label groups;
5. at least five test repairs are annotation-supported: the initial consensus
   is wrong, the final answer is correct, the selected packet contains the
   held-out annotated page, and Qwen cites evidence from that acquired page.

Required baselines are `KEEP`, fixed candidate 0, fixed candidate 1, fixed
both, retrieval-score routing, an unrestricted learned policy, and a
per-example available-action oracle marked diagnostic.

## 9. Claim boundary

Even a pass is a within-Qwen, Wikipedia-page-root result. Cross-model testing,
independent publishers, live retrieval, and transfer back to untouched BoolQ
roots remain separate future experiments. A failure is retained unchanged and
must not be repaired by retuning this version on dev or test outcomes.
