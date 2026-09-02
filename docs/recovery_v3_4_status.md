# Recovery V3.4 pre-formal status

V3.4 was stopped at its pre-call structural audit. It generated no Qwen
response, fitted router, formal test record, or outcome analysis.

- preregistration SHA-256:
  `29867f640da302e70e6edfe9f8cedcdc7db2a8b0872cecd0695a931f2cdac567`
- selection SHA-256:
  `496ac85ba7e4adb3a92a39b6aa174fcf9ec77868c2e97037b19afdd624daf58b`
- selection-audit SHA-256:
  `8377ee85ed4b62eae804b891f008a3b4821680db230310802239c81a9a69461f`

The frozen audit required at least 400 distinct Wikipedia page roots. The 483
eligible examples contained 363, so that gate failed. All other structural
gates passed, including exact sample and label counts, zero claim overlap with
the local AVeriTeC and FEVER material, balanced candidate order, and candidate-
role retrieval AUC 0.542.

V3.4.1 is a preregistered pre-formal correction. It changes only this impossible
gate from 400 to 350, still requiring over 96% of the observed 363 roots. No
method, threshold, sample, endpoint, statistical test, or success criterion is
changed.
