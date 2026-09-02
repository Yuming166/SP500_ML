# Recovery V3.5.2 status

Status: **development unlock failed; formal test remains untouched and locked**.

All 1,200 development certificates and 4,800 action rows passed validation.
Their SHA-256 values are respectively
`4d0dec7cf0050630cfbdce7aa3e1c55d21f62b80a8d49a7418dfd2945275a052`
and
`8e8e8b6d915be29bd895635551ffbe7e5a123eb73680a016087fefe3ae7d99cc`.
Nine certificate rows dropped 11 nonlocal/duplicate IDs; 16 decision rows
dropped 28. Projection did not change any answer or confidence.

Anchor-only accuracy was 410/600 (68.33%); 581 examples had at least 0.8
agreement. Of 190 baseline errors, 158 were repairable by at least one available
action. Across the two single-root paired actions there were 181 fix targets and
161 harm targets.

The raw certificate gate made 266 routes and selected the held-out annotated
root 263 times. It produced 118 fixes and 102 harms, net +16. However its gains
were sharply asymmetric: +38.33 percentage points on Supported and -33.00
points on Refuted, with 25.56% damage among initially correct high-consensus
examples. The main failure was semantic, not retrieval: scalar certificates
often labeled a conjunction as supported while overlooking one explicit wrong
attribute such as country, year, studio, or negation.

The frozen three-fold OOF fix/harm fitting found no nontrivial policy satisfying
the per-fold safety, both-label, net-fix, and annotation-supported-repair gates.
Accordingly no router manifest was written and all V3.5.2 formal EX-FEVER test
commands remain locked. These development results motivate an atomic,
proof-carrying certificate in the next protocol; thresholds will not be relaxed.
