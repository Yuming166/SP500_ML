# Detection V3.16.2 Ling pilot missingness rule

Date: 2026-09-03 (Asia/Shanghai)

Status: **frozen after the 160-call Ling transport smoke and before the
1,200-call Ling transfer pilot**.

The Ling smoke passed the registered transport gates with 158/160 final-valid
and 155/160 first-pass-valid rows. Its two terminal failures were isolated JSON
truncations in one reverse and one substitute response. Because the registered
validity gates permit a small failure rate, the transfer analysis must specify
how incomplete bundles are handled without dropping items after outcome access.

Frozen rule:

- original consensus uses valid original answers but agreement retains the
  fixed five-agent denominator;
- an item with any missing original or intervention response receives value
  `1.0` on every intervention-risk coordinate;
- items are never removed because an intervention response failed;
- fewer than four agreeing original agents cannot enter the high-consensus
  subset under the fixed 0.8 threshold; and
- the Qwen-selected weights and all pilot gates remain unchanged.

This is a fail-closed missing-data convention based only on smoke transport
behavior. It does not inspect any Ling development answer outcome and does not
authorize formal calls.
