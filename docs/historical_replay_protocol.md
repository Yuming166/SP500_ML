# Historical replay V0: data and signal-processing protocol

This is the real-data bridge after synthetic V2. It is a fixed-price historical
replay, not a closed-loop market simulator and not investment advice.

## As-of assumptions

- Bloomberg-sourced market, VIX, and macro values are treated as available only
  after the recorded day's close, hence usable at the next session.
- Google Trends sentiment has no row-level publication timestamp in this
  checkout, so the last known weekly observation is delayed by five trading
  days. This is conservative but remains an assumption to replace with release
  metadata when available.
- CBOE's full put/call workbook is daily and is treated like the other
  close-to-next-session sources. Its total, index, and stock series are one
  `cboe_options` root, rather than three independent votes. The earlier stock
  PCR CSV is therefore superseded inside new feature construction.
- ICI's weekly long-term mutual-fund flow workbook carries an observation date
  but not a row-level publication time. It is forward-filled only from prior
  observations and then delayed by five trading days. Total, domestic, and
  foreign flow are one `ici_mutual_fund_flow` root.
- Missing observations may be forward-filled only from prior values. No
  back-fill, centered filter, or future padding is allowed.
- The target is the following five-trading-day return and is excluded from all
  decision-time features.

## Signal-processing role

`log_price` is decomposed with the causal `(5, 20, 60)` filter bank. Its detail,
band, and trend components inherit the same `market_bloomberg` root source.
They are distinct time-scale representations, not independent evidence.

## Validation boundary

The next module will use expanding walk-forward splits with a five-day gap and
rule agents. It can test whether source overlap, staleness, and online
source-quality estimates improve selective routing in a real price path. It
cannot validate actual Bloomberg publication timestamps or claim that agents
caused historical prices to move.
