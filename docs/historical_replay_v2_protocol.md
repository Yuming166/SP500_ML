# Historical replay V2: source expansion protocol

V2 is a new exploratory-to-confirmatory replay specification. It leaves the
published V0/V1 reports unchanged and adds two locally available, real-data
sources: CBOE's full put/call ratios and ICI's weekly mutual-fund flows.

## As-of feature contract

- Daily market, VIX, macro, CBOE, and ETF observations are usable on the next
  session only.
- Google Trends and ICI weekly observations are forward-filled from the last
  observation and delayed five trading days. Neither series is back-filled.
- Each decision predicts the following five trading-day return. A five-day gap
  separates every training and test block.

## Agents and source roots

- Three market-scale agents share `market_bloomberg`.
- Three ETF-flow agents share `etf_flow_family`.
- VIX, macro, Trends sentiment, the CBOE option block, and the ICI flow block
  each have one root. CBOE's total/index/stock PCR features belong to one
  `cboe_options` root; ICI's total/domestic/foreign features belong to one
  `ici_mutual_fund_flow` root.

## Fixed comparison

Each outer expanding split has 504 training rows, a five-row label gap, and a
126-row test block. Logistic agents are fit only on that outer training data.
Their quality is estimated from nested chronological out-of-fold predictions
inside the training block and updated only after a five-day label has matured
in the test block.

- **Majority:** equal hard vote across agents.
- **Confidence:** majority action, with agreeing-agent probability confidence.
- **Recent performance:** agent probabilities weighted by inverse recent
  train-only Brier loss.
- **Provenance:** probabilities averaged inside each source root, then roots
  weighted by inverse root-level recent Brier loss. Its abstention score also
  penalizes disagreement among roots.

For every method the abstention threshold is its own 75th percentile risk on
the nested, train-only out-of-fold rows. This determines a roughly comparable
nominal training coverage without inspecting outer-test outcomes.

## Reporting boundary

This is a feature-expansion ablation. It can test whether grouping correlated
source families improves selective historical replay, but it cannot establish
the vendor's exact intraday publication time or investment profitability.
