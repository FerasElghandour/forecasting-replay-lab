# Findings — research prototype, October 14, 2026

**The prototype works, but the real-data experiment does not establish a reliable forecasting or trading advantage.** Keep the market as the operational probability baseline. Preserve this candidate and its failed results for comparison with future, independently informative features.

## Delivered and exercised

- Market-anchored probability model, identity-shrunk beta calibration, later-period blending, market-only calibration comparator, model serialization, and unlabeled paper forecast output.
- Event-purged, label-availability-safe expanding backtests with a one-hour embargo and untouched outer windows.
- Brier score, log loss, ten-bin reliability/ECE, per-fold results, and paired event/week cluster intervals.
- Cash-constrained, hold-to-settlement hypothetical positions with ask prices, slippage, entry fees, depth/freshness checks, and concurrent event limits.
- GET-only public Polymarket and Kalshi adapters, raw response provenance, strict snapshot validation, and offline reproduction.
- **36 passing automated tests**, plus live read-only API checks and end-to-end fit/predict commands. Tests cover future-feature/label leakage, related events, calibration identity, numerical optimization, model round trips, outcome mapping, pagination, candle end-time availability, fees, depth, quote age, cash lockup, event exposure, and verified-label requirements.

Run `python3 reproduce.py` from this folder to repeat tests and all five supplied evaluations offline with NumPy installed. Exact predictions, metrics and accounting are in `results/`. Numerical environment and seeds are recorded in `results/environment.json`.

## Real-data experiment

The bounded KXHIGHNY sample discovered 1,366 contracts across 228 daily events through both storage tiers. It retained **1,141 snapshots**, one per eligible contract at 24 hours before scheduled trading close. It excluded 222 contracts without valid two-sided quotes and three without a fresh enough pre-forecast candle. There were no recorded HTTP errors. Snapshot dates span February 28–October 13, 2026 UTC. This is a settled-survivor convenience sample, not the complete listing-time universe.

After warm-up, four outer windows cover June 22–October 13: **534 held-out contracts / 114 daily events**, all with fitted models. Every day's included contracts together have weight one. This avoids pretending that several temperature buckets are independent observations.

| Held-out probability output | Brier ↓ | Log loss ↓ | ECE, 10 bins ↓ |
|---|---:|---:|---:|
| Market midpoint | **0.156534** | 0.701536 | **0.094438** |
| Market-only beta calibration | 0.203300 | **0.586385** | 0.256871 |
| Raw correction | 0.333914 | 0.982624 | 0.377054 |
| Calibrated correction | 0.395226 | 1.328223 | 0.423707 |
| Selected market/model blend | 0.194270 | 0.603301 | 0.149766 |

The selected blend has **worse Brier loss and ECE**, but **better log loss**, than raw market prices. The simpler calibrated-market comparator has still better log loss. Thus the result depends on which proper score is used; it does not support a broad claim that the candidate predicts better. Calibration fitting itself worsened this candidate out of sample.

Market Brier minus selected-blend Brier is **−0.037736**. The paired event-bootstrap 95% interval is **[−0.066365, −0.011079]**. The seven-day calendar-block interval is wider: **[−0.108874, +0.027423]**, based on only 17 blocks. These are exploratory intervals and do not account for every dependence or future model search.

Selected model weights were **1.0, 1.0, 0.5, 0.0** across the four successive folds. The later selection set reverted to market prices. No settings were changed after observing these results. This pattern is consistent with instability; it does not identify its cause or guarantee that a recency tweak would help.

## Paper accounting

**Strict real-data replay: zero trades.** Every historical record lacks a verified fee vintage, and candle data also lacks order-book size. Zero simulated P&L here means insufficient execution inputs, not evidence that the strategy broke even.

An explicitly assumed-execution scenario permits ten-contract fills at candle quotes as old as two hours, assumes sufficient depth and a 0.07 taker fee coefficient, starts with $1,000, and holds positions to settlement. It is intentionally separate from strict results.

| Assumed slippage per share | Hypothetical trades | Net simulated P&L |
|---|---:|---:|
| $0.000 | 404 | +$409.04 |
| $0.005 | 387 | +$389.01 |
| $0.020 | 360 | +$323.94 |

These positive numbers are **not an executable-return estimate**. Quote age, unobserved depth, historical fee variation, stale candle prices, selection of survivors, and settlement timing can change the result. The forecasting candidate also performs worse on Brier loss. Good-looking hypothetical P&L therefore does not rescue the forecasting claim. Trade-level assumption flags and complete cash ledgers are saved for inspection.

## Synthetic controls

Each seeded control contains 1,200 independent generated events, with 600 evaluated after warm-up. They use the same pipeline. Synthetic prices, depth, fees and outcomes are constructed and do not represent either exchange.

| Control | Market Brier | Selected blend Brier | Interpretation |
|---|---:|---:|---|
| Genuine generated signal | 0.201646 | **0.175714** | Pipeline can recover an intentionally planted edge |
| No additional signal | 0.181112 | 0.180819 | Difference is not statistically clear; event CI includes zero |
| Signal reverses at warm-up boundary | **0.197706** | 0.218955 | Historical fitting and calibration fail under regime reversal |

The signal-control Brier improvement interval is [0.014861, 0.037322]; the no-signal interval is [−0.002372, 0.002934]. The reversal interval is [−0.028762, −0.013948]. These validate expected behaviors of the implementation, not a real financial opportunity. No-signal hypothetical P&L is −$34.08; reversal P&L is −$202.54.

## Polymarket integration status

The current Data API v2 endpoint returned **865 price observations** for the sampled October Fed contract over the requested three-day range. A separate public CTF order-book request produced a two-sided snapshot. Both are included as API smoke fixtures, not scored strategy results. Current metadata's advertised end date is explicitly marked as a placeholder requiring rule audit before modeling.

The importer can build historical Polymarket evaluation rows from current price history plus an independently verified resolution/schedule manifest. It refuses to infer final outcomes from prices or `closed` alone. **No real Polymarket predictive advantage was evaluated**, because an audited label/vintage universe was not available in this task. V2 identifier handling and pagination are fixture-tested; only CTF assets and the current Data API v2 were exercised against live Polymarket endpoints. Do not transfer the weather-trained model to the sampled Fed question.

## Material limitations

1. **Conditional historical time safety:** splitting and feature availability are enforced, but the Kalshi import assumes final `close_time` reproduces the listing-time fixed rule schedule and assumes a 60-second data publication lag. A listing-time schedule database and actual ingestion logs were not available. Hand-entered Polymarket evidence likewise needs independent audit.
2. **Coverage and selection:** bounded pagination, resolved survivors, exclusion of one-sided markets and one city/season constrain generalization. Voids, fractional payouts, disputed outcomes and cancellations are rejected or absent, not modeled as binary results.
3. **Calibration and nonstationarity:** small calibration sets and seasonal change can make fitted probabilities worse. ECE depends on bins; no individual probability is certified as true. Cluster intervals are approximate, with few calendar blocks.
4. **No independent real-data signal yet:** real runs use quote-derived features only. The external-vintage forecast interface is exercised synthetically. No news model, archived weather ensemble, economic vintage dataset, or LLM probabilities were invented.
5. **Binary marginal forecasts:** separate temperature buckets are grouped for leakage, uncertainty and exposure, but predictions are not reconciled to a jointly coherent distribution. Cross-platform contract equivalence is not inferred automatically.
6. **Execution is hypothetical:** no historical depth replay, queue position, partial fills, precise per-fill rounding, maker rebates, funding costs, mark-to-market equity or early exits. Current supported fee formulas must not be substituted for unknown older regimes. No live trading or geographic-access workaround was performed.

The next useful experiment is a frozen, prospective paper study in one recurring category with rule-matched, timestamped independent forecasts and live depth snapshots. Compare against both market baselines on a fresh holdout. The current real-data candidate should remain rejected as a generally superior forecaster.

Primary sources and the reasoning behind these choices are in [RESEARCH.md](RESEARCH.md); operating instructions and the snapshot contract are in [README.md](README.md).
