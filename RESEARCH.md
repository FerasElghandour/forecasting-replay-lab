# Research notes — October 14, 2026

Sources were checked for this task. API documentation and fee schedules can change; adapters fail on unfamiliar schemas rather than manufacturing compatibility. Claims in the referenced conversation about a large 2026 trading study were not used as evidence.

## Current APIs

**Polymarket:** Gamma supplies discovery, outcome labels and protocol identifiers. Current price history is also exposed through Data API v2 with cursor pagination and maximum 15-day explicit ranges. Outcome identifiers differ between CTF and Protocol V2. The prototype supports both identifier mappings and has an explicit legacy CLOB-history reader. A live Data API v2 read and CTF order-book read were exercised; Protocol V2 mapping and legacy schema are fixture-tested only. [Market details](https://docs.polymarket.com/market-data/market-details), [prices and order books](https://docs.polymarket.com/market-data/prices-order-books).

**Kalshi:** Public REST market data is documented on `external-api.kalshi.com/trade-api/v2`. Settled markets/candles move between current and historical storage according to published cutoffs. The importer checks both tiers, preserves pagination and ticker identities, and reads candle **end** timestamps. YES bids and asks imply a NO ask at one minus the YES bid; this does not establish available NO fill size. [Historical tier](https://docs.kalshi.com/getting_started/historical_data), [market candles](https://docs.kalshi.com/api-reference/market/get-market-candlesticks), [historical market candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks), [historical discovery](https://docs.kalshi.com/api-reference/historical/get-historical-markets).

**Costs:** Current Polymarket documentation describes a price-dependent taker curve with category-dependent rates and five-decimal rounding. Kalshi's July 2026 PDF specifies a multiplier-dependent taker curve, separate maker fees, and no settlement fee. Its centicent rounding text and cent-rounded example table warrant a precision sensitivity check; the prototype defaults conservatively to cents. These are present schedules, not evidence for every historical market date. [Polymarket fees](https://docs.polymarket.com/trading/fees), [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf).

## Forecasting methods considered

| Method | Assessment | Prototype choice |
|---|---|---|
| Raw market probability | Essential difficult-to-beat reference | Always scored |
| Market-only beta calibration | Simple comparator for systematic probability distortion | Always scored |
| Regularized market-anchored regression | Auditable, low capacity; can incorporate independent point-in-time forecasts | Implemented |
| Beta calibration | Parametric family includes identity; less flexible than isotonic on small samples | Identity-shrunk, later calibration set |
| Isotonic regression | Flexible monotone map; can overfit small datasets | Deferred |
| Domain model / expert ensemble | Can add independent information, but requires archived forecast vintages | Timestamped `evidence_p` interface; synthetic control only |
| Retrieval-augmented LLM forecasts | Promising for event questions; historical knowledge contamination is difficult to rule out | Deferred to prospective collection |
| Logical/multiclass consistency projection | Useful for exclusive buckets and nested contracts | Not implemented; binary outputs are marginal research forecasts |

Beta calibration's inclusion of the identity map motivates a conservative correction of existing probabilities; published advantages for particular classifiers do not establish superiority in prediction markets. [Kull, Silva Filho and Flach, AISTATS 2017](https://proceedings.mlr.press/v54/kull17a.html).

Brier score and logarithmic loss are proper scores for probability forecasts; report both alongside reliability rather than optimize classification hit rate. They measure more than calibration alone. ECE is a bin-dependent descriptive diagnostic, not a proper score or proof of calibration. [Gneiting and Raftery, 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

Rolling-origin evaluation uses only earlier information. Prediction markets additionally require purging by **when a final label became available**, not merely by when a forecast was issued. We add event-group separation and fitting-stage embargoes to that general approach. [Hyndman and Athanasopoulos, time-series cross-validation](https://otexts.com/fpptr/tscv.html).

Retrieval, forecasting and aggregation have produced competitive results on forecasting-platform questions, but that evidence does not establish profitable prediction-market execution. Modern LLMs can know the outcomes of historical questions even if retrieved documents are date-filtered. A future LLM branch should log prompts, model versions, document publication/ingestion dates, and predictions prospectively before outcomes occur. [Halawi et al., 2024](https://arxiv.org/abs/2402.18563), [ForecastBench](https://arxiv.org/abs/2409.19839).

## Evaluation protocol

The unit is a market at a predeclared horizon. Every event has total scoring weight one, so adding many correlated temperature buckets cannot overwhelm other days. Brier improvements use paired cluster resampling, both by event and by nonoverlapping seven-day calendar blocks. Positive differences mean lower model Brier loss. Intervals are exploratory percentile bootstrap intervals; they do not correct for repeated strategy research, longer dependence, or distribution shift.

The outer test is untouched by fitting, calibration, and blend selection. Blending is selected on log loss. An improvement in Brier score need not imply an improvement in log loss or financial utility. Paper decisions separately require a cost-adjusted edge and available execution inputs. All timestamps in files are Unix seconds, independent of the execution host's time zone.

## What would justify a stronger claim

Before selecting a candidate, freeze the target category, forecast horizon, data inclusion rules, features, cost model and decision rule. Add independently informative archived vintages—such as forecasts of the exact weather settlement measurement—without substituting realized measurements or revised observations. Use a complete listing-time universe including canceled, void and unresolved markets; score verified final binary outcomes while separately reporting coverage and abstentions.

Then preserve a new future holdout, compare with raw and calibrated market baselines, and require improvements that survive serial-dependence and cost sensitivity checks. Collect timestamped depth snapshots and prospective paper forecasts to evaluate slippage, latency, fill feasibility and rule interpretation. A lower Brier score on this bounded sample alone is insufficient for a profitability claim. No live trading or access-restriction bypass is necessary for this research workflow.
