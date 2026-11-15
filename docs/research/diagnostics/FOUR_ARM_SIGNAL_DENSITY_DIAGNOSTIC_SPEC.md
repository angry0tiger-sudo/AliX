# Four-Arm Signal-Density Diagnostic Specification

- **Status:** Design only; not executed
- **Scope:** Nobitex pre-hack confirmatory prefix only
- **Source cohort:** Run #10 artifact `14163527266`
- **Purpose:** Localize attrition between forecast coverage, threshold eligibility, entry eligibility, position availability, risk checks, and completed trades without creating or registering an economic trial.

## Evidence available now

The retained Run #10 artifact contains twelve final trial manifests, a run
report, and a forecast-cache payload. The recorded cache has 1,138 anchors and
1,138 finite inference results per model. The runner expands each cached
forecast across the following 24 decision bars. Therefore the code contract
implies 27,312 forecast-covered decision bars across the full 27,825-row
confirmatory prefix for B/C/D (1,138 × 24). This is a coverage count, not a
count of threshold-qualified signals and not a per-fold count.

The manifests expose completed trades and fold metrics only. They do not
persist the number of bars above the threshold, entry rejections, risk
rejections, signals suppressed while a position is open, or forecast-margin
histograms. The current artifact is therefore insufficient to compute those
counts. **شاهد کافی نیست** for the requested signal-density decomposition.

## Required diagnostic output

Produce one aggregate row per arm, threshold, and fold, plus a family summary.
Never write raw OHLCV, timestamps, prices, or per-bar forecasts to logs.

For B/C/D, record:

1. `valid_forecast_anchors`: finite cache anchors contributing to the fold.
2. `valid_forecast_bars`: decision bars covered by forward-filled finite cache entries.
3. `below_threshold_bars`: valid forecast bars whose applicable arm forecast is
   at or below the frozen threshold.
4. `eligible_signal_bars`: valid forecast bars strictly above the frozen
   threshold.
5. `eligible_signal_episodes`: false-to-true transitions in the bar signal
   stream, kept separate from signal bars so 24-bar forward filling is not
   mistaken for independent signals.
6. `entry_gate_rejections`, split by explicit reason: non-finite/invalid ATR,
   no valid next-bar entry inside the fold, and signal suppressed while an
   existing position is open.
7. `risk_gate_rejections`, split by named pre-entry risk rule. The current
   runner has no separately instrumented risk-rejection counter; barrier exits
   are completed trades, not risk-gate rejections. Do not report these as zero.
8. `completed_trades` and exit-reason counts, reconciled to the existing
   trade simulator.

For Arm A, report forecast fields as `NOT_APPLICABLE`; report Donchian
signal bars/episodes and the same entry/position/trade accounting separately.

## Forecast-return distance histogram

For each finite forecast bar, calculate the signed relative margin

`relative_margin = (forecast_return_bps - threshold_bps) / threshold_bps`.

Aggregate counts into preregistered bins:
`<= -100%`, `(-100%, -50%]`, `(-50%, -25%]`, `(-25%, 0%]`,
`(0%, 25%]`, `(25%, 50%]`, `(50%, 100%]`, and `>100%`.
This is a dimensionless distance from threshold, not a price series. Histogram
counts must sum to `valid_forecast_bars` for every B/C/D arm-threshold-fold
combination.

## Reconciliation and acceptance invariants

- Each decision bar is counted exactly once within its fold.
- `valid_forecast_bars = below_threshold_bars + eligible_signal_bars`.
- Every eligible signal bar is classified as entry attempted, entry-gate
  rejected, or suppressed by an already-open position.
- Completed trades reconcile exactly with the existing simulator; exit reasons
  are outcomes, not entry rejections.
- Cache identity and SHA-256 must match the retained Run #10 cache. Do not
  download model weights or invoke either inference model.
- Use only the exact 27,825-row pre-hack prefix and the frozen three fold
  ranges. Do not access any post-hack data or holdout.
- Output aggregate counts and histograms only. No raw OHLCV, raw prices,
  timestamps, or individual forecast values in artifacts or logs.

## Execution boundary

This document specifies a future diagnostic-only job, not an authorization to
run it. Any execution requires a separate owner directive and a separately
reviewed workflow path that cannot satisfy or arm the Four-Arm economic
execution trigger. No trial IDs, threshold values, or economic verdicts may be
changed by the diagnostic.
