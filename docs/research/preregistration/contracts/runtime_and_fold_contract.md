# Four-Arm Runtime, Fold, Signal and Execution Contract

**Registration:** NOBITEX-FOUR-ARM-KC-20261113  
**Date:** 2026-11-14  
**Scope:** Pre-hack BTCUSDT 1h dataset only; no post-hack inputs.

## 1. Model inputs and fixed adapters

- Kronos-Small is `NeoQuasar/Kronos-small` with its separately pinned `NeoQuasar/Kronos-Tokenizer-base` tokenizer.
- Chronos-2 is the actual small Chronos-2 checkpoint `autogluon/chronos-2-small`; Chronos-T5 is not substituted.
- All models run zero-shot, on CPU, with no fine-tuning, parameter updates, or learned combiner.
- Common context: exactly the latest 512 completed hourly candles through decision close (t).
- Forecast horizon: next 24 1h candles.
- Refresh cadence: every 24 bars, anchored at the start of each test fold. The latest forecast is held constant for the next 24 decision bars.
- Kronos adapter: input OHLCV and UTC timestamps; use the predicted 24th-step close; `kronos_return = forecast_close_24 / close_t - 1`.
- Chronos adapter: input univariate closes only; use the median (0.5 quantile) at step 24; `chronos_return = median_close_24 / close_t - 1`.
- B/C/D retain the same baseline entry predicate. Arm B additionally requires `kronos_return > 0`; arm C requires `chronos_return > 0`; arm D requires the arithmetic mean of the two percentage returns to be strictly positive.
- No extra tuned thresholds or per-arm normalization parameters are introduced. Percent return is the common forecast scale for the D arithmetic mean.

## 2. Deterministic three-fold chronological split

For dataset length (N=31,801), define prefix endpoints:

- Fold 1 prefix end: (lfloor 0.70N floor = 22,260)
- Fold 2 prefix end: (lfloor 0.85N floor = 27,030)
- Fold 3 prefix end: (N = 31,801)

For each prefix endpoint (P):
- `train_end = floor(0.70 * P)`
- `validation_end = floor(0.85 * P)`
- Train rows: `[0, train_end - 24)`
- Validation rows: `[train_end + 24, validation_end - 24)`
- Test rows: `[validation_end + 24, P - 24)`

The 24-bar gap on each side of both boundaries is the declared 24h purge/embargo separation. Train/validation/test ranges are chronological; each later fold expands its prefix and may include earlier fold observations only after they are historical. No shuffle is used. The 512-bar model context is past-only and is not scored as an observation. The final 24 rows of each prefix are not test decisions, preserving a full 24h outcome horizon.

This is a deterministic rolling-origin interpretation of the Owner's 70/15/15, three-fold chronological default. The exact integer boundaries are also emitted in every experiment manifest.

## 3. Baseline signal and trade mechanics

- Donchian entry decision at candle close (t): `close_t > max(high[t-20:t])`, using the prior 20 completed candles only.
- Entry: next candle open.
- ATR: Wilder ATR(14), computed only from current and prior candles.
- Initial stop: entry price minus (2 	imes ATR_t).
- Profit target: entry price plus (2 	imes ATR_t).
- Trailing stop: after a completed candle with no exit, ratchet to the maximum of the prior stop and highest completed high since entry minus (2 	imes ATR) as-of that completed candle. The updated stop is active from the next candle.
- Maximum holding: 24 hourly candles after entry; time exit at the final candle close.
- Only one position per arm may be open at a time. No pyramiding, leverage, short positions, or overlapping trades.
- Gap handling: if the next candle opens beyond a stop, fill at the open; if it opens beyond the target, fill at the open.
- Intrabar ambiguity: if both stop and target are touched in one 1h candle, apply the conservative stop-first convention. If only one barrier is touched, fill at that barrier.
- Model signals are decided at close (t) and can enter no earlier than the next candle open.

## 4. Costs and metrics

- Round-trip cost scenarios: exactly 50 bps and 90 bps.
- Net trade return = gross price return minus the specified all-in round-trip cost. The cost is not doubled again.
- Net profit factor = sum of positive net trade returns divided by the absolute sum of negative net trade returns. If the loss denominator is zero or no trades exist, report the metric as unavailable rather than infinity.
- MaxDD is measured on sequential compounded net trade returns, starting from equity 1.0.
- Fold stability uses the conservative rule that each of the three test folds must individually meet NetPF >= 1.10 at both costs and MaxDD <= 15%. Aggregate acceptance also requires NetPF >= 1.10 at both costs, at least 100 completed trades, and aggregate MaxDD <= 15%.
- DSR is not assigned a numeric value while `GLOBAL_N=UNKNOWN`. The manifest reports `dsr=null` and `dsr_status=INSUFFICIENT_EVIDENCE_GLOBAL_N_UNKNOWN`; such an arm cannot pass the DSR gate. No guessed trial count or synthetic DSR is permitted.
- An arm is PASS only when every aggregate gate, all three fold-stability gates, and a defensible DSR > 0 are established. Otherwise it is NO-GO / INSUFFICIENT_EVIDENCE, with the exact failed/unknown gates recorded.

## 5. Data provenance and safety

- Input file must be `nobitex-btcusdt-1h.csv` from the previously retained GitHub Actions pre-hack artifact, with CSV SHA-256 `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`, 31,801 rows, regular 1h grid at :30 UTC.
- The real runner does not call the public fetcher or any Nobitex endpoint. It verifies the source artifact manifest and file digest before parsing.
- The post-hack dataset, V2 artifact, spec v3 and trigger file are not runner inputs.
- The real runner validates the independent execution lock and the safety environment before opening the data manifest, dataset CSV, model snapshots, or any model runtime.
- `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, `private_api_calls=0`, `real_order_attempts=0`, `ALIX_DRY_RUN=true`, `ALIX_ALLOW_LIVE_TRADING=false`.
