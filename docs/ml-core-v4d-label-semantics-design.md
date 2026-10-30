# V4-D: Label Semantics and Multi-Horizon Target Design

**Status:** Research design only. No model training is authorized by this document.
**Parent revision:** `ac428dd94383173570fe3cf8af510e5b0941336e`
**Owner directive:** V4-D Transition Audit & Label Semantics.

## 1. Problem statement

V4-C compares two ATR Triple-Barrier horizons while keeping the barriers fixed:

| Scenario | Horizon | Take profit | Stop loss |
|---|---:|---:|---:|
| Baseline | 24 candles | 2.0 × ATR(14) | 1.0 × ATR(14) |
| Short_Horizon | 12 candles | 2.0 × ATR(14) | 1.0 × ATR(14) |

The 12-candle horizon changes the meaning of HOLD. A row that reaches TP or SL at candle 13-24 under Baseline becomes HOLD/HORIZON_EXPIRY under Short_Horizon. Treating all such rows as ordinary HOLD discards the fact that the original directional event did occur, only later.

V4-C artifacts contain row-level validation labels and timestamps for trained experiments, aggregate label audits for every experiment, and model diagnostics. They do **not** contain the underlying one-minute OHLCV candles. Exact late-touch time cannot be inferred from report JSON alone. The audit therefore requires the original matching BTCUSDT and ETHUSDT one-minute CSV inputs and fails closed if their row count, continuity, resampled bar counts, global exit-reason counts, validation distributions, or serialized validation labels do not match the V4-C report.

The audit is a validation-window label analysis. It does not fit a classifier, calculate model scores, access private exchange APIs, or change production targets.

## 2. Label definitions shared by both designs

Let (t) be the event candle, (ATR_t) the causal ATR(14), and the two barriers be:

- Upper barrier: (close_t + 2 ATR_t)
- Lower barrier: (close_t - 1 ATR_t)

The first barrier touched in candles (t+1) through (t+24) determines the event direction and touch time.

| Timing class | Definition |
|---|---|
| `HIT_FAST` | First TP/SL touch occurs in candles 1-12 |
| `HIT_LATE` | No touch in candles 1-12; first TP/SL touch occurs in candles 13-24 |
| `NO_HIT` | Neither barrier is touched through candle 24 |

An event without a complete future 24-candle window remains **unlabeled**. It must not be silently converted to `NO_HIT`. Incomplete ATR history likewise remains unlabeled.

The existing collision policy is retained unless a separate Owner Directive changes it: when TP and SL are both touched within the same candle, SL wins and the directional outcome is SELL. The precise collision rule must be explicit in any new target contract.

Timing and direction are separate information dimensions. The proposed three timing classes alone do not preserve BUY versus SELL. Direction must be retained as an orthogonal target/field or prediction head, with values `BUY` (TP), `SELL` (SL or collision under the current rule), and null for `NO_HIT`.

## 3. Design A: Three-class timing classifier

### Target

One multiclass target:

- `HIT_FAST`
- `HIT_LATE`
- `NO_HIT`

To retain direction, the training record must also retain an orthogonal direction target for `HIT_FAST` and `HIT_LATE`. The three-class output on its own is timing-only and is not a directional trading signal.

### Advantages

- One classifier and one primary decision boundary.
- Timing is learned directly as one mutually exclusive outcome.
- A single probability vector provides a straightforward view of fast, late, and no-hit likelihoods.
- Inference and deployment orchestration are simpler than a two-stage cascade.

### Disadvantages

- The three timing classes may be imbalanced, especially if `HIT_LATE` is sparse in some pairs/timeframes.
- The classifier must separate two distinct hit mechanisms from no-hit in one optimization problem.
- Direction is lost unless an additional direction head or target is added. Adding it creates a multi-task design rather than the strict single-target three-class design.
- A directional head trained only on hit rows has fewer samples and requires its own calibration and support gates.

### Required acceptance evidence

- Per-class support by pair and timeframe, including `HIT_LATE`.
- Confusion matrix and class-wise precision/recall/F1 for timing classes.
- Separate BUY/SELL direction metrics on hit events.
- Chronological, purged validation, with no overlap between label windows across partitions.
- Calibration and Brier Skill measured against the appropriate class-prevalence baseline.
- Explicit treatment of collision and incomplete-tail events.

## 4. Design B: Hierarchical two-stage classifier

### Target and inference

**Model 1: Event occurrence**

- `HIT`: TP or SL is touched within 24 candles.
- `NO_HIT`: neither barrier is touched within 24 candles.

**Model 2: Event timing, conditional on a hit**

- `FAST`: first touch occurs within candles 1-12.
- `LATE`: first touch occurs within candles 13-24.

The second model is trained only on rows where Model 1's ground-truth target is `HIT`. At inference, the unconditional timing probabilities are composed from the stages:

- (P(HIT\_FAST) = P(HIT) \times P(FAST \mid HIT))
- (P(HIT\_LATE) = P(HIT) \times P(LATE \mid HIT))
- (P(NO\_HIT) = 1 - P(HIT))

Direction remains a separate, explicit output for TP versus SL/collision. Neither stage should be treated as a directional signal by itself.

### Advantages

- Separates the question "does a barrier get hit?" from "how quickly?".
- Model 2 focuses on timing distinctions only among actual hit events.
- The occurrence model can be evaluated and calibrated independently from conditional timing.
- Stage-level diagnostics can show whether failure comes from event occurrence or timing.

### Disadvantages

- Two classifiers, two model lifecycles, and more inference orchestration.
- Model 2 has fewer training rows because it sees only hit events.
- Errors from Model 1 propagate: a missed hit cannot be recovered by Model 2.
- Composed probabilities can be miscalibrated even when each stage looks acceptable in isolation; end-to-end calibration must be measured on untouched forward data.
- The conditional Model 2 dataset is selected by the target outcome, so chronological split discipline and leakage controls are especially important.

### Required acceptance evidence

In addition to Design A's label, direction, and temporal gates:

- Model 1: precision/recall/F1 for `HIT` and `NO_HIT`.
- Model 2: conditional FAST/LATE metrics on ground-truth hit events.
- End-to-end metrics for composed `HIT_FAST`, `HIT_LATE`, and `NO_HIT`.
- End-to-end Brier/ECE on an untouched forward segment.
- Stage-wise and joint support gates, including the count of hit events available to Model 2.
- Explicit analysis of error propagation and decision-threshold stability.

## 5. Comparison and decision gate

| Dimension | Design A: three-class | Design B: hierarchical |
|---|---|---|
| Number of classifiers | 1 timing classifier, plus direction output if retained | 2 classifiers, plus direction output if retained |
| Main learning problem | Joint fast/late/no-hit separation | Hit occurrence, then conditional timing |
| Sample efficiency | All eligible rows train the timing classifier | Model 2 sees hit rows only |
| Error propagation | One-stage errors | Model 1 errors propagate into stage 2 |
| Probability composition | Direct class probabilities | Product/complement of stage probabilities |
| Operational complexity | Lower | Higher |
| Direction preservation | Requires an orthogonal direction target/head | Requires an orthogonal direction target/head |

No design is selected by this document. The transition audit must first establish the true FAST/LATE/NO_HIT and BUY/SELL frequencies by symbol and timeframe. A design decision then requires out-of-sample evidence; prevalence changes alone are not acceptance evidence.

## 6. V4-D safety and scope

- No LightGBM or other model training in the transition-audit path.
- No modification to `adaptive_targets.py` or Production target definitions.
- No change to CI coverage thresholds or quality gates.
- No Champion assignment, order permission, or trading authorization.
- Required safety output remains `Champion = null`, `order_permission = false`, `private_api_calls = 0`, `real_order_attempts = 0`, and `trading_authorization = NO-GO`.

## 7. Transition Audit CLI

The V4-C workflow uploads only the report JSON. It does not upload the source CSVs. Supply the exact matching, gap-free 730-day one-minute inputs:

```bash
python -m scripts.audit_ml_core_label_transitions \
  --report alix-real-data-classification-report.json \
  --btc-1m-csv BTCUSDT_1m.csv \
  --eth-1m-csv ETHUSDT_1m.csv \
  --output alix-v4d-label-transition-audit.json
```

The command rejects mismatched provenance or data. It does not synthesize missing candles, infer hit times from labels, or emit a partial event list as though it were complete.
