# Four-Arm Chronological Fold Plan — ADR-038

- Immutable source: 31,801 rows; source CSV SHA-256 remains registered in ADR-037.
- Confirmatory prefix: first 27,825 rows only, zero-based and end-exclusive.
- Final 3,976 source rows are excluded from feature construction, ATR, signals, forecast cache, model context, folds, trades and metrics.
- Confirmatory prefix endpoints: 19,477; 23,651; 27,825.
- Each fold is an expanding-prefix split with 70% training, 15% validation and 15% test of its prefix.
- 24 rows are purged/embargoed on both sides of train-validation and validation-test boundaries.
- Exact interval calculation is performed by build_walk_forward_folds in scripts/run_four_arm_experiment.py and repeated in each run manifest.
- All ranges are chronological; no rows are shuffled. Earlier fold OOS rows may be historical context in later expanding folds.
- The common 512-bar model context is past-only and is not scored as an observation.
- The global forecast cache is built once over the eligible 27,825-row prefix; folds and thresholds only read cached forecasts.
- Fold stability requires every test fold to meet NetPF >= 1.10 at both costs and MaxDD <= 15%.
