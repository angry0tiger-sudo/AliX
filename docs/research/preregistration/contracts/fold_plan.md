# Four-Arm Chronological Fold Plan

- Dataset rows: 31,801; zero-based, end-exclusive indexes.
- Prefix endpoints: 22,260; 27,030; 31,801.
- Each fold is an expanding-prefix split with 70% training, 15% validation and 15% test of its prefix.
- 24 rows are purged/embargoed on both sides of the train-validation and validation-test boundaries.
- The exact interval calculation is performed by `build_walk_forward_folds` in `scripts/run_four_arm_experiment.py` and repeated in each run manifest.
- Fold 1: prefix [0,22260), raw train end floor(0.70*22260), raw validation end floor(0.85*22260).
- Fold 2: prefix [0,27030), raw train end floor(0.70*27030), raw validation end floor(0.85*27030).
- Fold 3: prefix [0,31801), raw train end floor(0.70*31801), raw validation end floor(0.85*31801).
- All ranges are chronological and no rows are shuffled. Earlier fold OOS rows can be historical context in later expanding folds.
- The model context is past-only and the 512 warm-up rows are never counted as scored rows.
- Fold stability is conservative: every test fold must independently achieve NetPF >= 1.10 at both costs and MaxDD <= 15%.
