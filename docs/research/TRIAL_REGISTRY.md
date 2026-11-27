

---

## [APPEND] M7 Owner Selection and Preregistration — 2026-11-27

- **Owner decision:** R&D Path B (cross-asset lead-lag / cross-sectional portfolio rotation) selected; project remains OPEN.
- **Decision record:** ADR-070-m7-cross-sectional-owner-selection.md.
- **Previous focused trials:** 12 unique logical identities rejected across HGBQ-CQR (3), M3 OFI (2), Regime T Triple-Barrier (1), M5.2 Sentiment (4), and M6 Triple-Barrier three-class (2). These do not represent total historical AliX experimentation; GLOBAL_N=UNKNOWN and GLOBAL_DSR=NOT_COMPUTABLE remain.
- **New M7 identities:** exactly two registered candidates:
  - NOBITEX-M7-5PAIR-RIDGE-TOP2-01
  - NOBITEX-M7-10PAIR-RIDGE-TOP2-01
- **Status:** PREREGISTERED_LOCKED for model/policy and trial definitions; data qualification remains a separate fail-closed gate. The two universe-size variants are predeclared and are the only candidates. Folds/cost scenarios do not increase trial count.
- **Model:** StandardScaler + Ridge(alpha=1.0), four fixed features, next-four-hour log-return target, long-only top-two strictly positive predicted scores, 50% initial-equity weight per selected asset, unused capital remains in USDT.
- **Data:** public Nobitex Spot hourly PRE-HACK only; fixed interval 2021-10-31T20:30:00Z to 2025-06-17T20:30:00Z inclusive, 31,801 rows per symbol. BTCUSDT hash remains 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e. All ten altcoin assets require exact common timestamps, no gaps, no duplicates and SHA-256 manifests. No substitutions, interpolation or backfill.
- **Data acquisition authorization:** public /market/stats and /market/udf/history GET only; no credentials/private API. Failed symbols must be retained as blocked/rejected results. The five-pair and ten-pair candidates are blocked if their exact fixed universe is incomplete.
- **Economic gates:** NetPF >= 1.10; completed OOS trades >= 100; MaxDD <= 15%; at least 3 valid folds and 3 stable folds of 4; Cost-50 and Cost-90 separately. A stable fold requires >=20 completed trades, positive net PnL and fold PF >1.
- **Holdout:** the historical PRE-HACK terminal range is already contaminated by earlier AI-6/DEAP use. M7 remains exploratory. Holdout stays sealed and Post-Hack remains unread.
- **Safety:** Champion=null; order_permission=false; trading_authorization=NO-GO; private_api_calls=0; real_order_attempts=0; paper_orders=0; local_execution=false.
