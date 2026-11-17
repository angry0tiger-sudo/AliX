# ADR-052 — Regime R Execution Gate and Configuration Enforcement

- **Status:** Implemented; mock-only verification required by Standard CI and Quality Preflight.
- **Date:** 2026-11-17
- **Branch:** `research/nobitex-hf-strategy`
- **G3 preregistration SHA:** `2741a00625cd0248a2f61453a48874a8c07978e6`
- **Frozen JSON:** `config/regime_r/hgbq_cqr_preregistration.json`
- **Frozen JSON SHA-256:** `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- **Related:** ADR-050, ADR-051, G3.5 Owner Directive.

## 1. Decision

The `HGBQCQRAdapter` now loads the G3 preregistration JSON during initialization. There are no constructor-level hyperparameter overrides and no fallback defaults. A missing file, unreadable file, invalid JSON, changed byte-level SHA-256, or incompatible configuration contract raises a fail-closed error before an adapter instance can be used.

The runtime configuration is tied to the byte commitment recorded in ADR-051. The loader verifies the HGB estimator identity and all common model settings, the exact lower/median/upper quantile definitions, MAPIE 1.6.0 CQR API/order/confidence settings, and the registered 1/4/24-hour horizons. Model construction obtains its complete common parameter dictionary and each quantile from that loaded object.

## 2. Configuration-drift controls

The new `tests/unit/test_regime_r/test_configuration_drift.py` verifies:

- the exact SHA-256 commitment `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`;
- the live HGB estimator attributes for `learning_rate`, `max_depth`, `max_iter`, and the remaining committed common parameters against the JSON;
- each fitted estimator's quantile level against the JSON; and
- fail-closed behavior when one byte changes or when the config file is missing.

A formatting-only change is also a byte change and therefore invalidates the committed digest. Updating the locked configuration requires explicit written Owner approval, a new commitment, and a superseding ADR.

## 3. Owner execution authorization gate

Every call to `HGBQCQRAdapter.fit` now checks the process environment variable:

```text
ALIX_REGIME_R_EXECUTION_AUTHORIZED=true
```

The comparison is exact and case-sensitive. If the variable is absent or is anything other than the literal string `true`, `ExecutionAuthorizationError` is raised **before input validation or model construction**. The gate applies to all fit calls; G3.5 itself does not set the variable. Unit tests temporarily set it only around their deterministic synthetic arrays and restore the prior environment immediately afterwards.

This is a process-level Owner authorization switch, not a data-provenance classifier: the adapter accepts in-memory arrays and cannot independently determine whether a caller-supplied array originated from real or synthetic data. Consequently, the environment variable must remain unset except during an explicitly Owner-authorized execution window, and independent data-window/holdout controls remain mandatory. No network, exchange, data-source, Holdout, backtest, or order functionality is introduced here.

## 4. Mock-only verification and non-authorization

No real market data was read and no model was fitted on real data in G3.5. The only authorized test fits use deterministic synthetic arrays and temporary in-process test authorization. No Holdout access, economic backtest, fold evaluation, private API call, paper order, or live order was performed.

The gate and configuration checks establish enforcement of the frozen runtime contract; they do not establish forecast quality, empirical coverage, or profitability.

## 5. G4/G5 readiness boundary

The adapter is now wired to the exact G3.0 byte-committed model configuration, the drift tests guard the runtime estimator values, and the Owner gate blocks fit attempts unless explicitly enabled. The G3.5 engineering prerequisites for receiving a future G4/G5 authorization and approved data window are in place.

This ADR does **not** itself grant that authorization. Before any empirical run, the Owner must explicitly set the environment variable in the approved execution context and separately authorize the data source, exact UTC data window, and any non-Holdout evaluation. Holdout access and economic backtesting remain governed by separate Owner instructions and controls. No configuration tuning is permitted.

## 6. Preserved governance

- `main` remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.
- The G3 JSON and its SHA-256 commitment remain unchanged.
- All 30 G3 trial entries remain `PREREGISTERED_LOCKED` and unevaluated.
- Ruff, Strict Mypy, Xenon, coverage, and other quality thresholds are unchanged.

**G3.5 outcome:** configuration wiring, byte-level drift enforcement, and the Owner fit gate are implemented. This is engineering readiness for a separately authorized G4/G5 phase, not permission to access or fit real data.

Architecture First. Evidence Always. Fail Closed.
