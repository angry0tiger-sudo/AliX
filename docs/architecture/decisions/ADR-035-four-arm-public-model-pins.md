# ADR-035: Public Hugging Face Model Pins for the Four-Arm Experiment

- **Title:** Pin publicly available models for the four-arm experiment.
- **Date:** 2026-11-14.
- **Status:** Model revisions and primary weight digests captured by GitHub Actions; execution remains blocked pending the real runner and complete execution contracts.
- **Related:** MRD v0.9, ADR-032, ADR-034, ADR-033, FOUR_ARM_BLOCKERS.md.
- **Scope:** Public model artifacts only; pre-hack four-arm research only. No post-hack holdout access is authorized.

## Context and repository-name corrections

The Owner requested public Hugging Face models using the names `Salesforce/chronos-small` for Kronos-Small and `amazon/chronos-t5-small` for Chronos-2.

The public model identities were checked and corrected to preserve the intended model families:

1. `Salesforce/chronos-small` is not the requested Kronos-Small model. The public Kronos-Small checkpoint is `NeoQuasar/Kronos-small`.
2. `amazon/chronos-t5-small` is the original Chronos-T5 Small family, not Chronos-2. The actual Chronos-2 model family was selected as `autogluon/chronos-2-small`, the small 28M-parameter Chronos-2 checkpoint suitable for CPU-constrained inference.
3. Kronos-Small requires its separate tokenizer artifacts. The tokenizer repository is pinned independently and is part of the Kronos inference dependency.

These substitutions are recorded rather than silently treating Chronos-T5 as Chronos-2.

## Source and capture procedure

The immutable references below were captured by the GitHub Actions workflow `.github/workflows/four-arm-model-pin-capture.yml`, run #2, Run ID `43552152363`, on commit `64a40e15e06f4ba8c9b54e83164e795884fac96f`.

- **Capture timestamp (UTC):** `2026-11-14T03:32:39.099615+00:00`.
- **Source:** Hugging Face Hub model revision API: `/api/models/{repository}/revision/main?blobs=true`.
- **Requested ref:** `main`.
- **Resolved revision:** the full immutable 40-character commit SHA returned by the Hub.
- **Weight digest:** the 64-character SHA-256 LFS object digest returned by the Hub metadata API. The workflow validates the format and records file size and license metadata.
- **Workflow result:** `FOUR_ARM_MODEL_PIN_CAPTURE=PASS`; artifact upload succeeded.

## Immutable model pins

| Role | Hugging Face repository | Requested ref | Resolved immutable revision | Weight file | SHA-256 | Size (bytes) | Declared license |
|---|---|---|---|---|---|---:|---|
| Kronos-Small | `NeoQuasar/Kronos-small` | `main` | `901c26c1332695a2a8f243eb2f37243a37bea320` | `model.safetensors` | `b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020` | 98,980,656 | MIT |
| Kronos tokenizer | `NeoQuasar/Kronos-Tokenizer-base` | `main` | `0e0117387f39004a9016484a186a908917e22426` | `model.safetensors` | `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` | 15,842,368 | MIT |
| Chronos-2-Small | `autogluon/chronos-2-small` | `main` | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | `model.safetensors` | `492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd` | 111,749,048 | Apache-2.0 |

The full pin manifest is retained as the GitHub Actions artifact named `four-arm-model-pins-64a40e15e06f4ba8c9b54e83164e795884fac96f` from Run ID `43552152363`. The workflow output includes the source API URL, capture timestamp, revision, weight filename, digest, byte size and declared license for each entry.

## Runtime and usage boundary

1. The runner must load these exact full revisions, never the mutable `main` aliases.
2. The runner must verify the selected downloaded weight file(s) against the recorded SHA-256 before model use. The revision and weight digest are separate controls.
3. The Kronos tokenizer revision and digest are mandatory for Kronos inference. It is not interchangeable with the Kronos model weights.
4. Model inference is CPU-only and must respect the 8 GB runner memory limit. Models are not fine-tuned; this preregistered comparison uses inference only.
5. Any runtime package or source-code dependency must be independently version-locked. This ADR does not claim that pinning Hub weights also pins the inference implementation, tokenizer behavior, or Python environment.
6. No forecast threshold, adapter, scale normalization, or combination behavior may be selected by inspecting economic outcomes.

## Execution status

This ADR resolves model repository identity, immutable model revisions, and the Hub-reported primary weight digests. It does **not** authorize a run by itself and does not complete the execution lock.

The following remain mandatory before real pre-hack execution:

- Reviewed and content-addressed Kronos/Chronos forecast adapters, including forecast horizon, as-of alignment, units, zero-threshold conversion, and invalid-output behavior.
- Exact chronological fold plan with UTC boundaries and immutable pre-hack row indices; the 70/15/15 allocation and three-fold interpretation must be explicit.
- Frozen triple-barrier executable-price, intrabar collision, gap, fill, and expiry contract.
- Frozen 50 bps and 90 bps fee/fill simulator.
- Reproducible DSR implementation and conservative sensitivity treatment. `GLOBAL_N=UNKNOWN` must remain UNKNOWN; no numeric DSR PASS may be fabricated when its multiple-testing uncertainty is unresolved.
- Runtime dependency lock and a provenance/license/corpus-overlap disclosure for the exact model artifacts.
- A real runner with a mock-only synthetic test path, tests proving lock validation occurs before data/model I/O, and a workflow that uses only the pre-hack artifact whose CSV SHA-256 is `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.

The existing post-hack holdout remains sealed. No holdout row or value was accessed to discover or capture these model pins.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `main=FROZEN`

A successful model-pin capture is metadata evidence only. It is not a training run, economic evaluation, PASS, Paper Trading authorization, or Release Authorization.


## Repository evidence preservation (2026-11-14)

### Owner-request identity correction

The initial Owner directive incorrectly mapped the model identifiers to model families:

- `amazon/chronos-t5-small` is an older Chronos-T5 Small checkpoint, not Chronos-2. It is rejected for Arm C.
- Chronos-2 is pinned to `autogluon/chronos-2-small`, the actual small Chronos-2 model family.
- Kronos is a separate model family, pinned to `NeoQuasar/Kronos-small`, with its independent tokenizer `NeoQuasar/Kronos-Tokenizer-base`.
- The initial directive also named `Salesforce/chronos-small` as the Kronos-Small identifier; that is not the public Kronos checkpoint and is not used.

This correction is a model-identity correction only. It does not change the registered four arms or authorize any experiment.

### Artifact corroboration and retained repository evidence

The model pins were independently captured twice by GitHub Actions:

| Capture | Run ID | Artifact ID | Artifact name | Captured at (UTC) | Result |
|---|---:|---:|---|---|---|
| #2 | `43552152363` | `14096433464` | `four-arm-model-pins-64a40e15e06f4ba8c9b54e83164e795884fac96f` | `2026-11-14T03:32:39.099615+00:00` | All three repositories/revisions/weight SHA-256 values match capture #3 |
| #3 | `43552413845` | `14096931828` | `four-arm-model-pins-04897f6bf9a7340346a6807c3bffdb0771278a7d` | `2026-11-14T03:35:31.106972+00:00` | Selected canonical evidence for repository copy |

The two JSON payloads were downloaded and compared. The model revisions, model weight SHA-256 values, and tokenizer revision and SHA-256 are identical. The evidence has been copied into `config/four_arm/model_pins.json`; the repository copy is now the durable reference rather than the expiring CI artifacts.

The contract fingerprints were also captured twice:

| Capture | Run ID | Artifact ID | Artifact name | Captured at (UTC) |
|---|---:|---:|---|---|
| #1 | `43553975977` | `14097730906` | `four-arm-contract-fingerprints-9eb2db9655884c93af7655f1983a9588bfa57794` | `2026-11-14T03:54:28Z` |
| #2 | `43554204478` | `14096939228` | `four-arm-contract-fingerprints-c432c44c2e42d274394da5f094e04358f13e60e5` | `2026-11-14T03:57:23Z` |

Eight contract hashes are unchanged. Only `kronos_adapter_sha256` changed because the contract was amended to disclose the fixed, deterministic derivation of Kronos' required `amount` input from the available OHLCV fields. Capture #2 is the current fingerprint set and is retained in `config/four_arm/contract_fingerprints.json`; Capture #1 remains historical evidence and must not be substituted for the current Kronos adapter hash.

These repository copies preserve metadata and hashes only. They do not contain OHLCV data, holdout rows, model outputs, or experiment results.
