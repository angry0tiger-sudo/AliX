# AliX Historical Trial Registry

Status: INITIAL RECONCILIATION — family-level counts recorded; unresolved historical trial IDs remain UNKNOWN.
Last reconciled: 2026-11-12 UTC.
Registry basis: immutable GitHub Actions run history and retained result artifacts. Counts below distinguish
workflow executions, candidate variants, actually trained/evaluated candidates, and unique model/genome trials.
These are not interchangeable units and MUST NOT be summed into a fictitious global N.

## Global accounting rule

- `GLOBAL_N = UNKNOWN`.
- A single exact global N cannot be certified because several earlier failed/cancelled workflow executions
  do not retain complete candidate-level ledgers, and candidate variants, model runs and GP genomes are
  different trial units.
- Never replace UNKNOWN with a guessed number.
- For a new DSR/PBO cohort, include all earlier trials from the same hypothesis family, including rejected,
  failed, cancelled, partial, and tuning variants whenever the candidate reached evaluation.
- A workflow run that failed before candidate evaluation is an execution attempt, not automatically one
  economic model trial. Where logs/artifacts do not establish how far it progressed, count it as an
  execution attempt and mark its evaluation count UNKNOWN.
- Retries are separate attempts but not automatically separate candidate identities; record both run ID
  and attempt number.
- The counts in this file are conservative lower bounds for multiple-testing correction, not claims that
  the historic search space is fully reconstructed.

## 1. High-Frequency research (HF proxy)

- Workflow path: `.github/workflows/nobitex-hf-strategy.yml`
- Workflow file blob SHA at commit `bc493e24b9e3f849b6d9fbd0b776d15b39b8061f`:
  `453bca6bf29955a36a02d1831a0d7d65ecb27da2`.
- Workflow runs: **41 distinct run numbers** (#1–#41).
- Total recorded attempts including reruns: **42** (run #13 has attempt 2).
- All 41 current workflow-run records are failure conclusions; most recent exact-head example:
  [HF Research #41 / Run ID 43274590426](https://github.com/angry0tiger-sudo/AliX/actions/runs/43274590426),
  commit `bc493e24b9e3f849b6d9fbd0b776d15b39b8061f`.
- Run #41 failed at Step 9 (Binance Spot 15m) with
  `binance_proxy_page_gap_detected:15m:previous=1679661000:current=1679666400:delta=5400:expected_step=900`.
- Native Nobitex 1h data fetch in run #41 succeeded: 31,801 rows, zero missing intervals,
  CSV SHA-256 `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Trial accounting: `N_HF_EXECUTION_RECORDS = 41`; `N_HF_ATTEMPTS = 42`.
  Actual distinct economic configurations / successful backtest evaluations across all runs:
  **UNKNOWN**; lower bound for recorded execution attempts is 41. Do not describe 41 runs as
  41 proven successful economic evaluations.
- Run #41 native data artifact: Artifact ID `13971011148`, ZIP SHA-256
  `ae7a4d00fe2ec753b28868e4849fea4d45c88f1fd25b366b64298d7fec809b21`.
- The full execution ledger is below.

| Workflow run | Run ID | Commit SHA | Conclusion | Attempt |
|---:|---:|---|---|---:|
| 1 | [43162196636](https://github.com/angry0tiger-sudo/AliX/actions/runs/43162196636) | `a4e7690572d2d5689e6f2908c0010cd9d63afc5b` | failure | 1 |
| 2 | [43162589101](https://github.com/angry0tiger-sudo/AliX/actions/runs/43162589101) | `d6ea6e2f29df2764b2010719423322f6b42398c7` | failure | 1 |
| 3 | [43163049367](https://github.com/angry0tiger-sudo/AliX/actions/runs/43163049367) | `a91ca8c334b4871ed05f88f1f65406e68cf439bd` | failure | 1 |
| 4 | [43163142730](https://github.com/angry0tiger-sudo/AliX/actions/runs/43163142730) | `117161226482a2e84daf1c3ae947de05991c68ae` | failure | 1 |
| 5 | [43163238829](https://github.com/angry0tiger-sudo/AliX/actions/runs/43163238829) | `54928c4f3f8d757da0807e577f9430a73e0fdedc` | failure | 1 |
| 6 | [43163457233](https://github.com/angry0tiger-sudo/AliX/actions/runs/43163457233) | `e9ff50338f5f7351889284dd5d669a2aca1dacce` | failure | 1 |
| 7 | [43163642495](https://github.com/angry0tiger-sudo/AliX/actions/runs/43163642495) | `3052cf27c8042cad4da34d8a5f1b01156c12ced3` | failure | 1 |
| 8 | [43164165640](https://github.com/angry0tiger-sudo/AliX/actions/runs/43164165640) | `2e4bcb4230f8fee6720a46ff2474c7f1332d8ae0` | failure | 1 |
| 9 | [43164340856](https://github.com/angry0tiger-sudo/AliX/actions/runs/43164340856) | `5804bb4f6cc22ee5d35b1e8b93331f024cc91c2c` | failure | 1 |
| 10 | [43166538633](https://github.com/angry0tiger-sudo/AliX/actions/runs/43166538633) | `4a2d3f901a0c82d87d9d5a34da666644273e8fd8` | failure | 1 |
| 11 | [43167345366](https://github.com/angry0tiger-sudo/AliX/actions/runs/43167345366) | `2ac02dc3a861c8d04139929732c23df7501acfc9` | failure | 1 |
| 12 | [43167512895](https://github.com/angry0tiger-sudo/AliX/actions/runs/43167512895) | `71187961e3f7b1f677ef2db4366fb7377410bd11` | failure | 1 |
| 13 | [43195043147](https://github.com/angry0tiger-sudo/AliX/actions/runs/43195043147) | `448486e45e5a61fe2fe96a4d3ed7ccbc21325629` | failure | 2 |
| 14 | [43204228243](https://github.com/angry0tiger-sudo/AliX/actions/runs/43204228243) | `230e0528269bed49c6f1750eded9cae9c43e111a` | failure | 1 |
| 15 | [43204641970](https://github.com/angry0tiger-sudo/AliX/actions/runs/43204641970) | `109adcd0beffd437db604ba511f7ad424b425e6b` | failure | 1 |
| 16 | [43205062062](https://github.com/angry0tiger-sudo/AliX/actions/runs/43205062062) | `74825151df4e547e95d0b7b1d5b0a7948ad225cd` | failure | 1 |
| 17 | [43205341848](https://github.com/angry0tiger-sudo/AliX/actions/runs/43205341848) | `06e42d411ef165d65fa7548b788eb7d8881456ac` | failure | 1 |
| 18 | [43205844698](https://github.com/angry0tiger-sudo/AliX/actions/runs/43205844698) | `f0c981a2bb699e6f67b320b2cee163c75bdc2c4b` | failure | 1 |
| 19 | [43206685252](https://github.com/angry0tiger-sudo/AliX/actions/runs/43206685252) | `ba2c42ad1d73a1b931debba8f8c11234e0f8befc` | failure | 1 |
| 20 | [43213443503](https://github.com/angry0tiger-sudo/AliX/actions/runs/43213443503) | `575587c4a6d6066e12f7330c0a417c76e9bfdc10` | failure | 1 |
| 21 | [43219371916](https://github.com/angry0tiger-sudo/AliX/actions/runs/43219371916) | `9bd9b0aa4eaac1c8aa5fb329a8269bb4fbf4e198` | failure | 1 |
| 22 | [43219757400](https://github.com/angry0tiger-sudo/AliX/actions/runs/43219757400) | `3ace7290dad174645f3f5bf70444ce9ac016aafb` | failure | 1 |
| 23 | [43220432764](https://github.com/angry0tiger-sudo/AliX/actions/runs/43220432764) | `e94286a023d09f91649a80aff3d8ba3be53f8436` | failure | 1 |
| 24 | [43220639815](https://github.com/angry0tiger-sudo/AliX/actions/runs/43220639815) | `bd6de40ccb9ec05b92f1f16bf479afc46e13e10d` | failure | 1 |
| 25 | [43221054375](https://github.com/angry0tiger-sudo/AliX/actions/runs/43221054375) | `b5b4f9649f5bb12cb3a7269962bc3b4713b8e0cb` | failure | 1 |
| 26 | [43239496234](https://github.com/angry0tiger-sudo/AliX/actions/runs/43239496234) | `112d0382ed307ab2e43f4964ccb05c59eba0b9e0` | failure | 1 |
| 27 | [43239788710](https://github.com/angry0tiger-sudo/AliX/actions/runs/43239788710) | `90cf91283ec2b320a07161f2d9ee5f9a8725a5ee` | failure | 1 |
| 28 | [43243683636](https://github.com/angry0tiger-sudo/AliX/actions/runs/43243683636) | `85c5c790fdfbe23d99390648ae97e9c4aa114d72` | failure | 1 |
| 29 | [43244135160](https://github.com/angry0tiger-sudo/AliX/actions/runs/43244135160) | `8a631862e9a15c7242548fb4233af2c4f02036de` | failure | 1 |
| 30 | [43244932430](https://github.com/angry0tiger-sudo/AliX/actions/runs/43244932430) | `e74d44960c6a71278acc8173f5caccff480e2a5f` | failure | 1 |
| 31 | [43245629451](https://github.com/angry0tiger-sudo/AliX/actions/runs/43245629451) | `7b146fed7e1541037696f519d70cac4a6614bea8` | failure | 1 |
| 32 | [43245956378](https://github.com/angry0tiger-sudo/AliX/actions/runs/43245956378) | `277b5a9e92e636fee005711420702ca596f8f844` | failure | 1 |
| 33 | [43247760829](https://github.com/angry0tiger-sudo/AliX/actions/runs/43247760829) | `8b32491995df55440b257ef9690d8ac9ad3f4223` | failure | 1 |
| 34 | [43248084830](https://github.com/angry0tiger-sudo/AliX/actions/runs/43248084830) | `c573f661a57f1d20138b6859edcd29960c8e3107` | failure | 1 |
| 35 | [43250753163](https://github.com/angry0tiger-sudo/AliX/actions/runs/43250753163) | `5744e918cb865e4ce8fd7ec74f0a73a681d13508` | failure | 1 |
| 36 | [43252610139](https://github.com/angry0tiger-sudo/AliX/actions/runs/43252610139) | `7d84abc438fca94cc9431b52314ea608f163a9bf` | failure | 1 |
| 37 | [43258927760](https://github.com/angry0tiger-sudo/AliX/actions/runs/43258927760) | `609e6ed3459142e83aef9bc69e0412f57f56d0fa` | failure | 1 |
| 38 | [43259380731](https://github.com/angry0tiger-sudo/AliX/actions/runs/43259380731) | `997a300c55cd24be63eecadb93af99599eabd8d0` | failure | 1 |
| 39 | [43261726587](https://github.com/angry0tiger-sudo/AliX/actions/runs/43261726587) | `0e5e09d1b6cf6cbc36268b5cd0883fd3710cf821` | failure | 1 |
| 40 | [43273583026](https://github.com/angry0tiger-sudo/AliX/actions/runs/43273583026) | `877b0c5f31d2a2ef9f5a08697a94e1fcf416d7a6` | failure | 1 |
| 41 | [43274590426](https://github.com/angry0tiger-sudo/AliX/actions/runs/43274590426) | `bc493e24b9e3f849b6d9fbd0b776d15b39b8061f` | failure | 1 |

## 2. V4-T Custom Economic Score

- Workflow path: `.github/workflows/ml-core-v4t-economic-score.yml`
- Workflow blob SHA at successful commit `d5fe33c5cf32bb41cc083acc11a9935d30324173`:
  `fab1a50d5e6ab174446ab38e8e68afa46516af4e`.
- Workflow executions: **20** (#1–#20); successful final evidence:
  [Run ID 41625662854](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625662854), run #20,
  commit `d5fe33c5cf32bb41cc083acc11a9935d30324173`.
- Successful artifact: Artifact ID `13229820663`, ZIP SHA-256
  `7fddda74100eb554506c7a8716688eb528a7aa88bac5f1c549169c63f85a8997`.
- Exact report counts:
  - `candidate_count = 32` (32 candidate IDs in the retained JSON artifact).
  - `experiment_count = 16`.
  - `trained_candidate_count = 4`.
  - `trained_experiment_count = 2`.
  - `sample_gate_skipped_candidate_count = 28`.
  - `random_search_iterations = 9`.
- Trial accounting: **32 registered candidate variants**; 4 reached model training/evaluation, 28 were skipped
  by sample gates. The 9 inner random-search iterations are separately recorded; their overlap with candidate
  identities is not assumed. The 19 earlier failed workflow executions may have partial evaluations:
  **UNKNOWN**. Global V4-T evaluation N across all executions: **UNKNOWN, lower bound 32 candidate IDs**.
- Exact 32 candidate identities and status are retained in the JSON inside Artifact `13229820663`.

| Workflow run | Run ID | Commit SHA | Conclusion | Attempt |
|---:|---:|---|---|---:|
| 1 | [41587466180](https://github.com/angry0tiger-sudo/AliX/actions/runs/41587466180) | `017d86334dff6972fefe3637243466deb03f6616` | failure | 1 |
| 2 | [41587606129](https://github.com/angry0tiger-sudo/AliX/actions/runs/41587606129) | `af3d898d8dd92decf6be3d864776f00d8f56352c` | failure | 1 |
| 3 | [41587741328](https://github.com/angry0tiger-sudo/AliX/actions/runs/41587741328) | `9d2dd7e76872d5f094552402fcc7b7c3ee01d0f0` | failure | 1 |
| 4 | [41587953634](https://github.com/angry0tiger-sudo/AliX/actions/runs/41587953634) | `ccedb5ab881f68ab38e6f95d27eb943e85a98ebc` | failure | 1 |
| 5 | [41588062917](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588062917) | `6c4e84c7d2a07c5609c3fdfdf8e3ce06d601640c` | failure | 1 |
| 6 | [41588401053](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588401053) | `c80aa08d6e18412f9d7a59a5b76288ddf22b6c08` | failure | 1 |
| 7 | [41588473235](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588473235) | `ebee11de3d77d814cc78a1855c529bc8e81aa8d2` | failure | 1 |
| 8 | [41588613997](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588613997) | `8216a2f77fd1ee0e9cfe5089dcfc032e060bb403` | failure | 1 |
| 9 | [41588902869](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588902869) | `21c7746dc53b2f31441c92ae910fb5ea13830b1e` | failure | 1 |
| 10 | [41588995390](https://github.com/angry0tiger-sudo/AliX/actions/runs/41588995390) | `bb9c74b7b977c51e822b8484e4e1abdca642713f` | failure | 1 |
| 11 | [41589340003](https://github.com/angry0tiger-sudo/AliX/actions/runs/41589340003) | `99f56bce99e1568f4a2be19ddfc0f7693a968bd7` | failure | 1 |
| 12 | [41589454152](https://github.com/angry0tiger-sudo/AliX/actions/runs/41589454152) | `1ba1e6fd0f8a2b8240d8c7d7a41482957848ea5f` | failure | 1 |
| 13 | [41609379887](https://github.com/angry0tiger-sudo/AliX/actions/runs/41609379887) | `5db648add9a0abfd6647b984d53826a27b5c2196` | failure | 1 |
| 14 | [41609548646](https://github.com/angry0tiger-sudo/AliX/actions/runs/41609548646) | `1713c4d44c7bf51f4c2c3842f113ce2266a92cd1` | failure | 1 |
| 15 | [41609647180](https://github.com/angry0tiger-sudo/AliX/actions/runs/41609647180) | `dfead7307e2b9af49d21611b0da9a7017583ba53` | failure | 1 |
| 16 | [41609801661](https://github.com/angry0tiger-sudo/AliX/actions/runs/41609801661) | `414429d46853be74ccc83ac46e66eb9227e53cca` | failure | 1 |
| 17 | [41609992281](https://github.com/angry0tiger-sudo/AliX/actions/runs/41609992281) | `09dc89e91625e947574066739d8f08eb1d560165` | failure | 1 |
| 18 | [41625271331](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625271331) | `7126132b1ab8f666b15cf33504450c8dfb28a485` | failure | 1 |
| 19 | [41625419057](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625419057) | `4eeca5c4931dced795c34731a989f662f8acf587` | failure | 1 |
| 20 | [41625662854](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625662854) | `d5fe33c5cf32bb41cc083acc11a9935d30324173` | success | 1 |

## 3. V4-AP 36-cell primary diagnostic sweep

- Workflow path: `.github/workflows/ml-core-v4ap-diagnostic-sweep.yml`
- Workflow blob SHA at successful commit `798ef2d49bcbc715d35b16862a2cf83aa0865486`:
  `5436ff5cb177817bddf82f7c0b9f7c8f3ba94007`.
- Workflow executions: **26** (#1–#26); final successful evidence:
  [Run ID 42493418590](https://github.com/angry0tiger-sudo/AliX/actions/runs/42493418590), run #26,
  commit `798ef2d49bcbc715d35b16862a2cf83aa0865486`.
- Successful artifact: Artifact ID `13624018702`, ZIP SHA-256
  `24d040a7bf3ddf2ef6ee4667bafe0c042e7267e6f866ddd7cf0967c7efb97666`.
- Exact report counts:
  - `expected_combinations = 36`.
  - `candidates = 36`.
  - `dsr_cohort.trial_count = 36`.
  - `valid_sharpe_estimate_count = 36`.
  - `qualifying_candidate_ids = []`.
  - Conclusion: `CONCLUSION: NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET.`
- Trial accounting: **N_V4AP = 36 registered candidate cells** in the completed DSR cohort.
  The 25 earlier failed/cancelled workflow executions may contain partial model work not represented in the
  final report: additional candidate evaluations **UNKNOWN**. Do not assume they performed zero evaluations.
- Historical data source was Binance Spot 1m, not Nobitex native; the report discloses the 80-minute source
  discontinuity (2023-03-24 12:40–13:59 UTC), retained as a discontinuity and not interpolated.
  These cells are not evidence of transferability to Nobitex.
- Exact 36 candidate identities and metrics are retained in the JSON inside Artifact `13624018702`.

| Workflow run | Run ID | Commit SHA | Conclusion | Attempt |
|---:|---:|---|---|---:|
| 1 | [42452843589](https://github.com/angry0tiger-sudo/AliX/actions/runs/42452843589) | `7822d613d8587c6d02f1cd0f4ebcf6cfdb103d06` | failure | 1 |
| 2 | [42452967652](https://github.com/angry0tiger-sudo/AliX/actions/runs/42452967652) | `d72526c8ac15e27af51bb133eba5bafaf2c1fd8a` | cancelled | 1 |
| 3 | [42452969945](https://github.com/angry0tiger-sudo/AliX/actions/runs/42452969945) | `e22c81d8c96d97efb2b82168c56f0dbb3c5542ef` | failure | 1 |
| 4 | [42453073437](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453073437) | `b41a14a35cda6df906fe8cd72e38b75f0672c507` | failure | 1 |
| 5 | [42453163334](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453163334) | `330adb64844f0e7bceacbf2c82232e73528ed551` | failure | 1 |
| 6 | [42453268728](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453268728) | `547c54301959fef33ccab026c1e93f4443a90e42` | failure | 1 |
| 7 | [42453463675](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453463675) | `5a0ce4ae13f7b2fffa60622ff4c01f9c858a3189` | failure | 1 |
| 8 | [42453546102](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453546102) | `5b40908c24acb3c46229a054a5a84d0ef48bc351` | cancelled | 1 |
| 9 | [42453606303](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453606303) | `736b0eae4c9303c365d6bfd7feb3ee916ac1b1fe` | cancelled | 1 |
| 10 | [42453871322](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453871322) | `72b4ed0c4bdd0ee056e705d66e0f0ffa1449bb64` | cancelled | 1 |
| 11 | [42453886512](https://github.com/angry0tiger-sudo/AliX/actions/runs/42453886512) | `6821adefa5ab45b92ed46d7ff8145513274829e8` | failure | 1 |
| 12 | [42454327138](https://github.com/angry0tiger-sudo/AliX/actions/runs/42454327138) | `1d689e6535c606026b1231b99353545546d303ce` | failure | 1 |
| 13 | [42466931571](https://github.com/angry0tiger-sudo/AliX/actions/runs/42466931571) | `8000cba3c6af365c160ce09208673e598b3bbf16` | failure | 1 |
| 14 | [42467054496](https://github.com/angry0tiger-sudo/AliX/actions/runs/42467054496) | `da30b976459e5dd0ccc53f70ad1d4a3ea2ca5b5f` | failure | 1 |
| 15 | [42467212548](https://github.com/angry0tiger-sudo/AliX/actions/runs/42467212548) | `e09127c74b0b2f63ea7f966e5b1307b453a50a85` | failure | 1 |
| 16 | [42470454524](https://github.com/angry0tiger-sudo/AliX/actions/runs/42470454524) | `754ceaf8b3f999b8974c57b827205c5e030e183c` | cancelled | 1 |
| 17 | [42470520473](https://github.com/angry0tiger-sudo/AliX/actions/runs/42470520473) | `35c09443584658da96478adba024629862aeb932` | cancelled | 1 |
| 18 | [42470718083](https://github.com/angry0tiger-sudo/AliX/actions/runs/42470718083) | `fe3966dc230f3a110cd2cd92c748fac0f50f7f3e` | cancelled | 1 |
| 19 | [42470776630](https://github.com/angry0tiger-sudo/AliX/actions/runs/42470776630) | `931521128e1e630fc13bef3fd2122f21ccf41173` | failure | 1 |
| 20 | [42492108501](https://github.com/angry0tiger-sudo/AliX/actions/runs/42492108501) | `d85ed71a2b7d95b069011a65fb89a9c6b9b2802e` | failure | 1 |
| 21 | [42492260036](https://github.com/angry0tiger-sudo/AliX/actions/runs/42492260036) | `eaea94a7bc3259c16272cea92483d390015b4ee4` | failure | 1 |
| 22 | [42492399566](https://github.com/angry0tiger-sudo/AliX/actions/runs/42492399566) | `d72b88f4419f1b56ada4e40cf74124db2358df2a` | failure | 1 |
| 23 | [42492702456](https://github.com/angry0tiger-sudo/AliX/actions/runs/42492702456) | `7e241041b5db0382bfa85f1c43d521d0426344a2` | cancelled | 1 |
| 24 | [42492792203](https://github.com/angry0tiger-sudo/AliX/actions/runs/42492792203) | `65ab3dea30e7fce0e6e97b081d2140670e0fe966` | failure | 1 |
| 25 | [42493324845](https://github.com/angry0tiger-sudo/AliX/actions/runs/42493324845) | `c86d895ca51eb6b6c7bd0d5eb5f50bcd12522604` | cancelled | 1 |
| 26 | [42493418590](https://github.com/angry0tiger-sudo/AliX/actions/runs/42493418590) | `798ef2d49bcbc715d35b16862a2cf83aa0865486` | success | 1 |

## 4. AI-6 PPO risk-only research

- Workflow path: `.github/workflows/ai6-rl-risk-research.yml`
- Workflow blob SHA at final commit `7464e308d4e1efdb573a8225fb3a392ae7614199`:
  `e9ae139011da80c1ccca54d10d8db15165a78715`.
- Research protocol path: `docs/research/AI6-PPO-RISK-RESEARCH.md`,
  blob SHA `c5c6491835af99aaadafc1205154ce45dcb33b9c`.
- Final verdict path: `docs/phase-reports/AI6_RL_FINAL_VERDICT_2026-11-11.md`,
  blob SHA `bd535da99af29c53a4ec81366f8a3b20db2fc5c9`.
- Recorded workflow executions on `research/alternative-data-pivot`: **3** (#20–#22);
  one success (#22), two failures (#20–#21).
- Final run: [AI-6 RL #22 / Run ID 43000569350](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569350),
  commit `7464e308d4e1efdb573a8225fb3a392ae7614199`.
- Successful artifact: Artifact ID `13845902872`, ZIP SHA-256
  `dbfc2fc640fbf61bf34834b2a31ad46ee1945462259d1fe1ee57c74031847651`.
- Final training: 200,000 PPO timesteps, seed 42, 50 bps round-trip, fixed baseline/teacher.
- Final result: PPO accepted 0/234 validation signal episodes and 0/225 final-holdout signal episodes;
  completed PPO trades = 0; final decision NO-GO.
- A separate predecessor AI-6 experiment on commit `1f78ef845ac1c6336b0ec94027c1467fe116f3c0`
  is explicitly described in the research document as a prior run that skipped 228/228 validation and
  231/231 holdout signal episodes. Its run ID is not established by the retained final report.
- Trial accounting: recorded workflow execution records = 3; at least **2 distinct
  completed/evaluated AI-6 policy experiments** are documented (predecessor and final #22).
  Exact total unique model configurations / policy trials across older branches: **UNKNOWN**, lower bound 2.
  Do not count 200,000 timesteps as 200,000 independent trials.

| Workflow run | Run ID | Commit SHA | Conclusion | Attempt |
|---:|---:|---|---|---:|
| 20 | [42999889572](https://github.com/angry0tiger-sudo/AliX/actions/runs/42999889572) | `d3ce4431d46fca25f7034f3135c4296ceb6de4fe` | failure | 1 |
| 21 | [43000158057](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000158057) | `55f9d86c424eb3ac226e4047cfe69f915ba16dcb` | failure | 1 |
| 22 | [43000569350](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569350) | `7464e308d4e1efdb573a8225fb3a392ae7614199` | success | 1 |

## 5. DEAP genetic programming

- Workflow path: `.github/workflows/deap-genetic-programming.yml`
- Workflow blob SHA at successful commit `9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92`:
  `9c21ec86c04268cc094d62b25dd1da65f65bf44d`.
- Plan path: `docs/phase-reports/AI2_DEAP_REVISITED_PLAN.md`,
  blob SHA `56169f7ddad7dfc3794fffeab74116ee324731b2`.
- Workflow executions: **7** (#1–#7); final success:
  [DEAP #7 / Run ID 43022191014](https://github.com/angry0tiger-sudo/AliX/actions/runs/43022191014),
  commit `9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92`.
- Successful artifact: Artifact ID `13855652715`, ZIP SHA-256
  `e29926ca66e88d01ff61d211a3d9dc45d49c8acd3f31399d0b50ccfde8f1ab9b`.
- Final run configuration: population 500, 30 generations, seed 20261111.
- Actual report field: `evolution.unique_genomes_evaluated = 7842`.
- The generation log has 31 entries (generation 0 through 30) and the cumulative unique-evaluation
  count ends at 7,842. The final OOS result had 4 trades; therefore its apparently high PF does not
  pass the required trade-support gate.
- Trial accounting: **7,842 unique genome evaluations** are positively evidenced in the successful run.
  The six earlier failed workflow runs may have evaluated additional genomes; their totals are not
  reconstructed. Global DEAP N across all runs: **UNKNOWN, lower bound 7,842**.
- Do not infer 500 × 30 as the actual count; the artifact's measured `unique_genomes_evaluated`
  is the auditable number. Do not count repeat appearances of the same genome as unique variants.

| Workflow run | Run ID | Commit SHA | Conclusion | Attempt |
|---:|---:|---|---|---:|
| 1 | [43011613937](https://github.com/angry0tiger-sudo/AliX/actions/runs/43011613937) | `3edfe80acf4d8e8525546f2f33ace3eecf63a163` | failure | 1 |
| 2 | [43012302398](https://github.com/angry0tiger-sudo/AliX/actions/runs/43012302398) | `e297348a36becfc02eae90bf84f5741226e12682` | failure | 1 |
| 3 | [43018201079](https://github.com/angry0tiger-sudo/AliX/actions/runs/43018201079) | `40ad7e0dd2e1909023b1c0ada1fbd92db5f60549` | failure | 1 |
| 4 | [43018499674](https://github.com/angry0tiger-sudo/AliX/actions/runs/43018499674) | `4f54e8091f8181c2ccc4c4068fb6224b32d40521` | failure | 1 |
| 5 | [43019156000](https://github.com/angry0tiger-sudo/AliX/actions/runs/43019156000) | `6010534e546913e6b97e567ffa6b9daf105dfd1a` | failure | 1 |
| 6 | [43021422828](https://github.com/angry0tiger-sudo/AliX/actions/runs/43021422828) | `602f2018abdf87a28f579269c397039af4510b49` | failure | 1 |
| 7 | [43022191014](https://github.com/angry0tiger-sudo/AliX/actions/runs/43022191014) | `9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92` | success | 1 |

## Cross-family decision

| Family | Exact registered unit | Known lower bound | Execution attempts | Unknown remainder |
|---|---|---:|---:|---|
| HF | workflow execution records | 41 records / 42 attempts | 41 records | unique economic configurations/evaluations |
| V4-T | candidate variants | 32 candidates; 4 trained | 20 | partial work from 19 failed runs; inner-search overlap |
| V4-AP | candidate cells | 36 cells / 36 Sharpe estimates | 26 | partial work from 25 failed/cancelled runs |
| AI-6 | policy experiments | >=2 documented completed evaluations | 3 | older branch configurations / failed-run progress |
| DEAP | unique genomes evaluated | 7,842 | 7 | partial genome evaluations in 6 failed runs |

**Do not compute one grand N by adding these rows.** The units overlap conceptually and differ in granularity.
For new research, assign unique `trial_id` to each preregistered candidate/configuration and record
every candidate evaluation, including failed and rejected variants, in this file before evaluating
any new output.

No fresh trial is authorized by this registry. It is an audit of prior multiplicity and a prerequisite
for later pre-registration. All economic decisions remain subject to the absolute gates in QWEN.md.

## Immutable source artifacts and supporting files

| Family | Repository path + blob SHA | Final evidence run | Artifact ID | Artifact ZIP SHA-256 |
|---|---|---|---:|---|
| HF | `.github/workflows/nobitex-hf-strategy.yml` `453bca6bf29955a36a02d1831a0d7d65ecb27da2` | 43274590426 | 13971011148 | `ae7a4d00fe2ec753b28868e4849fea4d45c88f1fd25b366b64298d7fec809b21` |
| AI-6 | `.github/workflows/ai6-rl-risk-research.yml` `e9ae139011da80c1ccca54d10d8db15165a78715` | 43000569350 | 13845902872 | `dbfc2fc640fbf61bf34834b2a31ad46ee1945462259d1fe1ee57c74031847651` |
| DEAP | `.github/workflows/deap-genetic-programming.yml` `9c21ec86c04268cc094d62b25dd1da65f65bf44d` | 43022191014 | 13855652715 | `e29926ca66e88d01ff61d211a3d9dc45d49c8acd3f31399d0b50ccfde8f1ab9b` |
| V4-T | `.github/workflows/ml-core-v4t-economic-score.yml` `fab1a50d5e6ab174446ab38e8e68afa46516af4e` | 41625662854 | 13229820663 | `7fddda74100eb554506c7a8716688eb528a7aa88bac5f1c549169c63f85a8997` |
| V4-AP | `.github/workflows/ml-core-v4ap-diagnostic-sweep.yml` `5436ff5cb177817bddf82f7c0b9f7c8f3ba94007` | 42493418590 | 13624018702 | `24d040a7bf3ddf2ef6ee4667bafe0c042e7267e6f866ddd7cf0967c7efb97666` |


---
## [APPEND] Historical Trials & Holdout Commitment (Restored)
### V4-T (Triple-Barrier)
- 32 variants total: 4 TRAINED, 24 rejected (insufficient samples), 4 rejected (inner-CV failure).
- 36 nested configurations, 108 fold fits logged.
### V4-AP (Grid Sweep)
- 36 cells registered in statistical cohort.
### AI-6 (PPO)
- 2 records: final experiment and predecessor.
### DEAP (Genetic Programming)
- 1 verified record: 7,842 genomes.
- 1 owner-reported record: 828 genomes.
### DSR/PBO
- Non-HF subtotal: N >= 7,912. GLOBAL_N = UNKNOWN (pending full consolidation).
### HF Research Ledger (Updated with actual Run IDs)
- #44: Run ID 43306419521, SHA 5f3fae315c84149f15380c6fb5092915e4af2868
- #45: Run ID 43306747175, SHA e6af8b168a0fc49c981dfb79689e34fc2f8e9c61
### V4-T Workflow SHA Correction
- Corrected workflow blob SHA: fab1a50d5e6ab174446ab38e68afa46516af4e

---

## [APPEND] Data operation — non-holdout grid diagnosis (2026-11-13)

### DIAGNOSE #7 — Nobitex BTCUSDT 1h structural grid check

- Operation class: non-economic, non-holdout structural data diagnosis.
- Workflow run ID: `43398908810` (run #7), conclusion: `success`.
- Diagnostic commit SHA: `e3ed17137a1b01dbf68b1e19174837282ea66dce`.
- Diagnostic window: `2026-11-02T00:00:00Z` through `2026-11-12T00:00:00Z` (inclusive UTC hourly grid).
- HTTP status: `200`.
- Returned candle count: `240`.
- Off-grid count: `240`.
- Distinct offsets from the expected UTC hourly grid: `[1800]` seconds.
- Missing expected hourly grid timestamps: `241`.
- First returned timestamp (Unix seconds): `1793579400`.
- Last returned timestamp (Unix seconds): `1794439800`.
- Artifact: `nobitex-grid-diagnose-43398908810`, artifact ID `14024246276`.
- Structural interpretation: all returned timestamps were offset by 1,800 seconds; none aligned to the expected hourly grid. The non-holdout diagnosis therefore indicates a systematic 30-minute grid offset in this sampled interval.
- OHLCV values were transiently parsed in memory only to collect timestamp metadata; no prices or OHLCV fields were printed or persisted.
- The committed holdout window was not fetched or opened. No nonce was read, generated, or consumed by the DIAGNOSE mode.
- V2 commitment remained `PENDING_DISPATCH`; no spec, gap policy, pre-hack fetcher, or economic gate was changed.

---

## [APPEND] Data operation — Nobitex venue-grid revision v2 (2026-11-13)

### Structural evidence and dispatch governance

- Operation class: structural source-grid reconciliation and immutable spec revision; no economic evaluation.
- Previous nonce #1 `3764030460fa5cbdaebf522c97453b3b`: permanently retired.
- Previous nonce #2 `d86c6da4e1df4e7b4b042a5be940df24`: permanently retired.
- Previous nonce #3 `c1c9297040cc4bfcb28ac1f414c4ad07`: permanently retired.
- No retired nonce is valid for future dispatch; the fourth nonce must be newly generated by the authorized CI `NONCE_ROTATION` mode and then appended as the latest parser-compatible `dispatch_nonce:` line in V2.
- Pre-hack source contract constants: `PREHACK_START_TIMESTAMP=1635712200` and `PREHACK_END_TIMESTAMP=1750192200`; both have `mod 3600 = 1800`.
- Adjacent non-holdout DIAGNOSE Run #7: ID `43398908810`, conclusion `success`, artifact ID `14024246276`; HTTP 200, 240 returned candles, all 240 offset by 1,800 seconds relative to the previous :00 hourly grid, with 241 missing expected :00 timestamps.
- Revised spec v2: `config/holdout/posthack_window_spec.json`; grid convention is hourly candle-open timestamps at :30 UTC, corresponding to Iran local hour boundaries.
- Inclusive window boundaries are `2025-07-01T00:30:00Z` / `1751329800` and `2026-11-01T00:30:00Z` / `1793493000`. Both move by the same 30-minute convention offset; the nominal date span and duration are unchanged.
- Expected candle count remains `11,713`; minimum separation from the last pre-hack candle-open is `1,137,600` seconds / 316 hours. Purge and embargo controls remain active.
- This is a structural venue-convention correction, not window-shopping or selection based on observed holdout results.
- **No holdout data was observed.** The holdout window was not fetched or opened during this spec revision. No price/OHLCV values were printed or persisted.
- Gap policy, cost scenarios, economic gates, and data-use lock remain unchanged. `main` stays frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- Commitment remains `PENDING_DISPATCH` until the fourth nonce is approved and the one-time acquisition has an evidence-backed outcome.

---
## [APPEND] Data operation — Nobitex post-hack boundary revision v3 (2026-11-13)

### Structural evidence and revision record

- Operation class: boundary adjustment after documented gap; metadata-only reconciliation, not an economic trial or model evaluation.
- Decision reason: `boundary adjustment after documented gap`.
- Run #12: Workflow ID `43438007282`, commit `32a2b32df73d9fd1b135c112325639399e4fa9a9`, conclusion `failure`, terminal status `GAP_DETECTED`.
- Gap artifact: ID `14045205264`, file `nobitex-btcusdt-posthack-1h.gaps.json`.
- Observed structure: 11,700 rows returned vs 11,713 expected; 13 missing candles, all at the start boundary; `off_grid_count=0`, `grid_anomaly=false`; no backfill, interpolation, or masking.
- Missing timestamps (UTC): `2025-07-01T00:30:00Z`, `01:30:00Z`, `02:30:00Z`, `03:30:00Z`, `04:30:00Z`, `05:30:00Z`, `06:30:00Z`, `07:30:00Z`, `08:30:00Z`, `09:30:00Z`, `10:30:00Z`, `11:30:00Z`, `12:30:00Z`.
- Revised spec: `config/holdout/posthack_window_spec.json`, `spec_version=3`, start `2025-07-01T13:30:00Z` / `1751376600`, end unchanged `2026-11-01T00:30:00Z` / `1793493000`, inclusive expected count `11,700`.
- Derived minimum separation from `PREHACK_END_TIMESTAMP=1750192200` is `1,184,400` seconds / 329 hours.
- Canonical spec SHA-256: `4c2b5a55691d00da67a6ccd875ad66b5559b5d06408ec2f7f08ed5eae4a9790b`; verified by Standard CI #2349 / Run ID `43440748067`.
- Revision was based solely on the documented absence of the first 13 expected timestamps at the fixed window boundary. No OHLCV values or economic/model results were used; this is not window shopping.
- Fifth dispatch nonce: `5f5f3083a55f6c07bac0551473844853`, generated by Workflow Run #14 / ID `43441496367`, artifact ID `14044658921`. The four prior nonces are permanently retired: `3764030460fa5cbdaebf522c97453b3b`, `d86c6da4e1df4e7b4b042a5be940df24`, `c1c9297040cc4bfcb28ac1f414c4ad07`, `e3fb1601e654341a21fa13f00cd6271b`.
- Gap policy, cost scenarios, economic gates, purge/embargo controls, and data-use lock remain unchanged. The public fetcher and frozen `main` remain untouched.
- No price/OHLCV values were printed or persisted. No training, fitting, tuning, backtest, or economic evaluation was performed. V2 remains `PENDING_DISPATCH` until a complete one-time acquisition passes structural verification.


---

## [APPEND] Four-Arm Kronos/Chronos Experiment — preregistered 2026-11-13

- Registration: docs/research/preregistration/FOUR_ARM_KRONOS_CHRONOS.md.
- ADR: docs/architecture/decisions/ADR-032-four-arm-kronos-chronos-preregistration.md.
- Trial family: NOBITEX-FOUR-ARM-KC-20261113.
- Status: PREREGISTERED (documentation only; NOT_STARTED; zero evaluated candidates in this registration).
- Planned arms / trial IDs:
  - A / NOBITEX-KC26-A: Baseline — Donchian-20 + ATR(14) + 2-ATR trailing stop.
  - B / NOBITEX-KC26-B: Baseline + Kronos-Small.
  - C / NOBITEX-KC26-C: Baseline + Chronos-2.
  - D / NOBITEX-KC26-D: Baseline + Kronos-Small + Chronos-2.
- Baseline source: release/nobitex-mvp, commit 9f5e5f38d2145dc428173329a74c558b1f053ab4; implementation blob integrations/nobitex/mvp.py = 39deec55bdb8c8422efcbf84d34b1dd548cf8c6b.
- Data: pre-hack Nobitex BTC/USDT 1h only; 31,801 candles; November 2021 through 2025-06-17; SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e.
- Target / horizon: return[t+1 : t+24h]; 24h; triple-barrier TP=2×ATR(14), SL=2×ATR(14), maximum holding=24h.
- Leakage controls: purge=24h; embargo=24h; scaler fit on train segment only per fold; past-only as-of features; model warm-up=512 1h candles; model refresh=24h; minimum three chronological walk-forward folds.
- Acceptance gates: NetPF≥1.10; completed trades≥100; MaxDD≤15%; DSR>0; stability across at least three folds.
- Cost scenarios: 50 bps and 90 bps round-trip, reported separately for every arm.
- Decision rule: no PASS => documented NO-GO; exactly one PASS => eligible for separate paper-phase authorization; multiple PASS => highest DSR wins. Final holdout remains sealed until an arm passes and a separate Owner Release Authorization is recorded.
- DSR historical cohort: V4-T 32 variants; V4-AP 36 cells; DEAP 7,842 verified genomes plus 828 owner-reported genomes (overlap unresolved); AI-6 at least 2 evaluated policy experiments (3 execution records); HF 41 run records / 42 attempts with unique economic trial count UNKNOWN, plus ledger entries #44 and #45. These are heterogeneous counts and MUST NOT be blindly summed.
- GLOBAL_N = UNKNOWN pending full candidate-level consolidation. Four new trial IDs are planned and registered; none has been evaluated yet.
- Execution blockers: immutable model revisions/weight SHA-256 and corpus-overlap disclosures; forecast-to-decision mapping and combined-arm rule; exact fold boundaries and seed; exact return-price/barrier collision semantics; cost/fill simulator details; and DSR implementation/N sensitivity. These must be locked before any execution.
- Explicit exclusions: no post-hack holdout access; no training, fit, inference, tuning, backtest, or economic evaluation in the preregistration turn.
- Safety: Champion=null; order_permission=false; trading_authorization=NO-GO; private_api_calls=0; real_order_attempts=0.
- Registration was added before any evaluation; all four trial IDs must be retained in DSR/multiple-testing accounting if reached for evaluation.

## [APPEND] Four-arm infrastructure build — mock only, no economic evaluation (2026-11-13)

- Status: INFRASTRUCTURE_VALIDATED_MOCK_ONLY. The four registered trial IDs remain PREREGISTERED / NOT_STARTED: NOBITEX-KC26-A, NOBITEX-KC26-B, NOBITEX-KC26-C, NOBITEX-KC26-D.
- Governance: ADR-033. This record authorizes no real-data execution, model evaluation, paper trading, or live trading.
- Recovery: the red research HEAD 839bb2b was restored with normal commit 3b11a25b423f1fee434f0a5b0f5604fe4eabfcd4; its tree is identical to the last green preregistration baseline 55a378d. Recovery CI #2364, Run ID 43462802862, succeeded.
- Mock implementation commits: 3cef4f40cae6680c5dcd40de996d1ee422b011af (pure gates, deterministic synthetic fixture runner, and authorization validator); 1bc769d9122a4d90caeb2692c08e91c9c509a046 (Ruff formatting); 03910d97ec73a2d947dc125046630a71eddbe6b1 (strictly typed test mutators); 48918074671b4a0a062fe1279ac49f80767a3989 (dispatch-only workflow, unapproved execution-lock template, ADR-033, and four-arm workflow/lock contract tests); c47d66bf8de7d3e85cb4faa3ec4aeb7e64587d86 and f56e408010a5333a38ebd89980be5fc8df8985ba (Ruff-only formatting corrections).
- Final full-tree verification: Standard CI #2370, Run ID 43466295722, commit f56e408010a5333a38ebd89980be5fc8df8985ba — SUCCESS. Required jobs quality, postgres-integration, and freqtrade-runtime-smoke passed. Unit suite: 965 passed, 5 skipped; coverage 90.26% (required >=90%). Ruff lint/format, strict mypy, security, dependency audit, architecture, complexity and CI quality gates passed.
- Mock scope: in-memory synthetic OHLCV only, deterministic default seed 20261113; report exposes only structural metadata and a synthetic fixture fingerprint, never OHLCV values. Gate tests cover passing and failing fold-stability, DSR-sensitivity, economic-gate and winner-selection boundaries, including all four registered arm IDs.
- Execution boundary: ALIX_FOUR_ARM_EXECUTION_AUTHORIZED is false in the workflow; ALIX_DRY_RUN=true; ALIX_ALLOW_LIVE_TRADING=false. The committed lock template remains NOT_AUTHORIZED and has no owner authorization SHA, model revision/weight hashes, seed, or contract fingerprints. Its fail-closed validator checks owner authorization and safety flags before reading the lock path.
- Workflow: .github/workflows/four-arm-experiment.yml contains workflow_dispatch only, runs only the synthetic mock module, and was NOT dispatched in this task.
- Data/evaluation: no Nobitex or other market-data fetch, no pre-hack dataset read, no model download/load, no training/fit/inference/tuning/backtest, no economic evaluation, no post-hack holdout access, no private API call and no order attempt. The pre-hack SHA is present only as a required future lock field. V2, spec v3 and DISPATCH_TRIGGER remain unchanged.
- Outstanding ADR-032 blockers before any real run: immutable Kronos-Small/Chronos-2 revisions and weight SHA-256 plus license/provenance/corpus-overlap disclosure; B/C forecast-to-decision adapters and D combination rule; exact fold boundaries and seed; executable triple-barrier anchor/fill/collision/expiry semantics; frozen fill/cost simulator for 50/90 bps; exact DSR method and historical trial cohort reconciliation (GLOBAL_N remains UNKNOWN); complete owner-approved execution lock and a separate explicit Owner directive.
- Safety invariant at registration: Champion=null; order_permission=false; trading_authorization=NO-GO; private_api_calls=0; real_order_attempts=0; ALIX_DRY_RUN=true; ALIX_ALLOW_LIVE_TRADING=false; main=FROZEN.

