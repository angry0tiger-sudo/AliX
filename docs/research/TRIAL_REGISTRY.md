# AliX Historical Trial Registry

Status: RECONCILED — family-level counts recorded; unresolved historical trial IDs remain UNKNOWN; current M3 OFI lifecycle is CLOSED_NO_GO.
Last reconciled: 2026-11-27 UTC.
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



## 8. Four-arm forecast-driven amendment (2026-11-14; preregistered, not evaluated)

Authority: ADR-038. This amendment is prospective; no amended-arm or threshold result was viewed before registration.

Owner-supplied historical rationale: baseline PF 0.3743 / 0.4370; AI-6 produced zero trades; TTM PF 0.3731→0.3748; meta-labeling was worse than its teacher. MRD v0.9 §13.8 motivates replacing the teacher-filter path rather than trying to filter a negative-expectancy teacher into positive expectancy.

### Registered trial cells

Each arm × threshold cell is a distinct registered trial. Arm A is the same Donchian-20 control repeated as the matched control; the repetition is disclosed.

| Threshold | Registered trial IDs |
|---:|---|
| 120 bps | NOBITEX-KC26-FD-120-A, NOBITEX-KC26-FD-120-B, NOBITEX-KC26-FD-120-C, NOBITEX-KC26-FD-120-D |
| 180 bps | NOBITEX-KC26-FD-180-A, NOBITEX-KC26-FD-180-B, NOBITEX-KC26-FD-180-C, NOBITEX-KC26-FD-180-D |
| 240 bps | NOBITEX-KC26-FD-240-A, NOBITEX-KC26-FD-240-B, NOBITEX-KC26-FD-240-C, NOBITEX-KC26-FD-240-D |

Status of all 12 cells: **REGISTERED_NOT_EVALUATED**. Earlier NOBITEX-KC26-A/B/C/D entries are retained as **SUPERSEDED_BEFORE_EVALUATION**.

### Shared controls and cache

- Arms: A Donchian-20; B Kronos-Small forecast entry; C Chronos-2 forecast entry; D arithmetic-mean ensemble forecast entry.
- Shared risk: ATR(14), 2×ATR trailing stop, maximum holding 24 hours, maximum drawdown ≤15%.
- Forecast cache: one forecast per model per 24-bar anchor over the eligible confirmatory prefix; all folds and thresholds reuse it. Target about 1,140 forecast anchors rather than about 10,000 repeated model calls. Cache identity binds dataset/model/code/runtime fingerprints and mismatch fails closed.
- Thresholds: fixed 120 / 180 / 240 bps. The prior dynamic max(2×90 bps, ATR/close) rule is superseded because it is expected to collapse near 180 bps for owner-stated typical BTC 1h ATR/close values of 30–80 bps.
- Confirmatory data: first 27,825 rows only; final 3,976 source rows are excluded from all confirmatory analysis stages.
- DSR: GLOBAL_N remains UNKNOWN. Report lower bound, unresolved remainder and 1×/2×/5×/10× sensitivity.

### Terminal decision rule

If all four arms fail at every threshold, close the directional method family with NO-GO. One volatility-focused pivot is permitted based on the recorded TTM-R2 QLIKE 0.27786 versus 0.39348 (29.38% improvement). If that single pivot also fails, close the project with a documented NO-GO. No further directional tuning or additional pivot is authorized without a new owner directive and preregistration.

No experiment has been run for this amendment. The execution lock remains NOT_AUTHORIZED; Champion remains null; order_permission remains false; trading_authorization remains NO-GO; the post-hack holdout remains sealed.


---

## [APPEND] ADR-038 forecast-driven four-arm implementation registration (2026-11-14)

- Owner directive: OWNER-2026-11-14-FOURARM; scope is pre-hack Nobitex-only research.
- The following twelve cells are the complete preregistered candidate family; initial status is REGISTERED_NOT_EVALUATED:
  - 120 bps: NOBITEX-KC26-FD-120-A, NOBITEX-KC26-FD-120-B, NOBITEX-KC26-FD-120-C, NOBITEX-KC26-FD-120-D.
  - 180 bps: NOBITEX-KC26-FD-180-A, NOBITEX-KC26-FD-180-B, NOBITEX-KC26-FD-180-C, NOBITEX-KC26-FD-180-D.
  - 240 bps: NOBITEX-KC26-FD-240-A, NOBITEX-KC26-FD-240-B, NOBITEX-KC26-FD-240-C, NOBITEX-KC26-FD-240-D.
- Earlier NOBITEX-KC26-A/B/C/D records remain preserved and are superseded before evaluation; no registry entry is deleted.
- B/C/D are direct forecast-entry strategies at the fixed candidate threshold, not baseline filters. Arm A is the identical Donchian-20 control across three threshold cells.
- Arm A is computed once and referenced in all three cells; all three registered candidate IDs remain visible in DSR trial accounting.
- The immutable full-prefix forecast cache is computed once for 27,825 eligible rows; all folds and threshold cells reuse it.
- All economic records must include NetPF at 50/90 bps, completed trades, MaxDD, all per-fold metrics, DSR status, and 1×/2×/5×/10× sensitivity disclosure.
- GLOBAL_N remains UNKNOWN; numeric DSR is prohibited until the global candidate cohort is reconciled.


---

## [APPEND] Four-arm run #2 — fail-closed bootstrap stop before research I/O (2026-11-14)

- Execution workflow Run ID: \`43593275732\`; trigger commit: \`a540589da0db617d2cd0f3ffa24ba78a7245feb8\`.
- One-time trigger validation: PASS. Exact-SHA Standard CI #2402 / Run ID \`43593275721\`: SUCCESS. Preflight #10 / Run ID \`43593275720\`, artifact \`14113491751\`: seven tools all exit code 0; pytest \`980 passed, 5 skipped\`.
- Execution stopped at the lock-only preflight step before the lock validator or any research I/O. The exact bootstrap failure was \`ModuleNotFoundError: No module named 'numpy'\` at module-level import in \`scripts/run_four_arm_experiment.py:17\`.
- Dataset CSV and manifest were not opened; the pinned weights artifact was not downloaded by this execution run; no model source was checked out, no model was loaded, no inference or backtest occurred, and no post-hack holdout was accessed.
- Evaluated candidate count: **0**. All twelve ADR-038 cells remain \`REGISTERED_NOT_EVALUATED\`; no NetPF, trade count, MaxDD, fold metric or DSR result exists for this attempt. No candidate trial record is fabricated.
- Stop rule: **STOP — no automatic retry or second execution dispatch**. The earlier Experiment #1 was separately stopped at its PROBE guard and also performed no research I/O.
- Safety closeout: execution lock reverted to \`NOT_AUTHORIZED\`; \`owner_authorization_commit_sha=null\`; workflow authorization environment reverted to false; \`EXECUTION_TRIGGER=PROBE\`. \`ALIX_DRY_RUN=true\` and \`ALIX_ALLOW_LIVE_TRADING=false\` remain enforced.
- V2, Spec v3, persisted pre-hack CSV/manifest, public fetcher and frozen main were not modified. Holdout remains sealed under ADR-039; \`GLOBAL_N=UNKNOWN\` remains.
 
---

## [APPEND] Four-arm run #4 — one-time trigger replay guard rejection (2026-11-15)

- Trigger commit: `5eb9f26e3d7162a54a4140058d5d070fcc9eb77a`; execution workflow Run ID: `43681763697` (run #4).
- The workflow failed at Step 3, **Validate one-time execution trigger**, with the exact guard error `FOUR_ARM_EXECUTION_TRIGGER_REPLAYED`.
- The event parent was `fcf3fe4958f9fa95051afeb5a8000a84893a157a`. The guard's history scan correctly found a prior `EXECUTE` transition in commit `a540589da0db617d2cd0f3ffa24ba78a7245feb8` (2026-11-14); that earlier one-time authorization was subsequently revoked by `c1e4239bb3db60a1e070cd8c2326936fd575bf72`.
- This one-time trigger is therefore **consumed**. It is not safe to re-arm or retry using the same trigger contract. The failure occurred before lock-only validation, dependency installation, model-artifact download, Kronos checkout, inference or data access.
- The workflow's upload step found no result JSON files; no trial-manifest artifact was produced. The pre-hack CSV and manifest were not opened. No forecast cache, trade simulation, fold evaluation, economic metric or candidate result was produced. Post-hack holdout access remained forbidden and did not occur.
- Evaluated economic candidates: **0**. All twelve ADR-038 cells remain **REGISTERED_NOT_EVALUATED**. Do not append placeholder trial metrics or count this guard rejection as a candidate evaluation.
- This is an **engineering/authorization STOP before research I/O**, not an economic NO-GO for the directional family. `GLOBAL_N = UNKNOWN` remains unchanged; there is no basis for a family-level economic verdict or a volatility pivot execution.
- Safety closeout in the same append operation: execution lock `NOT_AUTHORIZED`; `owner_authorization_commit_sha=null`; workflow authorization environment `false`; `EXECUTION_TRIGGER=PROBE`; `ACQUISITION_TRIGGER=PROBE`; `ALIX_DRY_RUN=true`; `ALIX_ALLOW_LIVE_TRADING=false`.
- V2, Spec v3, the public fetcher, immutable pre-hack data, and frozen main were not modified. No price/OHLCV values or holdout bytes were accessed or logged.

---

## [APPEND] Four-arm run #6 — runtime stop at Kronos checkout (2026-11-15)

- Fresh trigger nonce: `7255cfe75d5d3a463cd305dcf10ec075`; trigger commit: `c30d137b279c5201c488297b27dd67893cc1af1e`.
- Execution workflow run #6: Run ID `43693421298`; conclusion **failure**.
- Exact-SHA gates before execution: Standard CI #2423 / Run ID `43693421306` — **SUCCESS**; Seven-tool Preflight #27 / Run ID `43693421279`, artifact `14153039640` — **SUCCESS**. The workflow's one-time nonce guard and immutable lock-only validation both passed.
- Runtime sequence: pinned CPU inference dependencies installed successfully; Acquisition #1 artifact `four-arm-model-weights-43590408032` (run `43590408032`, artifact `14111134121`) downloaded successfully. GitHub Actions reported archive SHA-256 `752df2962cf2bb6aac1e10ebbb66652f722cf6176519fdc0ad631a592a8eaf06`.
- **Failing step 8:** `Check out the pinned Kronos inference source`. Exact error: `Repository path '/home/runner/work/_temp/kronos-source' is not under '/home/runner/work/AliX/AliX'`. The external checkout action rejected a destination outside the GitHub workspace. This is a runner/workspace-path runtime failure, not a model or economic result.
- Steps 9 `Dependency-import preflight for all pinned inference modules` and 10 `Run all twelve preregistered trials on the verified pre-hack prefix` were **SKIPPED**. The pre-hack CSV/manifest were not opened; no forecast cache, model inference, trade simulation, fold evaluation or economic metrics were produced. No post-hack holdout access occurred.
- Upload step found no JSON result files. No trial-manifest artifact was produced; no result artifact ID exists for run #6.
- Evaluated economic candidates: **0**. All 12 ADR-038 cells remain `REGISTERED_NOT_EVALUATED`. No placeholder records or metrics were created; `GLOBAL_N=UNKNOWN` remains unchanged. This is an **engineering runtime STOP**, not economic NO-GO for the directional family. No volatility pivot is executed.
- Stop rule: **STOP — no retry or second dispatch** under this directive. Any future attempt requires a new owner directive and a corrected workspace-contained Kronos checkout path, followed by fresh exact-SHA CI/Preflight and a fresh one-time nonce.
- Safety closeout: execution lock `NOT_AUTHORIZED`; `owner_authorization_commit_sha=null`; workflow env `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false`; `EXECUTION_TRIGGER=PROBE`; `ACQUISITION_TRIGGER=PROBE`; `ALIX_DRY_RUN=true`; `ALIX_ALLOW_LIVE_TRADING=false`.
- V2, Spec v3, frozen main, public fetcher, and pre-hack data files were not changed. No price/OHLCV values were printed or logged. The holdout remained sealed.


---

## [APPEND] Four-arm run #8 — runtime stop before trial evaluation (2026-11-15)

- Fresh one-time nonce: `2b7b1d3ede8204e2cf3d5f7b02c879c5`; arming commit: `130d32fecc37e9515eddb784527a02cd25cc1e5e`; execution workflow Run ID: `43696399884` (run #8), conclusion **failure**.
- Exact-SHA gates: Standard CI #2426 / Run ID `43696399859` — **SUCCESS**; Seven-tool Preflight #30 / Run ID `43696399832`, artifact `14154681106` — **SUCCESS**. The seven tools each exited 0. One-time nonce guard, complete immutable lock-only validation and bare-env stdlib lock job all passed.
- Runtime sequence: pinned CPU-only inference dependencies installed successfully; Acquisition #1 weights artifact `four-arm-model-weights-43590408032` (run `43590408032`, artifact `14111134121`) downloaded with SHA-256 `752df2962cf2bb6aac1e10ebbb66652f722cf6176519fdc0ad631a592a8eaf06`; pinned Kronos checkout succeeded inside the GitHub workspace; dependency-import preflight passed for 11 modules.
- **Failing step 10:** `Run all twelve preregistered trials on the verified pre-hack prefix`. Exact traceback terminus: `ModuleNotFoundError: No module named 'research'` at `scripts/run_four_arm_experiment.py:17`, exit code `1`, while importing `research.four_arm_core.lock_validator`.
- The failure occurred during module import before the runner entered trial execution or opened the pre-hack CSV/manifest. No forecast cache, inference, trade simulation, fold evaluation or economic metric was produced. The upload step found no `four-arm-results/*.json`; no manifest artifact was created.
- Evaluated economic candidates: **0**. All twelve ADR-038 cells remain **REGISTERED_NOT_EVALUATED**. No NetPF@50/90, trades, MaxDD, per-fold metrics or DSR result exists; no placeholder result records are fabricated. `GLOBAL_N=UNKNOWN` remains unchanged. This is an **engineering/runtime STOP**, not an economic NO-GO for the directional family. No family-level closeout or volatility pivot is justified by this run.
- Stop rule: **STOP — no retry or second execution dispatch** under this directive. Any future attempt requires a new owner directive and a separately verified launcher/import-path correction, followed by fresh exact-SHA CI/Preflight and a fresh one-time nonce.
- Safety closeout: lock `NOT_AUTHORIZED`; `owner_authorization_commit_sha=null`; workflow env `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false`; `EXECUTION_TRIGGER=PROBE`; `ACQUISITION_TRIGGER=PROBE`; `ALIX_DRY_RUN=true`; `ALIX_ALLOW_LIVE_TRADING=false`.
- V2, Spec v3, frozen main, public fetcher and `data/nobitex/prehack/*` were not modified. No post-hack holdout access, private API call or real-order attempt occurred. No price/OHLCV values were printed or logged.


---

## [APPEND] Four-arm run #10 — first economic evaluation and authorization closeout (2026-11-15)

### Execution evidence

- Branch at execution: `research/nobitex-hf-strategy`; exact execution HEAD: `71bf644dc0dd0437cd577f1f2df515e9db95c5e1`; frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- Experiment workflow Run #10: [43712086361](https://github.com/angry0tiger-sudo/AliX/actions/runs/43712086361); Job `132805651365`; conclusion **SUCCESS**.
- Step 12, `Run all twelve preregistered trials on the verified pre-hack prefix`: **SUCCESS**, 2026-11-15 12:15:20Z–13:23:27Z, elapsed **1:08:07**.
- Step 13, `Upload twelve trial manifests and forecast-cache evidence`: **SUCCESS**, 13:23:27Z–13:23:28Z.
- Results artifact: ID `14163527266`, `four-arm-real-prehack-43712086361`, 55,340 bytes, SHA-256 `b12c3a807316ab52829e6b0b1aa450057133d76afc0aeb54f0ac259f38300679`, expires 2027-02-13T12:05:39Z.
- Separate technical smoke artifact: ID `14160749121`, `four-arm-real-smoke-43712086361`, 989 bytes, expires 2026-12-15T12:15:19Z. **SMOKE is not a trial and is not included in the registry records below.**
- The results ZIP contains exactly the twelve expected `trial-*.json` files, `four-arm-run-report.json`, and `forecast-cache.json`. All twelve IDs are unique and match the complete ADR-038 matrix; no required trial or fold field is missing.

### Data, cache and execution integrity

- Source dataset rows: **31,801**; confirmatory rows used: **27,825**; excluded tail: **3,976**.
- Dataset SHA-256 in every manifest: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Forecast cache: **1,138** anchors/entries; cache SHA-256 `aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868`; identity SHA-256 `f879e5c179cf6451485ef1ce432c7d921f5f4344146d80f08026f7f8cfcf2427`. Re-serialization SHA verification passed.
- Inference calls: **1,138 per model**; **2,276 total** (Kronos + Chronos). The shared cache was built once. All twelve manifests state `forecast_cache_computed_for_this_trial=false`; the nine forecast-driven B/C/D cells reference the same cache. No per-trial inference was repeated.
- Arm A is one threshold-invariant Donchian control: `arm_a_control_is_threshold_invariant=true` for the 120/180/240 A cells; their trade series and all fold metrics are byte-for-byte identical and computed once, then referenced three times. The run report records 12 registered cells, 10 unique signal rules, and 3 repeated Arm-A cells.
- Every one of the twelve manifests and the run report record `posthack_holdout_access=false`; `private_api_calls=0`; `real_order_attempts=0`. No holdout was opened.

### Trial results

`NetPF` is net of the listed round-trip cost. MaxDD is the maximum of the 50/90 bps drawdowns. DSR is JSON null in all twelve manifests. Numeric sensitivity values for multipliers 1×/2×/5×/10× are **MISSING**, not zero: `numeric_dsr_computed=false`, reason `GLOBAL_N_UNKNOWN`. The audited historical lower bound is 7,912; the global trial count remains **UNKNOWN**.

| Trial ID | NetPF@50 | NetPF@90 | Trades | MaxDD | DSR sensitivity | Per-fold (trades / PF50 / PF90 / MaxDD) | Classification |
|---|---:|---:|---:|---:|---|---|---|
| NOBITEX-KC26-FD-120-A | 0.340813 | 0.144630 | 291 | 94.32% | MISSING (×1/×2/×5/×10) | F1: 82/0.368/0.126/51.29%; F2: 86/0.411/0.182/53.98%; F3: 123/0.286/0.132/75.11% | **FAIL** |
| NOBITEX-KC26-FD-120-B | 0.476530 | 0.301477 | 28 | 26.58% | MISSING (×1/×2/×5/×10) | F1: 0/MISSING/MISSING/0.00%; F2: 20/0.257/0.145/24.02%; F3: 8/1.162/0.808/8.84% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-120-C | 0.401930 | 0.239210 | 70 | 52.64% | MISSING (×1/×2/×5/×10) | F1: 14/0.297/0.172/13.84%; F2: 34/0.492/0.328/31.97%; F3: 22/0.235/0.081/19.53% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-120-D | 0.398077 | 0.269117 | 41 | 42.18% | MISSING (×1/×2/×5/×10) | F1: 5/0.987/0.598/3.08%; F2: 30/0.325/0.229/40.57%; F3: 6/0.879/0.437/4.37% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-180-A | 0.340813 | 0.144630 | 291 | 94.32% | MISSING (×1/×2/×5/×10) | F1: 82/0.368/0.126/51.29%; F2: 86/0.411/0.182/53.98%; F3: 123/0.286/0.132/75.11% | **FAIL** |
| NOBITEX-KC26-FD-180-B | 0.632403 | 0.393601 | 11 | 9.70% | MISSING (×1/×2/×5/×10) | F1: 0/MISSING/MISSING/0.00%; F2: 6/0.484/0.260/5.92%; F3: 5/0.770/0.527/6.32% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-180-C | 0.523971 | 0.350488 | 35 | 30.76% | MISSING (×1/×2/×5/×10) | F1: 5/0.987/0.598/3.08%; F2: 21/0.512/0.370/24.86%; F3: 9/0.304/0.123/7.76% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-180-D | 0.397253 | 0.319292 | 7 | 17.42% | MISSING (×1/×2/×5/×10) | F1: 0/MISSING/MISSING/0.00%; F2: 7/0.397/0.319/17.42%; F3: 0/MISSING/MISSING/0.00% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-240-A | 0.340813 | 0.144630 | 291 | 94.32% | MISSING (×1/×2/×5/×10) | F1: 82/0.368/0.126/51.29%; F2: 86/0.411/0.182/53.98%; F3: 123/0.286/0.132/75.11% | **FAIL** |
| NOBITEX-KC26-FD-240-B | MISSING | MISSING | 1 | 0.00% | MISSING (×1/×2/×5/×10) | F1: 0/MISSING/MISSING/0.00%; F2: 0/MISSING/MISSING/0.00%; F3: 1/MISSING/MISSING/0.00% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-240-C | 0.572389 | 0.393050 | 24 | 21.88% | MISSING (×1/×2/×5/×10) | F1: 5/0.987/0.598/3.08%; F2: 14/0.548/0.405/18.42%; F3: 5/0.284/0.113/4.37% | **INSUFFICIENT_SAMPLE** |
| NOBITEX-KC26-FD-240-D | MISSING | MISSING | 0 | 0.00% | MISSING (×1/×2/×5/×10) | F1: 0/MISSING/MISSING/0.00%; F2: 0/MISSING/MISSING/0.00%; F3: 0/MISSING/MISSING/0.00% | **INSUFFICIENT_SAMPLE** |

All twelve trials have **0/3 folds passing** the registered per-fold stability gate.

### Per-fold detail

| Trial ID | Fold | Trades | NetPF@50 | NetPF@90 | MaxDD | Stability |
|---|---:|---:|---:|---:|---:|---|
| NOBITEX-KC26-FD-120-A | F1 | 82 | 0.368 | 0.126 | 51.29% | FAIL |
| NOBITEX-KC26-FD-120-A | F2 | 86 | 0.411 | 0.182 | 53.98% | FAIL |
| NOBITEX-KC26-FD-120-A | F3 | 123 | 0.286 | 0.132 | 75.11% | FAIL |
| NOBITEX-KC26-FD-120-B | F1 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-120-B | F2 | 20 | 0.257 | 0.145 | 24.02% | FAIL |
| NOBITEX-KC26-FD-120-B | F3 | 8 | 1.162 | 0.808 | 8.84% | FAIL |
| NOBITEX-KC26-FD-120-C | F1 | 14 | 0.297 | 0.172 | 13.84% | FAIL |
| NOBITEX-KC26-FD-120-C | F2 | 34 | 0.492 | 0.328 | 31.97% | FAIL |
| NOBITEX-KC26-FD-120-C | F3 | 22 | 0.235 | 0.081 | 19.53% | FAIL |
| NOBITEX-KC26-FD-120-D | F1 | 5 | 0.987 | 0.598 | 3.08% | FAIL |
| NOBITEX-KC26-FD-120-D | F2 | 30 | 0.325 | 0.229 | 40.57% | FAIL |
| NOBITEX-KC26-FD-120-D | F3 | 6 | 0.879 | 0.437 | 4.37% | FAIL |
| NOBITEX-KC26-FD-180-A | F1 | 82 | 0.368 | 0.126 | 51.29% | FAIL |
| NOBITEX-KC26-FD-180-A | F2 | 86 | 0.411 | 0.182 | 53.98% | FAIL |
| NOBITEX-KC26-FD-180-A | F3 | 123 | 0.286 | 0.132 | 75.11% | FAIL |
| NOBITEX-KC26-FD-180-B | F1 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-180-B | F2 | 6 | 0.484 | 0.260 | 5.92% | FAIL |
| NOBITEX-KC26-FD-180-B | F3 | 5 | 0.770 | 0.527 | 6.32% | FAIL |
| NOBITEX-KC26-FD-180-C | F1 | 5 | 0.987 | 0.598 | 3.08% | FAIL |
| NOBITEX-KC26-FD-180-C | F2 | 21 | 0.512 | 0.370 | 24.86% | FAIL |
| NOBITEX-KC26-FD-180-C | F3 | 9 | 0.304 | 0.123 | 7.76% | FAIL |
| NOBITEX-KC26-FD-180-D | F1 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-180-D | F2 | 7 | 0.397 | 0.319 | 17.42% | FAIL |
| NOBITEX-KC26-FD-180-D | F3 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-A | F1 | 82 | 0.368 | 0.126 | 51.29% | FAIL |
| NOBITEX-KC26-FD-240-A | F2 | 86 | 0.411 | 0.182 | 53.98% | FAIL |
| NOBITEX-KC26-FD-240-A | F3 | 123 | 0.286 | 0.132 | 75.11% | FAIL |
| NOBITEX-KC26-FD-240-B | F1 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-B | F2 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-B | F3 | 1 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-C | F1 | 5 | 0.987 | 0.598 | 3.08% | FAIL |
| NOBITEX-KC26-FD-240-C | F2 | 14 | 0.548 | 0.405 | 18.42% | FAIL |
| NOBITEX-KC26-FD-240-C | F3 | 5 | 0.284 | 0.113 | 4.37% | FAIL |
| NOBITEX-KC26-FD-240-D | F1 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-D | F2 | 0 | MISSING | MISSING | 0.00% | FAIL |
| NOBITEX-KC26-FD-240-D | F3 | 0 | MISSING | MISSING | 0.00% | FAIL |

### Adjudication and cohort verdict

- **3 FAIL:** the three registered Arm-A cells are the same Donchian control repeated across thresholds. Aggregate trades are 291, but NetPF is 0.340813/0.144630, MaxDD 94.32%, and fold stability 0/3. The failure is economic and repeated-control identity must not be mistaken for three independent strategies.
- **9 INSUFFICIENT_SAMPLE:** all forecast-driven B/C/D cells have fewer than 100 aggregate trades. Their numerical DSR values are also MISSING.
- **0 PASS.**
- Cohort is **MIXED: 3 FAIL + 9 INSUFFICIENT_SAMPLE**. It is not accurate to claim either “all twelve FAIL” or “all twelve INSUFFICIENT_SAMPLE”. The first cohort-level verdict is **INSUFFICIENT_SAMPLE / MIXED_COHORT — no candidate qualifies for progression**. The nine forecast-driven cells did not reach the minimum sample; therefore this run does **not** establish a family-level NO-GO for all directional methods. The conditional ADR-038 rule for an all-FAIL directional-family closure is not triggered.
- `GLOBAL_N=UNKNOWN` remains unchanged. No estimated N or numeric DSR was manufactured. TTM-R2 volatility pivot is **not initiated or recommended as an automatic next phase** under this mixed result; any separate pivot still requires the relevant owner directive and preregistration.

### Append-only status update

The twelve ADR-038 IDs are now recorded as evaluated by this appended event; the original `REGISTERED_NOT_EVALUATED` preregistration lines and all prior registry history remain intact. No record was deleted. Arm-A duplication is disclosed above and remains counted as three registered cells in the DSR accounting. The smoke artifact is technical evidence only and is not a thirteenth trial.

### Safety closeout

- Execution lock: `NOT_AUTHORIZED`; `owner_authorization_commit_sha=null`.
- `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false`; `ALIX_DRY_RUN=true`; `ALIX_ALLOW_LIVE_TRADING=false`.
- `EXECUTION_TRIGGER=PROBE`; `ACQUISITION_TRIGGER=PROBE`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`; `private_api_calls=0`; `real_order_attempts=0`.
- V2, Spec v3, public fetcher, `data/nobitex/prehack/*`, and frozen main remain unchanged. No holdout access occurred.


## [APPEND] Regime R HGBQ-CQR G3 — Preregistration Lock (2026-11-17)

- Preregistration: `config/regime_r/hgbq_cqr_preregistration.json`.
- SHA-256 commitment: `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`.
- Status of all 30 logical slots: **PREREGISTERED_LOCKED**; evaluated candidates at registration: **0**.
- The identifiers retain the legacy `NOBITEX-R1-NGBOOST` prefix because they are stable logical-slot identifiers established by the Regime R contract. The frozen implementation revision for these slots is `HGBQ-CQR-G3`; this does not create 30 additional slots.
- This append registers the 30 pre-existing logical slots as row-level entries. All historical registry sections above remain unchanged and append-only.
- No real-data fitting, Holdout access, economic backtest, network access, result generation, or candidate evaluation is authorized by this registration.

| Trial ID | Horizon | Slot | Implementation revision | Status | Evaluated |
|---|---:|---:|---|---|---|
| `NOBITEX-R1-NGBOOST-1h-01` | 1h | 1 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-02` | 1h | 2 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-03` | 1h | 3 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-04` | 1h | 4 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-05` | 1h | 5 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-06` | 1h | 6 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-07` | 1h | 7 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-08` | 1h | 8 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-09` | 1h | 9 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-1h-10` | 1h | 10 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-01` | 4h | 1 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-02` | 4h | 2 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-03` | 4h | 3 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-04` | 4h | 4 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-05` | 4h | 5 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-06` | 4h | 6 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-07` | 4h | 7 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-08` | 4h | 8 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-09` | 4h | 9 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-4h-10` | 4h | 10 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-01` | 24h | 1 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-02` | 24h | 2 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-03` | 24h | 3 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-04` | 24h | 4 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-05` | 24h | 5 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-06` | 24h | 6 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-07` | 24h | 7 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-08` | 24h | 8 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-09` | 24h | 9 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-R1-NGBOOST-24h-10` | 24h | 10 | `HGBQ-CQR-G3` | `PREREGISTERED_LOCKED` | No |



---

## [APPEND] Regime M3 OFI G1 — Preregistration Lock (2026-11-26)

- Preregistration: `config/regime_m3/ofi_preregistration.json`.
- Configuration SHA-256: `c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4`.
- Revision: `M3-OFI-G1`; status of both logical trials: **PREREGISTERED_LOCKED**.
- Registered horizons: **1h and 4h only**. The 24h horizon is explicitly excluded.
- Estimator: one fixed `StandardScaler + LogisticRegression` baseline; no model has been trained or evaluated.
- Registered feature set: best-level OFI, MLOFI levels 2–5, microprice deviation, signed API-type trade-volume ratio, and top-five order-book imbalance (eight numeric columns).
- Four chronological segment-based walk-forward folds are frozen; at least three of the four must satisfy the registered fold-stability rule.
- Both horizon trials independently require pooled NetPF >= 1.10, at least 100 completed OOS trades, absolute MaxDD <= 15%, and stability in at least three folds.
- The two trial IDs are new logical M3 cells; fold fits do not create additional trial variants:
  - `NOBITEX-M3-OFI-1h-LOGISTIC-BASELINE-01`
  - `NOBITEX-M3-OFI-4h-LOGISTIC-BASELINE-01`
- `GLOBAL_N=UNKNOWN` remains unchanged. No previous registry counts are rewritten or reinterpreted.
- **Evaluated candidates at registration: 0.** Both trials are `PREREGISTERED_LOCKED`; `Evaluated=No`.
- This append is registry-only evidence. It does not authorize feature/model implementation, model fitting, backtesting, Holdout/Post-Hack access, external data collection, paper trading, or order execution. A separate Owner directive is required before M3.5.

| Trial ID | Horizon | Slot | Implementation revision | Estimator | Status | Evaluated |
|---|---:|---:|---|---|---|---|
| `NOBITEX-M3-OFI-1h-LOGISTIC-BASELINE-01` | 1h | 1 | `M3-OFI-G1` | `LogisticRegression` | `PREREGISTERED_LOCKED` | No |
| `NOBITEX-M3-OFI-4h-LOGISTIC-BASELINE-01` | 4h | 1 | `M3-OFI-G1` | `LogisticRegression` | `PREREGISTERED_LOCKED` | No |



---

## [APPEND] Regime M3 OFI G1 — M3.5 Walk-Forward Evaluation (2026-11-27)

- Evaluation commit: `10c01b7988a87ca3e9ac0510a4f03e0d1260ee6b`.
- GitHub Actions evaluation: [Run #2586 / 45898505105](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105), `workflow_dispatch`, conclusion **success**.
- Standard CI on the evaluated commit: [Run #2585 / 45897855203](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855203), **success**; coverage 90.36%, 1,099 passed, 5 skipped.
- Independent quality preflight: [Run #125 / 45897855197](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855197), **success**.
- Result artifact: `regime-m3-ofi-evaluation-45898505105`, Artifact ID `15155002532`, ZIP SHA-256 `78c53dbb6f060b4d22117fb726b49750ace3b035d6bc5079357517a37e63edf7`.
- Preregistration ID: `ALIX-M3-OFI-G1`; exact locked JSON SHA-256: `c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4`. The evaluator verified the frozen config and all five M1.6 artifact IDs/digests before model fitting.
- Exactly **two logical M3 trials** were evaluated. The four walk-forward fold fits are evidence within each registered logical trial, not four extra candidates. `GLOBAL_N=UNKNOWN` remains unchanged.

### Final status of the two registered M3 logical trials

| Trial ID | Horizon | Status | Completed OOS cohorts | NetPF | MaxDD | Net PnL / initial equity | Stable folds |
|---|---:|---|---:|---:|---:|---:|---:|
| `NOBITEX-M3-OFI-1h-LOGISTIC-BASELINE-01` | 1h | **EVALUATED — NO-GO** | 665 | 0.0734699215 | 8.2759% | -7.9225% | 0/4 |
| `NOBITEX-M3-OFI-4h-LOGISTIC-BASELINE-01` | 4h | **EVALUATED — INSUFFICIENT_SAMPLE / NO-GO** | 82 | 0.2326940033 | 0.7494% | -0.1214% | 0/4 |

The locked acceptance gates applied independently to each horizon: NetPF >= 1.10, completed OOS cohorts >= 100, MaxDD <= 15%, and at least 3/4 stable folds. Neither trial passed.

### Fold accounting

| Trial | Fold | Training rows | Class 0 / Class 1 | Test rows | Completed cohorts | Fold NetPF | Fold Net PnL / initial equity | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1h | WF-01 | 220 | 131 / 89 | 285 | 90 | 0.6418592726 | -0.2143% | Unstable |
| 1h | WF-02 | 505 | 208 / 297 | 226 | 171 | 0.0144889973 | -1.8913% | Unstable |
| 1h | WF-03 | 731 | 277 / 454 | 263 | 209 | 0.0024871425 | -3.5877% | Unstable |
| 1h | WF-04 | 994 | 447 / 547 | 285 | 195 | 0.0850952660 | -2.2292% | Unstable |
| 4h | WF-01 | 0 | 0 / 0 | 105 | 0 | — | — | Insufficient; no fit |
| 4h | WF-02 | 105 | 0 / 105 | 46 | 0 | — | — | Insufficient; single training class |
| 4h | WF-03 | 151 | 0 / 151 | 83 | 0 | — | — | Insufficient; single training class |
| 4h | WF-04 | 234 | 83 / 151 | 105 | 82 | 0.2326940033 | -0.1214% | Fitted; unstable |

### Signal-count clarification

The evaluation artifact does **not** report 17 completed trades for either registered M3 trial. The registered completed-cohort count is 665 at 1h and 82 at 4h. With the fixed `P(up) >= 0.5` threshold, the 1h fold confusion matrices imply 665 positive predictions across 1,059 test rows; the only fitted 4h fold implies 82 positive predictions across 105 test rows. These are overlapping one-minute horizon cohorts and must not be interpreted as independent observations. The threshold was not raised or tuned after observing outcomes.

### Registry and safety closeout

- The earlier rows above remain an immutable record of the preregistration-time status `Evaluated=No`; this appendix records their later evaluation outcome rather than rewriting history.
- No additional model candidate or hyperparameter variant was created. No DeepLOB/Transformer/alternate estimator was evaluated.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.
- `private_api_calls=0`; `real_order_attempts=0`; `paper_orders=0`.
- `holdout_bytes_read=false`; `posthack_data_read=false`; `external_data_requests=0`; `local_execution=false`.
- The five accepted M1.6 public artifacts only were used; the 90 bps registered round-trip friction assumption was preserved.
- Final economic cohort decision: **INSUFFICIENT_SAMPLE / NO-GO**. Holdout remains sealed. Any later M4 proposal is a new owner-gated decision and is not authorized by this append.


---

## [APPEND] Regime M3 OFI G1 — Owner-approved lifecycle closure (2026-11-27)

- **Owner decision:** Option A approved; the M3 OFI baseline family is formally closed.
- **Current lifecycle status for both registered M3 logical trials: `CLOSED_NO_GO`.**
- This closure supersedes the prior M3.5 evaluation labels for current lifecycle status. The preregistration rows above remain historical snapshots of the registration-time state; no earlier trial accounting or preregistration history is deleted.
- **Reason:** no economic edge in the 1h registered baseline, plus inadequate fit/training sample and no economic pass at 4h. The two horizons failed their independent acceptance gates.

| Trial ID | Horizon | Current status | Evaluation evidence | Closure basis |
|---|---:|---|---|---|
| `NOBITEX-M3-OFI-1h-LOGISTIC-BASELINE-01` | 1h | **CLOSED_NO_GO** | 665 cohorts; NetPF 0.0734699215; MaxDD 8.2759%; net PnL -7.9225%; stable folds 0/4 | Adequate cohort count, but no economic edge and no stable fold |
| `NOBITEX-M3-OFI-4h-LOGISTIC-BASELINE-01` | 4h | **CLOSED_NO_GO** | 82 cohorts; NetPF 0.2326940033; MaxDD 0.7494%; net PnL -0.1214%; stable folds 0/4 | Insufficient training/evaluation sample under locked folds; only one fold fitted |

### Scope and safety

- M3 OFI is **CLOSED** on the existing five-segment M1.6 capture (29h 10m 20.393s). No DeepLOB, Transformer, higher-capacity model, alternate estimator, feature expansion, threshold optimization, or repeat attempt is permitted on this dataset.
- No additional data collection is authorized by this closure. Any materially new research family or data-acquisition phase requires a separate prospective Owner decision and preregistration.
- Holdout and Post-Hack remain sealed and unread. No access is granted.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`; `private_api_calls=0`; `real_order_attempts=0`; `paper_orders=0`; `local_execution=false`.
- Standard CI [Run #2591 / 45942283970](https://github.com/angry0tiger-sudo/AliX/actions/runs/45942283970) completed **SUCCESS** on branch head `a2eb76bbc0e083994df5d185eea348cb5eeb0231`; all four jobs and all recorded steps passed.
- Frozen `main` remains commit `cd9b13d42ac0cf0edd2df066ff81580e028146d6`, Tree SHA `5761c6c34143f8e6924f53a4e745c48932a5f791`.
- This is a documentation-only lifecycle update. `GLOBAL_N=UNKNOWN` remains unchanged.

