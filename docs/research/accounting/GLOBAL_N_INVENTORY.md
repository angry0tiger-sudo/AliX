# ممیزی موجودی GLOBAL_N — گردش‌کارها، واحد trial و قابلیت محاسبه DSR

- **تاریخ ممیزی:** 2026-11-15 UTC
- **مخزن/شاخه:** angry0tiger-sudo/AliX — research/nobitex-hf-strategy
- **HEAD مبنا:** 104bf2bb9bd54bdfe06c5f5895a846f28e29f652
- **Tree مبنا:** 07aa1499bb8c2019011f892e08c77852d100f57b
- **والد HEAD مبنا:** 7bf32a174e42ea8dee66726593229ed8e79af654
- **main:** 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb — FROZEN
- **آخرین ADR در مبنا:** ADR-044؛ ADR-045 شماره آزاد بعدی است.
- **وضعیت حسابداری:** **GLOBAL_N=UNKNOWN** — این سند عدد جایگزین تولید نمی‌کند.

## 1. دامنه و روش

در tree شاخه ۵۳ فایل YAML در .github/workflows وجود دارد. بررسی مستقل تأیید کرد:

- ۲۷ workflow با پیشوند ml-core-v4؛
- ۱۱ workflow پژوهشی/تشخیصی دیگر که در فهرست اولیه نام برده شده بودند؛
- یک workflow پژوهش اقتصادی واقعی چهار-بازویی: four-arm-real-experiment.yml.

بنابراین **۳۹ workflow پژوهشی یا پژوهش‌محور** در این inventory بررسی شده‌اند. ۱۴ فایل باقی‌مانده عمدتاً CI، preflight، probe، acquisition، persistence یا mock/test support هستند و در جدول candidateهای پژوهشی مستقل شمرده نشده‌اند.

این شمارش تعداد فایل‌های workflow است، نه تعداد استراتژی‌ها و نه GLOBAL_N. بعضی workflowها فقط آزمون/ممیزی اجرا می‌کنند؛ بعضی دیگر candidateهای workflow قبلی را دوباره replay می‌کنند؛ برخی grid و nested search دارند؛ و چند workflow از script/candidate-ID مشترک استفاده می‌کنند. بنابراین جمع سادهٔ سطرهای جدول، trial count معتبر ایجاد نمی‌کند.

منابع بررسی شامل YAMLها و scriptهای مرجع در همین HEAD، به‌علاوهٔ docs/research/TRIAL_REGISTRY.md و artifact/run references موجود در آن بوده‌اند. هرجا script/artifact تاریخی در HEAD قابل بازیابی نبود، یا تعداد candidateها از فایل صریحاً درنمی‌آمد، وضعیت «شاهد کافی نیست» حفظ شده است.

## 2. موجودی per-workflow

تمام cardinalityهای این جدول، ظرفیت یک اجرای کامل یا تعداد candidate rowهای گزارش‌شده‌اند؛ لزوماً اجرای تاریخی مستقل یا trial جدید نیستند.

| Workflow | ابعاد صریح در YAML/script مرجع | تعداد قابل‌اثبات در یک اجرای کامل | واحد شمارش و وضعیت DSR |
|---|---|---:|---|
| ai-artifact-training.yml | یک invocation روی BTCUSDT 1h و پنجره ۳۶۵ روز؛ دو artifact نام‌گذاری‌شده HMM و Meta Label. script مرجع scripts/train_ai_artifacts.py در HEAD مبنا موجود نیست. | ۱ execution؛ تعداد config/trial: **شاهد کافی نیست** | execution و artifact ≠ trial. پارامترهای داخلی از script غایب قابل اثبات نیست. |
| ai-artifact-shadow-validation.yml | یک shadow-validation روی training bundle قبلی و داده inference ۳۰روزه؛ script scripts/run_ai_shadow_validation.py در HEAD مبنا موجود نیست. | ۱ validation execution؛ config مستقل: **شاهد کافی نیست** | validation اجراست، نه candidate آموزش‌دیده جدید. |
| ai-ensemble-research.yml | compileall، Ruff و unit tests؛ مرحله train/evaluate در workflow وجود ندارد. | ۰ trial از workflow | engineering validation؛ executionهای CI را به N اضافه نکنید. |
| ai6-rl-risk-research.yml | PPO risk allocator؛ اجرای نهایی حداقل ۲۰۰٬۰۰۰ timestep. سه workflow execution (#20–#22)، دو failure و یک success؛ یک predecessor experiment جداگانه در پروتکل ذکر شده است. | حداقل ۲ policy experiment مستند؛ کل configها: **UNKNOWN** | واحد: policy/config؛ timestepها trial نیستند. پیشرفت دو run ناموفق کامل بازسازی نشده است. |
| deap-genetic-programming.yml | جمعیت ۵۰۰ و ۳۰ نسل در run نهایی؛ artifact مقدار واقعی unique_genomes_evaluated=7,842 را گزارش می‌کند. هفت execution، شش run شکست‌خورده پیش از run نهایی. | ۷٬۸۴۲ genome یکتای ارزیابی‌شده در run موفق؛ +۸۲۸ genome owner-reported با overlap نامعلوم | واحد: genome ارزیابی‌شده، نه population×generation و نه execution. genome تکراری یک بار شمرده می‌شود. partial work شش run شکست‌خورده بازسازی نشده است. |
| ttm-r2-risk-research.yml | یک مدل zero-shot ثابت TTM-R2 و پروتکل forecast؛ ۱۹۸ forecast در holdout، نه ۱۹۸ trial. grid صریح وجود ندارد. | ۱ مدل/فرضیه اجراشده | واحد: config مدل/فرضیه. forecast origin و trade شمار trial مستقل نیستند. |
| nobitex-hf-strategy.yml | داده ۱۵m و ۳۰m در دو step و سپس HF backtest؛ ۴۱ run number و ۴۲ attempt در ledger تاریخی. | ۴۱ execution record / ۴۲ attempt؛ تعداد configهای اقتصادی یکتا: **UNKNOWN** | execution/attempt قابل مشاهده‌اند؛ مدرکی نیست که هر run یک candidate اقتصادی مستقل و کامل باشد. |
| ml-core-real-data-baseline.yml | یک invocation از baseline classification pipeline؛ workflow grid مستقلی تعریف نمی‌کند. runner عمومی چند سناریو، profile و weighting را پشتیبانی می‌کند، اما ماتریس دقیق اجرای تاریخی از args/artifact همان run لازم است. | ۱ execution؛ تعداد configهای مدل: **شاهد کافی نیست** | classification diagnostic؛ تبدیل آن به trial اقتصادی/Sharpe بدون candidate-level artifact مجاز نیست. |
| ml-core-rebuild.yml | workflow اجرای test/validation و بازسازی quality؛ train یا economic candidate run ندارد. | ۰ trial | engineering workflow. |
| phase10-public-market-research.yml | یک invocation از scripts/run_freqai_baseline_walkforward.py؛ baseline walk-forward بدون grid صریح در workflow. | ۱ execution/config ثابت؛ folds، تکرار اعتبارسنجی‌اند نه trial مستقل | تعداد مدل/پارامترهای داخل artifact باید از report استخراج شود. |
| v4ai-empirical-gap-discovery.yml | یک gap-discovery diagnostic؛ sweep اقتصادی ندارد. | ۱ diagnostic execution؛ ۰ trial اقتصادی اثبات‌شده | واحد: diagnostic execution. |
| four-arm-real-experiment.yml | ۳ threshold ×۴ arm =۱۲ سلول ADR-038؛ Arm A در thresholdهای ۱۲۰/۱۸۰/۲۴۰ یک Donchian control واحد و با series یکسان است. | ۱۲ trial ID ثبت‌شده؛ ۱۰ signal rule مؤثر | برای provenance هر ۱۲ رکورد حفظ می‌شوند؛ سه ارجاع A یک مشاهده مستقل نیستند. حکم ADR-042 و ۱۲ رکورد Run #10 دست‌نخورده‌اند. |
| ml-core-v4ab-cost-frequency.yml | ۲ symbol ×۳ horizon (۲۴/۷۲/۱۲۰ bar) ×۳ exit policy | ۱۸ سلول | واحد: config سیگنال/افق/خروج. |
| ml-core-v4ac-cost-optimization.yml | ۲ symbol ×۴ timeframe (۱۵m/۳۰m/۱h/۴h) ×۲ exit policy ×۲ mode (PRIMARY_ONLY/META_FILTERED) | ۳۲ | واحد: candidate config؛ mode متفاوت، decision rule متفاوت است. |
| ml-core-v4ad-threshold-optimization.yml | ۲ symbol × policies (۳ policy برای ۱h و۲ برای ۴h) ×۴ mode (Primary + Meta Top 30/50/70) | ۴۰ | واحد: candidate/config. مدل متا بین thresholdها reuse می‌شود، threshold decision ruleها متفاوت‌اند. |
| ml-core-v4ae-trailing-optimization.yml | ۲ symbol ×۳ initial-stop ×۳ trailing-activation ×۳ trailing-step | ۵۴ (۲۷ برای هر symbol) | واحد: trailing-policy config؛ grid سه‌بعدی صریح. |
| ml-core-v4af-meta-final.yml | یک ETHUSDT/1h؛ سه target acceptance fraction (۳۰٪/۵۰٪/۷۰٪) به‌اضافه unfiltered control | ۴ | واحد: meta/threshold policy candidate. |
| ml-core-v4ah-advanced-ai.yml | ۴ policy candidate: unfiltered، fixed meta net expectancy، adaptive trail، adaptive trail + Kelly. افزون بر آن ۳ exit policy ×۲ cost scenario =۶ forward-validation diagnostic cell | ۴ candidate؛ ۶ diagnostic comparisons | شش سلول diagnostic-only و بدون meta model هستند؛ train مستقل نیستند. اگر برای انتخاب geometry استفاده شوند باید در selection log ثبت شوند. |
| ml-core-v4ai-maker-strategy.yml | script scripts/run_ml_core_v4ai_maker_strategy.py دارای ۳ candidate name: base control، ADX ablation، Meta Top 70 | ۳ candidate در هر run | Sharpe per-candidate در script محاسبه می‌شود؛ DSR فقط cohort همان run است. |
| ml-core-v4ao-donchian.yml | همان script مرجع V4-AI و همان ۳ candidate name؛ workflow title/data context متفاوت است. | ۳ candidate در هر run؛ overlap با V4-AI نیازمند تطبیق artifact/data identity | این دو workflow را ۶ trial مستقل فرض نکنید. script و candidate names مشترک‌اند؛ هویت دقیق data scope باید از artifactها تطبیق داده شود. |
| ml-core-v4ap-diagnostic-sweep.yml | ۲ asset ×۳ timeframe ×۲ strategy ×۳ exit geometry | ۳۶ سلول؛ ۳۶ Sharpe estimate در artifact/registry | واحد: candidate cell. گزارش استقلال را NOT_VERIFIED ثبت می‌کند. |
| ml-core-v4f-feature-ablation.yml | گزارش workflow صریحاً ۴۰ experiment و ۵ feature profile را تأیید می‌کند؛ دو asset و یک scenario ثابت؛ classification-only | ۴۰ experiment row | واحد: classification experiment/profile/split؛ Sharpe اقتصادی تولید نمی‌شود. |
| ml-core-v4g-trend-calibration.yml | گزارش workflow صریحاً ۸۰ experiment row، دو profile، دو asset و پنج walk-forward fold را تأیید می‌کند. | ۸۰ experiment row؛ ۵ fold زمانی | واحد: classification experiment row؛ foldها trial مستقل نیستند. |
| ml-core-v4h-tuning-geometry.yml | ۳ barrier scenario ×۲ feature profile ×۲ asset ×۴ timeframe =۴۸ experiment؛ هر experiment دو weighting variant؛ هر مدل ۹ inner-search candidate روی ۳ purged fold | ۴۸ experiment، ۹۶ model candidate، ۸۶۴ nested parameter candidate، ۲٬۵۹۲ fold fit | واحد اصلی: nested candidate config؛ fitهای foldی trial مستقل نیستند. ۴۸/۹۶/۸۶۴ را با هم جمع نکنید. |
| ml-core-v4i-economic-evaluation.yml | replay اقتصادی ۱۳ candidate پایدارشده از V4-H | ۱۳ replay | candidateهای inherited؛ خود workflow مدل/پارامتر جدید نمی‌آموزد. |
| ml-core-v4j-cost-isolation.yml | همان ۱۳ candidate ×۳ cost profile | ۳۹ candidate-cost evaluation pair | هزینه‌ها stress scenario هستند، نه ۳۹ استراتژی مستقل؛ فقط اگر cost profile برای انتخاب استفاده شود، multiplicity جدا ثبت می‌شود. |
| ml-core-v4k-trade-management.yml | replay ۱۳ candidate با confidence threshold ثابت ۰٫۵۵ و trade-management policy ثابت | ۱۳ candidate-policy replay | candidateها از V4-H به ارث می‌رسند؛ parent identity حفظ شود. |
| ml-core-v4l-confidence-sweep.yml | ۱۳ candidate ×۵ confidence threshold (۰٫۴۰/۰٫۴۵/۰٫۵۰/۰٫۵۵/۰٫۶۰) | ۶۵ candidate-threshold pair | واحد: candidate + threshold decision rule؛ انتخاب thresholdها multiplicity واقعی دارد. |
| ml-core-v4m-tp-sl-sweep.yml | ۱۳ candidate ×۴ TP/SL geometry | ۵۲ candidate-geometry pair | واحد: candidate + exit geometry؛ replay از سیگنال‌های V4-H، بدون train جدید. |
| ml-core-v4n-trailing-stop.yml | ۱۳ candidate با یک trailing configuration ثابت (SL=2 ATR، TP=3 ATR، threshold=0.60) | ۱۳ candidate-policy replay | یک policy ثابت روی ۱۳ candidate؛ parameter sweep ندارد. |
| ml-core-v4o-time-exit.yml | ۱۳ candidate ×۴ time-exit setting (None/3/6/9 bar) | ۵۲ candidate-time-exit pair | واحد: candidate + exit-time policy؛ ۴ variant ×۱۳ candidate، بدون ۵۲ model training. |
| ml-core-v4p-economic.yml | ۱۶ experiment، ۳۲ model، دو feature profile و ۹ random-search candidate برای هر model | ۳۲ top-level candidate؛ ۲۸۸ nested hyperparameter config | top-level و nested جدا هستند؛ fold fitها به‌تنهایی trial مستقل نیستند. |
| ml-core-v4q-economic.yml | ۱۶ experiment/۳۲ model و ۹ random-search candidate برای هر model؛ objective برابر BUY_F1 | ۳۲ top-level candidate؛ ۲۸۸ nested hyperparameter config | overlap با V4-P باید با nested config ID تطبیق داده شود. |
| ml-core-v4r-economic.yml | ۴ context ×۴ arm =۱۶ experiment؛ دو weighting policy =>۳۲ experiment/model pair و۶۴ candidate؛ random-search iterations=9 | ۶۴ top-level candidate؛ nested count مستقل: **شاهد کافی نیست** | واحد: candidate/model config. iteration به‌تنهایی تعداد unique hyperparameter combination را ثابت نمی‌کند. |
| ml-core-v4s-threshold-calibration.yml | audit از threshold calibration artifact و testها؛ model training/economic sweep ندارد. | ۰ trial جدید | audit-only؛ به N اضافه نشود. |
| ml-core-v4t-economic-score.yml | ۳۲ candidate ID؛ ۱۶ experiment؛ ۴ trained و۲۸ sample-gate skipped؛ random search ۹ iteration؛ registry نیز ۳۶ nested config و۱۰۸ fold fit ثبت کرده است. | ۳۲ top-level candidate؛ ۳۶ nested config؛ ۱۰۸ fold fit | overlap بین ۳۶ nested config و۳۲ ID حل نشده؛ آن‌ها را جمع نکنید. ۱۹ execution پیشین ممکن است partial work داشته باشند. |
| ml-core-v4v-binary-paradigm.yml | ۲ symbol ×۲ timeframe ×۲ arm ×۱ feature profile =۸ experiment؛ دو weighting =>۱۶ candidate؛ random search ۹ candidate/model | ۱۶ top-level candidate؛ ۱۴۴ nested config | fold fitها trial جدا نیستند. |
| ml-core-v4x-meta-labeling.yml | ۲ symbol ×۲ timeframe | ۴ candidate | واحد: symbol/timeframe candidate با Primary/Meta mapping ثابت. |
| ml-core-v4y-primary-signal.yml | ۲ symbol ×۳ timeframe (۱۵m/۳۰m/۱h) | ۶ candidate | واحد: symbol/timeframe primary signal candidate. |

### 2.1 Workflowهای support که trial محسوب نمی‌شوند

۱۴ فایل دیگر tree شامل ci.yml، quality-preflight.yml، format-assist.yml، audit-label-transitions.yml، bybit-spot-kline-probe.yml، four-arm-contract-fingerprints.yml، four-arm-experiment.yml (mock-only)، four-arm-model-acquisition.yml، four-arm-model-pin-capture.yml، nobitex-mvp-paper-24h.yml (شبیه‌سازی synthetic با seed ثابت و بدون شبکه)، nobitex-posthack.yml، persist-prehack-dataset.yml، phase10-public-price-observation.yml و probe-data-access.yml هستند. این‌ها به‌خودی‌خود candidate اقتصادی جدید ایجاد نمی‌کنند.

## 3. موجودی تاریخی و کران‌ها

### 3.1 کران پایین مستند قبلی

TRIAL_REGISTRY.md در snapshot مبنا این وضعیت را ثبت می‌کند:

| خانواده | شمارش مستند | نوع واحد | کمبود اصلی |
|---|---:|---|---|
| V4-T | ۳۲ top-level candidate؛ ۳۶ nested config جداگانه گزارش شده | candidate/config | overlap nested با۳۲ ID و۱۹ execution ناموفق |
| V4-AP | ۳۶ cell و۳۶ Sharpe estimate | candidate cell | ۲۵ اجرای قبلی ناموفق/cancelled ممکن است partial work داشته باشند |
| AI-6 | حداقل۲ policy experiment ارزیابی‌شده؛۳ execution | policy/config و execution | configهای دقیق قدیمی و پیشرفت دو run ناموفق |
| DEAP | ۷٬۸۴۲ genome verified +۸۲۸ owner-reported | genome | overlap ۸۲۸ با۷٬۸۴۲ و partial genomeهای شش run ناموفق |
| HF | ۴۱ run record و۴۲ attempt | execution/attempt | تعداد configهای اقتصادی یکتا نامعلوم |

Subtotal تاریخی غیر-HF در registry برابر **N ≥ 7,912** ثبت شده است. این عدد subtotal حسابداری تاریخی است، نه GLOBAL_N معتبر: واحدها ناهمگون‌اند، V4-T nested configها unresolved هستند، DEAP 828 overlap نامعلوم دارد، و HF هنوز به candidate/config identity تبدیل نشده است. این ممیزی subtotal ۷٬۹۱۲ را جایگزین یا اصلاح نمی‌کند.

Run #10 نیز ۱۲ trial ID ثبت‌شده دارد، اما سه شناسهٔ Arm A به یک Donchian signal rule و series واحد اشاره می‌کنند. بنابراین ۱۲ رکورد provenance برابر با ۱۲ مشاهده مستقل نیست؛ ۱۰ signal rule مؤثر در همان ماتریس گزارش شده است. هیچ‌یک از ۱۲ رکورد Run #10 در این ممیزی تغییر نکرده است.

### 3.2 کران بالای محافظه‌کارانه

**کران بالای عددی قابل‌دفاع برای GLOBAL_N در شواهد فعلی وجود ندارد.** علت:

1. تمام workflow executionهای تاریخی برای همه شاخه‌ها، همراه با attempt number و artifact ID، به یک ledger واحد نگاشت نشده‌اند.
2. runهای failure/cancelled در V4-T، V4-AP، DEAP، AI-6 و HF ممکن است پیش از شکست candidate را ارزیابی کرده باشند؛ progress دقیق همیشه حفظ نشده است.
3. ۳۶ nested config در V4-T در برابر۳۲ top-level candidate identity هنوز map نشده‌اند.
4. ۸۲۸ genome گزارش‌شدهٔ DEAP با۷٬۸۴۲ genome artifact و partial genomeها deduplicate نشده است.
5. HF ledger entries #44/#45 و۴۱ run record/۴۲ attempt به config ID تبدیل نشده‌اند.
6. workflowهای V4-I تا V4-O تعدادی candidate قدیمی را replay می‌کنند؛ candidate identity و data/decision-rule provenance باید حفظ شود تا replay دوباره شمرده نشود.
7. V4-AI و V4-AO از script و candidate names مشترک استفاده می‌کنند؛ overlap دقیق artifactها/data scope هنوز تطبیق نهایی نشده است.
8. V4-P/Q/R/V/H nested search candidateها دارند؛ برای همهٔ آن‌ها فهرست کامل config ID و parent/child graph در ledger تاریخی consolidated موجود نیست.
9. آزمایش‌های شاخه‌های archived و partial evaluationهای پیش از retention فعلی، کامل بازیابی نشده‌اند.

بنابراین «کران بالای محافظه‌کارانه» در این snapshot **ناموجود/غیرقابل‌اثبات از حیث شواهد** است؛ نه اینکه تعداد واقعی بی‌نهایت باشد، بلکه شواهد کافی برای maximum finite وجود ندارد. حاصل‌ضرب ابعاد یک workflow، upper bound همان ماتریس یک اجراست و جایگزین upper bound تاریخچهٔ پروژه نیست.

### 3.3 اسناد لازم برای بستن حسابداری

برای رسیدن به N قابل دفاع، حداقل این اسناد لازم‌اند:

- export کامل GitHub Actions run history برای هر ۳۹ workflow پژوهشی، همه attemptها، branch/SHA و conclusion؛
- artifact manifest و candidate-ID inventory برای هر run موفق و ناموفق/partial، شامل progress marker برای فهم اینکه آیا candidate evaluation شروع شده است؛
- crosswalk بین top-level candidate ID، nested hyperparameter config ID، genome hash، execution و workflow attempt؛
- artifact JSON کامل V4-T/V4-AP/V4-H/V4-P/Q/R/V و اسکریپت/گزارش‌های V4-F/G/I-O؛
- فهرست genome hashهای DEAP برای تطبیق۷٬۸۴۲ با۸۲۸ و partial genomeها؛
- mapping HF ledger #1–#45 به run ID و model/strategy/data/parameter configuration؛
- reconciliation candidate identity برای V4-AI/AO و replayهای V4-H؛
- inventory آزمایش‌های شاخه‌های historical/archived که قبل از Trial Registry انجام شدند؛
- canonical trial_id به‌همراه parent/duplicate references که replay، fold، fit، attempt و artifact را از candidate مستقل جدا کند.

تا تکمیل این موارد: **GLOBAL_N=UNKNOWN باقی می‌ماند.** تعداد ۳۹ workflow، subtotal ۷٬۹۱۲ یا ضرب ابعاد چند grid هیچ‌کدام جایگزین این مقدار نمی‌شوند.

## 4. تعریف واحد trial برای DSR

تعریف الزام‌آور در ADR-045 ثبت شده است. خلاصهٔ عملیاتی:

- **Trial مستقل** یک فرضیه/decision rule قابل شناسایی و از پیش ثبت‌شده روی data/protocol مشخص است که candidate-defining مشخصاتش ثابت‌اند: مدل/نسخه، feature/label، timeframe/horizon، threshold، entry/exit/risk policy، parameter set و execution/cost contractی که در انتخاب candidate دخیل است.
- دو مقدار متفاوت از پارامتر candidate-defining دو trial/config جدا هستند، حتی اگر در یک grid و یک workflow باشند.
- fold fit، optimizer step، forecast row، bar، trade، generation number یا workflow attempt به‌تنهایی trial مستقل نیست.
- اجرای مجدد همان config روی همان data/protocol، execution/attempt جدید است نه trial جدید؛ تغییر candidate-defining parameter/model/data/decision rule باید به‌عنوان candidate جدید یا replication با lineage ثبت شود.
- ارجاع‌های تکراری Arm A در سه threshold یک candidate/Sharpe observation مؤثرند؛ سه ledger row برای provenance باقی می‌مانند، اما برای N مستقل dedup می‌شوند.
- run ناموفق فقط وقتی trial محسوب می‌شود که candidate evaluation، hyperparameter comparison یا candidate output واقعاً شروع شده باشد. setup/import/preflight failure قبل از آن execution/attempt است، نه trial. اگر progress نامعلوم باشد، UNKNOWN_PARTIAL_PROGRESS ثبت شود نه صفر.

## 5. دسترس‌پذیری Sharpe per-trial

DSR علاوه بر N به Sharpe مشاهده‌شدهٔ candidate و پراکندگی Sharpe بین trialهای cohort نیاز دارد. صرف داشتن N کافی نیست.

| خانواده | N/candidate قابل‌اثبات در artifact | Sharpe per-trial در شواهد retained؟ | منبع/مرز |
|---|---:|---|---|
| V4-AP | ۳۶ | **بله، ۳۶ Sharpe estimate** | scripts/run_ml_core_v4ap_diagnostic_sweep.py؛ TRIAL_REGISTRY.md §3؛ run 42493418590 / artifact 13624018702. independence در report برابر NOT_VERIFIED است. |
| V4-AI/AO | ۳ candidate در هر اجرای script | **بله، برای cohort همان run** | scripts/run_ml_core_v4ai_maker_strategy.py؛ فیلد sharpe_ratio_per_trade و DSR درون-run candidate/exit geometry. خود script تصریح می‌کند historical trials داخل cohort نیستند و independence تأیید نشده؛ overlap AI/AO نیز باید بررسی شود. |
| V4-T | ۳۲ top-level؛۳۶ nested با overlap نامعلوم | **شاهد کافی نیست** | artifact 13229820663 و TRIAL_REGISTRY.md §2؛ Sharpe per candidate در ledger retained ثبت نشده است. |
| V4-H/F/G | ۴۸ experiment /۴۰ /۸۰ classification row | **شاهد کافی نیست؛ Sharpe اقتصادی معیار report نیست** | workflow/scriptها classification metrics و nested tuning دارند؛ این‌ها Sharpe per-trial نیستند. |
| V4-AB/AC/AD/AE/AF/AH | ماتریس‌های جدول بالا | **شاهد کافی نیست** | script/workflowها شواهد candidate-level Sharpe تاریخی را در registry retained ثبت نکرده‌اند. |
| V4-I/J/K/L/M/N/O | replay candidateهای V4-H با management/cost/threshold variants | **شاهد کافی نیست** | workflowها/scripts replay؛ distribution کامل Sharpe بین همهٔ trialهای تاریخی در inventory موجود نیست. |
| V4-P/Q/R/V/X/Y | candidateهای جدول بالا و nested search | **شاهد کافی نیست** | در retained registry، Sharpe per-trial کامل و متصل به همهٔ parent/nested IDs در دسترس نیست. |
| AI-6 | حداقل۲ policy experiment | **شاهد کافی نیست** | docs/research/AI6-PPO-RISK-RESEARCH.md و TRIAL_REGISTRY.md §4؛ economic result هست، Sharpe list cohort-level retained نیست. |
| DEAP | ۷٬۸۴۲ genome verified +۸۲۸ owner-reported | **شاهد کافی نیست** | TRIAL_REGISTRY.md §5؛ genome fitness با Sharpe-per-trial تطبیق داده نشده است. |
| TTM-R2 | ۱ مدل/فرضیه؛۱۹۸ forecast origin | **شاهد کافی نیست** | docs/ttm-r2-risk-evaluation.md و docs/ttm_r2_closure.md؛ QLIKE، PF و return گزارش شده‌اند، نه توزیع Sharpe بین trialها. |
| HF | ۴۱ run record /۴۲ attempt؛ unique config UNKNOWN | **شاهد کافی نیست** | TRIAL_REGISTRY.md §1؛ executionها به Sharpe/config candidate-level نگاشت نشده‌اند. |
| Run #10 | ۱۲ ثبت /۱۰ signal rule مؤثر | **شاهد کافی نیست؛ DSR صریحاً محاسبه نشده** | TRIAL_REGISTRY.md Run #10 appendix و ADR-042؛ DSR در manifestها null و numeric sensitivityها missing هستند. |

### 5.1 نتیجه DSR و گزینه‌های محافظه‌کارانه

- **مانع اول: N.** GLOBAL_N=UNKNOWN است؛ subtotal تاریخی ≥7,912 فقط کران پایین ناهمگون/حل‌نشده است.
- **مانع دوم: Sharpe distribution.** فقط V4-AP و cohort درون-run V4-AI/AO شواهد صریح Sharpe per-candidate دارند. برای خانواده‌های عمدهٔ دیگر، Sharpe list کامل و deduplicated موجود نیست.
- بنابراین حتی اگر N دقیقاً مشخص شود، برای خانواده‌هایی که Sharpe per-trial آن‌ها غایب است، DSR **محاسبه‌پذیر نیست**. Sharpe را از PF/return/win rate/QLIKE/trade count استنتاج نکنید.
- گزینه اول: DSR را NOT_COMPUTABLE / MISSING_INPUT نگه دارید و هیچ candidate را بر مبنای DSR تأیید نکنید.
- گزینه دوم: Sharpeهای candidate را از artifactهای immutable و ledgerها بازسازی و deduplicate کنید؛ برای runهای partial وضعیت evaluated/not-evaluated/unknown حفظ شود.
- گزینه سوم: پس از تکمیل distribution، sensitivity DSR را برای Nهای مختلف مجاز و رو به افزایش محاسبه کنید؛ فقط پس از تثبیت cohort و Sharpe series.
- «فرض بدترین پراکندگی Sharpe» بدون داده یا محدودیت ریاضی معتبر، عدد متناهی تولید نمی‌کند. «کران بالای N» نیز finite و قابل اثبات نیست. هیچ‌کدام با عدد ساختگی جایگزین نمی‌شوند.

## 6. حدود و ایمنی

این ممیزی هیچ trial، model inference، dispatch، nonce، threshold mutation یا holdout access انجام نداده است. هیچ رکوردی به TRIAL_REGISTRY.md اضافه یا حذف نشده است. ADR-042 و۱۲ رکورد Run #10 دست‌نخورده‌اند.

- GLOBAL_N=UNKNOWN
- Champion=null
- order_permission=false
- trading_authorization=NO-GO
- lock=NOT_AUTHORIZED
- EXECUTION_TRIGGER=PROBE
- ACQUISITION_TRIGGER=PROBE
- ALIX_DRY_RUN=true
- ALIX_ALLOW_LIVE_TRADING=false
- private_api_calls=0
- real_order_attempts=0
- main=FROZEN
- Holdout remains sealed.

Architecture First. Evidence Always. Fail Closed.


---

## اصلاحیهٔ حسابداری — شرح ابعاد V4-R (2026-11-15)

در جدول per-workflow، عبارت ابعاد V4-R نیاز به دقت بیشتر داشت. در script مرجع، V4R_EXPECTED_CONTEXTS=4 حاصل دو timeframe در دو feature profile است؛ این چهار context برای هر یک از دو symbol اجرا می‌شوند. بنابراین ابعاد کامل چنین‌اند:

- ۲ symbol × (۲ timeframe ×۲ feature profile =۴ context به‌ازای هر symbol) ×۴ arm = **۳۲ experiment**؛
- دو weighting policy برای هر experiment => **۶۴ candidate/model config**؛
- random-search iterations=9 گزارش شده، اما تعداد unique nested configهای مشاهده‌شده و overlap آن‌ها با candidateهای سطح بالا در artifactهای retained این ممیزی به‌صورت مستقل تثبیت نشده است: **شاهد کافی نیست**.

این اصلاحیه، عبارت cardinality ردیف V4-R در بخش ۲ را supersede می‌کند. تعدادهای ۳۲ experiment و۶۴ candidate از constantهای صریح V4-R گرفته شده‌اند؛ این correction مجوز جمع‌زدن آن‌ها با subtotalهای تاریخی یا اعلام GLOBAL_N نیست.
