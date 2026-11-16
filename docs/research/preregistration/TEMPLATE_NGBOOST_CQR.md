# پیش‌ثبت پژوهشی — NGBoost + Conformalized Quantile Regression (CQR)

**Hypothesis / Family ID:** `NOBITEX-R1-NGBOOST`  
**Regime:** R  
**Status:** `R1_R2_DRAFT / MOCK_ONLY / NOT_READY_FOR_FIT`  
**Parent baseline:** `32cdd57b7a31f113110eff443dd204120c2163b4`  
**Owner-selected family:** NGBoost conditional distribution + MAPIE CQR  
**N_budget:** **30** (حداکثر ۳۰ کاندید؛ ۱۰ کاندید برای هر افق)  
**Data authority:** هیچ دادهٔ واقعی یا holdout در این فاز مجاز نیست.

این سند یک قالب پیش‌ثبت با ۳۰ شناسهٔ رزروشده و قرارداد اولیه است. تا زمانی که تمام پارامترهای با برچسب BLOCKER تکمیل، hash و توسط مالک تأیید نشده‌اند، این سند برای fit، tuning، inference یا ارزیابی اقتصادی قابل اجرا نیست.

---

## 1. فرضیهٔ قابل ابطال

مدل NGBoost توزیع شرطی بازده ناخالصِ قابل‌اجرا را برای افق‌های ۱، ۴ و ۲۴ کندل یک‌ساعته تخمین می‌زند. CQR بازهٔ پیش‌بینی را بر اساس مجموعهٔ کالیبراسیون زمانی تنظیم می‌کند. سیگنال پژوهشی تنها زمانی مجاز است که **حد پایین پس از CQR** از هزینهٔ round-trip مربوط بالاتر باشد.

فرضیه رد می‌شود یا شواهد ناکافی تلقی می‌شود اگر حد پایین کالیبره‌شده به‌ندرت از هزینه عبور کند، coverage/width در foldها ناپایدار باشد، تعداد معاملات/NetPF/MaxDD گیت‌ها را پاس نکنند، یا استقلال و حسابداری trialها قابل دفاع نباشد.

**خروجی مورد انتظار:** توزیع شرطی و بازهٔ پیش‌بینی، نه فقط `predict()` نقطه‌ای. میانگین یا میانه به‌تنهایی معیار تصمیم نیست.

## 2. Target و زمان‌بندی

واحد داده: کندل کامل یک‌ساعته.

در زمان بسته‌شدن کندل (t):

- Featureها فقط تا close کندل (t) قابل مشاهده‌اند.
- ورود مرجع: open کندل (t+1).
- خروج مرجع: close کندل (t+h)، برای (h\in\{1,4,24\}).
- بازده هدف، simple gross return است:

```text
gross_return(t,h) = close[t+h] / open[t+1] - 1
label_end_index = t + h
```

هر هدف فقط پس از بسته‌شدن کندل (t+h) بالغ است. پنجرهٔ ناقص، gap یا قیمت نامعتبر باید unlabeled/invalid بماند؛ صفرسازی، forward-fill و برچسب‌گذاری مصنوعی ممنوع است.

هزینه از target gross کم نمی‌شود؛ هزینه جداگانه در policy و ledger اعمال می‌شود تا دوبار کسر نشود.

## 3. Feature schema اولیه

همهٔ ستون‌ها باید causal، right-aligned و مبتنی بر کندل بسته‌شده باشند:

| Feature | تعریف قراردادی |
|---|---|
| `log_return_1` | بازده لگاریتمی یک کندل گذشته |
| `log_return_4` | بازده لگاریتمی چهار کندل گذشته |
| `log_return_24` | بازده لگاریتمی ۲۴ کندل گذشته |
| `realized_volatility_6` | نوسان تحقق‌یافتهٔ trailing ۶ کندل |
| `realized_volatility_24` | نوسان تحقق‌یافتهٔ trailing ۲۴ کندل |
| `range_fraction_1` | دامنهٔ high-low تقسیم بر قیمت مرجع close |
| `atr_fraction_14` | ATR(14) نرمال‌شده با close |
| `log_volume_change_1` | تغییر لگاریتمی حجم، فقط گذشته |
| `volume_zscore_24` | z-score حجم در پنجرهٔ trailing ۲۴ کندل |
| `hour_sin`, `hour_cos` | کدگذاری چرخه‌ای ساعت تصمیم |
| `weekday_sin`, `weekday_cos` | کدگذاری چرخه‌ای روز هفته |

**BLOCKER:** قرارداد دقیق فرمول هر ستون، warm-up، رفتار volume=0، timezone و serialization باید در feature schema نسخه‌دار و SHA-256 شود. هر تغییر feature/window/normalization یک candidate جدید است.

## 4. کاندیدها و Trial IDs

تمام ۳۰ ID رزرو شده‌اند. سه افق، هر کدام دقیقاً ۱۰ candidate دارند. کل مجموعه یک selection family است؛ انتخاب بهترین افق از میان سه افق نیز بخشی از multiplicity همین خانواده است.

- `NOBITEX-R1-NGBOOST-1h-01`
- `NOBITEX-R1-NGBOOST-1h-02`
- `NOBITEX-R1-NGBOOST-1h-03`
- `NOBITEX-R1-NGBOOST-1h-04`
- `NOBITEX-R1-NGBOOST-1h-05`
- `NOBITEX-R1-NGBOOST-1h-06`
- `NOBITEX-R1-NGBOOST-1h-07`
- `NOBITEX-R1-NGBOOST-1h-08`
- `NOBITEX-R1-NGBOOST-1h-09`
- `NOBITEX-R1-NGBOOST-1h-10`
- `NOBITEX-R1-NGBOOST-4h-01`
- `NOBITEX-R1-NGBOOST-4h-02`
- `NOBITEX-R1-NGBOOST-4h-03`
- `NOBITEX-R1-NGBOOST-4h-04`
- `NOBITEX-R1-NGBOOST-4h-05`
- `NOBITEX-R1-NGBOOST-4h-06`
- `NOBITEX-R1-NGBOOST-4h-07`
- `NOBITEX-R1-NGBOOST-4h-08`
- `NOBITEX-R1-NGBOOST-4h-09`
- `NOBITEX-R1-NGBOOST-4h-10`
- `NOBITEX-R1-NGBOOST-24h-01`
- `NOBITEX-R1-NGBOOST-24h-02`
- `NOBITEX-R1-NGBOOST-24h-03`
- `NOBITEX-R1-NGBOOST-24h-04`
- `NOBITEX-R1-NGBOOST-24h-05`
- `NOBITEX-R1-NGBOOST-24h-06`
- `NOBITEX-R1-NGBOOST-24h-07`
- `NOBITEX-R1-NGBOOST-24h-08`
- `NOBITEX-R1-NGBOOST-24h-09`
- `NOBITEX-R1-NGBOOST-24h-10`

| افق | تعداد slot | بازه شناسه | سهم از N_budget |
|---|---:|---|---:|
| 1h | 10 | 01–10 | 10 |
| 4h | 10 | 01–10 | 10 |
| 24h | 10 | 01–10 | 10 |
| **کل خانواده** | **30** | | **30** |

**قاعده:** شناسهٔ رزرو‌شده مساوی با trial اجراشده نیست. برای هر ID، status، config hash، fit-start، candidate-evaluated، run ID و attempt ID به‌صورت append-only ثبت شود. تمام candidateهایی که خروجی‌شان مشاهده یا در انتخاب مقایسه شده است در N مربوط وارد می‌شوند؛ retry یا execution attempt به‌تنهایی لزوماً candidate جدید نیست، مطابق ADR-045.

## 5. NGBoost و CQR

### 5.1 NGBoost

- مدل باید توزیع شرطی بازده را تخمین بزند.
- خروجی باید حداقل امکان استخراج lower، median و upper quantiles را داشته باشد.
- distribution family، base learner، تعداد estimator، learning rate، regularization، seed و تمام پارامترهای مدل candidate-defining هستند.

**BLOCKER:** ماتریس پیکربندی دقیق ۱۰ candidate در هر افق هنوز تعیین نشده است. هیچ sweep یا جست‌وجوی آزاد پس از مشاهدهٔ نتیجه مجاز نیست.

### 5.2 CQR / MAPIE

- CQR باید روی quantileهای حاصل از توزیع NGBoost اعمال شود.
- طبق API مستند MAPIE، `ConformalizedQuantileRegressor` در حالت prefit سه estimator quantile می‌گیرد. اتصال یک توزیع NGBoost به آن به adapterهای sklearn-compatible برای lower/upper/median نیاز دارد.
- ترتیب ورودی estimatorها باید با API نسخهٔ pin‌شده و تست integration در GitHub Actions تأیید شود.
- calibration فقط از دادهٔ زمانی گذشته و کاملاً پیش از evaluation fold استفاده می‌کند.
- استفاده از split تصادفی یا calibration روی test ممنوع است.
- CQR به‌تنهایی تضمین conditional coverage در سری زمانی وابسته و non-stationary نمی‌دهد. coverage، interval width، coverage drift و نتایج regime/fold باید گزارش شوند.

**BLOCKERهای اجباری:** nominal coverage، quantile levels، conformity score، finite-sample quantile rule، calibration-window length، rolling/block calibration policy، حداقل اندازهٔ calibration و cadence بازکالیبراسیون.

## 6. قاعدهٔ تصمیم مبتنی بر Lower Bound

حد پایین مورد استفاده، lower endpoint پس از اعمال CQR correction است، نه raw quantile مدل.

| سناریو | شرط پذیرش سیگنال پژوهشی |
|---|---|
| Cost-50 | `calibrated_lower_bound_gross > 0.005` |
| Cost-90 | `calibrated_lower_bound_gross > 0.009` |

شرط strict است؛ مساوی‌شدن با هزینه به معنای عدم ورود است. برای هر سناریو ledger و metrics جداگانه ساخته می‌شود. عبور از Cost-50 به‌تنهایی pass در Cost-90 نیست.

**BLOCKER:** تعریف کامل portfolio execution (position fraction، یک موقعیت هم‌زمان یا عدم هم‌پوشانی، skip accounting، fill/exit، terminal liquidation) باید پیش از fit ثابت شود. هیچ معامله‌ای در این فاز انجام نمی‌شود.

## 7. Walk-forward و نشت زمانی

- حداقل سه fold زمانی chronological و غیرتصادفی.
- مرز هر train/calibration/test با UTC timestamp و row index ثبت شود.
- train، calibration و test در هر fold تفکیک‌شده باشند؛ calibration آیندهٔ test را نمی‌بیند.
- label هر نمونهٔ train باید پیش از آغاز test بالغ شده باشد: `label_end_index < test_start_index`.
- purge/embargo حداقل باید افق همان target را پوشش دهد؛ مقادیر دقیق و مرزها قبل از fit ثبت شوند.
- scaler/feature transforms فقط روی train fit شوند.
- calibration set و test set هرگز برای انتخاب hyperparameter، threshold یا checkpoint استفاده نشوند.
- داده‌های post-hack و holdout مطلقاً sealed هستند.

**BLOCKER:** timestampها، row ranges، purge/embargo exact و SHA-256 هر fold plan هنوز باید تکمیل شوند.

## 8. N_budget و حسابداری Sharpe/DSR

- `N_budget = 30`، سقف ازپیش‌تعیین‌شدهٔ این خانواده است؛ این عدد برابر `GLOBAL_N` تاریخی نیست.
- تمام ۳۰ slot در یک خانوادهٔ انتخاب قرار دارند. افق‌ها را برای کوچک‌نمایی multiplicity به سه خانوادهٔ مستقل تقسیم نکنید، چون انتخاب بهترین horizon میان آن‌ها انجام می‌شود.
- Candidateهای واقعاً ارزیابی‌شده، attemptها، failureها، cancelled/partialها و duplicateها طبق ADR-045 ثبت و تفکیک شوند.
- برای هر candidate ارزیابی‌شده، return series خالص، Sharpe با فرکانس مشترک، تعداد نمونه، skewness/kurtosis و ورودی‌های لازم estimator DSR نگهداری شود.
- وابستگی/همبستگی میان candidateها و horizonها باید گزارش و در sensitivity لحاظ شود.
- family DSR فقط با N واقعیِ قابل‌دفاع و توزیع کامل Sharpe قابل محاسبه است. `N_budget=30` نیز به‌عنوان sensitivity محافظه‌کارانه گزارش شود.
- `GLOBAL_N=UNKNOWN` و `GLOBAL_DSR=NOT_COMPUTABLE` تاریخی تا زمان بازسازی مستقل تغییر نمی‌کنند.

## 9. معیارهای پذیرش اقتصادی

معیارها باید برای هر افق و هر سناریوی هزینه جداگانه گزارش شوند:

| معیار | Cost-50 | Cost-90 |
|---|---:|---:|
| Net Profit Factor | `>= 1.10` | `>= 1.10` |
| Completed trades | `>= 100` | `>= 100` |
| Max Drawdown | `<= 15%` | `<= 15%` |

- حداقل سه chronological fold؛ نتیجهٔ pooled نباید fold ناموفق یا sample ناکافی را پنهان کند.
- NetPF و MaxDD از equity ledger پس از هزینه محاسبه شوند، نه از target labels یا forecast scores.
- تعداد forecastها، سیگنال‌های عبورکرده از lower-bound gate، سیگنال‌های ردشده، سیگنال‌های skip‌شده، trade count و interval coverage جداگانه گزارش شوند.
- NLL/log score، CRPS (در صورت تعریف مناسب)، quantile loss، coverage، interval width و calibration drift معیارهای تشخیصی‌اند؛ جایگزین acceptance اقتصادی نیستند.
- عدم کفایت نمونه، metric نامعین، fold ناقص، data leakage یا dependency/provenance نامعتبر برابر `INSUFFICIENT_EVIDENCE / NO-GO` برای candidate/family است.

## 10. Dependency و CI

وضعیت snapshot فعلی:
- `ngboost` و `MAPIE` در `requirements.txt`، `requirements.lock` و CI dependency install فهرست نشده‌اند.
- نسخهٔ عمومی گزارش‌شده برای NGBoost برابر 0.6.0 است، ولی release provenance در PyPI موجود نیست و سازگاری دقیق با `scikit-learn==1.9.0` و `numpy==2.5.4` فعلاً اثبات نشده است.
- MAPIE 1.6.0 release provenance دارد و API CQR ارائه می‌کند؛ compatibility و CQR prefit adapter باید در Actions آزموده شوند.
- R1/R2 mock tests نباید این پکیج‌ها را import کنند یا نیاز به نصب آن‌ها داشته باشند.

**No install / no dependency update in this phase.** R3 باید نسخه‌های قطعی، dependency closure، hash/provenance، license، pip-audit و سازگاری API را در GitHub Actions تأیید کند. نسخهٔ ناقص/ناسازگار بدون دستور مالک به نسخهٔ دیگری downgrade نمی‌شود.

## 11. وضعیت مجوز و Safety Invariants

```text
PHASE = MOCK_ONLY
FIT_AUTHORIZED = false
REAL_DATA_AUTHORIZED = false
HOLDOUT_ACCESS = FORBIDDEN
DISPATCH = FORBIDDEN
NONCE = FORBIDDEN
INFERENCE = FORBIDDEN
BACKTEST = FORBIDDEN
```

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `EXECUTION_TRIGGER=PROBE`
- `ACQUISITION_TRIGGER=PROBE`
- `main=FROZEN`

## 12. چک‌لیست لازم پیش از R3

- [ ] ماتریس دقیق ۳۰ config و config SHA ثبت شود.
- [ ] quantile levels و nominal coverage تصویب شوند.
- [ ] MAPIE/NGBoost adapter و conformity score نسخه‌گذاری شوند.
- [ ] temporal calibration/block policy و minimum sample تعیین شوند.
- [ ] fold timestamps، row indices، purge و embargo freeze شوند.
- [ ] target/feature schema و dataset boundary hash شوند.
- [ ] cost/fill و position sizing policy تصویب شوند.
- [ ] تمام candidate Sharpe/DSR inputs و dependence treatment تعریف شوند.
- [ ] CI و security/dependency audit در Actions سبز شود.
- [ ] مالک preregistration کامل و مجوز اجرای جداگانه را امضا کند.

**نتیجه:** این سند N_budget و ۳۰ شناسه را ثبت می‌کند، اما به‌علت blockerهای فوق هنوز READY_FOR_FIT نیست. فاز بعدی تنها با دستور مستقل مالک آغاز می‌شود.
