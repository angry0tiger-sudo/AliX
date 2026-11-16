# ADR-047 — Regime R: احیای پروژه و چرخش راهبردی به خانواده‌های مدل جدید

- **Status:** Accepted برای جهت‌گیری راهبردی؛ انتخاب مدل، preregistration نهایی و اجرای آزمایش همچنان نیازمند دستور مستقل مالک است.
- **Date:** 2026-11-16
- **Branch:** `research/nobitex-hf-strategy`
- **Parent / evidence baseline:** `722c3c30ea9b68d421291b5c2a9c5f49e0209e5b`
- **Related:** ADR-012, ADR-032, ADR-036, ADR-042, ADR-043, ADR-045, ADR-046
- **Owner override:** دستور مالک در 2026-11-16، خاتمهٔ Regime P را در سطح برنامه لغو و Regime R را برای طراحی و ارزیابی گزینه‌های جدید مجاز می‌کند.
- **Execution authority:** هیچ fit، tuning، inference، dispatch، data acquisition، backtest، holdout access یا تغییر runtime در این ADR مجاز نشده است.

---

## 1. تصمیم و دامنه

پروژه از وضعیت بستهٔ قبلی در سطح برنامه به وضعیت راهبردی زیر منتقل می‌شود:

```text
PROJECT_STATUS = PIVOTING_TO_NEW_MODELS
PIVOT_STATUS = DESIGN_ONLY / OWNER_MODEL_SELECTION_PENDING
VALIDATION_STATUS =
  NOT_VALIDATABLE_UNDER_CURRENT_GOVERNANCE / INSUFFICIENT_EVIDENCE
GLOBAL_N = UNKNOWN
GLOBAL_DSR = NOT_COMPUTABLE
Champion = null
order_permission = false
trading_authorization = NO-GO
```

این تغییر فقط یک **تغییر جهت برنامه‌ای** است. نتایج تاریخی Run #10، ADR-045، ADR-046، شمارش trialها، آلودگی داده و محدودیت‌های استقلال مدل‌ها بازنویسی نمی‌شوند. «مدل جدید» به‌تنهایی یک خانوادهٔ آماری مستقل یا راهی برای پاک‌کردن سابقهٔ انتخاب مدل نیست.

این ADR چهار گزینه را اولویت‌بندی می‌کند؛ هیچ‌کدام انتخاب یا اجرا نمی‌شوند. در این مرحله تنها خروجی مجاز، تصمیم راهبردی مالک دربارهٔ یک گزینه و سپس تدوین preregistration مستقل آن است.

## 2. تحلیل ریشه‌ای شکست‌های قبلی

### 2.1 مدل‌های OHLCV: PPO و DEAP

**PPO در AliX مدل پیش‌بینی جهت نبود.** دامنهٔ AI-6 صریحاً آن را risk allocator معرفی می‌کند: teacher ensemble سیگنال ورود را تولید می‌کرد و PPO فقط انتخاب می‌کرد یک سیگنال را بپذیرد یا رد کند. teacher از LightGBM، XGBoost، causal HMM و Logistic Regression تشکیل شده بود.

شواهد اقتصادی teacher، پیش از نسبت‌دادن شکست به PPO، ضعیف بود:

- Validation baseline: NetPF = 0.3743، MaxDD = 66.56%، 234 معامله.
- Historical terminal segment baseline: NetPF = 0.4370، MaxDD = 60.74%، 225 معامله.
- PPO هیچ signal episode را نپذیرفت: 0 از 228 در validation و 0 از 231 در قطعهٔ نهایی.

بنابراین صفر معاملهٔ PPO شاهد «مدیریت ریسک موفق» یا آزمون مستقل توان پیش‌بینی جهت نیست. نتیجهٔ مستند، ترکیب یک teacher کم‌لبه با policy participation failure است. محیط PPO از قبل تصمیم دودویی SKIP/TAKE داشت؛ پس شواهد، مشکل را به صرفاً پیوسته‌بودن فضای action نسبت نمی‌دهند.

**DEAP از OHLCV به یک جست‌وجوی نمادین محدود استفاده کرد.** قرارداد ثبت‌شده شامل 500 عضو جمعیت، 30 نسل، و هفت ورودی مشتق‌شده از OHLCV (RSI، MACD، ATR، فاصلهٔ Close از EMA100، حجم استانداردشده، ROC و Range) بود. هر درخت قاعدهٔ BUY/NO_BUY می‌ساخت و موقعیت سیگنال را تا خاموش‌شدن قاعده یا سقف 48 کندل نگه می‌داشت. تابع fitness به NetPF پس از هزینه، drawdown و حداقل تکرار معامله وابسته بود. نتیجهٔ OOS گزارش‌شده فقط 4 معامله، حدود 3.694 معامله در سال بود؛ در مقابل، قرارداد پذیرش بیش از 200 معامله در سال می‌خواست.

این نتیجه شکست تعمیم/تکرارپذیری اقتصادی و ناکفایتی نمونه را نشان می‌دهد؛ اما ثابت نمی‌کند یک اندیکاتور خاص علت شکست بوده است. جمعیت بزرگ و نسل‌های متعدد روی دادهٔ محدود می‌توانند قواعد درون‌نمونه‌ای متناسب با نویز انتخاب کنند و روی آینده پایدار نمانند.

**تشخیص مشترک دربارهٔ OHLCV:**

1. 31,801 کندل یک دارایی در تایم‌فریم یک‌ساعته، برای آزمودن تعداد زیادی مدل، ژنوم، هدف و افق اطلاعات مستقلی به اندازهٔ تعداد ردیف‌ها فراهم نمی‌کند. خودهمبستگی و هم‌پوشانی افق‌های برچسب، اندازهٔ مؤثر نمونه را کمتر می‌کند.
2. RSI، MACD، ATR، بازده و z-score حجم، تبدیل‌های مختلف همان OHLCV هستند؛ افزایش تعداد ستون‌ها لزوماً افزایش اطلاعات مستقل نیست.
3. OHLCV فاقد تغییرات صف سفارش، لغو/ثبت سفارش، aggressor flow، spread لحظه‌ای و عدم‌تعادل عمق است. این فقدان برای افق کوتاه می‌تواند تعیین‌کننده باشد.
4. بخشی از targetهای پیشین بر BUY/NO_BUY یا عبور از موانع ATR متمرکز بودند. این‌ها لزوماً با توزیع بازده **قابل‌اجرا و خالص از همهٔ هزینه‌ها**، فرصت نگهداری سرمایه و ریسک دنباله‌ای یکسان نیستند.
5. قراردادهای موجود، هزینهٔ ثابت 50 و 90 bps را بررسی کرده‌اند؛ target و fill فرضی باید علاوه بر کارمزد، spread، slippage، latency و market impact را نیز با زمان تصمیم و قیمت قابل‌اجرا هم‌راستا کنند.
6. در Run #10 بخش بزرگی از پیش‌بینی‌ها بسیار کوچک بود: در diagnostics، حدود 85.7% تا 91.6% پیش‌بینی‌ها در محدودهٔ بازده 24ساعتهٔ حداکثر 60 bps قرار می‌گرفتند. تنها 0% تا 5.26% کندل‌ها از thresholdهای ثبت‌شده عبور می‌کردند. از کندل‌های واجد شرایط نیز 1,439 از 1,656 مورد (حدود 86.9%) در زمان بازبودن موقعیت مسدود شده بودند. پس کمبود سیگنال، threshold، و position occupancy هم‌زمان نقش داشته‌اند.
7. بازار پرنویز و non-stationary است، اما دادهٔ فعلی ثابت نمی‌کند «نویز ذاتی» به‌تنهایی توضیح شکست است. شواهد قوی‌تر به نبود اطلاعات microstructure، هم‌راستایی ناقص target با net utility، sample support ناکافی و انتخاب مکرر مدل اشاره دارند.

### 2.2 مدل‌های بنیادی Chronos و Kronos

Run #10 به 0 PASS، سه FAIL برای یک کنترل Donchian تکراری، و 9 INSUFFICIENT_SAMPLE رسید. بهترین NetPF مشاهده‌شده در هزینهٔ 50 bps برابر 0.6324 و در 90 bps برابر 0.3936 بود؛ هر 36 بررسی پایداری fold ناموفق ماند. این نتایج از وجود edge اقتصادی قابل ارتقا پشتیبانی نمی‌کنند.

Chronos-2 در ارزیابی قبلی با baseline PF برابر 0.4077 و candidate PF برابر 0.4139، در کنار MaxDD حدود 99.8%، بهبود معنی‌دار اقتصادی نشان نداد. در TTM-R2، بهبود QLIKE از 0.39347957 به 0.27785937 کیفیت پیش‌بینی واریانس را نشان داد، اما replay معاملاتی 50 bps همچنان NetPF برابر 0.3748، drawdown برابر 18.26% و بازده خالص منفی 18.17% داشت.

توضیح‌های محتمل، اما نه اثبات‌شده:

- **هدف آموزش در برابر هدف اقتصادی:** loss پیش‌بینی سری زمانی، QLIKE یا خطای forecast مستقیماً سود خالص پس از spread، fee، slippage، latency و policy را بهینه نمی‌کند.
- **عدم تناسب دامنه و افق:** مدل عمومی سری زمانی می‌تواند الگوهای میانگین‌گیرانه یا پیش‌بینی‌های کوچک تولید کند؛ edge موردنیاز BTC Spot ممکن است کوتاه‌افق، شرطی به flow و شکننده در رژیم‌های متفاوت باشد.
- **کمبود مشاهدهٔ market microstructure:** OHLCV تجمیع‌شده مسیر تشکیل قیمت را پنهان می‌کند.
- **تراکم پایین forecastهای قابل‌معامله:** shrinkage و عبور کم از threshold، تعداد فرصت‌ها را کاهش داد؛ lowering threshold نیز در Run #10 برای چند arm با افت PF همراه بود.

**تفکیک حیاتی:** ابهام pretraining و احتمال هم‌پوشانی corpus، مسئلهٔ استقلال و اعتبار آزمون است؛ شواهد فعلی ثابت نمی‌کند این هم‌پوشانی علت اقتصادی PF پایین بوده است. برعکس، شفافیت provenance نیز به‌تنهایی استقلال را ثابت نمی‌کند. ADR-046 همچنان حاکم است.

## 3. گزینه‌های جدید و اولویت‌بندی

جست‌وجوی ساختار فعلی شاخه، پیاده‌سازی HMM سه‌حالته و استفادهٔ آن در teacher را نشان می‌دهد. در مسیرهای ثبت‌شده اثری از PatchTST، NGBoost، Conformalized Quantile Regression، BOCPD یا DeepLOB/OFI یافت نشد. این بررسی، مبنای انتخاب گزینه‌های زیر است؛ پیش از هر preregistration باید جست‌وجوی کامل dependency و workflow نیز تکرار شود.

### اولویت 1 — Order-Flow Imbalance و مدل ریزساختار بازار (OFI / DeepLOB)

**فرضیه:** عدم‌تعادل جریان سفارش/معامله و تغییرات صف خرید و فروش می‌تواند فشار کوتاه‌مدت قیمت را آشکار کند که در OHLCV یک‌ساعته محو شده است.

**مدل/رویکرد:** ابتدا baselineهای قابل‌تفسیر مانند order-flow imbalance، queue imbalance، microprice و signed trade imbalance؛ در صورت عبور از data gate، مقایسه با DeepLOB یا مدل مشابه روی توالی‌های L2. شروع با DeepLOB بدون baseline ساده ممنوع است.

**دادهٔ لازم:** L2 order-book updates، best bid/ask، عمق چند سطح، تغییرات/لغو سفارش‌ها، معاملات با timestamp و aggressor side، sequence integrity و spread. OHLCV فعلی کافی نیست. مستندات عمومی Nobitex endpointهای اردربوک و معاملات را توصیف می‌کنند؛ برای نگهداری اردربوک WebSocket را توصیه می‌کنند و polling را به‌علت cache به فاصلهٔ 1 تا 10 ثانیه محدود می‌کنند. endpoint اردربوک REST تا 300 درخواست در دقیقه و endpoint معاملات تا 60 درخواست در دقیقه مستند شده‌اند. با این حال، پشتیبانی دقیق نماد BTCUSDT، قابلیت نگهداری تاریخچه، عمق/کیفیت واقعی updateها و مجوز capture باید جداگانه تأیید شود. در این نوبت هیچ endpoint فراخوانی نشده است.

**نقش ریسک:** پیش‌بینی جهت/فشار قیمت در افق کوتاه و همچنین برآورد هزینهٔ اجرای ورود؛ مدل به‌تنهایی اجازهٔ سفارش یا اجرای خودکار نمی‌دهد.

**مزیت مورد انتظار:** افزودن منبع اطلاعاتی متفاوت، نه صرفاً بازآرایی همان کندل‌ها.

**ریسک/مانع:** کیفیت timestamp، gap/reconnect، hidden liquidity، spoofing/cancel noise، نرخ به‌روزرسانی، queue position، latency و هزینهٔ واقعی. اگر فقط snapshotهای REST با فاصلهٔ چندثانیه در دسترس باشند، عنوان DeepLOB ممکن است از ظرفیت واقعی داده فراتر رود.

### اولویت 2 — NGBoost با هدف بازده خالص و کالیبراسیون Conformal (CQR)

**فرضیه:** به‌جای پیش‌بینی یک عدد یا کلاس BUY، مدل توزیع شرطی بازده خالص و عدم‌قطعیت آن را تخمین بزند؛ فقط زمانی اجازهٔ سیگنال پژوهشی بدهد که حد پایین بازهٔ پیش‌بینی از تمام هزینه‌های ثبت‌شده و حاشیهٔ ایمنی بالاتر باشد.

**مدل/رویکرد:** NGBoost برای پیش‌بینی توزیع شرطی؛ کالیبراسیون بازه با Conformalized Quantile Regression یا روش conformal سازگار با سری زمانی. این مسیر به‌جای تکرار logistic meta-labeling، uncertainty و calibration را خروجی اصلی می‌کند.

**دادهٔ لازم:** از نظر schema می‌تواند با OHLCV و ویژگی‌های علّی موجود آغاز شود؛ اما pre-hack قبلاً مصرف شده و فقط برای توسعه/تشخیص گذشته قابل استفاده است، نه اثبات OOS مستقل. ارزیابی تأییدی نیازمند دادهٔ جدید و مجاز است. هدف باید next-executable net return باشد، نه فقط جهت close-to-close.

**نقش ریسک:** هم تخمین جهت/اندازهٔ بازده و هم کالیبراسیون عدم‌قطعیت؛ تصمیم باید selective باشد. خروجی آن به‌صورت probability/interval است، نه BUY قطعی.

**مزیت مورد انتظار:** عدم‌قطعیت و ناهمسانی واریانس را صریح می‌کند و می‌تواند از معامله در جایی که edge از هزینه عبور نمی‌کند جلوگیری کند.

**ریسک/مانع:** پوشش conformal در حضور non-stationarity و وابستگی زمانی به‌صورت خودکار تضمین نمی‌شود؛ calibration باید با بلوک‌های زمانی، پنجرهٔ rolling و آزمون drift انجام شود. بازهٔ وسیع ممکن است به no-trade منجر شود؛ این شکست نیست، مگر آنکه هدف فرکانس معامله را نیز شرط کرده باشد.

### اولویت 3 — Bayesian Online Changepoint Detection (BOCPD) به‌عنوان کنترل رژیم

**فرضیه:** بازار میان رژیم‌های کم‌نوسان، پرنوسان و تغییر ساختاری جابه‌جا می‌شود؛ آشکارسازی علّی نقطهٔ تغییر می‌تواند از اعمال یک policy ثابت در رژیم نامتناسب جلوگیری کند.

**مدل/رویکرد:** BOCPD یا نسخهٔ robust آن با run-length posterior و hazard ازپیش‌ثبت‌شده. HMM موجود دوباره اجرا یا صرفاً تغییر نام داده نمی‌شود؛ BOCPD بر توزیع طول رژیم و posterior تغییر ساختاری تمرکز دارد.

**دادهٔ لازم:** در نسخهٔ نخست همان OHLCV کفایت می‌کند: log return، realized volatility، range و volume surprise؛ ورودی‌ها فقط از گذشته ساخته شوند. برای نسخهٔ microstructure می‌توان flow و spread را بعداً اضافه کرد.

**نقش ریسک:** عمدتاً risk/regime overlay؛ کاهش exposure، افزایش حداقل confidence یا توقف سیگنال در زمان احتمال بالای change point. این گزینه به‌تنهایی جهت بازار را پیش‌بینی یا BUY ایجاد نمی‌کند.

**مزیت مورد انتظار:** کمک به کنترل شکست رژیم و کاهش انتقال اشتباه پارامتر/سیاست میان دوره‌های متفاوت.

**ریسک/مانع:** false changepoint، حساسیت به hazard/prior و تأخیر تشخیص؛ اگر تغییر رژیم دیر شناسایی شود، زیان قبلاً رخ داده است. پارامترهای BOCPD نیز اگر روی دادهٔ ارزیابی انتخاب شوند trial محسوب می‌شوند.

### اولویت 4 — PatchTST با آموزش نظارت‌شدهٔ محلی

**فرضیه:** به‌جای اتکا به forecast صفر-شات یک مدل بنیادی، encoder زمانی محلی با patchهای causal می‌تواند شکل حرکت و رابطهٔ چندمتغیرهٔ OHLCV را برای target اقتصادی مشخص یاد بگیرد.

**مدل/رویکرد:** PatchTST supervised، ترجیحاً از initialization تصادفی و بدون checkpoint مبهم pretraining؛ ورودی‌ها فقط پنجره‌های گذشته و خروجی‌ها توزیع یا quantile بازده در افق ازپیش‌ثابت باشد. مقایسه با baseline خطی/ساده و NGBoost الزامی است.

**دادهٔ لازم:** از نظر فرمت با OHLCV موجود قابل اجراست، اما 31,801 کندل یک‌ساعته برای آزمایش چندین معماری/patch/horizon کافی تلقی نمی‌شود. pre-hack برای توسعهٔ اکتشافی و تحلیل گذشته است، نه holdout تازه؛ ارزیابی مستقل به دادهٔ جدید نیاز دارد.

**نقش ریسک:** مدل directional/distributional؛ خروجی نقطه‌ای close یا جهت به‌تنهایی معیار پذیرش نیست.

**مزیت مورد انتظار:** patching توالی را در قطعات زمانی مدل می‌کند و encoder محلی supervised است؛ نسبت به foundation forecast عمومی، target و آموزش با دامنهٔ خاص پروژه هم‌راستاتر می‌شود.

**ریسک/مانع:** ظرفیت مدل در برابر حجم کوچک داده، overfit، non-stationarity و پیچیدگی تنظیمات. تغییر Transformer به Mamba بدون افزودن اطلاعات یا اصلاح target، اولویت ندارد.

### گزینه‌هایی که فعلاً توصیه نمی‌شوند

- **HMM ساده:** از قبل در teacher وجود داشته؛ نوآوری محسوب نمی‌شود.
- **Graph Neural Network:** مسئلهٔ فعلی یک دارایی/یک جفت و یک venue است. GNN بدون پنل هم‌زمان چنددارایی/چندvenue و فرضیهٔ روابط lead-lag ساختگی خواهد بود. فقط پس از تصمیم مالک برای گسترش universe قابل بررسی است.
- **Mamba به‌عنوان جایگزین فوری Transformer:** معماری جدید به‌تنهایی کمبود داده یا target نامتناسب را حل نمی‌کند؛ پس از ایجاد دادهٔ بیشتر و اثبات ارزش temporal encoder قابل بررسی است.

## 4. معماری اجرایی پیشنهادی بدون بدهی فنی و بدون تضعیف CI

### 4.1 قراردادهای لایه‌ای

Pipeline باید به اجزای مستقل و versioned تقسیم شود:

1. **Data contract:** منبع، venue، symbol، timezone، timestamp precision، cadence، sequence/reconnect، snapshot identity، raw SHA و کیفیت داده.
2. **Feature contract:** schema نسخه‌دار، point-in-time و causal؛ هیچ feature با آیندهٔ label یا دادهٔ آینده ساخته نشود.
3. **Target contract:** horizon، زمان دسترسی label، next executable price، barrier/tie/expiry semantics، تمام هزینه‌ها و label-end index.
4. **Model adapter:** ورودی feature schema و خروجی forecast schema ثابت؛ مدل direction، distribution و risk-only از یکدیگر تفکیک شوند.
5. **Decision policy:** مدل forecast تولید می‌کند؛ policy جداگانه با هزینه و محدودیت ریسک تصمیم پژوهشی می‌گیرد.
6. **Economic simulator:** ledger یکسان، fill در timestamp قابل‌اجرا، spread/fee/slippage/latency/impact و liquidate/terminal accounting صریح.
7. **Evaluation/report:** walk-forward folds، calibration، sample support، fold stability، NetPF، net return، MaxDD و Sharpe/DSR با manifest قابل بازتولید.

هیچ مدل جدیدی نباید مستقیماً به runtime، Freqtrade strategy، risk supervisor یا exchange adapter متصل شود.

### 4.2 CI و شاخه‌بندی

- `main` منجمد می‌ماند و هیچ تغییر یا merge به آن مجاز نیست.
- این commit فقط ADR است. توسعهٔ انتخاب‌شده باید پس از تصمیم مالک روی شاخهٔ تحقیقاتی اختصاصی و جدا از runtime انجام شود.
- Standard CI روی commit دقیق باید سبز باشد؛ CI مهندسی از اجرای اقتصادی جدا می‌ماند.
- unit و contract tests از synthetic fixtures استفاده می‌کنند؛ Standard CI نباید dataset واقعی، model weights یا holdout را بارگذاری کند.
- هر workflow پژوهشی جدید باید دستی/محدود، فقط روی branch مجاز، دارای permission حداقلی، concurrency کنترل‌شده و lock-before-I/O باشد.
- workflow جدید نباید به‌طور ضمنی با هر push اجرا شود. اجرای واقعی فقط پس از انتخاب مدل، preregistration immutable و Owner Authorization مستقل ممکن است.
- dependencyهای پرهزینه/اختیاری مدل جدید وارد base requirements و runtime production نمی‌شوند؛ محیط پژوهشی جدا و lockfile مستقل دارد.
- شکست CI با حذف تست، شرط، branch protection، secret، یا `[skip ci]` حل نمی‌شود.

## 5. Trial Registry و Preregistration جدید

### 5.1 ثبت append-only

برای هر خانواده، فیلدهای زیر پیش از اولین fit اجباری‌اند:

- `hypothesis_id`, `trial_family_id`, canonical `trial_id`, `parent_trial_id`, `duplicate_of`
- `attempt_id`, workflow run ID، commit SHA و candidate-evaluated status
- model/architecture، hyperparameters، seed و seed schedule، data SHA/snapshot ID، feature schema SHA
- target schema، horizon، label-end، fold-plan، purge/embargo
- cost/fill contract SHA و measured/assumed cost splits
- مدل/وزن provenance، pretrained checkpoint/corpus disclosure و training status
- candidate-level net return series، Sharpe و moments لازم DSR، calibration metrics و economic verdict
- وضعیت complete/partial/failed/cancelled/skipped با توضیح اینکه آیا candidate واقعاً ارزیابی شده است یا صرفاً execution attempt بوده است.

Workflow run، attempt، seed، genome، مدل و trial یک مفهوم واحد نیستند. موارد دقیقاً تکراری طبق ADR-045 deduplicate می‌شوند، اما تمام attemptها در تاریخچهٔ append-only حفظ می‌شوند. هر variant که در انتخاب مدل/پارامتر مؤثر شود باید در trial accounting وارد شود.

### 5.2 Preregistration

پیش از اولین fit/tuning باید موارد زیر immutable شوند:

1. یک فرضیهٔ اقتصادی قابل ابطال و مکانیزم edge؛
2. مدل/نسخه و دلیل انتخاب، همهٔ hyperparameterها و دامنهٔ جست‌وجو؛
3. feature/target schema، horizon، label maturity و point-in-time semantics؛
4. dataset window و exposure classification؛
5. سه یا چند chronological walk-forward fold، مرز UTC/row، purge و embargo متناسب با افق label؛
6. train-only scaler/calibrator و منع shuffle؛
7. integer `N_budget` و فهرست candidate slots پیش از مشاهدهٔ هر outcome؛
8. seed یا seed schedule؛
9. baselineها و انتخاب معیارها؛
10. 50/90 bps scenario به‌علاوهٔ مدل هزینهٔ واقعی venue، spread، slippage، impact و latency؛
11. حداقل trade/support، fold stability، drawdown و معیارهای توقف؛
12. روش calibration، Sharpe، DSR، وابستگی trialها و sensitivity؛
13. مرز final prospective holdout و قانون یک‌بارمصرف‌بودن آن.

### 5.3 عبور درست از GLOBAL_N=UNKNOWN

**GLOBAL_N با انتخاب مدل جدید صفر یا معلوم نمی‌شود.** سه قاعده:

- تا زمانی که تمام trialهای تاریخی و توزیع Sharpe آن‌ها بازسازی نشده‌اند، `GLOBAL_N=UNKNOWN` و `GLOBAL_DSR=NOT_COMPUTABLE` باقی می‌مانند.
- یک خانوادهٔ جدید فقط در صورتی می‌تواند `N_family` مشخص داشته باشد که از ابتدا با hypothesis/data boundary مستقل تعریف شود، تمام candidateهای همان خانواده شمارش شوند، و از ابتدا Sharpe/return inputs کامل ثبت شوند. اگر چند مدل پیشنهادی هم‌زمان روی outcomeهای یک dataset اجرا و بهترین آن‌ها انتخاب شوند، همهٔ آن مدل‌ها و variantهای مقایسه‌شده یک selection cohort مشترک‌اند؛ جداکردن نام خانوادگی مدل‌ها N را پاک نمی‌کند.
- `DSR_family` احتمالی فقط برای cohort جدید و کامل گزارش می‌شود؛ این مقدار جایگزین global DSR نیست و نباید به‌عنوان «رفع» مشکل تاریخی معرفی شود.

نام‌گذاری یک مدل یا ایجاد branch تازه، independence ایجاد نمی‌کند. اگر pre-hack برای انتخاب معماری/threshold استفاده شود، نتایج آن به‌عنوان دادهٔ تاریخی مصرف‌شده علامت می‌خورند؛ ارزیابی تأییدی باید بر دادهٔ out-of-time جدید و مجاز انجام شود. post-hack holdout همچنان sealed است و در این ADR هیچ مجوزی برای استفاده از آن صادر نشده است.

## 6. نقشهٔ راه گام‌به‌گام

| مرحله | خروجی لازم | شرط عبور |
|---|---|---|
| R0 — Owner selection | انتخاب دقیق یک گزینه از چهار گزینه و فرضیهٔ اقتصادی | هیچ fit یا data access قبل از این تصمیم |
| R1 — Feasibility/data audit | اثبات دسترسی قانونی/فنی به دادهٔ لازم، quality contract، هزینه و coverage | برای OFI: تایید symbol و WebSocket/event quality؛ برای OHLCV: دادهٔ prospective مستقل |
| R2 — Target & baseline design | target خالص از هزینه، horizon واحد، baselineهای منصفانه و risk role | target و fill semantics قابل آزمون و immutable باشند |
| R3 — Preregistration | hypothesis ID، family ID، candidate IDs، N_budget، folds، seeds، DSR و stop rules | بازبینی و تأیید مالک قبل از اولین fit |
| R4 — Isolated implementation | adapter جدید در research namespace، tests روی synthetic data و CI lock | عدم تغییر main/runtime/fetcher/holdout و Standard CI سبز |
| R5 — محدودسازی پژوهش | ابتدا baseline و candidate از پیش ثبت‌شده، بدون sweep باز | هیچ candidate خارج از N_budget یا انتخاب پس از مشاهدهٔ OOS |
| R6 — Walk-forward evidence | گزارش per-fold، trade support، NetPF@50/90، drawdown، net return، calibration و Sharpe | حداقل سه fold؛ معیار اقتصادی و sample support ثبت‌شده برقرار باشد |
| R7 — Owner review | گزارش مستقل و verdict | PASS فقط eligibility برای تصمیم بعدی مالک؛ نه Champion و نه trading authorization |

**پیشنهاد معیار اقتصادی اولیه:** تا زمان تصویب amendment جدید، از گیت‌های محافظه‌کار موجود به‌عنوان baseline استفاده شود: NetPF حداقل 1.10 در هر دو سناریوی 50 و 90 bps، حداقل 100 معاملهٔ اقتصادی قابل‌ارزیابی، MaxDD حداکثر 15%، حداقل سه fold زمانی پایدار و DSR مثبت برای cohortی که N و Sharpe آن کامل و قابل‌دفاع است. برای مدل risk-only، معیار forecast مانند QLIKE/MAE فقط گیت تشخیصی است؛ PASS آن بدون replay اقتصادی، اثبات edge معاملاتی نیست. افق microstructure ممکن است به تعریف sample threshold مخصوص خود نیاز داشته باشد و این تغییر باید پیش از آزمون ثبت و تأیید شود.

## 7. قفل‌ها و ممنوعیت‌ها

این ADR هیچ‌یک از این مقادیر را تغییر نمی‌دهد:

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `authorization_status=NOT_AUTHORIZED`
- `EXECUTION_TRIGGER=PROBE`
- `ACQUISITION_TRIGGER=PROBE`
- `main=FROZEN`
- post-hack holdout sealed / locked.

هیچ dispatch، nonce، inference، train، backtest، private API، order attempt، تغییر fetcher عمومی، Spec v3 یا V2 در مرحلهٔ طراحی مجاز نیست.

## 8. تصمیم بعدی مالک

برای شروع Regime R، مالک باید فقط **یک گزینه** را برای فاز امکان‌سنجی انتخاب کند. انتخاب یک گزینه به‌تنهایی اجازهٔ fit یا مصرف داده نیست؛ پس از آن، data boundary، هزینه و target همان گزینه مشخص و preregistration کامل برای تأیید جداگانه ارائه می‌شود.

**Engineering recommendation:** اگر مالک حاضر به سرمایه‌گذاری برای capture دادهٔ رویدادی و رسیدگی به پیچیدگی کیفیت آن است، اولویت اقتصادی با OFI/market microstructure است؛ چون اطلاعات جدید می‌آورد. اگر فعلاً فقط OHLCV مجاز باشد، NGBoost + cost-aware target و calibration مسیر کم‌هزینه‌تر و قابل‌تفسیرتری نسبت به افزایش بی‌هدف ظرفیت شبکه است. BOCPD نقش مکمل risk-only دارد، نه موتور alpha. PatchTST تا زمانی که دادهٔ مستقل و sample support کافی فراهم نشده، گزینهٔ چهارم و کم‌اولویت‌تر است.

---

## 9. منابع فنی

### شواهد داخلی AliX

- `docs/research/AI6-PPO-RISK-RESEARCH.md`
- `docs/phase-reports/AI2_DEAP_REVISITED_PLAN.md`
- `docs/research/TRIAL_REGISTRY.md`
- `docs/research/decisions/RUN10_DIRECTIONAL_FAMILY_DECISION_MEMO.md`
- `docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_HARVEST_REPORT.md`
- `docs/ttm-r2-risk-evaluation.md`
- `research/ai_ensemble/hmm.py`
- `research/ai_ensemble/features.py`
- `research/ai_ensemble/labels.py`
- ADR-045 و ADR-046.

### منابع بیرونی

- Nobitex official market-data documentation: https://github.com/nobitex/docs-api/blob/master/source/includes/_market_data.md
- DeepLOB: https://arxiv.org/abs/1808.03668
- Order-flow imbalance and cryptocurrency returns (2026): https://doi.org/10.1016/j.finmar.2026.101047
- NGBoost, ICML/PMLR 2020: https://proceedings.mlr.press/v119/duan20a.html
- Conformalized Quantile Regression, NeurIPS 2019: https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
- Bayesian Online Changepoint Detection: https://arxiv.org/abs/0710.3742
- PatchTST official implementation / ICLR 2023: https://github.com/PatchTST/PatchTST

**Architecture First. Evidence Always. Fail Closed.**
