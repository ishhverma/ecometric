# Global Macro Forecasting with ML, Econometrics, and Real‑Time Data

A reproducible, end‑to‑end pipeline that combines time‑series econometrics and modern machine learning to forecast macroeconomic aggregates (GDP growth, inflation, unemployment, financial conditions, yields, commodities) using real‑time macro and financial data. The project is intended as a teaching/ research prototype for nowcasting, medium‑term forecasting, and signal generation for macro trading or policy analysis.

Contents
- Executive summary
- Motivation & goals
- Pipeline (diagram + description)
- Data: sources, series table, and vintage notes
- Quick start (Colab / local)
- File layout and reproducibility
- Data engineering: harmonization & vintage handling
- Feature engineering & dimension reduction
- Models & modeling workflow
- Evaluation, uncertainty & backtesting checklist
- Operationalization & deployment suggestions
- Common pitfalls and limitations
- Suggested experiments & extensions
- Dependencies & environment
- Contact / license

Executive summary
This repository demonstrates how to:
- download and cache macro and market series from FRED and Yahoo Finance;
- unify mixed‑frequency and mixed‑release data into a harmonized monthly panel;
- perform cleaning, diagnostic testing (stationarity, breaks), and robust feature engineering;
- build and compare econometric baselines (ARIMA/VAR, MIDAS, state‑space/DFM) and modern ML models (tree‑based, regularized linear, neural nets, Bayesian models with PyMC);
- evaluate point and probabilistic forecasts with time‑series aware validation and honest backtesting using vintage data where available.

Motivation & goals
- Combine interpretable econometrics (causal structure, inference) with flexible ML (nonlinear interactions, regularization) to improve forecasts.
- Work with realistic, release‑aware data (avoid look‑ahead due to revisions).
- Provide a reusable, modular pipeline for research and teaching: data → features → models → evaluation → deployment.
- Encourage best practices: time‑aware CV, vintage experiments, reproducible caching, unit tests for parsers.

Pipeline (diagram + description)
The following shows the high‑level pipeline. Use this pipeline to reason about where look‑ahead can be introduced and where vintage management matters most.

Mermaid flow (rendered on GitHub with mermaid enabled):
```mermaid
flowchart LR
  A[APIs: FRED / ALFRED, Yahoo / Market Data] --> B[Raw Downloads (data/raw/)]
  B --> C[Parser & Validator (src/ingest/)]
  C --> D[Cache & Version (snapshots per run)]
  D --> E[Frequency Harmonization & Alignment (monthly panel)]
  E --> F[Cleaning & Diagnostics]
  F --> G[Feature Engineering (lags, growth, PCA)]
  G --> H[Modeling: Econometrics | ML | Bayesian]
  H --> I[Evaluation: rolling CV, calibration, backtest]
  I --> J[Model Persist & Serve (artifacts, API)]
  J --> K[Monitor & Retrain]
```

ASCII fallback pipeline:
Raw Downloads (data/raw/) -> Parser & Validator -> Cache/Vintages -> Harmonize to Monthly Panel -> Cleaning & Diagnostics -> Feature Engineering -> Modeling (Econometrics / ML / Bayesian) -> Evaluation & Backtest -> Persist & Serve -> Monitor & Retrain

Key pipeline notes
- Keep raw downloads immutable: store untouched API responses with timestamp and query metadata.
- Parsers must validate units and date indices and emit a standardized CSV/Parquet per series.
- Harmonization: choose an indexing convention (e.g., month‑end or release date) and stick to it in downstream steps.
- Vintage experiments: when possible, create panels that reflect what was available at each run date.

Data: sources, series table, and vintage notes
Primary sources
- FRED (Federal Reserve Economic Data) — macro series (use ALFRED for vintage-aware experiments).
- Yahoo Finance (via yfinance) — daily market data and indices.

Suggested series (example list; update per your research needs):
- FRED codes (monthly / quarterly)
  - GDP (quarterly): GDP
  - Real GDP (quarterly): GDPC1
  - CPI (monthly): CPIAUCSL (headline)
  - Core CPI (monthly): CPILFESL
  - Unemployment rate (monthly): UNRATE
  - Industrial Production (monthly): INDPRO
  - Retail Sales (monthly): RSAFS
  - Federal Funds Effective Rate (monthly): FEDFUNDS
  - Oil price (monthly): DCOILWTICO
  - Personal Consumption Expenditures (quarterly/monthly variants): PCE / PCEC
  - GDP deflator: GDPDEF
- Yahoo tickers (daily -> resample to monthly)
  - S&P 500: ^GSPC (use Adj Close)
  - 10‑year yield proxy: ^TNX (or US Treasury data)
  - VIX: ^VIX
  - Gold: GC=F or GLD
  - Crude Oil: CL=F or USO
  - Other ETFs: TLT, SPY, GLD as proxies

Vintage and ALFRED
- ALFRED provides vintage series snapshots. For honest historical performance, use ALFRED to reconstitute datasets that reflect what was known at each date.
- If you cannot obtain vintage data for all series, document the missing vintages and report that backtests use final/revised values (which inflates historical skill).

Quick start (Colab / local)
1) Open the main notebook:
   - Global_Macro_Forecasting_ML-_Econometrics-_and_Real_Time_Data-1.ipynb

2) Install dependencies (Colab / first run):
```bash
pip install -r env/requirements.txt
```

3) Configure API keys:
- FRED API key: set as environment variable FRED_API_KEY or place in a config file `.env` (not checked in).
- (Optional) If using an alternate data provider, configure credentials similarly.

4) Run data ingestion cell to download & cache raw series to data/raw/. Example snippet (notebook cell):
```python
from src.ingest.fred import fetch_fred_series
from src.ingest.yahoo import fetch_yahoo_series

# Example
fetch_fred_series("CPIAUCSL", api_key=os.getenv("FRED_API_KEY"), out_dir="data/raw/fred/")
fetch_yahoo_series("^GSPC", start="1990-01-01", end="2025-01-01", out_dir="data/raw/yahoo/")
```

5) Build monthly panel:
- Execute harmonization cell to align frequencies, resample daily market data to month‑end returns, and save to data/processed/monthly_panel.parquet.

File layout (recommended)
- notebooks/
  - Global_Macro_Forecasting_ML-_Econometrics-_and_Real_Time_Data-1.ipynb
  - 01_data_ingest.ipynb (optional split)
  - 02_feature_engineering.ipynb
  - 03_modeling_and_evaluation.ipynb
- src/
  - ingest/ (FRED, ALFRED, Yahoo parsers)
  - features/ (transformations, lagging, PCA)
  - models/ (wrappers for sklearn, statsmodels, PyMC)
  - utils/ (validators, date helpers)
- data/
  - raw/ (immutable API dumps)
  - processed/ (aligned monthly panels)
  - vintages/ (snapshots & ALFRED stores)
- env/
  - requirements.txt or environment.yml
- README.md
- LICENSE

Data engineering: harmonization & vintage handling (detailed)
- Frequency harmonization strategies:
  - For daily market data -> monthly: use month‑end adjusted close returns or compute monthly realized volatility from daily returns.
  - For quarterly macro targets (e.g., GDP): either forecast at quarterly horizons or use MIDAS / temporal disaggregation to integrate monthly predictors.
- Timestamping:
  - Choose a canonical indexing scheme: "availability date" (vintage indexed) or "reference period end" (e.g., month‑end).
  - For nowcasting, index features by their release dates/availability — critical to avoid look‑ahead.
- Missing values & imputation:
  - Conservative: forward‑fill only when appropriate; prefer missingness indicators and model robustness to missing features.
  - Impute using only data with availability <= forecast date; store imputation method and parameters in metadata.
- Unit consistency:
  - Convert price levels to log‑levels or returns where appropriate.
  - Convert percentages to decimals or keep consistent units across features.

Feature engineering & dimension reduction
- Lags: generate 1, 3, 6, 12 period lags for persistent macro variables.
- Growth transforms: month-over-month and year-over-year changes (log diffs).
- Rolling stats: moving averages, volatilities, and momentum signals for markets.
- Aggregation: dynamic factor models (DFM) or PCA on standardized panels to extract common components.
- Interaction features: spreads (e.g., 10y–3m), volatility × returns, policy×macro interactions.
- Normalization: standardize features using training‑set statistics; save scalers for inference.

Models & modeling workflow
- Baseline bench: naive (last value), historical mean, AR(1).
- Econometrics:
  - ARIMA/SARIMA for univariate baselines.
  - VAR/VECM for system forecasting; cointegration via Johansen tests.
  - MIDAS for mixed frequencies.
  - Kalman/state‑space for time‑varying parameters and dynamic factors.
- Machine learning:
  - Regularized linear: Lasso, Ridge, ElasticNet.
  - Tree ensembles: RandomForest, XGBoost, LightGBM — good default for tabular macro + market data.
  - Neural nets: LSTM/TCN only with sufficient data or careful transfer learning/regularization.
  - Ensembles/stacking: combine econometric and ML forecasts using meta‑learners.
- Bayesian:
  - Time‑varying parameter Bayesian regressions and hierarchical models in PyMC4 for full posterior uncertainty; use ArviZ for diagnostics.
- Validation:
  - Use rolling window or expanding window cross‑validation for time series.
  - Nested CV for hyperparameter tuning where feasible.

Evaluation, uncertainty & backtesting checklist
- Point metrics: RMSE, MAE, MAPE; compare to benchmarks (relative RMSE).
- Probabilistic: prediction interval coverage, CRPS, log predictive density.
- Directional: sign accuracy and ROC/AUC for binary classification of events (e.g., recession).
- Economic backtest:
  - Convert forecasts into signals (e.g., allocation or hedge rules) and simulate P&L with transaction costs and slippage.
  - Evaluate Sharpe ratio, maximum drawdown, turnover.
- Robustness checks:
  - Feature importance (SHAP or permutation), sensitivity to training window length, sensitivity to missing‑data imputation choices.
- Backtesting checklist (honest):
  1. Use vintage data for training & evaluation or clearly note final-data limitation.
  2. Use only information available at the forecast date for features.
  3. Use time‑aware CV (rolling/expanding) — no random k‑fold.
  4. Tune hyperparameters inside the CV loop to avoid leakage.
  5. Store and version both the trained model and the dataset snapshot used to train it.

Operationalization & deployment
- Scheduling: run forecasts with Airflow/Prefect or simple cron jobs for periodic tasks.
- Persist artifacts: model binaries (joblib/pickle), scalers, and metadata (training end date, feature set).
- Serving: lightweight API (FastAPI) to expose latest forecasts or export CSVs for downstream processes.
- Monitoring: drift detection on inputs and performance; create alerts if model performance degrades beyond thresholds.
- Retraining policy: define retrain cadence and triggers (fixed cadence or performance‑triggered).

Common pitfalls and limitations
- Small-sample overfitting: macro datasets have limited observations; prefer simpler models or strong regularization.
- Revision bias: training on final revised series overstates historical skill; use vintage data where possible.
- Data snooping and p‑hacking: extensive feature hunting inflates apparent out‑of‑sample performance unless properly validated.
- Interpretability tradeoffs: tree ensembles and neural nets can be opaque—use SHAP or partial dependence to explain relationships.

Suggested experiments & extensions
- ALFRED vintage nowcasting experiment (construct a true real‑time dataset).
- Mixed‑frequency models (MIDAS, mixed VAR) to exploit daily/weekly financial features for monthly/quarterly targets.
- Ensemble strategy: combine DFM (econometric) + LightGBM (ML) + Bayesian model and evaluate stacked forecast improvements.
- Regime switching or change‑point detection to build regime‑conditional models.
- Alternative data: aggregate credit card spending, mobility, or news sentiment for short‑term nowcasts.

Dependencies & environment
- Core: Python 3.8+; numpy, pandas, scipy, matplotlib, seaborn
- Data: yfinance, fredapi (or custom HTTP clients)
- Modeling: scikit‑learn, statsmodels, xgboost/lightgbm, pymc, arviz
- Utilities: joblib, pyarrow (for parquet), prefect/airflow (optional)
- Example: env/requirements.txt (pin exact versions for reproducibility)

Security & data governance
- Do not commit API keys or raw proprietary data.
- If saving datasets with PII or sensitive content, follow your organization's governance rules.

Contributing, reproducibility and tests
- Keep raw downloads in data/raw/ and never alter them in place.
- Add unit tests for ingestion/parsers in src/ingest/tests to ensure time index and unit consistency.
- Provide reproducible environment via environment.yml or pinned requirements.txt.
- If adding datasets, include metadata: source, original API request, timestamp, and notes on adjustments.

License & attribution
- Choose a permissive license if you intend public reuse (e.g., MIT) or a more restrictive one for internal/commercial projects.
- Cite data providers (FRED, Yahoo) and libraries (PyMC, ArviZ) in research outputs.
- 
