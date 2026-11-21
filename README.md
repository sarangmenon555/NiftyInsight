# NiftyInsight: NSE daily swing trading with machine learning

An end to end algorithmic trading system for Indian equities. It downloads daily NSE data, builds one combined feature table for the 700 most liquid stocks, trains tree based models to find stocks likely to rise at least 3 percent over the next 5 trading days, simulates trading rules on those predictions, and publishes a fresh list of candidate trades every evening.

## 1. Problem

Retail investors in India can buy and hold delivery shares but cannot cheaply short them, so the useful question each evening is: which stocks should I buy at tomorrow's open and hold for one week?

- Market: NSE cash equities, long only, delivery (CNC) orders.
- Frequency: daily bars, signals created after the close, entry at the next open.
- Prediction target: `target_cls` is 1 when the return from the next open to the close 5 trading days later is at least 3 percent. `target_reg` is the return itself. With `ALGO_TARGET_MODE=excess` the target is the return relative to the average return of the universe on the same date (at least 2 percent above it), which removes market timing from the problem. `ret_fwd` always holds the raw return.
- Use of the output: `reports/latest_signals.csv` lists ranked candidates with a stop loss price. A dashboard shows signals, performance and model health. An optional broker module turns signals into Zerodha Kite orders (dry run by default).

This is a new problem compared with the course setup: a non US market, a daily horizon of one week, a point in time liquidity universe instead of the largest stocks, and a data source based on NSE bhavcopy files, including the delivery percentage that Yahoo Finance does not provide.

## 2. Data sources

| Source | What it gives | Access |
|---|---|---|
| NSE bhavcopy with delivery data (`sec_bhavdata_full_DDMMYYYY.csv`) | Daily open, high, low, close, previous close, volume, traded value, trade count, delivery quantity and delivery percentage for every equity | Free, scraped from NSE archives with a browser like session |
| NSE index close files (`ind_close_all_DDMMYYYY.csv`) | Nifty 50, Nifty 500, Nifty Midcap 100, Nifty Smallcap 100, India VIX, plus Nifty 50 PE, PB and dividend yield | Free, same archive |
| NSE Nifty 500 list (`ind_nifty500list.csv`) | Industry per symbol, used for sector features | Free |

No Yahoo Finance and no FRED data are used.

Size: the delivery data file exists for about 1,850 trading days, and a 700 stock liquidity universe gives 1,236,778 feature rows (the build step prints the count).

Feature dictionary: [docs/FEATURES.md](docs/FEATURES.md). There are 65 numeric features plus about 60 dummy and binned features (day of week, month, sector, six sets of bins, nine market condition flags). Highlights:

- Momentum, volatility, RSI, MACD, Bollinger, moving average distances and 52 week distances.
- Liquidity features: traded value ratios, Amihud illiquidity, average trade size.
- Delivery features: delivery percentage, its ratio to the 21 day average, its trend. These come from the NSE file and are the main new information over price only data.
- Market features: equal weighted universe return and volatility, market breadth, Nifty 50 returns, India VIX level and percentile, Nifty valuation (PE, PB, dividend yield), midcap versus large cap spread.
- Cross sectional ranks and sector relative returns on each date.

## 3. Data transformations

1. Raw daily files are cached on disk and parsed into `data/processed/base.parquet` (incremental, only new files are parsed).
2. Split and bonus adjustment: bhavcopy prices are raw. NSE publishes `PREV_CLOSE` adjusted for corporate actions on the ex date, so daily returns use `close / prev_close - 1` and an adjusted price series is built from the cumulative product. Daily moves beyond 50 percent are treated as data artefacts and neutralised.
3. Universe: each day the 700 most liquid stocks by 60 day median traded value with price above 20 rupees. This avoids a fixed list of current index members, which would leak future information into the past.
4. Features are computed on wide date by symbol panels and stacked into one table (`features.parquet`). Targets are added.
5. Dummies and bins are added at modelling time (`add_derived`). Feature sets are defined in `features.py`: `numeric_columns`, `dummy_columns`, `TO_PREDICT`.
6. Incremental update: new days are appended by recomputing only the last 15 days plus a 420 day warm up window. A test checks that incremental and full rebuilds agree.

Exploratory analysis (`src/eda.py`) writes the target distribution, positive rate by year and month, Spearman correlations of every feature with the targets, and plots to `reports/`.

## 4. Models and decision rules

- Decision tree, random forest and XGBoost, all tuned with `RandomizedSearchCV` using a predefined train and validation split in time order (average precision as the score).
- Time split: train up to 2022-12-31 (`ALGO_TRAIN_END`), validation up to 2023-12-31 (`ALGO_VALID_END`), test afterwards, with a 10 day purge so labels do not overlap splits.
- An ensemble column `pred_ens` averages the random forest and XGBoost probabilities.
- Decision rule: the probability threshold is chosen on validation data only, by scanning the top 10 percent down to the top 0.5 percent of probabilities and keeping the cut with the best t statistic of net return after costs. Higher probability events are therefore targeted, and the threshold never sees the test period.
- For live use the models are refit on all labelled data and saved in `data/models`.

## 5. Trading simulation

Costs: 0.15 percent per side by default (taxes, charges and slippage), configurable with `ALGO_COST_PER_SIDE`.

Vector simulation (`sim1_`, `sim2_`, `sim3_` columns for each prediction): threshold only, top 5 per day, top 5 per day in a healthy market regime. Capital is split into five weekly tranches for an approximate return on capital.

Exact simulation (`simulation.run_exact`): day by day loop with cash, reinvestment of gains, a cap on concurrent positions, entry at the next open, stop loss checked against the daily low with gap handling, time exit after the horizon, costs on both sides. Three strategies:

| Strategy | Rules |
|---|---|
| `sim1_equal_weight` | Up to 10 positions, equal weight, hold 5 days |
| `sim2_stop_weighted` | Up to 10 positions, size scaled by predicted probability, 8 percent stop loss |
| `sim3_regime_concentrated` | At most 5 positions, only when the equal weighted universe is above its 200 day average and VIX is at or below 25, probability sizing and stop loss |

Benchmarks: Nifty 50 and Nifty 500 price indices and an equal weighted universe index. Metrics: total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, win rate, average exposure, rolling one year return and drawdown plots.

Error analysis (`src/analysis.py`): precision and lift by sector, year, month, volatility bucket, liquidity bucket and market regime, the worst false positives, and the features that separate false positives from true positives. `reports/analysis_findings.md` summarises them.

## 6. How to run

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that NSE endpoints respond from your network before the long download:

```
python -m src.pipeline check-sources
```

Full workflow with real data (first run takes a while, the download is about 5,500 small files):

```
python -m src.pipeline all --source download
```

Individual steps: `download`, `build`, `eda`, `train`, `simulate`, `analyze`, `latest`, `trade`. Add `--fast` to `train` or `all` for a quick run with smaller samples.

Pipeline demo without any network access, on synthetic data in a separate folder:

```
bash scripts/run_demo.sh
```

Synthetic data is only a plumbing test. Its performance numbers mean nothing.

Notebooks in `notebooks/` call the same functions: `00_end_to_end`, `01_data_and_eda`, `02_models`, `03_simulation_and_analysis`. They are shipped without outputs. Run them on your real data and commit the executed versions.

### Two regimes

- Download mode (default): `python -m src.pipeline run-daily` downloads missing days, updates the tables incrementally, retrains when models are older than 30 days, and writes the latest signals.
- File mode: `python -m src.pipeline run-daily --mode file` skips download and feature building and works from an existing `data/processed/features.parquet`. Copy a prebuilt file there to replicate results quickly.

### Automation with cron

`scripts/run_daily.sh` runs the daily job and appends to `logs/run_daily.log`. NSE publishes bhavcopy in the evening, so schedule it after 20:00 India time:

```
CRON_TZ=Asia/Kolkata
30 20 * * 1-5 cd /path/to/niftyinsight && bash scripts/run_daily.sh
```

Raw files, processed tables and models live in `data/` (not intended for git). Output reports live in `reports/`.

### Docker

```
docker compose build
docker compose run --rm full-run
docker compose run --rm pipeline
docker compose up dashboard
```

### Dashboard

`streamlit run dashboard/app.py` shows data freshness, latest signals, equity curves against benchmarks, model metrics, feature importance, error analysis and the orders log. To make it live, run it with Docker on a server and put authentication in front of it (see `docs/DEPLOYMENT.md`).

### Broker module

`python -m src.pipeline trade` prints and logs the orders it would place (dry run). Live orders need `pip install -r requirements-broker.txt`, the `--live` flag, `ENABLE_LIVE_TRADING=1`, `KITE_API_KEY` and `KITE_ACCESS_TOKEN`. The live path is untested against a real account.

### Tests

`pip install -r requirements-dev.txt` and `python -m pytest -q`. The tests cover split adjustment, target construction, absence of look ahead in features, incremental versus full rebuild, hand computed simulator cases (costs, stop loss, gap stop, reinvestment), metrics, the order builder and an end to end synthetic run.

## 7. Results and discussion

### 7.1 Setup

- Data: 1,236,778 daily rows for the 700 most liquid NSE stocks on each date. The bhavcopy delivery file was available for about 1,850 trading days (the endpoint returns nothing for earlier dates), so the history is shorter than the 2015 start date in the configuration.
- Split in time: train up to 2022-12-31, validation 2023, test from 2024-01-01 to November 2026 (471,437 labelled test rows). The last 5 days have no label yet.
- Two experiments with the same code and data: the absolute target (return of at least 3 percent in 5 days) and the excess target (return at least 2 percent above the universe average on the same date, `ALGO_TARGET_MODE=excess`).
- Trading rules: entry at the next open, 5 day hold, 0.15 percent cost per side, strategies as in section 5. Benchmarks are price indices without dividends.
- Important: all tables below come from fast mode runs (3 tuning iterations per model, small training samples). They are a working baseline, not the final tuned numbers.

### 7.2 Model quality on the test period

| Target | Model | AUC | Precision in top 1 percent | Base rate |
|---|---|---|---|---|
| Absolute | Decision tree | 0.469 | 0.249 | 0.237 |
| Absolute | Random forest | 0.603 | 0.473 | 0.237 |
| Absolute | XGBoost | 0.577 | 0.429 | 0.237 |
| Absolute | Ensemble (RF and XGB) | 0.594 | 0.470 | 0.237 |
| Excess | Decision tree | 0.530 | 0.308 | 0.274 |
| Excess | Random forest | 0.554 | 0.391 | 0.274 |
| Excess | XGBoost | 0.550 | 0.405 | 0.274 |
| Excess | Ensemble (RF and XGB) | 0.554 | 0.403 | 0.274 |

Validation and test AUC are close (for example random forest, absolute: 0.580 and 0.603), so the models are not simply overfitted to the validation period.

### 7.3 Trading results on the test period (exact simulation, CAGR and maximum drawdown)

| Strategy | Absolute CAGR | Absolute max DD | Excess CAGR | Excess max DD |
|---|---|---|---|---|
| Decision tree, sim1 equal weight | -22.8% | -53.7% | -3.0% | -41.5% |
| Decision tree, sim2 stop and probability sizing | -25.5% | -57.9% | -20.4% | -62.2% |
| Decision tree, sim3 regime filter, 5 positions | -19.0% | -45.7% | -9.0% | -40.3% |
| Random forest, sim1 | -4.8% | -46.1% | -2.1% | -42.6% |
| Random forest, sim2 | -13.6% | -49.2% | -5.9% | -58.1% |
| Random forest, sim3 | -19.3% | -57.5% | +6.0% | -46.8% |
| XGBoost, sim1 | +8.0% | -27.3% | -12.9% | -50.2% |
| XGBoost, sim2 | +2.3% | -25.7% | -23.0% | -64.8% |
| XGBoost, sim3 | +4.1% | -29.5% | -14.9% | -55.7% |
| Ensemble, sim1 | -6.1% | -40.3% | -8.3% | -54.1% |
| Ensemble, sim2 | -7.1% | -39.0% | -6.8% | -56.1% |
| Ensemble, sim3 | +0.5% | -31.9% | -12.4% | -57.3% |
| Nifty 50 | +6.1% | -15.8% | +6.1% | -15.8% |
| Nifty 500 | +7.7% | -18.8% | +7.7% | -18.8% |
| Equal weight universe | +9.8% | -24.3% | +9.8% | -24.3% |

### 7.4 Discussion

1. **The models have a small but real ability to rank stocks.** Random forest and XGBoost beat the base rate in the top 1 percent of predictions (47 percent against 24 percent for the absolute target). The decision tree is no better than chance and is unstable, so it is shown only as the lecture baseline.
2. **Most of the absolute target edge was market timing, not stock picking.** In the error analysis the features that separate false positives from true positives are market level: 21 day market volatility (standardized gap 0.35), India VIX, market breadth and the market trend. A fixed +3 percent target is easier to reach when the whole market is volatile, so the model learns when the market is likely to bounce. When the target is changed to the return relative to the universe, those gaps fall below 0.1 and test AUC drops from about 0.59 to 0.55. What is left is a weak stock selection signal.
3. **No strategy has demonstrated a reliable profit.** Only 5 of the 24 strategy runs have a positive CAGR. None beats the equal weight universe (9.8 percent). The best absolute result (XGBoost sim1, 8.0 percent) is slightly above Nifty 50 (6.1 percent) and Nifty 500 (7.7 percent), but with a deeper drawdown (27 percent against 16 and 19 percent), and it is the best of 24 attempts, so it is optimistic. An earlier fast run with the same settings gave -8.8 percent for this exact strategy, which shows that results move a lot with small changes in the tuned model. The best excess result (random forest sim3, 6.0 percent) is about equal to Nifty 50 with a 47 percent drawdown.
4. **Costs matter.** Strategies make from 190 to about 1,500 trades in about 2.9 years. At 0.3 percent round trip cost, a small edge disappears. Win rates are 35 to 49 percent.
5. **The 8 percent stop loss hurt in 7 of 8 comparisons** between sim2 and sim1. A likely reason is that many selected stocks are volatile small caps that touch the stop and then recover, but this was not tested. The regime filter (sim3) improved CAGR over sim2 in 6 of 8 cases while keeping only 24 to 61 percent of capital invested on average, but the effect is not consistent enough to rely on.
6. **The probability threshold does not transfer from validation to test.** It was chosen to select about the top 10 percent of validation rows, yet it selects 24 percent (absolute) and 18 percent (excess) of test rows, because predicted probabilities drift upward in the test period. Ranking stocks within each day is more robust than a fixed threshold.
7. **Signals look better on average than the strategies that trade them.** The average gross 5 day return across all test signals is 0.62 percent (absolute) and 0.88 percent (excess), but the simulated strategies, which take only the highest ranked names under position limits, lose money. This was not investigated. Possible causes are that the highest probability names are the most volatile small caps, gap risk at the open, and position limits. A table of average return by predicted probability decile would settle it.
8. **The approximate vector simulation and the exact simulation disagree strongly** (for example random forest sim2 on the excess target: 26 percent approximate CAGR against -5.9 percent exact). The vector version has no position limit, no stop and an idealised capital split, so it is only useful for screening. The exact simulation is the one to trust.
9. **Where predictions work best.** Lift over the base rate was highest in 2025 and for low volatility stocks (lift 1.66 in the lowest volatility quartile for the absolute target), and weakest for the Diversified sector (1.06, and 0.76 for the excess target). Sector information is missing for many stocks because the sector list covers only current Nifty 500 members.
10. **Data quality issue found.** The liquidity universe includes a few ETFs (for example silver ETFs) and thinly traded small caps. They behave differently from ordinary stocks and should be excluded in a next version.

### 7.5 Conclusion

On NSE daily data the pipeline reproduces a realistic outcome: a modest predictive signal, much of which comes from market conditions, that does not survive transaction costs as a trading strategy. The project's value is in the full workflow and in the diagnostics that explain why. It does not claim a profitable strategy.

### 7.6 Next experiments

- Average return by predicted probability decile, to test whether higher confidence picks are really better.
- Exclude ETFs and the least liquid names, and trade the top few stocks per day by rank instead of using a fixed threshold.
- Drop the stop loss or widen it, and test a volatility based stop.
- Walk forward retraining with repeated runs and different seeds, reporting the spread of results instead of the best one.
- Longer history through the older price only bhavcopy format, which has no delivery data.

## 8. Limitations and known difficulties

- Results are unstable between runs in fast mode. Report ranges, not the best strategy.
- ETFs and very thin small caps enter the liquidity universe and should be filtered.
- Bhavcopy prices are not dividend adjusted and the corporate action adjustment relies on NSE `PREV_CLOSE`. Check a few known splits in your data.
- NSE blocks clients that do not look like browsers. The downloader uses a session with cookies, retries and a small worker pool. If NSE changes URLs or file layouts, `check-sources` is the first place to look.
- The endpoint URLs and column mappings follow the published NSE file layouts but were written without live access to NSE from the build environment. Run `check-sources` and a short date range first, and fix `src/data_nse.py` if a header differs.
- Stocks that were delisted may be missing from NSE current files, so some survivorship bias remains.
- Simulations assume fills at the open and at the stop price. Circuit limits, partial fills and market impact are not modelled.
- Long only delivery trading. Shorting would need futures and different costs.
- Taxes (short term capital gains) are not included beyond the flat cost assumption.

## 9. Ideas for improvement

- Walk forward retraining instead of one fixed train, validation, test split.
- Add FII and DII flow data, corporate action calendars, earnings dates and news sentiment.
- Predict excess return over the market so signals do not just follow the market regime.
- Add LSTM or temporal convolution models and compare with the trees.
- Meta labelling: a second model that decides whether to act on each signal.
- Portfolio construction by volatility targeting or risk parity instead of equal slots.
- Use the error analysis tables to drop sectors or liquidity buckets where precision is below the base rate.

## 10. Project layout

```
src/            data loaders, features, models, simulation, analysis, broker, CLI
dashboard/      Streamlit monitor
notebooks/      research notebooks calling src
scripts/        cron entry point and synthetic demo
tests/          unit and end to end tests
docs/           feature dictionary and deployment guide
Dockerfile, docker-compose.yml, requirements*.txt, Makefile
```

## 11. Course rubric checklist

| Area | Item | Where |
|---|---|---|
| Problem | README, context, end goal | Sections 1 and 5 |
| Problem | New problem definition | Indian equities, daily, one week horizon |
| Problem | Clear step description and findings | Sections 3, 6, 8 |
| Data | Lecture style sources and features | Price, volume, technical features |
| Data | 20+ new features documented | docs/FEATURES.md |
| Data | New data source | NSE bhavcopy, index files |
| Data | More than one million rows | 1,236,778 rows |
| Transformations | One data frame, feature sets | `features.py` |
| Transformations | 5+ new features, dummies, bins | `add_derived` |
| Transformations | EDA | `eda.py`, notebook 01 |
| Modeling | Tree models, more than one | `models.py` |
| Modeling | Custom decision rules | `choose_thresholds`, top k and regime rules |
| Modeling | Hyperparameter tuning | `tune_model` |
| Modeling | New model | XGBoost |
| Simulation | Vector, multiple strategies | `add_vector_simulations` |
| Simulation | Exact simulation with reinvestment | `run_exact` |
| Simulation | Metrics against benchmark | `metrics.py`, `simulation_summary.csv` |
| Simulation | Stop loss, sizing by probability, regime filter | `sim2_`, `sim3_` |
| Simulation | Better than benchmark | Not achieved reliably, see section 7 |
| Simulation | Deep analysis by sector, year and more | `analysis.py` |
| Automation | Scripts plus end to end notebook | `src/`, `notebooks/00_end_to_end.ipynb` |
| Automation | Dependencies | `requirements*.txt` |
| Automation | Cron | `scripts/run_daily.sh` |
| Automation | Two regimes | `run-daily --mode` |
| Automation | Incremental loading, storage outside git | `data/` parquet |
| Bonus | Modules, broker code, dashboard, Docker | `src/`, `broker.py`, `dashboard/`, `Dockerfile` |
| Bonus | Cloud deployment | Free dashboard hosting guide in docs/DEPLOYMENT.md, deploy it yourself |
