from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config, features, metrics


@dataclass(frozen=True)
class Strategy:
    name: str
    max_positions: int
    stop_loss: Optional[float]
    weight_by_proba: bool
    regime_filter: bool
    max_new_per_day: int
    description: str


STRATEGIES = [
    Strategy(
        "sim1_equal_weight",
        10,
        None,
        False,
        False,
        3,
        "Up to 10 concurrent positions, equal weight, hold for the full horizon.",
    ),
    Strategy(
        "sim2_stop_weighted",
        10,
        config.STOP_LOSS,
        True,
        False,
        3,
        "Up to 10 positions, size scaled by predicted probability, stop loss on the daily low.",
    ),
    Strategy(
        "sim3_regime_concentrated",
        5,
        config.STOP_LOSS,
        True,
        True,
        2,
        "At most 5 positions, only when the market regime is healthy, probability sizing and stop loss.",
    ),
]


def add_vector_simulations(preds, thresholds, cost=None, top_k=None):
    cost = config.COST_PER_SIDE if cost is None else cost
    top_k = config.TOP_K_PER_DAY if top_k is None else top_k
    out = preds.copy()
    round_trip = 2.0 * cost
    for column in config.PREDICTION_COLUMNS:
        short = column.replace("pred_", "")
        threshold = thresholds[column]["threshold"]
        rank = out.groupby("date")[column].rank(method="first", ascending=False)
        signal1 = out[column] >= threshold
        signal2 = signal1 & (rank <= top_k)
        signal3 = signal2 & (out["flag_regime_ok"] == 1)
        for number, signal in enumerate((signal1, signal2, signal3), start=1):
            out[f"sim{number}_{short}_signal"] = signal
            out[f"sim{number}_{short}_ret"] = np.where(signal, out["ret_fwd"] - round_trip, np.nan)
    return out


def vector_summary(preds_with_sims, horizon=None):
    horizon = config.HORIZON if horizon is None else horizon
    test = preds_with_sims[preds_with_sims["split"] == "test"]
    all_dates = pd.Index(sorted(test["date"].unique()))
    rows = []
    for column in config.PREDICTION_COLUMNS:
        short = column.replace("pred_", "")
        for number in (1, 2, 3):
            ret_col = f"sim{number}_{short}_ret"
            chosen = test[test[f"sim{number}_{short}_signal"]]
            if len(chosen) == 0:
                rows.append({"prediction": column, "strategy": f"sim{number}", "trades": 0})
                continue
            per_day = chosen.groupby("date")[ret_col].mean().reindex(all_dates).fillna(0.0)
            equity = (1.0 + per_day / horizon).cumprod()
            equity.index = pd.DatetimeIndex(all_dates)
            summary = metrics.equity_metrics(equity)
            rows.append(
                {
                    "prediction": column,
                    "strategy": f"sim{number}",
                    "trades": int(len(chosen)),
                    "precision": float(chosen["target_cls"].mean()),
                    "avg_net_return_per_trade": float(chosen[ret_col].mean()),
                    "approx_cagr": summary["cagr"],
                    "approx_sharpe": summary["sharpe"],
                    "approx_max_drawdown": summary["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def load_panels(symbols, start):
    base = pd.read_parquet(config.BASE_PATH, filters=[("date", ">=", pd.Timestamp(start))])
    base = base[base["symbol"].isin(symbols)]
    panels = adjusted_panels_for_sim(base)
    return panels


def adjusted_panels_for_sim(base):
    panels = features.adjusted_panels(base)
    panels["close_ff"] = panels["close"].ffill()
    return panels


def build_entry_queue(signals, dates):
    queue = {}
    if len(signals) == 0:
        return queue
    ordered = signals.sort_values("proba", ascending=False)
    positions = dates.get_indexer(ordered["date"]) + 1
    for position, symbol, proba in zip(positions, ordered["symbol"], ordered["proba"]):
        if position <= 0 or position >= len(dates):
            continue
        queue.setdefault(int(position), []).append((symbol, float(proba)))
    return queue


def run_exact(signals, panels, strategy, threshold, cost=None, capital=None, horizon=None):
    cost = config.COST_PER_SIDE if cost is None else cost
    capital = config.INITIAL_CAPITAL if capital is None else capital
    horizon = config.HORIZON if horizon is None else horizon
    dates = panels["close"].index
    symbol_index = {symbol: i for i, symbol in enumerate(panels["close"].columns)}
    open_a = panels["open"].to_numpy()
    low_a = panels["low"].to_numpy()
    close_a = panels["close_ff"].to_numpy()
    queue = build_entry_queue(signals, dates)

    cash = float(capital)
    positions = []
    trades = []
    equity = np.empty(len(dates))
    invested = np.empty(len(dates))

    for i in range(len(dates)):
        if i in queue and len(positions) < strategy.max_positions:
            held = {p["idx"] for p in positions}
            previous_close = close_a[i - 1] if i > 0 else None
            marked = sum(p["shares"] * previous_close[p["idx"]] for p in positions) if previous_close is not None else 0.0
            equity_open = cash + marked
            slot = equity_open / strategy.max_positions
            added = 0
            for symbol, proba in queue[i]:
                if len(positions) >= strategy.max_positions or added >= strategy.max_new_per_day:
                    break
                idx = symbol_index.get(symbol)
                if idx is None or idx in held:
                    continue
                price = open_a[i, idx]
                if not np.isfinite(price) or price <= 0:
                    continue
                allocation = slot
                if strategy.weight_by_proba:
                    allocation *= min(max(proba / threshold, 1.0), 1.5)
                allocation = min(allocation, cash)
                if allocation < 0.25 * slot:
                    continue
                shares = allocation * (1.0 - cost) / price
                cash -= allocation
                positions.append(
                    {"idx": idx, "symbol": symbol, "shares": shares, "entry_price": float(price), "entry_i": i, "cost_basis": allocation, "proba": proba}
                )
                held.add(idx)
                added += 1

        remaining = []
        for position in positions:
            idx = position["idx"]
            held_days = i - position["entry_i"] + 1
            exit_price, reason = None, None
            if strategy.stop_loss is not None:
                stop_price = position["entry_price"] * (1.0 - strategy.stop_loss)
                low = low_a[i, idx]
                day_open = open_a[i, idx]
                if i > position["entry_i"] and np.isfinite(day_open) and day_open <= stop_price:
                    exit_price, reason = float(day_open), "stop_gap"
                elif np.isfinite(low) and low <= stop_price:
                    exit_price, reason = float(stop_price), "stop"
            if exit_price is None and held_days >= horizon:
                exit_price, reason = float(close_a[i, idx]), "time"
            if exit_price is None:
                remaining.append(position)
                continue
            proceeds = position["shares"] * exit_price * (1.0 - cost)
            cash += proceeds
            trades.append(
                {
                    "symbol": position["symbol"],
                    "entry_date": dates[position["entry_i"]],
                    "exit_date": dates[i],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "days_held": held_days,
                    "invested": position["cost_basis"],
                    "pnl": proceeds - position["cost_basis"],
                    "ret": proceeds / position["cost_basis"] - 1.0,
                    "reason": reason,
                    "proba": position["proba"],
                }
            )
        positions = remaining
        marked_close = sum(p["shares"] * close_a[i, p["idx"]] for p in positions)
        equity[i] = cash + marked_close
        invested[i] = marked_close / equity[i] if equity[i] > 0 else 0.0

    curve = pd.Series(equity, index=dates, name=strategy.name)
    exposure = pd.Series(invested, index=dates, name="exposure")
    return {"equity": curve, "exposure": exposure, "trades": pd.DataFrame(trades)}


def signals_for(preds, column, threshold, regime_filter):
    chosen = preds[preds[column] >= threshold]
    if regime_filter:
        chosen = chosen[chosen["flag_regime_ok"] == 1]
    return chosen[["date", "symbol", column]].rename(columns={column: "proba"})


def load_benchmarks(dates, capital=None):
    capital = config.INITIAL_CAPITAL if capital is None else capital
    dates = pd.DatetimeIndex(dates)
    series = {}
    if config.MARKET_PATH.exists():
        market = pd.read_parquet(config.MARKET_PATH).set_index("date").sort_index()
        for column, label in (("nifty50_close", "Nifty 50"), ("nifty500_close", "Nifty 500")):
            if column in market.columns:
                values = market[column].reindex(dates).ffill().dropna()
                if len(values):
                    series[label] = values / values.iloc[0] * capital
    if config.FEATURES_PATH.exists():
        ew = pd.read_parquet(config.FEATURES_PATH, columns=["date", "ew_ret_1"]).groupby("date")["ew_ret_1"].first()
        ew = ew.reindex(dates).fillna(0.0)
        index = (1.0 + ew).cumprod()
        series["Equal weight universe"] = index / index.iloc[0] * capital
    return series


def run_all_simulations(preds, thresholds):
    test = preds[preds["split"] == "test"].copy()
    start = test["date"].min() - pd.Timedelta(days=10)
    panels = load_panels(test["symbol"].unique(), start)
    test_dates = panels["close"].index[panels["close"].index >= test["date"].min()]
    results = {}
    rows = []
    for column in config.PREDICTION_COLUMNS:
        threshold = thresholds[column]["threshold"]
        for strategy in STRATEGIES:
            signals = signals_for(test, column, threshold, strategy.regime_filter)
            outcome = run_exact(signals, panels, strategy, threshold)
            equity = outcome["equity"].loc[test_dates]
            key = f"{column}__{strategy.name}"
            results[key] = {"equity": equity, "exposure": outcome["exposure"].loc[test_dates], "trades": outcome["trades"]}
            summary = metrics.equity_metrics(equity)
            summary.update(metrics.trade_metrics(outcome["trades"]))
            summary["avg_exposure"] = float(outcome["exposure"].loc[test_dates].mean())
            rows.append({"key": key, "prediction": column, "strategy": strategy.name, **summary})
    benchmarks = load_benchmarks(test_dates)
    benchmark_rows = []
    for label, series in benchmarks.items():
        series = series.loc[series.index.intersection(test_dates)]
        if len(series) > 10:
            benchmark_rows.append({"key": label, "prediction": "benchmark", "strategy": label, **metrics.equity_metrics(series)})
    summary_table = pd.DataFrame(rows + benchmark_rows)
    return results, benchmarks, summary_table
