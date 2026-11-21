import numpy as np
import pandas as pd
import pytest

from src import metrics, simulation


def make_panels(open_values, low_values, close_values, symbol="AAA"):
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    frame = lambda values: pd.DataFrame({symbol: values}, index=dates, dtype="float64")
    panels = {"open": frame(open_values), "low": frame(low_values), "close": frame(close_values), "high": frame(close_values)}
    panels["close_ff"] = panels["close"].ffill()
    return panels, dates


def one_signal(dates, symbol="AAA", proba=0.6, position=0):
    return pd.DataFrame({"date": [dates[position]], "symbol": [symbol], "proba": [proba]})


def strategy(**overrides):
    base = dict(name="t", max_positions=1, stop_loss=None, weight_by_proba=False, regime_filter=False, max_new_per_day=1, description="")
    base.update(overrides)
    return simulation.Strategy(**base)


def test_time_exit_with_costs():
    prices = [100.0] * 10
    panels, dates = make_panels(prices, prices, [100, 100, 100, 100, 100, 110, 110, 110, 110, 110])
    outcome = simulation.run_exact(one_signal(dates), panels, strategy(), 0.5, cost=0.001, capital=100000.0, horizon=5)
    trades = outcome["trades"]
    assert len(trades) == 1
    assert trades.loc[0, "reason"] == "time"
    assert trades.loc[0, "days_held"] == 5
    assert trades.loc[0, "entry_date"] == dates[1]
    assert trades.loc[0, "exit_date"] == dates[5]
    expected = (110.0 / 100.0) * (1 - 0.001) * (1 - 0.001) - 1.0
    assert trades.loc[0, "ret"] == pytest.approx(expected, rel=1e-9)
    assert outcome["equity"].iloc[-1] == pytest.approx(100000.0 * (1 + expected), rel=1e-9)


def test_stop_loss_exits_at_stop_price():
    opens = [100.0] * 10
    lows = [100, 100, 100, 90, 100, 100, 100, 100, 100, 100]
    closes = [100, 100, 100, 95, 100, 100, 100, 100, 100, 100]
    panels, dates = make_panels(opens, lows, closes)
    outcome = simulation.run_exact(one_signal(dates), panels, strategy(stop_loss=0.08), 0.5, cost=0.0, capital=100000.0, horizon=5)
    trades = outcome["trades"]
    assert trades.loc[0, "reason"] == "stop"
    assert trades.loc[0, "exit_price"] == pytest.approx(92.0)
    assert trades.loc[0, "ret"] == pytest.approx(-0.08)


def test_gap_below_stop_exits_at_open():
    opens = [100, 100, 100, 85, 100, 100, 100, 100, 100, 100]
    lows = [100, 100, 100, 80, 100, 100, 100, 100, 100, 100]
    closes = [100] * 10
    panels, dates = make_panels(opens, lows, closes)
    outcome = simulation.run_exact(one_signal(dates), panels, strategy(stop_loss=0.08), 0.5, cost=0.0, capital=100000.0, horizon=5)
    assert outcome["trades"].loc[0, "reason"] == "stop_gap"
    assert outcome["trades"].loc[0, "exit_price"] == pytest.approx(85.0)


def test_gains_are_reinvested_and_slots_respected():
    opens = [100.0] * 12
    closes = [100, 100, 100, 100, 100, 120, 120, 120, 120, 120, 120, 120]
    panels, dates = make_panels(opens, opens, closes)
    signals = pd.DataFrame({"date": [dates[0], dates[5]], "symbol": ["AAA", "AAA"], "proba": [0.6, 0.6]})
    outcome = simulation.run_exact(signals, panels, strategy(), 0.5, cost=0.0, capital=100000.0, horizon=5)
    trades = outcome["trades"]
    assert len(trades) == 2
    assert trades.loc[1, "invested"] == pytest.approx(120000.0)


def test_no_signal_keeps_capital():
    prices = [100.0] * 8
    panels, dates = make_panels(prices, prices, prices)
    empty = pd.DataFrame({"date": [], "symbol": [], "proba": []})
    outcome = simulation.run_exact(empty, panels, strategy(), 0.5, cost=0.001, capital=50000.0, horizon=5)
    assert (outcome["equity"] == 50000.0).all()


def test_metrics_cagr_and_drawdown():
    index = pd.bdate_range("2020-01-01", periods=2)
    index = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")])
    equity = pd.Series([100.0, 200.0], index=index)
    summary = metrics.equity_metrics(equity, risk_free=0.0)
    assert summary["cagr"] == pytest.approx(1.0, rel=0.01)
    curve = pd.Series([100.0, 120.0, 90.0, 100.0], index=pd.bdate_range("2020-01-01", periods=4))
    assert metrics.drawdown(curve).min() == pytest.approx(-0.25)
