import numpy as np
import pandas as pd

from . import config


def drawdown(equity):
    return equity / equity.cummax() - 1.0


def equity_metrics(equity, risk_free=None):
    risk_free = config.RISK_FREE_RATE if risk_free is None else risk_free
    equity = equity.dropna()
    daily = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(daily.std() * np.sqrt(252.0))
    excess = daily - risk_free / 252.0
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252.0)) if excess.std() > 0 else float("nan")
    downside = daily[daily < 0].std()
    sortino = float(excess.mean() / downside * np.sqrt(252.0)) if downside and downside > 0 else float("nan")
    max_dd = float(drawdown(equity).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "years": float(years),
    }


def rolling_returns(equity, window=252):
    return equity / equity.shift(window) - 1.0


def monthly_returns(equity):
    month_end = equity.resample("ME").last()
    return month_end.pct_change().dropna()


def trade_metrics(trades):
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": float("nan"), "avg_trade_return": float("nan"), "avg_holding_days": float("nan"), "stop_exits": 0}
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades["ret"] > 0).mean()),
        "avg_trade_return": float(trades["ret"].mean()),
        "avg_holding_days": float(trades["days_held"].mean()),
        "stop_exits": int(trades["reason"].str.startswith("stop").sum()),
    }
