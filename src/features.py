import re

import numpy as np
import pandas as pd

from . import config

FEATURE_DOCS = {
    "ret_1": "One day return on the split-adjusted close.",
    "ret_2": "Two day return.",
    "ret_3": "Three day return.",
    "ret_5": "One trading week return.",
    "ret_10": "Two trading week return.",
    "ret_21": "One month momentum (21 trading days).",
    "ret_63": "Three month momentum.",
    "ret_126": "Six month momentum.",
    "gap": "Opening gap versus the previous adjusted close.",
    "intraday_range": "High minus low divided by close, a daily volatility proxy.",
    "body": "Close minus open divided by open, the candle body.",
    "close_location": "Position of the close inside the day range, 0 is the low and 1 is the high.",
    "vol_5": "Standard deviation of daily returns over 5 days.",
    "vol_21": "Standard deviation of daily returns over 21 days.",
    "vol_63": "Standard deviation of daily returns over 63 days.",
    "vol_ratio_5_63": "Short term over long term volatility, flags volatility expansion.",
    "atr_pct_14": "Average true range over 14 days as a share of price.",
    "rsi_14": "Relative strength index with a 14 day Wilder smoothing.",
    "macd_hist": "MACD histogram (12, 26, 9) scaled by price.",
    "dist_sma_20": "Distance of price from its 20 day simple moving average.",
    "dist_sma_50": "Distance of price from its 50 day simple moving average.",
    "dist_sma_200": "Distance of price from its 200 day simple moving average.",
    "dist_52w_high": "Distance of price from its 52 week high, always zero or negative.",
    "dist_52w_low": "Distance of price from its 52 week low, always zero or positive.",
    "bb_pctb": "Bollinger percent B over 20 days, how far price sits inside the band.",
    "max_ret_21": "Largest single day gain in the last 21 days, a lottery-like feature.",
    "min_ret_21": "Largest single day loss in the last 21 days.",
    "skew_63": "Skewness of daily returns over 63 days.",
    "log_turnover": "Log of the traded value in rupees for the day.",
    "turnover_ratio_5_21": "5 day average traded value over the 21 day average.",
    "volume_ratio_5_21": "5 day average share volume over the 21 day average.",
    "deliv_per": "Share of traded quantity that was marked for delivery, taken from NSE bhavcopy.",
    "deliv_ratio_21": "Delivery percentage divided by its own 21 day average.",
    "deliv_trend_5_21": "5 day average delivery percentage minus the 21 day average.",
    "log_trades": "Log of the number of trades executed in the day.",
    "avg_trade_value": "Log of average rupees per trade, separates retail and institutional activity.",
    "log_amihud_21": "Log scaled Amihud illiquidity: mean absolute return per rupee traded over 21 days.",
    "beta_63": "Rolling 63 day beta against the equal weighted universe return.",
    "ew_ret_1": "Equal weighted universe return for the day.",
    "ew_ret_5": "Equal weighted universe return over 5 days.",
    "ew_ret_21": "Equal weighted universe return over 21 days.",
    "ew_dist_sma_200": "Distance of the equal weighted universe index from its 200 day average.",
    "ew_vol_21": "Annualised 21 day volatility of the equal weighted universe.",
    "breadth_sma50": "Share of universe stocks trading above their 50 day average.",
    "breadth_adv": "Share of universe stocks that closed up on the day.",
    "nifty50_ret_1": "Nifty 50 one day return, from NSE index closing files.",
    "nifty50_ret_5": "Nifty 50 five day return.",
    "nifty50_ret_21": "Nifty 50 21 day return.",
    "nifty50_dist_sma_200": "Distance of the Nifty 50 from its 200 day average.",
    "vix": "India VIX closing level.",
    "vix_chg_5": "Five day percentage change of India VIX.",
    "vix_pctile_252": "Percentile rank of India VIX within the last year.",
    "nifty50_pe": "Nifty 50 trailing price to earnings ratio.",
    "nifty50_pb": "Nifty 50 price to book ratio.",
    "nifty50_dy": "Nifty 50 dividend yield.",
    "midcap_rel_21": "Nifty Midcap 100 21 day return minus Nifty 50 21 day return, a risk appetite proxy.",
    "cs_rank_ret_21": "Cross sectional percentile of the 21 day return on that date.",
    "cs_rank_ret_63": "Cross sectional percentile of the 63 day return on that date.",
    "cs_rank_vol_21": "Cross sectional percentile of 21 day volatility on that date.",
    "cs_rank_deliv_per": "Cross sectional percentile of delivery percentage on that date.",
    "cs_rank_turnover_ratio": "Cross sectional percentile of the 5 to 21 day traded value ratio.",
    "rel_ret_21_sector": "21 day return minus the average 21 day return of the stock's sector.",
    "rel_strength_21": "21 day return minus the equal weighted universe 21 day return.",
    "vol_adj_mom_63": "63 day return divided by its volatility scaled for the horizon.",
    "mom_deliv": "Interaction of 21 day momentum and the delivery ratio.",
}

DUMMY_PREFIXES = ("dow_", "month_", "sector_", "bin_", "flag_")

DUMMY_DOCS = {
    "dow_*": "Day of week dummies, Monday to Friday.",
    "month_*": "Calendar month dummies.",
    "sector_*": "Sector dummies from the NSE Nifty 500 industry list, one per sector.",
    "bin_ret21_*": "Five bins of 21 day momentum: crash, down, flat, up, surge.",
    "bin_vol21_*": "Four bins of 21 day volatility: low, mid, high, extreme.",
    "bin_rsi_*": "Five RSI zones from oversold to overbought.",
    "bin_hi52_*": "Four bins of distance from the 52 week high.",
    "bin_vix_*": "Four India VIX regimes: calm, normal, elevated, stress.",
    "bin_deliv_*": "Three bins of the delivery ratio: low, normal, high.",
    "flag_above_sma200": "Price above the 200 day average.",
    "flag_near_high52": "Price within 5 percent of the 52 week high.",
    "flag_near_low52": "Price within 10 percent of the 52 week low.",
    "flag_gap_up": "Opening gap above 2 percent.",
    "flag_gap_down": "Opening gap below minus 2 percent.",
    "flag_volume_spike": "Traded value over twice its 21 day average.",
    "flag_high_delivery": "Delivery ratio above 1.5.",
    "flag_oversold": "RSI below 30.",
    "flag_regime_ok": "Market regime filter: universe above its 200 day average and VIX at or below 25.",
}

TO_PREDICT = list(config.TARGETS)


def pivot(base, column, index=None, columns=None):
    frame = base.pivot(index="date", columns="symbol", values=column).sort_index()
    if index is not None or columns is not None:
        frame = frame.reindex(index=index if index is not None else frame.index, columns=columns if columns is not None else frame.columns)
    return frame


def adjusted_panels(base, index=None, columns=None):
    close = pivot(base, "close", index, columns).astype("float64")
    prev_close = pivot(base, "prev_close", index, columns).astype("float64")
    raw_ret = close / prev_close - 1.0
    valid = prev_close.gt(0) & raw_ret.abs().le(0.5)
    ret1 = raw_ret.where(valid)
    growth = (1.0 + ret1.fillna(0.0)).cumprod()
    adj_close = (100.0 * growth).where(close.notna())
    ratio = adj_close / close
    panels = {
        "close": adj_close,
        "open": pivot(base, "open", index, columns).astype("float64") * ratio,
        "high": pivot(base, "high", index, columns).astype("float64") * ratio,
        "low": pivot(base, "low", index, columns).astype("float64") * ratio,
        "ret1": ret1,
    }
    return panels


def build_features(base, market=None, sectors=None, keep_from=None):
    base = base.sort_values(["date", "symbol"])
    turnover_full = pivot(base, "turnover")
    close_full = pivot(base, "close")
    liquidity = turnover_full.rolling(config.LIQUIDITY_WINDOW, min_periods=20).median()
    rank = liquidity.rank(axis=1, ascending=False)
    member = (rank <= config.UNIVERSE_SIZE) & (close_full >= config.MIN_PRICE)
    columns = member.columns[member.any(axis=0)]
    dates = member.index
    member = member[columns]
    del turnover_full, close_full, liquidity, rank

    base = base[base["symbol"].isin(columns)]
    panels = adjusted_panels(base, dates, columns)
    adj = panels["close"].ffill(limit=5)
    open_, high, low = panels["open"], panels["high"], panels["low"]
    r = panels["ret1"]
    n_dates, n_cols = member.shape

    present = panels["close"].notna().to_numpy()
    mask = member.to_numpy() & present
    if keep_from is not None:
        mask = mask & np.asarray(dates >= pd.Timestamp(keep_from))[:, None]
    flat = mask.ravel()

    feats = {}

    def emit(name, frame):
        feats[name] = frame.to_numpy(dtype="float32").ravel()[flat]

    def emit_series(name, series):
        values = series.reindex(dates).to_numpy(dtype="float32")
        feats[name] = np.broadcast_to(values[:, None], (n_dates, n_cols)).ravel()[flat]

    wide = {}
    for n in (1, 2, 3, 5, 10, 21, 63, 126):
        wide[f"ret_{n}"] = adj / adj.shift(n) - 1.0
        emit(f"ret_{n}", wide[f"ret_{n}"])

    emit("gap", open_ / adj.shift(1) - 1.0)
    emit("intraday_range", (high - low) / panels["close"])
    emit("body", (panels["close"] - open_) / open_)
    spread = high - low
    emit("close_location", (panels["close"] - low) / spread.where(spread > 0))

    for n in (5, 21, 63):
        wide[f"vol_{n}"] = r.rolling(n, min_periods=max(3, n // 2)).std()
        emit(f"vol_{n}", wide[f"vol_{n}"])
    emit("vol_ratio_5_63", wide["vol_5"] / wide["vol_63"])

    prev_adj = adj.shift(1)
    true_range = np.fmax(np.fmax(high - low, (high - prev_adj).abs()), (low - prev_adj).abs())
    emit("atr_pct_14", true_range.rolling(14, min_periods=7).mean() / adj)

    delta = adj.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    wide["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / loss)
    emit("rsi_14", wide["rsi_14"])
    ema_fast = adj.ewm(span=12, adjust=False).mean()
    ema_slow = adj.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    emit("macd_hist", (macd - macd.ewm(span=9, adjust=False).mean()) / adj)

    sma = {}
    for n in (20, 50, 200):
        sma[n] = adj.rolling(n, min_periods=int(n * 0.7)).mean()
        wide[f"dist_sma_{n}"] = adj / sma[n] - 1.0
        emit(f"dist_sma_{n}", wide[f"dist_sma_{n}"])
    emit("dist_52w_high", adj / adj.rolling(252, min_periods=126).max() - 1.0)
    emit("dist_52w_low", adj / adj.rolling(252, min_periods=126).min() - 1.0)
    std20 = adj.rolling(20, min_periods=14).std()
    emit("bb_pctb", (adj - sma[20]) / (2.0 * std20.where(std20 > 0)))
    emit("max_ret_21", r.rolling(21, min_periods=10).max())
    emit("min_ret_21", r.rolling(21, min_periods=10).min())
    emit("skew_63", r.rolling(63, min_periods=30).skew())

    turnover = pivot(base, "turnover", dates, columns).astype("float64")
    volume = pivot(base, "volume", dates, columns).astype("float64")
    trades = pivot(base, "trades", dates, columns).astype("float64")
    deliv = pivot(base, "deliv_per", dates, columns).astype("float64")
    emit("log_turnover", np.log1p(turnover))
    wide["turnover_ratio_5_21"] = turnover.rolling(5, min_periods=3).mean() / turnover.rolling(21, min_periods=10).mean()
    emit("turnover_ratio_5_21", wide["turnover_ratio_5_21"])
    emit("volume_ratio_5_21", volume.rolling(5, min_periods=3).mean() / volume.rolling(21, min_periods=10).mean())
    emit("deliv_per", deliv)
    wide["deliv_ratio_21"] = deliv / deliv.rolling(21, min_periods=10).mean()
    emit("deliv_ratio_21", wide["deliv_ratio_21"])
    emit("deliv_trend_5_21", deliv.rolling(5, min_periods=3).mean() - deliv.rolling(21, min_periods=10).mean())
    emit("log_trades", np.log1p(trades))
    emit("avg_trade_value", np.log1p(turnover / trades.where(trades > 0)))
    amihud = (r.abs() / turnover.where(turnover > 0)).rolling(21, min_periods=10).mean()
    emit("log_amihud_21", np.log1p(amihud * 1e9))

    universe_ret = r.where(member)
    market_ret = universe_ret.mean(axis=1)
    market_index = (1.0 + market_ret.fillna(0.0)).cumprod() * 100.0
    covariance = r.rolling(63, min_periods=40).cov(market_ret)
    variance = market_ret.rolling(63, min_periods=40).var()
    emit("beta_63", covariance.div(variance, axis=0))

    emit_series("ew_ret_1", market_ret)
    emit_series("ew_ret_5", market_index / market_index.shift(5) - 1.0)
    ew_ret_21 = market_index / market_index.shift(21) - 1.0
    emit_series("ew_ret_21", ew_ret_21)
    emit_series("ew_dist_sma_200", market_index / market_index.rolling(200, min_periods=120).mean() - 1.0)
    emit_series("ew_vol_21", market_ret.rolling(21, min_periods=10).std() * np.sqrt(252.0))
    above50 = (wide["dist_sma_50"] > 0).astype("float64").where(member & wide["dist_sma_50"].notna())
    emit_series("breadth_sma50", above50.mean(axis=1))
    up = (universe_ret > 0).astype("float64").where(universe_ret.notna())
    emit_series("breadth_adv", up.mean(axis=1))

    if market is not None and len(market):
        m = market.set_index("date").sort_index().reindex(dates).ffill(limit=3)
        if "nifty50_close" in m.columns:
            n50 = m["nifty50_close"]
            emit_series("nifty50_ret_1", n50.pct_change(1, fill_method=None))
            emit_series("nifty50_ret_5", n50.pct_change(5, fill_method=None))
            emit_series("nifty50_ret_21", n50.pct_change(21, fill_method=None))
            emit_series("nifty50_dist_sma_200", n50 / n50.rolling(200, min_periods=120).mean() - 1.0)
            if "midcap_close" in m.columns:
                emit_series("midcap_rel_21", m["midcap_close"].pct_change(21, fill_method=None) - n50.pct_change(21, fill_method=None))
        if "vix_close" in m.columns:
            vix = m["vix_close"]
            emit_series("vix", vix)
            emit_series("vix_chg_5", vix / vix.shift(5) - 1.0)
            emit_series("vix_pctile_252", vix.rolling(252, min_periods=60).rank(pct=True))
        for column in ("nifty50_pe", "nifty50_pb", "nifty50_dy"):
            if column in m.columns:
                emit_series(column, m[column])

    for name, source in (
        ("cs_rank_ret_21", wide["ret_21"]),
        ("cs_rank_ret_63", wide["ret_63"]),
        ("cs_rank_vol_21", wide["vol_21"]),
        ("cs_rank_deliv_per", deliv),
        ("cs_rank_turnover_ratio", wide["turnover_ratio_5_21"]),
    ):
        emit(name, source.where(member).rank(axis=1, pct=True))

    sector_map = {}
    if sectors is not None and len(sectors):
        sector_map = dict(zip(sectors["symbol"], sectors["sector"]))
    labels = np.array([sector_map.get(symbol, "Unknown") for symbol in columns], dtype=object)
    ret21_member = wide["ret_21"].where(member)
    relative = np.full((n_dates, n_cols), np.nan)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        block = ret21_member.iloc[:, idx]
        relative[:, idx] = block.sub(block.mean(axis=1), axis=0).to_numpy()
    emit("rel_ret_21_sector", pd.DataFrame(relative, index=dates, columns=columns))

    emit("price", pivot(base, "close", dates, columns))

    entry = open_.shift(-1)
    exit_price = adj.shift(-config.HORIZON)
    forward = exit_price / entry - 1.0
    emit("ret_fwd", forward)

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_cols)[flat],
            "symbol": np.tile(columns.to_numpy(dtype=object), n_dates)[flat],
            "sector": np.tile(labels, n_dates)[flat],
        }
    )
    for name, values in feats.items():
        frame[name] = values
    frame = frame.reset_index(drop=True)
    if config.TARGET_MODE == "excess":
        market_forward = frame.groupby("date")["ret_fwd"].transform("mean")
        target_reg = (frame["ret_fwd"] - market_forward).astype("float32")
        threshold = config.TARGET_EXCESS_THRESHOLD
    else:
        target_reg = frame["ret_fwd"]
        threshold = config.TARGET_THRESHOLD
    frame["target_reg"] = target_reg
    frame["target_cls"] = np.where(target_reg.isna(), np.nan, (target_reg >= threshold).astype("float32")).astype("float32")
    return frame


def update_features(base, market=None, sectors=None, full=False):
    config.ensure_dirs()
    if config.FEATURES_PATH.exists() and not full:
        existing = pd.read_parquet(config.FEATURES_PATH)
        recompute_from = existing["date"].max() - pd.Timedelta(days=config.INCREMENTAL_RECOMPUTE_DAYS)
        window_start = recompute_from - pd.Timedelta(days=config.INCREMENTAL_BUFFER_DAYS)
        window = base[base["date"] >= window_start]
        fresh = build_features(window, market, sectors, keep_from=recompute_from)
        kept = existing[existing["date"] < recompute_from]
        combined = pd.concat([kept, fresh], ignore_index=True)
    else:
        combined = build_features(base, market, sectors)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    combined.to_parquet(config.FEATURES_PATH, index=False)
    return combined


def bin_dummies(series, edges, labels, prefix):
    cut = pd.cut(series, bins=edges, labels=labels)
    return pd.get_dummies(cut, prefix=prefix, dtype="uint8")


def add_derived(df):
    df = df.copy()
    df["rel_strength_21"] = df["ret_21"] - df["ew_ret_21"]
    df["vol_adj_mom_63"] = df["ret_63"] / (df["vol_63"] * np.sqrt(63.0)).replace(0, np.nan)
    df["mom_deliv"] = df["ret_21"] * df["deliv_ratio_21"]

    parts = []
    dow = df["date"].dt.dayofweek
    month = df["date"].dt.month
    parts.append(pd.DataFrame({f"dow_{d}": (dow == d).astype("uint8") for d in range(5)}))
    parts.append(pd.DataFrame({f"month_{m}": (month == m).astype("uint8") for m in range(1, 13)}))
    clean = df["sector"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    parts.append(pd.get_dummies(clean, prefix="sector", dtype="uint8"))

    specs = [
        ("ret_21", [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf], ["crash", "down", "flat", "up", "surge"], "bin_ret21"),
        ("vol_21", [-np.inf, 0.01, 0.02, 0.03, np.inf], ["low", "mid", "high", "extreme"], "bin_vol21"),
        ("rsi_14", [-np.inf, 30, 45, 55, 70, np.inf], ["oversold", "weak", "neutral", "strong", "overbought"], "bin_rsi"),
        ("dist_52w_high", [-np.inf, -0.4, -0.2, -0.05, np.inf], ["deep", "far", "near", "at_high"], "bin_hi52"),
        ("vix", [-np.inf, 13, 18, 25, np.inf], ["calm", "normal", "elevated", "stress"], "bin_vix"),
        ("deliv_ratio_21", [-np.inf, 0.8, 1.2, np.inf], ["low", "normal", "high"], "bin_deliv"),
    ]
    for column, edges, labels, prefix in specs:
        if column in df.columns:
            parts.append(bin_dummies(df[column], edges, labels, prefix))

    if "vix" in df.columns:
        calm = np.where(df["vix"].notna(), df["vix"] <= 25, df["ew_vol_21"] <= 0.25)
    else:
        calm = (df["ew_vol_21"] <= 0.25).to_numpy()
    flags = pd.DataFrame(
        {
            "flag_above_sma200": df["dist_sma_200"] > 0,
            "flag_near_high52": df["dist_52w_high"] > -0.05,
            "flag_near_low52": df["dist_52w_low"] < 0.10,
            "flag_gap_up": df["gap"] > 0.02,
            "flag_gap_down": df["gap"] < -0.02,
            "flag_volume_spike": df["turnover_ratio_5_21"] > 2.0,
            "flag_high_delivery": df["deliv_ratio_21"] > 1.5,
            "flag_oversold": df["rsi_14"] < 30,
            "flag_regime_ok": (df["ew_dist_sma_200"] > 0).to_numpy() & calm,
        }
    ).astype("uint8")
    parts.append(flags)
    parts = [part.set_axis(df.index, axis=0) for part in parts]
    return pd.concat([df] + parts, axis=1)


def numeric_columns(df):
    return [column for column in FEATURE_DOCS if column in df.columns]


def dummy_columns(df):
    return [column for column in df.columns if column.startswith(DUMMY_PREFIXES)]


def feature_columns(df):
    return numeric_columns(df) + dummy_columns(df)


def load_modeling_frame():
    df = pd.read_parquet(config.FEATURES_PATH)
    return add_derived(df)


def feature_docs_markdown():
    lines = ["# Feature dictionary", "", "## Numeric features", "", "| Feature | Description |", "|---|---|"]
    for name, text in FEATURE_DOCS.items():
        lines.append(f"| {name} | {text} |")
    lines += ["", "## Dummy and binned features", "", "| Feature | Description |", "|---|---|"]
    for name, text in DUMMY_DOCS.items():
        lines.append(f"| {name} | {text} |")
    lines += [
        "",
        "## Targets",
        "",
        "| Target | Description |",
        "|---|---|",
        "| ret_fwd | Forward return from the next day open to the close H trading days later, split adjusted. |",
        "| target_reg | Equals ret_fwd by default. With ALGO_TARGET_MODE=excess it is ret_fwd minus the average ret_fwd of the universe on the same date. |",
        "| target_cls | 1 when target_reg is at or above the configured threshold (absolute or excess), otherwise 0. |",
        "",
    ]
    return "\n".join(lines)
