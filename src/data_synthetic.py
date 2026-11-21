import numpy as np
import pandas as pd

from . import config

SECTOR_NAMES = [
    "Financial Services",
    "Information Technology",
    "Healthcare",
    "Automobile",
    "Fast Moving Consumer Goods",
    "Capital Goods",
    "Metals and Mining",
    "Oil Gas and Consumable Fuels",
    "Chemicals",
    "Realty",
    "Construction",
    "Services",
]


def generate(n_symbols=150, start=None, end=None, seed=7):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start or config.START_DATE)
    end = pd.Timestamp(end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)))
    days = pd.bdate_range(start, end)
    days = days[rng.random(len(days)) > 0.035]
    steps = len(days)
    count = n_symbols

    phase = np.arange(steps) / 160.0
    market_vol = 0.007 + 0.005 * np.sin(phase) ** 2
    stress = np.zeros(steps)
    for _ in range(max(1, steps // 500)):
        center = rng.integers(0, steps)
        width = rng.integers(20, 60)
        lo, hi = max(0, center - width), min(steps, center + width)
        stress[lo:hi] += 0.012
    market_vol = market_vol + stress
    market_ret = rng.normal(0.0004, market_vol)

    sector_ids = rng.integers(0, len(SECTOR_NAMES), count)
    sector_ret = rng.normal(0.0, 0.005, (steps, len(SECTOR_NAMES)))
    beta = rng.uniform(0.6, 1.4, count)

    latent = np.zeros((steps, count))
    shocks = rng.normal(0.0, 0.0005, (steps, count))
    for t in range(1, steps):
        latent[t] = 0.97 * latent[t - 1] + shocks[t]
    idio = rng.standard_t(4, (steps, count)) * 0.011
    ret = beta * market_ret[:, None] + sector_ret[:, sector_ids] + latent + idio
    ret = np.clip(ret, -0.2, 0.2)

    start_price = rng.uniform(40, 3000, count)
    adj_close = start_price * np.cumprod(1.0 + ret, axis=0)
    prev_adj = np.vstack([start_price[None, :], adj_close[:-1]])

    split_factor = np.ones((steps, count))
    for _ in range(max(3, count // 15)):
        stock = rng.integers(0, count)
        day = rng.integers(steps // 5, steps - 5)
        split_factor[day:, stock] *= 2.0

    close = adj_close / split_factor
    prev_close = prev_adj / split_factor
    overnight = 0.35 * ret + rng.normal(0.0, 0.004, (steps, count))
    open_ = prev_adj * (1.0 + overnight) / split_factor
    spread_hi = np.abs(rng.normal(0.0, 0.006, (steps, count)))
    spread_lo = np.abs(rng.normal(0.0, 0.006, (steps, count)))
    high = np.maximum(open_, close) * (1.0 + spread_hi)
    low = np.minimum(open_, close) * (1.0 - spread_lo)

    size = np.exp(rng.normal(19.0, 1.6, count))
    turnover = size * np.exp(rng.normal(0.0, 0.5, (steps, count))) * (1.0 + 25.0 * np.abs(ret))
    volume = turnover / close
    trades = turnover / np.exp(rng.normal(10.5, 0.4, (steps, count)))
    deliv_per = np.clip(45.0 + 2500.0 * latent + rng.normal(0.0, 8.0, (steps, count)), 5.0, 95.0)
    deliv_qty = volume * deliv_per / 100.0

    exists = np.ones((steps, count), dtype=bool)
    late = rng.random(count) < 0.3
    first = np.where(late, rng.integers(0, int(steps * 0.4), count), 0)
    early_exit = rng.random(count) < 0.1
    last = np.where(early_exit, rng.integers(int(steps * 0.6), steps, count), steps)
    index = np.arange(steps)[:, None]
    exists = (index >= first[None, :]) & (index < last[None, :])

    symbols = np.array([f"SYN{i:03d}" for i in range(count)], dtype=object)
    date_grid = np.repeat(days.values, count).reshape(steps, count)
    symbol_grid = np.tile(symbols, steps).reshape(steps, count)
    flat = exists.ravel()

    def pick(array):
        return array.ravel()[flat]

    base = pd.DataFrame(
        {
            "date": pd.to_datetime(pick(date_grid)),
            "symbol": pick(symbol_grid),
            "open": pick(open_).astype("float32"),
            "high": pick(high).astype("float32"),
            "low": pick(low).astype("float32"),
            "close": pick(close).astype("float32"),
            "prev_close": pick(prev_close).astype("float32"),
            "volume": pick(volume).astype("float32"),
            "turnover": pick(turnover).astype("float32"),
            "trades": pick(trades).astype("float32"),
            "deliv_qty": pick(deliv_qty).astype("float32"),
            "deliv_per": pick(deliv_per).astype("float32"),
        }
    )

    nifty50 = 10000.0 * np.cumprod(1.0 + market_ret)
    nifty500 = 8000.0 * np.cumprod(1.0 + market_ret * 1.05)
    midcap = 12000.0 * np.cumprod(1.0 + market_ret * 1.2 + rng.normal(0, 0.002, steps))
    vix = 100.0 * market_vol * np.sqrt(252.0) * (1.0 + rng.normal(0.0, 0.05, steps))
    pe = 20.0 + np.cumsum(rng.normal(0.0, 0.08, steps))
    pe = np.clip(pe, 12.0, 32.0)
    market = pd.DataFrame(
        {
            "date": days,
            "nifty50_close": nifty50,
            "nifty500_close": nifty500,
            "midcap_close": midcap,
            "vix_close": vix,
            "nifty50_pe": pe,
            "nifty50_pb": pe / 4.5,
            "nifty50_dy": 1.3 + rng.normal(0.0, 0.02, steps),
        }
    )
    sectors = pd.DataFrame({"symbol": symbols, "sector": [SECTOR_NAMES[i] for i in sector_ids]})
    return base, market, sectors


def write_synthetic(n_symbols=None, seed=7):
    n_symbols = n_symbols or config.SYNTHETIC_SYMBOLS
    config.ensure_dirs()
    if config.BASE_PATH.exists() and not config.SYNTHETIC_MARKER.exists():
        raise RuntimeError(
            "Refusing to overwrite real data in "
            + str(config.DATA_DIR)
            + ". Set ALGO_DATA_DIR to a separate directory for synthetic runs."
        )
    base, market, sectors = generate(n_symbols=n_symbols, seed=seed)
    base.to_parquet(config.BASE_PATH, index=False)
    market.to_parquet(config.MARKET_PATH, index=False)
    sectors.to_csv(config.SECTOR_PATH, index=False)
    config.SYNTHETIC_MARKER.write_text("synthetic\n")
    for stale in (config.FEATURES_PATH, config.PREDICTIONS_PATH):
        if stale.exists():
            stale.unlink()
    return base, market, sectors
