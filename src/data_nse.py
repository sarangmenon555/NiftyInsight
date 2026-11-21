import datetime as dt
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import config

HOME_URL = "https://www.nseindia.com"
BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{stamp}.csv"
INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{stamp}.csv"
SECTOR_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": HOME_URL,
}

BASE_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover",
    "trades",
    "deliv_qty",
    "deliv_per",
]
BHAV_RENAME = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "PREV_CLOSE": "prev_close",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "CLOSE_PRICE": "close",
    "TTL_TRD_QNTY": "volume",
    "TURNOVER_LACS": "turnover_lacs",
    "NO_OF_TRADES": "trades",
    "DELIV_QTY": "deliv_qty",
    "DELIV_PER": "deliv_per",
}
NUMERIC_BASE = ["open", "high", "low", "close", "prev_close", "volume", "turnover_lacs", "trades", "deliv_qty", "deliv_per"]

INDEX_KEYS = {
    "nifty 50": "nifty50",
    "nifty 500": "nifty500",
    "india vix": "vix",
    "nifty bank": "bank",
    "nifty midcap 100": "midcap",
    "nifty smallcap 100": "smallcap",
}
INDEX_RENAME = {
    "closing index value": "close",
    "p/e": "pe",
    "p/b": "pb",
    "div yield": "dy",
}

_local = threading.local()


def get_session():
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            session.get(HOME_URL, timeout=20)
        except requests.RequestException:
            pass
        _local.session = session
    return session


def reset_session():
    _local.session = None


def fetch_file(url, path, day, retries=4):
    marker = path.with_suffix(".missing")
    if path.exists():
        return "cached"
    if marker.exists():
        return "missing"
    for attempt in range(retries):
        try:
            response = get_session().get(url, timeout=40)
        except requests.RequestException:
            reset_session()
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code == 200 and len(response.content) > 200:
            path.write_bytes(response.content)
            return "downloaded"
        if response.status_code == 404:
            if day < pd.Timestamp.today().normalize() - pd.Timedelta(days=7):
                marker.touch()
                return "missing"
            return "pending"
        reset_session()
        time.sleep(1.5 * (attempt + 1))
    return "failed"


def business_days(start, end):
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))


def download_range(start, end, workers=4, include_indices=True, pause=0.15):
    config.ensure_dirs()
    tasks = []
    for day in business_days(start, end):
        stamp = f"{day:%d%m%Y}"
        tasks.append((BHAV_URL.format(stamp=stamp), config.BHAV_DIR / f"{day:%Y%m%d}.csv", day, "bhav"))
        if include_indices:
            tasks.append((INDEX_URL.format(stamp=stamp), config.INDEX_DIR / f"{day:%Y%m%d}.csv", day, "index"))

    def run(task):
        url, path, day, kind = task
        status = fetch_file(url, path, day)
        if status == "downloaded":
            time.sleep(pause)
        return kind, status

    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind, status in pool.map(run, tasks):
            counts[(kind, status)] += 1
    return counts


def read_bhav_file(path):
    frame = pd.read_csv(path, skipinitialspace=True, dtype=str)
    frame.columns = [column.strip().upper() for column in frame.columns]
    frame = frame.rename(columns=BHAV_RENAME)
    frame["series"] = frame["series"].str.strip()
    frame = frame[frame["series"] == "EQ"].copy()
    frame["symbol"] = frame["symbol"].str.strip()
    frame["date"] = pd.Timestamp(path.stem)
    for column in NUMERIC_BASE:
        frame[column] = pd.to_numeric(frame[column].str.strip(), errors="coerce")
    frame["turnover"] = frame["turnover_lacs"] * 1e5
    frame = frame[BASE_COLUMNS]
    float_columns = [c for c in BASE_COLUMNS if c not in ("date", "symbol")]
    frame[float_columns] = frame[float_columns].astype("float32")
    return frame


def update_base():
    config.ensure_dirs()
    existing = pd.read_parquet(config.BASE_PATH) if config.BASE_PATH.exists() else None
    last = existing["date"].max() if existing is not None else None
    files = sorted(config.BHAV_DIR.glob("*.csv"))
    if last is not None:
        files = [path for path in files if pd.Timestamp(path.stem) > last]
    frames = []
    for path in files:
        try:
            frames.append(read_bhav_file(path))
        except (KeyError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    if not frames and existing is None:
        raise RuntimeError("No usable bhavcopy files found in " + str(config.BHAV_DIR))
    parts = ([existing] if existing is not None else []) + frames
    base = pd.concat(parts, ignore_index=True)
    base = base.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    base.to_parquet(config.BASE_PATH, index=False)
    return base


def read_index_file(path):
    frame = pd.read_csv(path, skipinitialspace=True, dtype=str)
    frame.columns = [column.strip().lower() for column in frame.columns]
    name_column = next((c for c in frame.columns if c in ("index name", "index")), None)
    if name_column is None:
        raise ValueError("index name column missing")
    frame["key"] = frame[name_column].str.strip().str.lower().map(INDEX_KEYS)
    frame = frame.dropna(subset=["key"]).rename(columns=INDEX_RENAME)
    row = {"date": pd.Timestamp(path.stem)}
    for _, item in frame.iterrows():
        key = item["key"]
        for column in ("close", "pe", "pb", "dy"):
            if column in frame.columns:
                row[f"{key}_{column}"] = pd.to_numeric(item[column], errors="coerce")
    return row


def update_market():
    config.ensure_dirs()
    existing = pd.read_parquet(config.MARKET_PATH) if config.MARKET_PATH.exists() else None
    last = existing["date"].max() if existing is not None else None
    files = sorted(config.INDEX_DIR.glob("*.csv"))
    if last is not None:
        files = [path for path in files if pd.Timestamp(path.stem) > last]
    rows = []
    for path in files:
        try:
            rows.append(read_index_file(path))
        except (KeyError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    parts = ([existing] if existing is not None else []) + ([pd.DataFrame(rows)] if rows else [])
    if not parts:
        return pd.DataFrame()
    market = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    market.to_parquet(config.MARKET_PATH, index=False)
    return market


def download_sectors():
    config.ensure_dirs()
    if config.SECTOR_PATH.exists():
        return True
    for url in SECTOR_URLS:
        try:
            response = get_session().get(url, timeout=30)
        except requests.RequestException:
            continue
        if response.status_code == 200 and len(response.content) > 500:
            config.SECTOR_PATH.write_bytes(response.content)
            return True
    return False


def load_sectors():
    if not config.SECTOR_PATH.exists():
        return pd.DataFrame(columns=["symbol", "sector"])
    frame = pd.read_csv(config.SECTOR_PATH, dtype=str)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "symbol" in frame.columns and "industry" in frame.columns:
        frame = frame.rename(columns={"industry": "sector"})
    if "symbol" not in frame.columns or "sector" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "sector"])
    frame["symbol"] = frame["symbol"].str.strip()
    frame["sector"] = frame["sector"].str.strip()
    return frame[["symbol", "sector"]].drop_duplicates("symbol")


def check_sources():
    session = get_session()
    today = pd.Timestamp.today().normalize()
    days = [d for d in pd.bdate_range(today - pd.Timedelta(days=12), today)][::-1]
    report = {}
    for label, template in (("bhavcopy", BHAV_URL), ("index_close", INDEX_URL)):
        report[label] = "no response"
        for day in days:
            url = template.format(stamp=f"{day:%d%m%Y}")
            try:
                response = session.get(url, timeout=30)
            except requests.RequestException as error:
                report[label] = f"network error: {type(error).__name__}"
                break
            if response.status_code == 200 and len(response.content) > 200:
                head = response.text.splitlines()[0][:160]
                report[label] = f"ok {day:%Y-%m-%d} header: {head}"
                break
            report[label] = f"http {response.status_code}"
    report["sector_list"] = "unreachable"
    for url in SECTOR_URLS:
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            report["sector_list"] = "ok " + url
            break
    return report
