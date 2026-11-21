import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import analysis, broker, config, data_nse, data_synthetic, eda, features, metrics, models, reporting, simulation


def log(message):
    print(message, flush=True)


def load_thresholds():
    return json.loads(config.THRESHOLDS_PATH.read_text())


def step_download(start=None, end=None, workers=4):
    config.ensure_dirs()
    if start is None:
        cached = sorted(config.BHAV_DIR.glob("*.csv"))
        start = (pd.Timestamp(cached[-1].stem) - pd.Timedelta(days=3)) if cached else pd.Timestamp(config.START_DATE)
    end = end or pd.Timestamp.today().normalize()
    log(f"Downloading NSE files from {pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}")
    counts = data_nse.download_range(start, end, workers=workers)
    for key, value in sorted(counts.items()):
        log(f"  {key[0]} {key[1]}: {value}")
    if not data_nse.download_sectors():
        log("  sector list could not be downloaded, sector features fall back to Unknown")
    got = sum(v for (kind, status), v in counts.items() if kind == "bhav" and status in ("downloaded", "cached"))
    if got == 0 and not list(config.BHAV_DIR.glob("*.csv")):
        raise RuntimeError("No bhavcopy files available. Run check-sources to see which endpoint failed.")
    return counts


def step_build(source, full=False):
    config.ensure_dirs()
    if source == "synthetic":
        log("Generating synthetic data")
        base, market, sectors = data_synthetic.write_synthetic()
        frame = features.update_features(base, market, sectors, full=True)
    elif source == "download":
        log("Building base table from raw files")
        base = data_nse.update_base()
        market = data_nse.update_market()
        sectors = data_nse.load_sectors()
        log(f"  base rows {len(base):,}, market rows {len(market):,}, sector rows {len(sectors):,}")
        frame = features.update_features(base, market, sectors, full=full)
    else:
        if not config.FEATURES_PATH.exists():
            raise RuntimeError("Mode file needs " + str(config.FEATURES_PATH) + ". Copy a prebuilt features.parquet there or use the download mode.")
        frame = pd.read_parquet(config.FEATURES_PATH)
    log(f"Feature table rows {len(frame):,}, columns {frame.shape[1]}, last date {frame['date'].max().date()}")
    return frame


def step_eda():
    df = features.load_modeling_frame()
    result = eda.run_eda(df)
    log("EDA written to " + str(config.REPORT_DIR))
    return result


def step_train(fast=False):
    df = features.load_modeling_frame()
    log(f"Training on {len(df):,} rows and {len(features.feature_columns(df))} features, fast={fast}")
    result = models.train_all(df, fast=fast)
    log(result["metrics"].to_string(index=False))
    return result


def step_simulate():
    preds = pd.read_parquet(config.PREDICTIONS_PATH)
    thresholds = load_thresholds()
    with_sims = simulation.add_vector_simulations(preds, thresholds)
    vector = simulation.vector_summary(with_sims)
    vector.to_csv(config.REPORT_DIR / "vector_simulation_summary.csv", index=False)
    sim_columns = [c for c in with_sims.columns if c.startswith("sim")]
    with_sims[["date", "symbol", "split"] + sim_columns].to_parquet(config.PROCESSED_DIR / "vector_simulations.parquet", index=False)

    results, benchmarks, table = simulation.run_all_simulations(preds, thresholds)
    table.to_csv(config.REPORT_DIR / "simulation_summary.csv", index=False)
    equity = pd.DataFrame({key: value["equity"] for key, value in results.items()})
    for label, series in benchmarks.items():
        equity[label] = series.reindex(equity.index)
    equity.to_csv(config.REPORT_DIR / "equity_curves.csv")

    strategies = table[table["prediction"] != "benchmark"]
    best = strategies.sort_values("cagr", ascending=False).iloc[0]
    best_info = {key: (float(best[key]) if isinstance(best[key], (int, float, np.floating, np.integer)) else str(best[key])) for key in ("key", "cagr", "sharpe", "max_drawdown", "trades", "total_return")}
    (config.REPORT_DIR / "best_strategy.json").write_text(json.dumps(best_info, indent=2))
    results[best["key"]]["trades"].to_csv(config.REPORT_DIR / "best_strategy_trades.csv", index=False)

    primary = config.PRIMARY_PREDICTION
    curves = {k.split("__")[1]: v["equity"] for k, v in results.items() if k.startswith(primary + "__")}
    bench_curves = {label: series.reindex(next(iter(curves.values())).index).ffill() for label, series in benchmarks.items()}
    reporting.save_equity_report(curves, bench_curves)
    log(table[["key", "cagr", "sharpe", "max_drawdown", "trades", "win_rate"]].round(4).to_string(index=False))
    log("Best strategy by CAGR: " + str(best["key"]))
    return table, vector


def step_analyze():
    preds = pd.read_parquet(config.PREDICTIONS_PATH)
    tables, gaps, findings = analysis.run_analysis(preds, load_thresholds())
    log(findings)
    return tables, gaps


def latest_signals(top_n=25):
    config.ensure_dirs()
    last = pd.read_parquet(config.FEATURES_PATH, columns=["date"])["date"].max()
    day = pd.read_parquet(config.FEATURES_PATH, filters=[("date", "==", last)])
    day = features.add_derived(day).reset_index(drop=True)
    scored = models.score_frame(day)
    thresholds = load_thresholds()
    primary = config.PRIMARY_PREDICTION
    threshold = thresholds[primary]["threshold"]
    out = day[["date", "symbol", "sector", "price", "flag_regime_ok"]].join(scored)
    out = out.rename(columns={"date": "signal_date", "flag_regime_ok": "regime_ok"})
    out["above_threshold"] = out[primary] >= threshold
    out["threshold"] = threshold
    out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["stop_loss_price"] = (out["price"] * (1.0 - config.STOP_LOSS)).round(2)
    out["hold_trading_days"] = config.HORIZON
    out = out.head(top_n)
    out.to_csv(config.REPORT_DIR / "latest_signals.csv", index=False)
    return out


def step_latest(top_n=25):
    out = latest_signals(top_n)
    shown = out[["rank", "symbol", "sector", "price", config.PRIMARY_PREDICTION, "above_threshold", "regime_ok", "stop_loss_price"]].head(10)
    log(f"Latest signals for {out['signal_date'].iloc[0].date()} (entry at next open)")
    log(shown.to_string(index=False))
    return out


def write_status():
    status = {"run_at_utc": datetime.now(timezone.utc).isoformat()}
    if config.FEATURES_PATH.exists():
        dates = pd.read_parquet(config.FEATURES_PATH, columns=["date"])["date"]
        status["data_end"] = str(dates.max().date())
        status["feature_rows"] = int(len(dates))
    if config.MODEL_META_PATH.exists():
        status["model"] = json.loads(config.MODEL_META_PATH.read_text())
    latest_path = config.REPORT_DIR / "latest_signals.csv"
    if latest_path.exists():
        latest = pd.read_csv(latest_path)
        status["signals_above_threshold"] = int(latest["above_threshold"].sum())
    (config.REPORT_DIR / "status.json").write_text(json.dumps(status, indent=2))
    return status


def step_trade(live, capital, max_positions):
    path = config.REPORT_DIR / "latest_signals.csv"
    if not path.exists():
        raise RuntimeError("No latest_signals.csv. Run the latest command first.")
    signals = pd.read_csv(path)
    primary = config.PRIMARY_PREDICTION
    signals = signals[signals["above_threshold"] & (signals["regime_ok"] == 1)].copy()
    signals["proba"] = signals[primary]
    orders = broker.build_orders(signals, capital, max_positions, weight_by_proba=True)
    log(orders.to_string(index=False) if len(orders) else "No qualifying signals today")
    result = broker.KiteBroker(dry_run=not live).place(orders)
    log(result.to_string(index=False) if len(result) else "No orders sent")
    return result


def run_daily(mode):
    config.ensure_dirs()
    if mode == "download":
        step_download()
        step_build("download")
    retrain = (not models.models_exist()) or ((models.models_age_days() or 0) > config.RETRAIN_AFTER_DAYS)
    if retrain:
        log("Retraining models")
        step_train(fast=False)
        step_simulate()
        step_analyze()
        reporting.write_results_markdown()
    step_latest()
    return write_status()


def run_all(source, fast):
    if source == "download":
        step_download()
    step_build(source)
    step_eda()
    step_train(fast=fast)
    step_simulate()
    step_analyze()
    reporting.write_results_markdown()
    step_latest()
    return write_status()


def parser():
    root = argparse.ArgumentParser(prog="python -m src.pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check-sources")
    download = commands.add_parser("download")
    download.add_argument("--start")
    download.add_argument("--end")
    download.add_argument("--workers", type=int, default=4)
    build = commands.add_parser("build")
    build.add_argument("--source", choices=["download", "synthetic", "file"], default="download")
    build.add_argument("--full", action="store_true")
    commands.add_parser("eda")
    train = commands.add_parser("train")
    train.add_argument("--fast", action="store_true")
    commands.add_parser("simulate")
    commands.add_parser("analyze")
    latest = commands.add_parser("latest")
    latest.add_argument("--top", type=int, default=25)
    trade = commands.add_parser("trade")
    trade.add_argument("--live", action="store_true")
    trade.add_argument("--capital", type=float, default=config.INITIAL_CAPITAL)
    trade.add_argument("--max-positions", type=int, default=10)
    daily = commands.add_parser("run-daily")
    daily.add_argument("--mode", choices=["download", "file"], default="download")
    everything = commands.add_parser("all")
    everything.add_argument("--source", choices=["download", "synthetic", "file"], default="download")
    everything.add_argument("--fast", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command == "check-sources":
        report = data_nse.check_sources()
        for key, value in report.items():
            log(f"{key}: {value}")
        return 0 if all(str(v).startswith("ok") for v in report.values()) else 1
    if command == "download":
        step_download(args.start, args.end, args.workers)
    elif command == "build":
        step_build(args.source, args.full)
    elif command == "eda":
        step_eda()
    elif command == "train":
        step_train(args.fast)
    elif command == "simulate":
        step_simulate()
    elif command == "analyze":
        step_analyze()
    elif command == "latest":
        step_latest(args.top)
        write_status()
    elif command == "trade":
        step_trade(args.live, args.capital, args.max_positions)
    elif command == "run-daily":
        run_daily(args.mode)
    elif command == "all":
        run_all(args.source, args.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
