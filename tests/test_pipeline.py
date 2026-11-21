import json

import pandas as pd

from src import config, pipeline


def test_end_to_end_synthetic():
    exit_code = pipeline.main(["all", "--source", "synthetic", "--fast"])
    assert exit_code == 0
    for name in ("latest_signals.csv", "simulation_summary.csv", "model_metrics.csv", "results.md", "status.json", "equity_curves.png"):
        assert (config.REPORT_DIR / name).exists(), name
    summary = pd.read_csv(config.REPORT_DIR / "simulation_summary.csv")
    assert (summary["prediction"] == "benchmark").any()
    assert summary.loc[summary["prediction"] != "benchmark", "trades"].sum() > 0
    thresholds = json.loads(config.THRESHOLDS_PATH.read_text())
    assert set(thresholds) == set(config.PREDICTION_COLUMNS)


def test_file_mode_and_trade_dry_run():
    assert pipeline.main(["build", "--source", "file"]) == 0
    assert pipeline.main(["latest", "--top", "5"]) == 0
    assert pipeline.main(["trade"]) == 0
    path = config.REPORT_DIR / "orders_log.csv"
    if path.exists():
        assert set(pd.read_csv(path)["mode"]) == {"dry_run"}


def test_order_builder_sizes_positions():
    from src import broker

    signals = pd.DataFrame(
        {"symbol": ["A", "B"], "price": [100.0, 2500.0], "stop_loss_price": [92.0, 2300.0], "proba": [0.5, 0.5]}
    )
    orders = broker.build_orders(signals, capital=100000.0, max_positions=5)
    assert list(orders["quantity"]) == [200, 8]
    log = broker.KiteBroker(dry_run=True).place(orders)
    assert set(log["status"]) == {"dry_run_only"}


def test_live_trading_requires_explicit_enable(monkeypatch):
    import pytest

    from src import broker

    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    with pytest.raises(RuntimeError):
        broker.KiteBroker(dry_run=False)
