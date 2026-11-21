import numpy as np
import pandas as pd
import pytest

from src import config, data_synthetic, features


@pytest.fixture(scope="module")
def built():
    base, market, sectors = data_synthetic.write_synthetic()
    frame = features.update_features(base, market, sectors, full=True)
    return base, market, sectors, frame


def test_every_documented_feature_exists(built):
    derived = features.add_derived(built[3])
    missing = [name for name in features.FEATURE_DOCS if name not in derived.columns]
    assert missing == []
    assert len(features.numeric_columns(derived)) >= 20
    assert len(features.dummy_columns(derived)) >= 20


def test_split_adjustment_removes_artificial_jumps(built):
    frame = built[3]
    assert frame["ret_1"].min() > -0.3
    assert frame["ret_1"].max() < 0.3


def test_target_matches_manual_computation(built):
    base, _, _, frame = built
    panels = features.adjusted_panels(base)
    adj = panels["close"].ffill(limit=5)
    sample = frame.dropna(subset=["target_reg"]).iloc[1000]
    assert sample["target_reg"] == pytest.approx(sample["ret_fwd"])
    dates = adj.index
    position = dates.get_loc(sample["date"])
    entry = panels["open"].iloc[position + 1][sample["symbol"]]
    exit_price = adj.iloc[position + config.HORIZON][sample["symbol"]]
    assert sample["target_reg"] == pytest.approx(exit_price / entry - 1.0, rel=1e-4)
    assert sample["target_cls"] == float(sample["target_reg"] >= config.TARGET_THRESHOLD)


def test_features_use_only_past_information(built):
    base, market, sectors, frame = built
    cutoff = frame["date"].sort_values().iloc[len(frame) // 2]
    truncated = base[base["date"] <= cutoff]
    partial = features.build_features(truncated, market[market["date"] <= cutoff], sectors)
    check = ["ret_21", "rsi_14", "vol_21", "dist_sma_50", "deliv_ratio_21"]
    left = frame[frame["date"] == cutoff].set_index("symbol")[check]
    right = partial[partial["date"] == cutoff].set_index("symbol")[check]
    common = left.index.intersection(right.index)
    assert len(common) > 10
    np.testing.assert_allclose(left.loc[common].to_numpy(), right.loc[common].to_numpy(), rtol=1e-3, atol=1e-4, equal_nan=True)


def test_incremental_update_matches_full_rebuild(built):
    base, market, sectors, full = built
    cutoff = full["date"].sort_values().iloc[int(len(full) * 0.8)]
    features.update_features(base[base["date"] <= cutoff], market, sectors, full=True)
    updated = features.update_features(base, market, sectors, full=False)
    assert len(updated) == len(full)
    merged = updated.merge(full, on=["date", "symbol"], suffixes=("_inc", "_full"))
    assert len(merged) == len(full)
    np.testing.assert_allclose(merged["ret_21_inc"].to_numpy(), merged["ret_21_full"].to_numpy(), rtol=1e-3, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(merged["target_reg_inc"].to_numpy(), merged["target_reg_full"].to_numpy(), rtol=1e-3, atol=1e-4, equal_nan=True)


def test_excess_target_is_demeaned_per_date(built, monkeypatch):
    base, market, sectors, _ = built
    monkeypatch.setattr(config, "TARGET_MODE", "excess")
    frame = features.build_features(base[base["date"] >= base["date"].max() - pd.Timedelta(days=700)], market, sectors)
    labeled = frame.dropna(subset=["target_reg"])
    daily_mean = labeled.groupby("date")["target_reg"].mean()
    assert daily_mean.abs().max() < 1e-5
    assert labeled["target_cls"].isin([0.0, 1.0]).all()
    assert (labeled["ret_fwd"] != labeled["target_reg"]).any()
