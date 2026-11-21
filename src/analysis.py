import numpy as np
import pandas as pd

from . import config, features, reporting


def precision_table(df, group_column, column, threshold):
    selected = df[df[column] >= threshold]
    grouped = selected.groupby(group_column, observed=True).agg(
        signals=("target_cls", "size"),
        precision=("target_cls", "mean"),
        avg_forward_return=("ret_fwd", "mean"),
    )
    base = df.groupby(group_column, observed=True).agg(observations=("target_cls", "size"), base_rate=("target_cls", "mean"))
    table = base.join(grouped, how="left").reset_index()
    table["lift"] = table["precision"] / table["base_rate"]
    table["avg_net_return"] = table["avg_forward_return"] - 2.0 * config.COST_PER_SIDE
    return table


def false_positive_gaps(selected, feature_frame):
    feature_frame = feature_frame.drop(columns=[c for c in ("target_cls", "target_reg", "ret_fwd", "sector") if c in feature_frame.columns])
    merged = selected[["date", "symbol", "target_cls"]].merge(feature_frame, on=["date", "symbol"], how="left")
    columns = features.numeric_columns(merged)
    wrong = merged[merged["target_cls"] == 0]
    right = merged[merged["target_cls"] == 1]
    rows = []
    for column in columns:
        spread = merged[column].std()
        if not spread or np.isnan(spread):
            continue
        rows.append(
            {
                "feature": column,
                "mean_false_positive": float(wrong[column].mean()),
                "mean_true_positive": float(right[column].mean()),
                "standardized_gap": float((wrong[column].mean() - right[column].mean()) / spread),
            }
        )
    table = pd.DataFrame(rows)
    table["abs_gap"] = table["standardized_gap"].abs()
    return table.sort_values("abs_gap", ascending=False).reset_index(drop=True)


def run_analysis(preds, thresholds, column=None):
    column = column or config.PRIMARY_PREDICTION
    threshold = thresholds[column]["threshold"]
    test = preds[preds["split"] == "test"].copy()
    test["year"] = test["date"].dt.year
    test["month"] = test["date"].dt.month
    test["volatility_bucket"] = pd.qcut(test["vol_21"], 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    test["liquidity_bucket"] = pd.qcut(test["log_turnover"], 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    test["regime"] = np.where(test["flag_regime_ok"] == 1, "healthy", "weak")

    tables = {}
    for group in ("sector", "year", "month", "volatility_bucket", "liquidity_bucket", "regime"):
        table = precision_table(test, group, column, threshold)
        tables[group] = table
        table.to_csv(config.REPORT_DIR / f"analysis_precision_by_{group}.csv", index=False)

    reporting.save_grouped_bars(tables["year"], "year", "precision", "analysis_precision_by_year.png", "Precision of signals by year", "base_rate")
    sectors = tables["sector"].dropna(subset=["precision"]).sort_values("precision", ascending=False)
    reporting.save_grouped_bars(sectors, "sector", "precision", "analysis_precision_by_sector.png", "Precision of signals by sector", "base_rate")

    selected = test[test[column] >= threshold]
    wrong = selected[selected["target_cls"] == 0].sort_values(column, ascending=False)
    wrong.head(100)[["date", "symbol", "sector", column, "target_reg", "ret_fwd", "vol_21"]].to_csv(config.REPORT_DIR / "analysis_worst_false_positives.csv", index=False)

    gaps = pd.DataFrame()
    if config.FEATURES_PATH.exists() and len(selected):
        feature_frame = pd.read_parquet(config.FEATURES_PATH, filters=[("date", ">=", test["date"].min())])
        gaps = false_positive_gaps(selected, feature_frame)
        gaps.to_csv(config.REPORT_DIR / "analysis_false_positive_gaps.csv", index=False)

    findings = write_findings(tables, gaps, selected, column, threshold)
    (config.REPORT_DIR / "analysis_findings.md").write_text(findings)
    return tables, gaps, findings


def write_findings(tables, gaps, selected, column, threshold):
    lines = [f"# Findings for {column} at threshold {threshold:.4f}", ""]
    if len(selected):
        lines.append(f"Signals on the test period: {len(selected)}. Precision {selected['target_cls'].mean():.3f}. Average forward return {selected['ret_fwd'].mean():.4f}.")
    for name in ("year", "regime", "volatility_bucket", "liquidity_bucket"):
        table = tables[name].dropna(subset=["precision"])
        if len(table) == 0:
            continue
        ranked = table.sort_values("lift", ascending=False)
        best_label, worst_label = ranked[name].iloc[0], ranked[name].iloc[-1]
        lines.append(f"By {name}: best lift {ranked['lift'].iloc[0]:.2f} at {best_label}, worst lift {ranked['lift'].iloc[-1]:.2f} at {worst_label}.")
    sector = tables["sector"].dropna(subset=["precision"])
    sector = sector[sector["signals"] >= 50]
    if len(sector):
        best = sector.sort_values("lift", ascending=False).iloc[0]
        worst = sector.sort_values("lift").iloc[0]
        lines.append(f"By sector (50 or more signals): best lift {best['lift']:.2f} at {best['sector']}, worst lift {worst['lift']:.2f} at {worst['sector']}.")
    if len(gaps):
        lines.append("")
        lines.append("Features that differ most between false positives and true positives:")
        for _, row in gaps.head(8).iterrows():
            lines.append(f"- {row['feature']}: standardized gap {row['standardized_gap']:.2f}")
    return "\n".join(lines) + "\n"
