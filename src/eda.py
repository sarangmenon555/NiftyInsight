import numpy as np
import pandas as pd

from . import config, features, reporting


def target_summary(df):
    labeled = df[df["target_reg"].notna()]
    describe = labeled["target_reg"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("target_reg")
    describe.loc["positive_rate"] = labeled["target_cls"].mean()
    by_year = labeled.groupby(labeled["date"].dt.year).agg(
        rows=("target_cls", "size"),
        positive_rate=("target_cls", "mean"),
        mean_forward_return=("target_reg", "mean"),
        median_forward_return=("target_reg", "median"),
    )
    by_month = labeled.groupby(labeled["date"].dt.month).agg(positive_rate=("target_cls", "mean"), mean_forward_return=("target_reg", "mean"))
    return describe, by_year, by_month


def correlations(df, sample=400_000):
    columns = features.numeric_columns(df)
    labeled = df[df["target_reg"].notna()]
    if len(labeled) > sample:
        labeled = labeled.sample(sample, random_state=config.RANDOM_STATE)
    rows = []
    for column in columns:
        series = labeled[column]
        if series.notna().sum() < 100 or series.std() == 0:
            continue
        rows.append(
            {
                "feature": column,
                "spearman_target_reg": float(series.corr(labeled["target_reg"], method="spearman")),
                "spearman_target_cls": float(series.corr(labeled["target_cls"], method="spearman")),
            }
        )
    table = pd.DataFrame(rows)
    table["abs_corr"] = table["spearman_target_reg"].abs()
    return table.sort_values("abs_corr", ascending=False).reset_index(drop=True), labeled


def run_eda(df):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    describe, by_year, by_month = target_summary(df)
    describe.to_csv(config.REPORT_DIR / "eda_target_summary.csv")
    by_year.to_csv(config.REPORT_DIR / "eda_target_by_year.csv")
    by_month.to_csv(config.REPORT_DIR / "eda_target_by_month.csv")
    table, labeled = correlations(df)
    table.to_csv(config.REPORT_DIR / "eda_correlations.csv", index=False)

    reporting.save_histogram(labeled["target_reg"], "eda_target_hist.png", "Forward return distribution (clipped at 1 and 99 percent)", clip=tuple(labeled["target_reg"].quantile([0.01, 0.99])))
    top = table.head(25)
    reporting.save_barh(top["feature"], top["spearman_target_reg"], "eda_top_correlations.png", "Spearman correlation with forward return")
    top_features = list(table.head(14)["feature"])
    matrix = labeled[top_features].corr(method="spearman")
    reporting.save_heatmap(matrix, "eda_correlation_heatmap.png", "Correlation between the top features")
    reporting.save_grouped_bars(by_year.reset_index(), "date", "positive_rate", "eda_positive_rate_by_year.png", "Share of observations reaching the target, by year")
    return {"target": describe, "by_year": by_year, "by_month": by_month, "correlations": table}
