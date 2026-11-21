import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, metrics

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def ensure():
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_histogram(series, name, title, bins=80, clip=None):
    ensure()
    values = series.dropna()
    if clip is not None:
        values = values.clip(*clip)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(values, bins=bins, color=PALETTE[0], alpha=0.85)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(config.REPORT_DIR / name, dpi=130)
    plt.close(figure)


def save_barh(labels, values, name, title, top=25):
    ensure()
    pairs = sorted(zip(labels, values), key=lambda item: abs(item[1]), reverse=True)[:top][::-1]
    figure, axis = plt.subplots(figsize=(8, max(3, 0.28 * len(pairs) + 1)))
    axis.barh([p[0] for p in pairs], [p[1] for p in pairs], color=PALETTE[0])
    axis.set_title(title)
    axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    figure.savefig(config.REPORT_DIR / name, dpi=130)
    plt.close(figure)


def save_heatmap(matrix, name, title):
    ensure()
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(matrix.index, fontsize=7)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.04)
    figure.tight_layout()
    figure.savefig(config.REPORT_DIR / name, dpi=130)
    plt.close(figure)


def save_lines(curves, name, title, log=False, ylabel=""):
    ensure()
    figure, axis = plt.subplots(figsize=(10, 5))
    for i, (label, series) in enumerate(curves.items()):
        axis.plot(series.index, series.values, label=label, color=PALETTE[i % len(PALETTE)], linewidth=1.4)
    if log:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(config.REPORT_DIR / name, dpi=130)
    plt.close(figure)


def save_equity_report(strategy_curves, benchmark_curves, prefix="equity"):
    combined = {**strategy_curves, **benchmark_curves}
    save_lines(combined, f"{prefix}_curves.png", "Equity curves on the test period", log=True, ylabel="Capital")
    save_lines({k: metrics.drawdown(v) for k, v in combined.items()}, f"{prefix}_drawdown.png", "Drawdown", ylabel="Drawdown")
    save_lines(
        {k: metrics.rolling_returns(v).dropna() for k, v in combined.items()},
        f"{prefix}_rolling_1y.png",
        "Rolling 1 year return",
        ylabel="Return",
    )


def frame_to_markdown(frame, digits=4):
    if frame is None or len(frame) == 0:
        return "No data.\n"
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if np.isnan(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_results_markdown():
    ensure()
    parts = ["# Results", ""]
    sections = [
        ("Model quality", "model_metrics.csv"),
        ("Vector simulation (approximate)", "vector_simulation_summary.csv"),
        ("Exact simulation versus benchmarks", "simulation_summary.csv"),
    ]
    for title, name in sections:
        path = config.REPORT_DIR / name
        if path.exists():
            frame = pd.read_csv(path)
            if name == "simulation_summary.csv":
                keep = [c for c in ("key", "total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "trades", "win_rate", "avg_exposure") if c in frame.columns]
                frame = frame[keep]
            parts += [f"## {title}", "", frame_to_markdown(frame)]
    findings = config.REPORT_DIR / "analysis_findings.md"
    if findings.exists():
        parts += ["## Error analysis", "", findings.read_text()]
    (config.REPORT_DIR / "results.md").write_text("\n".join(parts))


def save_grouped_bars(table, label_column, value_column, name, title, baseline_column=None):
    ensure()
    figure, axis = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(table))
    axis.bar(positions, table[value_column], color=PALETTE[0], alpha=0.85, label=value_column)
    if baseline_column is not None:
        axis.plot(positions, table[baseline_column], color=PALETTE[1], marker="o", label=baseline_column)
    axis.set_xticks(positions)
    axis.set_xticklabels(table[label_column].astype(str), rotation=45, ha="right", fontsize=8)
    axis.set_title(title)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(config.REPORT_DIR / name, dpi=130)
    plt.close(figure)
