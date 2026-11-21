import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config, metrics

st.set_page_config(page_title="NSE Algo Trading Monitor", layout="wide")


@st.cache_data(ttl=300)
def read_csv(name):
    path = config.REPORT_DIR / name
    return pd.read_csv(path) if path.exists() else None


def show_image(name, caption):
    path = config.REPORT_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)


def status_panel():
    path = config.REPORT_DIR / "status.json"
    if not path.exists():
        st.warning("No status.json yet. Run the pipeline first.")
        return
    status = json.loads(path.read_text())
    columns = st.columns(4)
    columns[0].metric("Data through", status.get("data_end", "n/a"))
    columns[1].metric("Feature rows", f"{status.get('feature_rows', 0):,}")
    columns[2].metric("Signals above threshold", status.get("signals_above_threshold", "n/a"))
    model = status.get("model", {})
    columns[3].metric("Model trained", str(model.get("trained_at_utc", "n/a"))[:10])
    st.caption("Last run (UTC): " + status.get("run_at_utc", "n/a"))


st.title("NSE daily trading system monitor")
status_panel()

tab_signals, tab_perf, tab_models, tab_errors, tab_orders = st.tabs(
    ["Latest signals", "Performance", "Model quality", "Error analysis", "Orders log"]
)

with tab_signals:
    signals = read_csv("latest_signals.csv")
    if signals is None:
        st.info("latest_signals.csv not found.")
    else:
        only_active = st.checkbox("Only signals above threshold and in a healthy regime", value=True)
        view = signals
        if only_active:
            view = signals[signals["above_threshold"] & (signals["regime_ok"] == 1)]
        st.write(f"Signal date {signals['signal_date'].iloc[0]}. Entry at the next session open, hold {int(signals['hold_trading_days'].iloc[0])} trading days.")
        st.dataframe(view, use_container_width=True)

with tab_perf:
    summary = read_csv("simulation_summary.csv")
    curves = read_csv("equity_curves.csv")
    if summary is None or curves is None:
        st.info("Run the simulate step first.")
    else:
        curves = curves.rename(columns={curves.columns[0]: "date"})
        curves["date"] = pd.to_datetime(curves["date"])
        curves = curves.set_index("date")
        chosen = st.multiselect("Curves", list(curves.columns), default=[c for c in curves.columns if c.startswith("pred_ens")] + [c for c in curves.columns if c in ("Nifty 50", "Nifty 500")])
        if chosen:
            normalized = curves[chosen] / curves[chosen].iloc[0]
            st.line_chart(normalized)
            st.line_chart(pd.DataFrame({c: metrics.drawdown(curves[c]) for c in chosen}))
        st.dataframe(summary, use_container_width=True)
        best = config.REPORT_DIR / "best_strategy.json"
        if best.exists():
            st.json(json.loads(best.read_text()))
        show_image("equity_rolling_1y.png", "Rolling one year return")

with tab_models:
    metrics_table = read_csv("model_metrics.csv")
    if metrics_table is not None:
        st.dataframe(metrics_table, use_container_width=True)
    importance = read_csv("feature_importance.csv")
    if importance is not None:
        model_name = st.selectbox("Model", sorted(importance["model"].unique()), index=1)
        st.bar_chart(importance[importance["model"] == model_name].head(25).set_index("feature")["importance"])
    vector = read_csv("vector_simulation_summary.csv")
    if vector is not None:
        st.subheader("Vector simulation")
        st.dataframe(vector, use_container_width=True)

with tab_errors:
    findings = config.REPORT_DIR / "analysis_findings.md"
    if findings.exists():
        st.markdown(findings.read_text())
    for name, caption in (("analysis_precision_by_year.png", "Precision by year"), ("analysis_precision_by_sector.png", "Precision by sector")):
        show_image(name, caption)
    gaps = read_csv("analysis_false_positive_gaps.csv")
    if gaps is not None:
        st.dataframe(gaps.head(20), use_container_width=True)

with tab_orders:
    orders = read_csv("orders_log.csv")
    if orders is None:
        st.info("No orders logged yet.")
    else:
        st.dataframe(orders.tail(200), use_container_width=True)
