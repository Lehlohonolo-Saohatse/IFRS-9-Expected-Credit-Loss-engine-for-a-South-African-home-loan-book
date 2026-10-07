"""Monitoring dashboard."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"
st.set_page_config(page_title="SA home-loan IFRS 9 ECL monitor", layout="wide")
st.title("SA home-loan IFRS 9 ECL monitor")
st.warning("SYNTHETIC portfolio: results show methodology, not real South African credit risk. By Lehlohonolo Saohatse")

res = json.loads((OUT / "results.json").read_text())
loans = pd.read_csv(OUT / "ecl_by_loan.csv.gz")
rd = lambda n: pd.read_csv(OUT / n)

st.sidebar.header("Scenario weights (illustrative)")
wb = st.sidebar.slider("Base", 0, 100, 50); wu = st.sidebar.slider("Upside", 0, 100 - wb, min(20, 100 - wb))
wd = 100 - wb - wu
st.sidebar.write(f"Downside = {wd}%")
view = st.sidebar.selectbox("View", ["Weighted", "base", "upside", "downside"])
w = dict(base=wb / 100, upside=wu / 100, downside=wd / 100)
loans["ecl_w"] = sum(w[k] * loans[f"ecl_{k}"] for k in w)
col = "ecl_w" if view == "Weighted" else f"ecl_{view}"
loans["ecl_sel"] = loans[col] + (loans["overlay"] if view == "Weighted" else 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reporting date", res["reporting_date"]); c2.metric("Gross exposure (ZAR bn)", f"{loans.bal.sum()/1e9:.1f}")
c3.metric("ECL (ZAR m)", f"{loans.ecl_sel.sum()/1e6:,.0f}"); c4.metric("Coverage", f"{loans.ecl_sel.sum()/loans.bal.sum()*100:.2f}%")

t1, t2, t3, t4, t5 = st.tabs(["ECL by stage", "Stage migration", "Predicted vs actual", "Scenarios & sensitivity", "Validation"])
with t1:
    g = loans.groupby("stage").agg(loans=("loan_idx", "size"), ead=("bal", "sum"), ecl=("ecl_sel", "sum"))
    g["coverage_%"] = g.ecl / g.ead * 100
    st.dataframe(g.style.format({"ead": "{:,.0f}", "ecl": "{:,.0f}", "coverage_%": "{:.2f}"}))
    st.bar_chart(g["ecl"] / 1e6)
    a, b = st.columns(2)
    loans["ltv_band"] = pd.cut(loans.ltv0, [0, .7, .8, .9, 1.01], labels=["<=70%", "70-80%", "80-90%", "90-100%"])
    x = loans.groupby("ltv_band", observed=True).apply(lambda d: d.ecl_sel.sum() / d.bal.sum() * 100, include_groups=False)
    a.subheader("Coverage % by LTV at origination"); a.bar_chart(x)
    y = loans.groupby("bureau_band").apply(lambda d: d.ecl_sel.sum() / d.bal.sum() * 100, include_groups=False)
    b.subheader("Coverage % by bureau band (A best)"); b.bar_chart(y)
with t2:
    st.dataframe(rd("stage_migration_pct.csv").set_index("from").style.format("{:.1%}").background_gradient(cmap="Blues"))
    st.caption("Dec-2024 stage (rows) to Dec-2025 stage (columns). 'Exited' = prepaid, matured or resolved.")
with t3:
    st.subheader("12m PD: predicted vs observed (out-of-time)")
    cy = rd("validation_calibration_year.csv").set_index("obs_year")[["predicted", "observed"]]; st.line_chart(cy)
    st.subheader("By bureau band"); st.bar_chart(rd("validation_calibration_band.csv").set_index("bureau_band")[["predicted", "observed"]])
    st.subheader("6-month ECL backtest by vintage"); st.dataframe(rd("validation_ecl_backtest.csv"))
    st.subheader("LGD backtest by cLTV band"); st.dataframe(rd("validation_lgd_backtest.csv"))
with t4:
    st.subheader("Scenario paths"); st.dataframe(rd("scenario_paths.csv"))
    st.subheader("ECL by scenario (ZAR)"); st.dataframe(rd("ecl_by_scenario.csv"))
    st.subheader("Scenario-weight sensitivity"); st.dataframe(rd("sensitivity_scenario_weights.csv"))
    st.subheader("Staging thresholds sensitivity"); st.dataframe(rd("sensitivity_staging.csv"))
with t5:
    st.dataframe(rd("validation_discrimination.csv")); st.dataframe(rd("validation_stability.csv"))
    st.subheader("Recovery check: fitted vs true DGP coefficients"); st.dataframe(rd("validation_recovery_coefficients.csv"))
