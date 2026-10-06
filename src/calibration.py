"""Calibration diagnostics: simulated default / arrears rates vs documented targets."""
import numpy as np
import pandas as pd
from .utils import load_config, ROOT


def load_processed(cfg=None):
    cfg = cfg or load_config()
    p = ROOT / cfg["paths"]["processed"]
    macro = pd.read_csv(p / "macro_monthly.csv", index_col=0)
    macro.index = pd.PeriodIndex(macro.index, freq="M")
    return (pd.read_csv(p / "loans.csv"), pd.read_csv(p / "panel.csv.gz"),
            pd.read_csv(p / "defaults.csv"), macro)


def annual_stats(panel, defaults, macro):
    idx = pd.PeriodIndex(macro.index, freq="M")
    yr = np.array([idx[i].year for i in range(len(idx))])
    pn = panel.copy(); pn["year"] = yr[pn["m"]]
    perf = pn[pn["st"] < 3]
    # annual default rate = defaults entering in year / avg performing loans-months*12
    g = pn.groupby("year")
    out = pd.DataFrame({
        "loan_months_performing": perf.groupby("year").size(),
        "new_defaults": pn.groupby("year")["new_def"].sum(),
        "arrears_30plus_share": g.apply(lambda x: (x["st"] >= 1).mean(), include_groups=False),
        "default_stock_share": g.apply(lambda x: (x["st"] == 3).mean(), include_groups=False),
    })
    out["annual_default_rate"] = out["new_defaults"] / out["loan_months_performing"] * 12
    return out


if __name__ == "__main__":
    loans, panel, defaults, macro = load_processed()
    print(annual_stats(panel, defaults, macro).round(4))
    d = defaults
    print(d["outcome"].value_counts(normalize=True).round(3))
    print("mean realised LGD (sold):", d["realised_lgd"].mean().round(3))
