"""Monthly SA macro dataset, 2012-01 .. 2025-12.

Real data first: drop CSVs (columns: date,value) in data/raw/ and they are used.
    repo.csv  (SARB repo rate, %)          cpi_yoy.csv (Stats SA CPI y/y, %)
    unemployment.csv (QLFS official rate, % quarterly)   gdp_yoy.csv (real GDP y/y, %)
    hpi.csv (house price index LEVEL, any base; optional)
If a file is missing, an APPROXIMATE built-in series is used and flagged in the lineage
table. The built-in series are typed from memory: REPLACE THEM with real downloads.
Repo decisions (2012-2025) are the exception: they match published SARB MPC decisions.
"""
import numpy as np
import pandas as pd
from .utils import ROOT, load_config, month_index

REPO_CHANGES = [  # (effective date, repo %)
    ("2012-01-01", 5.50), ("2012-07-20", 5.00), ("2014-01-30", 5.50), ("2014-07-18", 5.75),
    ("2015-07-24", 6.00), ("2016-01-29", 6.25), ("2016-03-25", 7.00), ("2017-07-21", 6.75),
    ("2018-03-29", 6.50), ("2018-11-22", 6.75), ("2019-07-19", 6.50), ("2020-01-17", 6.25),
    ("2020-03-20", 5.25), ("2020-04-15", 4.25), ("2020-05-22", 3.75), ("2020-07-24", 3.50),
    ("2021-11-19", 3.75), ("2022-01-28", 4.00), ("2022-03-25", 4.25), ("2022-05-20", 4.75),
    ("2022-07-22", 5.50), ("2022-09-23", 6.25), ("2022-11-25", 7.00), ("2023-01-27", 7.25),
    ("2023-03-31", 7.75), ("2023-05-26", 8.25), ("2024-09-20", 8.00), ("2024-11-23", 7.75),
    ("2025-01-31", 7.50), ("2025-05-30", 7.25), ("2025-08-01", 7.00), ("2025-11-21", 6.75),
]
# APPROXIMATE annual-average CPI y/y and real GDP y/y (mid-year anchors) - replace with Stats SA.
CPI_ANNUAL = {2012: 5.7, 2013: 5.8, 2014: 6.1, 2015: 4.6, 2016: 6.3, 2017: 5.3, 2018: 4.6, 2019: 4.1,
              2020: 3.3, 2021: 4.5, 2022: 6.9, 2023: 6.0, 2024: 4.4, 2025: 3.2}
GDP_ANNUAL = {2012: 2.4, 2013: 2.5, 2014: 1.4, 2015: 1.3, 2016: 0.7, 2017: 1.2, 2018: 1.6, 2019: 0.3,
              2020: -6.2, 2021: 4.9, 2022: 2.1, 2023: 0.8, 2024: 0.5, 2025: 1.1}
# APPROXIMATE QLFS official unemployment rate by quarter (year, q): % - replace with Stats SA.
UNEMP_ANNUAL_PRE2020 = {2012: 24.9, 2013: 24.7, 2014: 25.1, 2015: 25.2, 2016: 26.5, 2017: 27.5,
                        2018: 27.2, 2019: 28.8}
UNEMP_Q = {(2020, 1): 30.1, (2020, 2): 23.3, (2020, 3): 30.8, (2020, 4): 32.5,
           (2021, 1): 32.6, (2021, 2): 34.4, (2021, 3): 34.9, (2021, 4): 35.3,
           (2022, 1): 34.5, (2022, 2): 33.9, (2022, 3): 33.7, (2022, 4): 32.9,
           (2023, 1): 32.9, (2023, 2): 32.6, (2023, 3): 31.9, (2023, 4): 32.1,
           (2024, 1): 32.9, (2024, 2): 33.5, (2024, 3): 33.2, (2024, 4): 31.9,
           (2025, 1): 32.9, (2025, 2): 33.2, (2025, 3): 31.9, (2025, 4): 31.8}
QLFS_PUBLICATION_LAG_MONTHS = 2   # quarter Q is first usable 2 months after quarter-end (no look-ahead)


def _read_raw(name):
    p = ROOT / "data" / "raw" / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df.groupby("date")["value"].last()


def _annual_to_monthly(d, idx):
    """Mid-year anchors, linear interpolation."""
    s = pd.Series({pd.Period(f"{y}-07", "M"): v for y, v in d.items()})
    s = s.reindex(idx.union(s.index)).astype(float).interpolate(limit_direction="both")
    return s.reindex(idx)


def build_macro(cfg=None, save=True):
    cfg = cfg or load_config()
    idx = month_index("2012-01", cfg["portfolio"]["obs_end"])
    lineage = []

    # --- repo / prime ---
    raw = _read_raw("repo.csv")
    if raw is not None:
        repo = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
        lineage.append(("repo", "data/raw/repo.csv (SARB)", "user download", "month-end, ffill"))
    else:
        ch = pd.Series({pd.Timestamp(d).to_period("M"): v for d, v in REPO_CHANGES})
        repo = ch.reindex(idx.union(ch.index)).ffill().reindex(idx)
        lineage.append(("repo", "built-in SARB MPC decision table", "typed in code", "effective-month step series; VERIFY vs SARB"))
    prime = repo + 3.5
    lineage.append(("prime", "derived", "-", "prime = repo + 3.5pp"))

    # --- CPI ---
    raw = _read_raw("cpi_yoy.csv")
    if raw is not None:
        cpi = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
        lineage.append(("cpi_yoy", "data/raw/cpi_yoy.csv (Stats SA)", "user download", "none"))
    else:
        cpi = _annual_to_monthly(CPI_ANNUAL, idx)
        lineage.append(("cpi_yoy", "built-in APPROXIMATE annual CPI", "typed in code", "interpolated; REPLACE with Stats SA P0141"))

    # --- GDP ---
    raw = _read_raw("gdp_yoy.csv")
    if raw is not None:
        gdp = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
        lineage.append(("gdp_yoy", "data/raw/gdp_yoy.csv", "user download", "ffill"))
    else:
        gdp = _annual_to_monthly(GDP_ANNUAL, idx)
        lineage.append(("gdp_yoy", "built-in APPROXIMATE annual GDP", "typed in code", "interpolated; REPLACE with Stats SA P0441"))

    # --- unemployment (quarterly -> monthly, publication-lagged) ---
    raw = _read_raw("unemployment.csv")
    if raw is not None:
        q = raw.copy()
        q.index = q.index.asfreq("M")
        lineage.append(("unemployment", "data/raw/unemployment.csv (QLFS)", "user download",
                        f"quarterly -> monthly, available {QLFS_PUBLICATION_LAG_MONTHS}m after quarter-end"))
        qdates = [(p.to_timestamp() + pd.offsets.QuarterEnd(0)).to_period("M") for p in q.index]
        qs = pd.Series(q.values, index=qdates)
    else:
        rows = {}
        for y, v in UNEMP_ANNUAL_PRE2020.items():
            for k in range(1, 5):
                rows[(y, k)] = v
        rows.update(UNEMP_Q)
        qs = pd.Series({pd.Period(f"{y}Q{k}", "Q").asfreq("M", "end"): v for (y, k), v in rows.items()})
        lineage.append(("unemployment", "built-in APPROXIMATE QLFS rates", "typed in code",
                        f"quarterly -> monthly, available {QLFS_PUBLICATION_LAG_MONTHS}m after quarter-end; REPLACE with Stats SA QLFS"))
    avail = qs.copy()
    avail.index = avail.index + QLFS_PUBLICATION_LAG_MONTHS
    unemp = avail.reindex(idx.union(avail.index)).ffill().bfill().reindex(idx)

    # --- house price index (level) ---
    raw = _read_raw("hpi.csv")
    if raw is not None:
        hpi = raw.reindex(idx.union(raw.index)).interpolate().ffill().bfill().reindex(idx)
        hpi = 100 * hpi / hpi.iloc[0]
        lineage.append(("hpi", "data/raw/hpi.csv", "user download", "rebased to 100 at 2012-01"))
    else:
        yoy = (cpi - 1.0) / 100.0   # CPI-linked assumption: real house prices fall ~1% a year
        lvl = (1 + yoy) ** (1 / 12)
        hpi = 100 * lvl.cumprod() / lvl.iloc[0]
        lineage.append(("hpi", "assumption: HPI y/y = CPI y/y - 1pp", "-", "compounded monthly; replace with FNB/ABSA/BIS index if available"))

    income_idx = (1 + (cpi.clip(lower=0) - 0.5) / 100.0) ** (1 / 12)
    income_idx = 100 * income_idx.cumprod() / income_idx.iloc[0]

    macro = pd.DataFrame({"repo": repo, "prime": prime, "cpi_yoy": cpi, "gdp_yoy": gdp,
                          "unemp": unemp, "hpi": hpi, "income_idx": income_idx}, index=idx)
    macro.index.name = "month"
    macro["unemp_chg12"] = macro["unemp"] - macro["unemp"].shift(12).bfill()
    lin = pd.DataFrame(lineage, columns=["series", "source", "pulled", "transformation"])
    lin["date_pulled"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    if save:
        out = ROOT / cfg["paths"]["processed"]
        out.mkdir(parents=True, exist_ok=True)
        macro.to_csv(out / "macro_monthly.csv")
        (ROOT / cfg["paths"]["docs"]).mkdir(exist_ok=True)
        lin.to_csv(ROOT / cfg["paths"]["docs"] / "data_lineage.csv", index=False)
    return macro, lin


if __name__ == "__main__":
    m, l = build_macro()
    print(m.tail(3)); print(l.to_string())
