"""Macro scenarios for ECL: base / upside / downside, 36 months of narrative then mean reversion.

Weights (config) are ILLUSTRATIVE. Narratives are stylised SA stories, not forecasts.
"""
import numpy as np
import pandas as pd

H_NARRATIVE = 36          # months of explicit scenario path
H_REVERT = 24             # months of linear mean reversion afterwards
LONG_RUN = {"prime": 10.50, "unemp": 32.0, "hpi_yoy": 3.5, "cpi": 4.5}

# knots: offset (months after reporting date) -> value. Offset 0 = last actual.
SCENARIOS = {
    "base": {"narrative": "Rates on hold, slow disinflation, unemployment drifts down slightly.",
             "prime": {12: 10.50, 36: 10.50}, "unemp": {36: 31.0}, "hpi_yoy": {12: 3.0, 36: 3.5}, "cpi": {12: 4.0, 36: 4.5}},
    "upside": {"narrative": "Inflation anchored at 3%, further rate cuts, jobs recovery, firmer house prices.",
               "prime": {12: 9.25, 36: 9.25}, "unemp": {36: 29.5}, "hpi_yoy": {12: 6.0, 36: 7.0}, "cpi": {12: 3.5, 36: 3.5}},
    "downside": {"narrative": "Rate-hiking cycle (+250bp) on an inflation / rand shock, unemployment +3.5pp, house prices fall ~8%.",
                 "prime": {12: 12.75, 24: 12.75, 36: 11.75}, "unemp": {12: 34.0, 24: 35.5, 36: 34.5},
                 "hpi_yoy": {12: -4.0, 24: -4.0, 36: 1.0}, "cpi": {12: 6.5, 36: 5.0}},
}


def _path(cur, knots, long_run, total):
    kn = {0: cur, **knots}
    if H_NARRATIVE not in kn:
        kn[H_NARRATIVE] = kn[max(knots)]
    kn[H_NARRATIVE + H_REVERT] = long_run
    kn[total + 1] = long_run
    xs = sorted(kn)
    return np.interp(np.arange(1, total + 1), xs, [kn[k] for k in xs])


def build_future(macro, name, total=260):
    """Future monthly frame (columns prime, unemp, hpi, income_idx, cpi_yoy, hpi_yoy) after the last actual month."""
    sc = SCENARIOS[name]
    last = macro.iloc[-1]
    cur_hpi_yoy = (macro["hpi"].iloc[-1] / macro["hpi"].iloc[-13] - 1) * 100
    prime = _path(last["prime"], sc["prime"], LONG_RUN["prime"], total)
    unemp = _path(last["unemp"], sc["unemp"], LONG_RUN["unemp"], total)
    hyoy = _path(cur_hpi_yoy, sc["hpi_yoy"], LONG_RUN["hpi_yoy"], total)
    cpi = _path(last["cpi_yoy"], sc["cpi"], LONG_RUN["cpi"], total)
    hpi = last["hpi"] * np.cumprod((1 + hyoy / 100) ** (1 / 12))
    inc = last["income_idx"] * np.cumprod((1 + (np.clip(cpi, 0, None) - 0.5) / 100) ** (1 / 12))
    idx = pd.period_range(pd.Period(str(macro.index[-1]), "M"), periods=total + 1, freq="M")[1:]
    return pd.DataFrame({"prime": prime, "unemp": unemp, "hpi": hpi, "income_idx": inc, "cpi_yoy": cpi, "hpi_yoy": hyoy}, index=idx)


def scenario_table(macro, years=(2026, 2027, 2028)):
    rows = []
    for name in SCENARIOS:
        f = build_future(macro, name)
        for y in years:
            sel = f[[p.year == y for p in f.index]]
            rows.append(dict(scenario=name, year=y, prime_end=sel["prime"].iloc[-1], unemp_end=sel["unemp"].iloc[-1],
                             hpi_yoy_end=sel["hpi_yoy"].iloc[-1], cpi_end=sel["cpi_yoy"].iloc[-1]))
    return pd.DataFrame(rows)
