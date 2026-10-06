"""Single source of truth for model features (used in training AND projection: no train/serve skew).

Look-ahead control: every macro input is the value AVAILABLE at the observation month
(unemployment is publication-lagged in macro.py); labels use strictly later months.
"""
import numpy as np
import pandas as pd

FEATURES = ["s1", "s2", "s3", "s4", "score_z", "ltv0", "cltv", "dti", "self_emp", "contract",
            "prime_shock", "unemp", "du12", "probation"]


def make_X(f):
    """f: dict of equal-length arrays with keys age, score_z, ltv0, cltv, dti, emp, prime_shock, unemp, du12, msc."""
    age = np.asarray(f["age"], dtype=float)
    msc = np.asarray(f["msc"])
    cols = [np.minimum(age, 12) / 12, np.clip(age - 12, 0, 24) / 12, np.clip(age - 36, 0, 48) / 12,
            np.clip(age - 84, 0, 156) / 12, f["score_z"], f["ltv0"], np.minimum(f["cltv"], 2.0), np.minimum(f["dti"], 1.0),
            (np.asarray(f["emp"]) == 1).astype(float), (np.asarray(f["emp"]) == 2).astype(float),
            np.asarray(f["prime_shock"]) / 2.0, (np.asarray(f["unemp"]) - 30.0) / 4.0, np.asarray(f["du12"]) / 2.0,
            ((msc >= 0) & (msc <= 12)).astype(float)]
    return np.column_stack(cols).astype(np.float64)


def feature_frame(loans, panel, macro, drop_default_rows=False):
    """Loan-month feature table with labels."""
    hpi, inc = macro["hpi"].values, macro["income_idx"].values
    prime, u, du = macro["prime"].values, macro["unemp"].values, macro["unemp_chg12"].values
    li = panel["loan_idx"].values
    m = panel["m"].values.astype(int)
    om = loans["orig_m"].values[li]
    score_z = ((loans["bureau_score"].values - 670) / 75)[li]
    df = pd.DataFrame({
        "loan_idx": li, "m": m, "age": panel["age"].values, "st": panel["st"].values, "msc": panel["msc"].values,
        "score_z": score_z, "ltv0": loans["ltv0"].values[li], "emp": loans["emp_code"].values[li],
        "cltv": panel["bal"].values / (loans["property_value"].values[li] * hpi[m] / hpi[om]),
        "dti": panel["inst"].values / (loans["income0"].values[li] * inc[m] / inc[om]),
        "prime_shock": (prime[m] - macro["prime"].values[om]) , "unemp": u[m], "du12": du[m],
        "bal": panel["bal"].values, "rate": panel["rate"].values, "true_eta": panel["true_eta"].values,
        "bureau_band": loans["bureau_band"].values[li], "orig_m": om, "new_def": panel["new_def"].values,
    })
    df["prime_shock"] = df["prime_shock"] * 1.0  # percentage points (macro prime is in %)
    # ---- next-month state (NaN if loan closed or censored) ----
    nxt = panel[["loan_idx", "m", "st"]].copy(); nxt["m"] = nxt["m"].astype(int) - 1
    nxt = nxt.rename(columns={"st": "st_next"})
    df = df.merge(nxt, on=["loan_idx", "m"], how="left")
    # ---- 12-month default label (only for loans performing at t, window complete) ----
    ev = panel.loc[panel["new_def"] == 1, ["loan_idx", "m"]].astype(int)
    ev_key = np.sort(ev["loan_idx"].values.astype(np.int64) * 10000 + ev["m"].values)
    key = df["loan_idx"].values.astype(np.int64) * 10000 + df["m"].values
    pos = np.searchsorted(ev_key, key, side="right")
    pos_c = np.minimum(pos, len(ev_key) - 1)
    nd = ev_key[pos_c]
    same = (pos < len(ev_key)) & ((nd // 10000) == df["loan_idx"].values)
    gap = np.where(same, (nd % 10000) - df["m"].values, 999)
    last_m = len(macro) - 1
    df["d12"] = np.where(df["m"].values + 12 <= last_m, (gap <= 12).astype(float), np.nan)
    df.loc[df["st"] == 3, "d12"] = np.nan
    return df


def split_mask(df, macro, lo=None, hi=None):
    idx = {str(p): i for i, p in enumerate(pd.PeriodIndex(macro.index, freq="M"))}
    mm = df["m"].values
    mask = np.ones(len(df), dtype=bool)
    if lo is not None:
        mask &= mm >= idx[lo]
    if hi is not None:
        mask &= mm <= idx[hi]
    return mask
