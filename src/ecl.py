"""ECL = sum_j  P(default in month j) x LGD_j x EAD_j x DF_j   (DF at the loan's EIR), per scenario, then weighted.

Stage 1: 12-month horizon. Stage 2 and cure-probation: lifetime (remaining contractual term).
Defaulted (90+dpd): PD = 1, ECL = EAD x LGD x DF(assumed months to sale).
"""
import numpy as np
import pandas as pd
from .projection import MacroPath, project, snapshot
from .scenarios import SCENARIOS, build_future
from .staging import assign_stage
from .utils import discount_factor


def pd12_at_origination(model, loans, macro, path, smm=0.004):
    out = np.zeros(len(loans))
    for om, g in loans.groupby("orig_m"):
        sn = pd.DataFrame({
            "loan_idx": g["loan_idx"].values, "age": 0, "bal": g["loan_amount"].values.astype(float), "st": 0, "msc": -1,
            "score_z": ((g["bureau_score"].values - 670) / 75), "ltv0": g["ltv0"].values, "emp": g["emp_code"].values,
            "spread": g["spread"].values, "prop0": g["property_value"].values, "income0": g["income0"].values,
            "orig_m": g["orig_m"].values, "dti": g["dti0"].values})
        out[g["loan_idx"].values] = project(model, sn, path, int(om), macro, horizon=12, flat=True, smm=smm)["pd12"]
    return out


def current_pd12(model, snap, path, m0, macro, smm=0.004):
    """12m PD with macro held flat at the snapshot date (scenario-independent; used for SICR)."""
    return project(model, snap.assign(st=np.minimum(snap["st"], 2)), path, m0, macro, horizon=12, flat=True, smm=smm)["pd12"]


def months_in_default(defaults, loan_idx, m0):
    d = defaults[defaults["def_m"] <= m0].sort_values("def_m").groupby("loan_idx")["def_m"].last()
    return (m0 - pd.Series(loan_idx).map(d)).fillna(0).values


def stage_snapshot(model, loans, ff, macro, cfg, m0, pd12_orig):
    """Snapshot at month m0 with PDs and IFRS 9 stage."""
    snap = snapshot(loans, ff, m0)
    snap["pd12_orig"] = pd12_orig[snap["loan_idx"].values]
    path = MacroPath.from_history(macro)
    snap["pd12_now"] = current_pd12(model, snap, path, m0, macro, cfg["ecl"]["prepay_smm"])
    snap["stage"], snap["stage_reason"] = assign_stage(snap["st"].values, snap["msc"].values, snap["pd12_now"].values,
                                                       snap["pd12_orig"].values, cfg["staging"])
    return snap


def ecl_scenario(model, lgd, snap, path, m0, defaults, cfg_ecl, cure_window=9):
    """Loan-level 12m and lifetime ECL under one macro path (staging picks which applies)."""
    n = len(snap)
    e12, elife = np.zeros(n), np.zeros(n)
    pd12, pdl, lgd_avg = np.zeros(n), np.zeros(n), np.zeros(n)
    dflt = snap["st"].values >= 3
    perf = ~dflt
    if perf.any():
        sp = snap[perf].reset_index(drop=True)
        r = project(model, sp, path, m0, None, horizon=cfg_ecl["max_horizon"], flat=False, smm=cfg_ecl["prepay_smm"])
        H = r["mass"].shape[1]
        t = m0 + 1 + np.arange(H)[None, :]
        om = sp["orig_m"].values.astype(int)[:, None]
        cltv = r["ead"] / (sp["prop0"].values[:, None] * path.hpi[t] / path.hpi[om])
        l = lgd.expected_lgd(cltv, np.broadcast_to(path.unemp[t], cltv.shape), sp["score_z"].values[:, None], r["rate"])
        loss = r["mass"] * l * r["ead"] * r["dfac"]
        e12[perf] = loss[:, :12].sum(axis=1)
        elife[perf] = loss.sum(axis=1)
        pd12[perf] = r["mass"][:, :12].sum(axis=1)
        pdl[perf] = r["mass"].sum(axis=1)
        lgd_avg[perf] = (r["mass"] * l).sum(axis=1) / np.maximum(r["mass"].sum(axis=1), 1e-12)
    if dflt.any():
        sd = snap[dflt]
        mid = months_in_default(defaults, sd["loan_idx"].values, m0)
        k = cfg_ecl["stage3_sale_months"]
        om = sd["orig_m"].values.astype(int)
        cltv = sd["bal"].values / (sd["prop0"].values * path.hpi[m0 + k] / path.hpi[om])
        u = np.full(len(sd), path.unemp[m0 + k])
        pc = lgd.p_cure(cltv, u, sd["score_z"].values) * (mid <= cure_window)   # cure only possible early in workout
        l = (1 - pc) * lgd.lgd_no_cure(cltv, u, sd["rate"].values)
        e = sd["bal"].values * l * discount_factor(sd["rate"].values, k)
        e12[dflt] = e; elife[dflt] = e; pd12[dflt] = 1.0; pdl[dflt] = 1.0; lgd_avg[dflt] = l
    return pd.DataFrame({"ecl12": e12, "ecl_life": elife, "pd12": pd12, "pd_life": pdl, "lgd": lgd_avg})


def run_ecl(model, lgd_model, loans, ff, macro, defaults, cfg, pd12_orig, m0=None):
    ce = cfg["ecl"]
    m0 = len(macro) - 1 if m0 is None else m0
    snap = stage_snapshot(model, loans, ff, macro, cfg, m0, pd12_orig)
    stage = snap["stage"].values
    fut = {k: build_future(macro, k) for k in SCENARIOS}
    w = ce["scenario_weights"]
    out = snap[["loan_idx", "stage", "stage_reason", "bal", "rate", "st", "msc", "pd12_now", "pd12_orig", "ltv0", "cltv",
                "dti", "score_z", "bureau_band", "emp", "age"]].copy()
    for name in SCENARIOS:
        p = MacroPath.from_history(macro, fut[name])
        r = ecl_scenario(model, lgd_model, snap, p, m0, defaults, ce)
        out[f"ecl12_{name}"] = r["ecl12"].values
        out[f"ecllife_{name}"] = r["ecl_life"].values
        out[f"ecl_{name}"] = np.where(stage == 1, r["ecl12"], r["ecl_life"])
        out[f"pd_{name}"] = np.where(stage == 1, r["pd12"], r["pd_life"])
        out[f"lgd_{name}"] = r["lgd"].values
    out["ecl_weighted"] = sum(w[k] * out[f"ecl_{k}"] for k in SCENARIOS)
    ov = ce["overlay"]
    out["overlay"] = np.where((out["stage"] < 3) & (out["dti"] > ov["dti_threshold"]), ov["uplift"] * out["ecl_weighted"], 0.0)
    out["ecl_final"] = out["ecl_weighted"] + out["overlay"]
    return out
