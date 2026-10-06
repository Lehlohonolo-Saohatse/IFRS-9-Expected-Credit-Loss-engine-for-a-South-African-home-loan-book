"""Independent-style validation of the PD / LGD / ECL models. Everything here uses only OUT-OF-TIME data
except the explicit 'recovery check', which compares fitted models with the known synthetic DGP."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .features import make_X, split_mask, FEATURES
from .pd_model import gini, ks_stat, gbm_predict
from .projection import MacroPath, project, snapshot
from .utils import sigmoid


def month_idx(macro, label):
    return list(macro.index).index(pd.Period(label, "M"))


def hazard_pd_frame(model, loans, ff, macro, months, smm=0.004):
    """Pool snapshot rows (performing, 12m label complete) with the hazard-implied 12m PD (flat macro, no look-ahead)."""
    path = MacroPath.from_history(macro)
    parts = []
    for m0 in months:
        sn = snapshot(loans, ff, m0)
        sn = sn[(sn["st"] < 3) & sn["d12"].notna()].reset_index(drop=True)
        if len(sn) == 0:
            continue
        sn["pd_haz"] = project(model, sn, path, m0, macro, horizon=12, flat=True, smm=smm)["pd12"]
        parts.append(sn)
    return pd.concat(parts, ignore_index=True)


def discrimination(frames, model_scorecard, gbm):
    rows = []
    for sample, fr in frames.items():
        y = fr["d12"].values
        preds = {"Hazard (Markov, macro-aware)": fr["pd_haz"].values, "WoE scorecard (benchmark)": model_scorecard.predict(fr),
                 "Gradient boosting (challenger)": gbm_predict(gbm, fr)}
        for name, p in preds.items():
            rows.append(dict(sample=sample, model=name, n=len(fr), default_rate=y.mean(), gini=gini(y, p), ks=ks_stat(y, p)))
    return pd.DataFrame(rows)


def calibration_tables(fr, macro, col="pd_haz"):
    yrs = np.array([p.year for p in macro.index])
    fr = fr.assign(obs_year=yrs[fr["m"].values], vintage=yrs[fr["orig_m"].values.astype(int)])
    def agg(g):
        return pd.Series(dict(n=len(g), predicted=g[col].mean(), observed=g["d12"].mean(),
                              ratio=g[col].mean() / max(g["d12"].mean(), 1e-9)))
    return (fr.groupby("obs_year").apply(agg, include_groups=False).reset_index(),
            fr.groupby("bureau_band").apply(agg, include_groups=False).reset_index().set_index("bureau_band").loc[["A", "B", "C", "D", "E"]].reset_index(),
            fr.groupby("vintage").apply(agg, include_groups=False).reset_index())


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def stability(train_fr, test_fr):
    rows = [dict(item="PD score (hazard 12m)", psi=psi(train_fr["pd_haz"].values, test_fr["pd_haz"].values))]
    for c in ["score_z", "cltv", "dti", "age", "unemp", "ltv0", "prime_shock"]:
        rows.append(dict(item=f"CSI {c}", psi=psi(train_fr[c].values, test_fr[c].values)))
    t = pd.DataFrame(rows)
    t["flag"] = np.where(t["psi"] < 0.10, "stable", np.where(t["psi"] < 0.25, "monitor", "shift"))
    return t


def ecl_backtest(model, lgd, loans, ff, defaults, macro, start="2022-12", months=6, smm=0.004):
    """6-month-ahead expected loss at `start` vs realised loss on defaults in the following `months` (all resolved by obs end)."""
    m0 = month_idx(macro, start)
    sn = snapshot(loans, ff, m0)
    sn = sn[sn["st"] < 3].reset_index(drop=True)
    path = MacroPath.from_history(macro)
    r = project(model, sn, path, m0, macro, horizon=months, flat=True, smm=smm)
    om = sn["orig_m"].values.astype(int)[:, None]
    cltv = r["ead"] / (sn["prop0"].values[:, None] * path.hpi[m0] / path.hpi[om])
    l = lgd.expected_lgd(cltv, np.full(cltv.shape, path.unemp[m0]), sn["score_z"].values[:, None], r["rate"])
    sn["exp_defaults"] = r["mass"].sum(axis=1)
    sn["exp_loss"] = (r["mass"] * l * r["ead"]).sum(axis=1)   # value at default date (no reporting-date discounting)
    d = defaults[(defaults["def_m"] > m0) & (defaults["def_m"] <= m0 + months)]
    d = d[d["loan_idx"].isin(sn["loan_idx"])].copy()
    d["loss"] = np.where(d["outcome"] == "cured", 0.0, d["ead"] * d["realised_lgd"])
    unresolved = int((d["outcome"] == "open").sum())
    d = d[d["outcome"] != "open"]
    yrs = np.array([p.year for p in macro.index])
    ac = d.groupby(yrs[loans["orig_m"].values[d["loan_idx"].values]]).agg(real_defaults=("loan_idx", "size"), real_loss=("loss", "sum"))
    pr = sn.groupby(yrs[sn["orig_m"].values.astype(int)]).agg(exp_defaults=("exp_defaults", "sum"), exp_loss=("exp_loss", "sum"), exposure=("bal", "sum"))
    t = pr.join(ac).fillna(0).rename_axis("vintage").reset_index()
    tot = t.drop(columns="vintage").sum().to_frame().T; tot.insert(0, "vintage", "ALL")
    t = pd.concat([t, tot], ignore_index=True)
    t["loss_ratio_pred_vs_real"] = t["exp_loss"] / t["real_loss"].replace(0, np.nan)
    t.attrs["unresolved_excluded"] = unresolved
    return t


def lgd_backtest(lgd, defaults, loans, train_end_m, last_m, cure_window=9):
    d = defaults.merge(loans[["loan_idx", "bureau_score"]], on="loan_idx")
    d["score_z"] = (d["bureau_score"] - 670) / 75
    s = d[d["outcome"].isin(["vol_sale", "sale_exec"]) & (d["res_m"] > train_end_m)].copy()
    s["pred"] = lgd.lgd_no_cure(s["cltv_def"].values, s["unemp_def"].values, s["rate_def"].values)
    s["band"] = pd.cut(s["cltv_def"], [0, 0.6, 0.8, 1.0, 1.2, 9], labels=["<60%", "60-80%", "80-100%", "100-120%", ">120%"])
    tab = s.groupby("band", observed=True).agg(n=("pred", "size"), predicted_lgd=("pred", "mean"), realised_lgd=("realised_lgd", "mean"),
                                                true_fsd=("true_fsd", "mean")).reset_index()
    tab.loc[len(tab)] = ["ALL", len(s), s["pred"].mean(), s["realised_lgd"].mean(), s["true_fsd"].mean()]
    c = d[(d["def_m"] > train_end_m) & (d["def_m"] + cure_window <= last_m)].copy()
    pc = lgd.p_cure(c["cltv_def"].values, c["unemp_def"].values, c["score_z"].values)
    cure = dict(n=len(c), predicted_cure_rate=float(pc.mean()), observed_cure_rate=float((c["outcome"] == "cured").mean()))
    return tab, cure


def recovery_check(model, ff, test_mask, cfg_dgp):
    """Compare fitted hazard model with the known DGP."""
    t = ff[test_mask & (ff["st"] == 0)]
    X = make_X(t)
    p01 = model.p01(X)
    rho = spearmanr(p01, t["true_eta"].values).statistic
    dec = pd.qcut(t["true_eta"].values, 10, labels=False)
    nxt = t["st_next"].notna().values
    y = (t["st_next"].values == 1)
    dtab = pd.DataFrame({"decile": dec[nxt], "fitted_p": p01[nxt], "observed": y[nxt], "true_p": sigmoid(t["true_eta"].values[nxt])})
    dtab = dtab.groupby("decile").mean().reset_index()
    # coefficient recovery (per natural unit of the feature), model m0
    sc, lr = model.m0.sc, model.m0.m
    raw = lr.coef_[0] / sc.scale_
    coef = dict(zip(FEATURES, raw))
    truth = {"score_z": -cfg_dgp["b_score"], "dti": cfg_dgp["b_dti"] / 0.08, "self_emp": cfg_dgp["b_self_emp"],
             "contract": cfg_dgp["b_contract"], "prime_shock": cfg_dgp["b_prime_shock"],
             "unemp": cfg_dgp["b_unemp"], "du12": 0.0, "probation": cfg_dgp["b_probation"]}
    ct = pd.DataFrame([dict(feature=k, fitted=coef[k], true_dgp=v) for k, v in truth.items()])
    return dict(spearman_fitted_vs_true=float(rho)), dtab, ct
