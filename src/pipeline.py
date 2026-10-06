"""End-to-end run:  python -m src.pipeline   (about 2-4 minutes on a laptop)"""
import json, pickle, time
import numpy as np
import pandas as pd
from .calibration import annual_stats
from .ecl import run_ecl, pd12_at_origination, stage_snapshot
from .features import feature_frame, split_mask
from .generator import simulate
from .lgd import LGDModel
from .macro import build_macro
from .pd_model import MarkovPD, WoEScorecard, fit_gbm
from .projection import MacroPath
from .scenarios import scenario_table, SCENARIOS
from .staging import assign_stage, migration_matrix
from . import validation as V
from .utils import load_config, ROOT


def main(regenerate=True):
    t0 = time.time()
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["outputs"]; out_dir.mkdir(exist_ok=True)
    (ROOT / cfg["paths"]["docs"] / "figures").mkdir(parents=True, exist_ok=True)
    macro, lineage = build_macro(cfg)
    macro.index = pd.PeriodIndex(macro.index, freq="M")
    loans, panel, defaults, _ = simulate(cfg, macro)
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    log(f"simulated {len(loans)} loans, {len(panel)} loan-months, {len(defaults)} defaults")
    ff = feature_frame(loans, panel, macro)
    S = cfg["splits"]; mi = lambda s: V.month_idx(macro, s)
    results = {}

    # ---------------- calibration table (targets) ----------------
    cal = annual_stats(panel, defaults, macro).reset_index()
    cal.to_csv(out_dir / "calibration_by_year.csv", index=False)

    # ---------------- train ----------------
    hz = MarkovPD().fit(ff[split_mask(ff, macro, None, S["hazard_train_end"])])
    tr12 = split_mask(ff, macro, None, S["pd12_train_end"]) & (ff["st"] < 3) & ff["d12"].notna()
    sc = WoEScorecard().fit(ff[tr12], ff.loc[tr12, "d12"].values)
    gbm = fit_gbm(ff[tr12], ff.loc[tr12, "d12"].values)
    lgd = LGDModel().fit(defaults, loans, macro, mi(S["lgd_train_end"]), cure_window=cfg["dgp"]["cure_window"], last_m=len(macro) - 1)
    sc.iv_table.to_csv(out_dir / "scorecard_woe_bins.csv", index=False)
    sc.iv().rename("information_value").reset_index().to_csv(out_dir / "scorecard_iv.csv", index=False)
    with open(ROOT / cfg["paths"]["processed"] / "models.pkl", "wb") as f:
        pickle.dump(dict(hazard=hz, scorecard=sc, gbm=gbm, lgd=lgd), f)
    log("models trained")

    # ---------------- ECL ----------------
    pd12_orig = pd12_at_origination(hz, loans, macro, MacroPath.from_history(macro), cfg["ecl"]["prepay_smm"])
    ecl = run_ecl(hz, lgd, loans, ff, macro, defaults, cfg, pd12_orig)
    ecl.to_csv(out_dir / "ecl_by_loan.csv.gz", index=False, float_format="%.5g")
    log("ECL done")
    w = cfg["ecl"]["scenario_weights"]
    ecl["ltv_band"] = pd.cut(ecl["ltv0"], [0, .7, .8, .9, 1.01], labels=["<=70%", "70-80%", "80-90%", "90-100%"]).astype(str)
    stg = ecl.groupby("stage").agg(loans=("loan_idx", "size"), ead=("bal", "sum"), ecl_base=("ecl_base", "sum"),
                                   ecl_upside=("ecl_upside", "sum"), ecl_downside=("ecl_downside", "sum"),
                                   ecl_weighted=("ecl_weighted", "sum"), overlay=("overlay", "sum"), ecl_final=("ecl_final", "sum")).reset_index()
    tot = stg.drop(columns="stage").sum().to_frame().T; tot.insert(0, "stage", "Total")
    stg = pd.concat([stg.astype({"stage": object}), tot], ignore_index=True)
    stg["coverage_pct"] = stg["ecl_final"] / stg["ead"] * 100
    stg.to_csv(out_dir / "ecl_by_stage.csv", index=False)
    for name, col in [("ltv_band", "ltv_band"), ("score_band", "bureau_band")]:
        g = ecl.groupby(col).agg(loans=("loan_idx", "size"), ead=("bal", "sum"), ecl_final=("ecl_final", "sum")).reset_index()
        g["coverage_pct"] = g["ecl_final"] / g["ead"] * 100
        g.to_csv(out_dir / f"ecl_by_{name}.csv", index=False)
    scen = pd.DataFrame({k: [ecl[f"ecl_{k}"].sum()] for k in SCENARIOS}); scen["weighted"] = ecl["ecl_weighted"].sum()
    scen["overlay"] = ecl["overlay"].sum(); scen["final"] = ecl["ecl_final"].sum(); scen["ead"] = ecl["bal"].sum()
    scen.to_csv(out_dir / "ecl_by_scenario.csv", index=False)
    scenario_table(macro).to_csv(out_dir / "scenario_paths.csv", index=False)

    # ---------------- sensitivities (cheap re-staging / re-weighting) ----------------
    rows = []
    wl = {k: ecl[f"ecl12_{k}"] for k in SCENARIOS}
    ll = {k: ecl[f"ecllife_{k}"] for k in SCENARIOS}
    ecl12_w = sum(w[k] * wl[k] for k in SCENARIOS); eclL_w = sum(w[k] * ll[k] for k in SCENARIOS)
    for mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for ab in [0.0025, 0.005, 0.01]:
            c = dict(cfg["staging"], sicr_pd_multiple=mult, sicr_pd_absolute=ab)
            stg_, _ = assign_stage(ecl["st"], ecl["msc"], ecl["pd12_now"], ecl["pd12_orig"], c)
            e = np.where(stg_ == 1, ecl12_w, eclL_w)
            rows.append(dict(test="SICR thresholds", multiple=mult, absolute_pp=ab * 100, stage2_share_pct=(stg_ == 2).mean() * 100,
                             stage2_ead_pct=ecl.loc[stg_ == 2, "bal"].sum() / ecl["bal"].sum() * 100, total_ecl=e.sum()))
    for pm in [6, 12, 18]:
        c = dict(cfg["staging"], probation_months=pm)
        stg_, _ = assign_stage(ecl["st"], ecl["msc"], ecl["pd12_now"], ecl["pd12_orig"], c)
        e = np.where(stg_ == 1, ecl12_w, eclL_w)
        rows.append(dict(test="Probation months", multiple=pm, stage2_share_pct=(stg_ == 2).mean() * 100,
                         stage2_ead_pct=np.nan, total_ecl=e.sum(), absolute_pp=np.nan))
    pd.DataFrame(rows).to_csv(out_dir / "sensitivity_staging.csv", index=False)
    wrows = []
    for nm, ww in {"Illustrative 50/20/30": (.5, .2, .3), "Base-heavy 70/10/20": (.7, .1, .2), "Equal 33/33/33": (1/3, 1/3, 1/3),
                   "Downside-heavy 40/10/50": (.4, .1, .5), "Optimistic 40/40/20": (.4, .4, .2), "100% base": (1, 0, 0), "100% downside": (0, 0, 1)}.items():
        e = ww[0] * ecl["ecl_base"].sum() + ww[1] * ecl["ecl_upside"].sum() + ww[2] * ecl["ecl_downside"].sum()
        wrows.append(dict(weights=nm, base=ww[0], upside=ww[1], downside=ww[2], ecl=e, coverage_pct=e / ecl["bal"].sum() * 100))
    pd.DataFrame(wrows).to_csv(out_dir / "sensitivity_scenario_weights.csv", index=False)

    # ---------------- stage migration Dec-2024 -> Dec-2025 ----------------
    m1 = len(macro) - 1; m_prev = m1 - 12
    prev = stage_snapshot(hz, loans, ff, macro, cfg, m_prev, pd12_orig)
    cur = ecl.set_index("loan_idx")["stage"]
    to = prev["loan_idx"].map(cur).fillna(0).astype(int).values
    cnt, mig = migration_matrix(prev["stage"].values, to)
    cnt.to_csv(out_dir / "stage_migration_counts.csv"); mig.to_csv(out_dir / "stage_migration_pct.csv")
    log("sensitivities + migration done")

    # ---------------- validation ----------------
    def months(a, b, step=1):
        return list(range(mi(a), mi(b) + 1, step))
    fr_tr = V.hazard_pd_frame(hz, loans, ff, macro, months("2013-06", S["pd12_train_end"], 3))
    fr_va = V.hazard_pd_frame(hz, loans, ff, macro, months(*S["pd12_valid"]))
    fr_te = V.hazard_pd_frame(hz, loans, ff, macro, months(*S["pd12_test"]))
    disc = V.discrimination({"train (2013-2021)": fr_tr, "validation (2022)": fr_va, "out-of-time test (2023-24)": fr_te}, sc, gbm)
    disc.to_csv(out_dir / "validation_discrimination.csv", index=False)
    cal_y, cal_b, cal_v = V.calibration_tables(pd.concat([fr_va, fr_te]), macro)
    cal_y.to_csv(out_dir / "validation_calibration_year.csv", index=False)
    cal_b.to_csv(out_dir / "validation_calibration_band.csv", index=False)
    cal_v.to_csv(out_dir / "validation_calibration_vintage.csv", index=False)
    V.stability(fr_tr, fr_te).to_csv(out_dir / "validation_stability.csv", index=False)
    bt = V.ecl_backtest(hz, lgd, loans, ff, defaults, macro, start=S["hazard_train_end"], months=6, smm=cfg["ecl"]["prepay_smm"])
    bt.to_csv(out_dir / "validation_ecl_backtest.csv", index=False)
    lt, cure = V.lgd_backtest(lgd, defaults, loans, mi(S["lgd_train_end"]), len(macro) - 1, cfg["dgp"]["cure_window"])
    lt.to_csv(out_dir / "validation_lgd_backtest.csv", index=False)
    test_mask = split_mask(ff, macro, S["hazard_test"][0], S["hazard_test"][1])
    rc, dtab, ctab = V.recovery_check(hz, ff, test_mask, cfg["dgp"])
    dtab.to_csv(out_dir / "validation_recovery_deciles.csv", index=False)
    ctab.to_csv(out_dir / "validation_recovery_coefficients.csv", index=False)
    log("validation done")

    # ---------------- headline numbers ----------------
    te = disc[disc["sample"].str.startswith("out-of-time")].set_index("model")
    results.update(
        reporting_date=cfg["ecl"]["reporting_date"], n_loans=len(loans), n_open=len(ecl), total_ead=float(ecl["bal"].sum()),
        ecl_final=float(ecl["ecl_final"].sum()), coverage_pct=float(ecl["ecl_final"].sum() / ecl["bal"].sum() * 100),
        ecl_by_scenario={k: float(ecl[f"ecl_{k}"].sum()) for k in SCENARIOS}, weights=w,
        stage_share={int(k): float(v) for k, v in ecl.groupby("stage")["bal"].sum().div(ecl["bal"].sum()).items()},
        gini_test={k: float(v) for k, v in te["gini"].items()}, ks_test={k: float(v) for k, v in te["ks"].items()},
        cure_backtest=cure, spearman_recovery=rc["spearman_fitted_vs_true"],
        lgd_pred=float(lt.iloc[-1]["predicted_lgd"]), lgd_real=float(lt.iloc[-1]["realised_lgd"]),
        backtest_pred_vs_real=float(bt.iloc[-1]["loss_ratio_pred_vs_real"]),
        lgd_samples=dict(cure=lgd.n_cure, lgd=lgd.n_lgd), data_mode="SYNTHETIC",
        macro_real_files=[p.name for p in (ROOT / "data" / "raw").glob("*.csv")],
        runtime_s=round(time.time() - t0))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    make_figures(cfg, macro, cal, ecl, mig, disc, cal_y, dtab, lt)
    log("done")
    return results


def make_figures(cfg, macro, cal, ecl, mig, disc, cal_y, dtab, lt):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = ROOT / cfg["paths"]["docs"] / "figures"
    x = [p.to_timestamp() for p in macro.index]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for a, c, t in zip(ax.ravel(), ["prime", "unemp", "cpi_yoy", "hpi"], ["Prime %", "Unemployment % (publication-lagged)", "CPI y/y % (approx.)", "House price index (assumption)"]):
        a.plot(x, macro[c]); a.set_title(t, fontsize=9); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fd / "macro.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(cal["year"], cal["annual_default_rate"] * 100, marker="o", label="annual default rate")
    ax.plot(cal["year"], cal["arrears_30plus_share"] * 100, marker="s", label="30+ dpd share")
    ax.plot(cal["year"], cal["default_stock_share"] * 100, marker="^", label="90+ dpd stock share")
    ax.set_ylabel("%"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("Synthetic book: calibration to plausibility targets", fontsize=9)
    fig.tight_layout(); fig.savefig(fd / "calibration_targets.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(cal_y["obs_year"].astype(str), cal_y["observed"] * 100, width=.4, align="edge", label="observed")
    ax.bar(cal_y["obs_year"].astype(str), cal_y["predicted"] * 100, width=-.4, align="edge", label="predicted (hazard, flat macro)")
    ax.set_ylabel("12m default rate %"); ax.legend(fontsize=8); ax.set_title("Out-of-time calibration by observation year", fontsize=9)
    fig.tight_layout(); fig.savefig(fd / "calibration_oot.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mig.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(mig.shape[1])); ax.set_xticklabels(mig.columns, fontsize=8); ax.set_yticks(range(mig.shape[0])); ax.set_yticklabels(mig.index, fontsize=8)
    for i in range(mig.shape[0]):
        for j in range(mig.shape[1]):
            ax.text(j, i, f"{mig.values[i, j]:.0%}", ha="center", va="center", fontsize=8)
    ax.set_title("Stage migration Dec-2024 to Dec-2025", fontsize=9); fig.tight_layout(); fig.savefig(fd / "stage_migration.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    g = ecl.groupby("stage")[["ecl_upside", "ecl_base", "ecl_downside"]].sum() / 1e6
    g.plot.bar(ax=ax); ax.set_ylabel("ZAR m"); ax.set_title("ECL by stage and scenario", fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fd / "ecl_by_stage_scenario.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dtab["observed"] * 100, dtab["fitted_p"] * 100, "o-", label="fitted"); ax.plot(dtab["observed"] * 100, dtab["true_p"] * 100, "s--", label="true DGP")
    ax.set_xlabel("observed monthly arrears-entry %"); ax.set_ylabel("monthly arrears-entry %"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Recovery check by true-risk decile", fontsize=9); fig.tight_layout(); fig.savefig(fd / "recovery_check.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
