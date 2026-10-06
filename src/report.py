"""Builds docs/Model_Development_Document.md with numbers pulled from outputs/ (so it never goes stale)."""
import json
import pandas as pd
from .utils import ROOT, load_config


def md_table(df, fmt=None):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: f"{x:,.4g}" if abs(x) < 1000 else f"{x:,.0f}")
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + \
        "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)


def main():
    cfg = load_config(); O = ROOT / "outputs"
    r = json.loads((O / "results.json").read_text()); rd = lambda n: pd.read_csv(O / n)
    stg, disc, caly = rd("ecl_by_stage.csv"), rd("validation_discrimination.csv"), rd("validation_calibration_year.csv")
    lgdt, bt, rc, stab = rd("validation_lgd_backtest.csv"), rd("validation_ecl_backtest.csv"), rd("validation_recovery_coefficients.csv"), rd("validation_stability.csv")
    w = cfg["ecl"]["scenario_weights"]; S = cfg["staging"]
    txt = f"""# Model Development Document: IFRS 9 ECL, South African home loans

> **SYNTHETIC DATA.** The loan portfolio is simulated with a known default and recovery process. Results demonstrate
> methodology, not real South African credit risk. Macro series are real or flagged approximations (see `docs/data_lineage.csv`).

Version 1.1 | Reporting date {r['reporting_date']} | Run time {r['runtime_s']}s | Seed {cfg['seed']}

## 1. Purpose and scope
Estimate IFRS 9 expected credit loss for a variable-rate (prime-linked), 20-year residential mortgage book of {r['n_open']:,} open loans
(gross exposure ZAR {r['total_ead']/1e9:.1f}bn): staging, 12-month and lifetime PD, LGD, EAD, three macro scenarios, probability weighting, overlay.
Out of scope: stage-3 individual assessments, off-balance-sheet, COVID-style payment relief, real bank data.

## 2. Data and lineage
Macro: SARB repo/prime (prime = repo + 3.5pp), CPI, QLFS unemployment (quarterly, usable 2 months after quarter end so no look-ahead), real GDP, house price index.
Series flagged APPROXIMATE in `docs/data_lineage.csv` are placeholders to replace with Stats SA / SARB / FRED downloads (`scripts/download_fred.py`).
Portfolio: 30,000 loans originated 2012-2024, observed monthly to Dec-2025 (~2.0m loan-months). Drivers of the true default process: bureau score, instalment/income,
current LTV, employment type, seasoning hump (peak 36 months), unemployment, payment shock from prime changes, a 2020 stress shock, an unobserved AR(1) factor,
and post-cure re-default risk. Recovery process: 25% voluntary sales (9-15 months), 75% sales in execution (18-30 months), forced-sale discount ~22% (rising with unemployment), legal costs 4-9% of EAD,
cures from arrears and from default (3% a month for 9 months). Calibration targets and results: `outputs/calibration_by_year.csv`.

## 3. Methodology
**Splits (time-ordered, never random).** 12m PD: train observation months to 2021-12 (labels known by end-2022), validate 2022, test 2023-24. Hazard transitions: train to 2022-12. LGD: defaults/resolutions to 2022-12.

**PD, production: multi-state discrete-time hazard model.** Three logistic transition models on loan-month data (0->30dpd; 30dpd->{{cure, stay, 60dpd}}; 60dpd->{{30dpd, stay, default}}) with seasoning hinges, bureau score, LTV, current LTV, affordability,
employment, payment shock, unemployment and its 12m change, cure-probation flag. 12m and lifetime PD = cumulative probability mass entering default when the chain is propagated with the loan's
contractual amortisation and the macro path; prepayment at {cfg['ecl']['prepay_smm']:.1%} a month reduces survival. *Deviation from the original plan:* a single one-step default hazard is ~0 for current loans (default needs three consecutive roll-ups), so the plan's "1 - prod(1 - hazard)" is applied through the chain rather than to one hazard.
**Benchmark:** WoE logistic scorecard (no macro). **Challenger:** gradient boosting on the 12m default flag.

**LGD.** LGD = (1 - P(cure)) x LGD given no cure. Cure: logistic on current LTV, unemployment, score. LGD given no cure: linear model of 1/cLTV, cLTV, unemployment and EIR on realised economic LGD (recoveries discounted at the EIR over the workout). Point-in-time, no downturn add-on. Cured loans assumed to lose nothing.

**EAD.** Contractual amortisation at the prime-linked rate path; EAD at default month j = balance at the start of month j.

**Staging.** Stage 3: 90+dpd, or within {S['probation_months']} months of curing from default. Stage 2: 30+dpd backstop, or SICR = 12m PD now >= {S['sicr_pd_multiple']}x origination 12m PD AND an absolute rise >= {S['sicr_pd_absolute']*100:.2f}pp (both on flat-macro PD so staging is scenario-independent). Otherwise Stage 1.

**ECL.** sum over months of PD mass x LGD x EAD x DF at the loan's EIR. Stage 1: 12 months; Stage 2 and probation: remaining term; Stage 3: PD = 1, EAD x LGD x DF({cfg['ecl']['stage3_sale_months']} months to sale).
Scenarios: base / upside / downside over 36 months then linear reversion to long-run values over 24 months (`src/scenarios.py`). **Weights {w['base']:.0%}/{w['upside']:.0%}/{w['downside']:.0%} are illustrative.** Overlay: {cfg['ecl']['overlay']['name']}, +{cfg['ecl']['overlay']['uplift']:.0%} on Stage 1-2 loans with instalment/income above {cfg['ecl']['overlay']['dti_threshold']:.0%}.

## 4. Results
Total ECL ZAR {r['ecl_final']/1e6:,.0f}m, coverage {r['coverage_pct']:.2f}% of gross exposure.

{md_table(stg[['stage','loans','ead','ecl_base','ecl_upside','ecl_downside','ecl_final','coverage_pct']])}

Scenario ECL (ZAR m): base {r['ecl_by_scenario']['base']/1e6:,.0f}, upside {r['ecl_by_scenario']['upside']/1e6:,.0f}, downside {r['ecl_by_scenario']['downside']/1e6:,.0f}. See `outputs/sensitivity_*.csv` for weights and SICR threshold sensitivity.

## 5. Performance (out-of-time)
{md_table(disc)}

Calibration by observation year (hazard model, flat macro, no look-ahead):

{md_table(caly)}

LGD backtest (resolved defaults after the training cut-off; cure-rate predicted {r['cure_backtest']['predicted_cure_rate']:.1%} vs observed {r['cure_backtest']['observed_cure_rate']:.1%}):

{md_table(lgdt)}

6-month expected loss at Dec-2022 vs realised loss on defaults of Jan-Jun 2023 (ALL row: predicted/realised = {r['backtest_pred_vs_real']:.2f}):

{md_table(bt.tail(4))}

Recovery check against the true process: Spearman(fitted arrears-entry probability, true latent risk) = {r['spearman_recovery']:.2f}. Coefficients:

{md_table(rc)}

Stability (PSI/CSI):

{md_table(stab)}

## 6. Assumptions
Prepayment constant; probability-weighted scenario weights illustrative; staging uses flat-macro PD; cure loss zero; Stage 3 sale timing {cfg['ecl']['stage3_sale_months']} months; HPI is a CPI-linked assumption unless `data/raw/hpi.csv` is supplied; income grows with CPI less 0.5pp.

## 7. Limitations (read these first)
1. Synthetic data: validation of a model on data from a process of the same family flatters it. The point is the process and the checks, not the numbers.
2. Macro sensitivities are only weakly identified from one rate cycle: the recovery check shows borrower effects recovered closely but payment-shock and unemployment effects attenuated, so the downside scenario is probably too mild.
3. The backtest under-predicts 6-month loss (ratio above); the LGD model initially missed EIR-driven discounting (fixed in v1.1) and is linear in cLTV.
4. LGD training only sees resolved workouts (censoring toward fast resolutions).
5. Gradient boosting over-fits (train Gini well above test): kept as challenger only.
6. PSI flags for macro covariates are expected (different regime) but the PD-score PSI shows the book has aged and re-priced.
7. Approximate macro series must be replaced before any external use.

## 8. Monitoring plan
Monthly: ECL by stage and migration, predicted vs observed default by band/vintage, PSI/CSI (amber 0.10, red 0.25), scenario refresh quarterly, annual recalibration, backtest of ECL vs realised loss every half-year, overlay review each close. Dashboard: `app/streamlit_app.py`.

## 9. Governance and independence
Development code is `src/` (excluding `validation.py`); validation is `src/validation.py` + `notebooks/03_independent_validation.ipynb`, using out-of-time data and the true-process recovery check only. Staging thresholds live in `config.yaml` and are justified by `outputs/sensitivity_staging.csv`.

## 10. Version history
| Version | Change |
|---|---|
| 1.0 | First build. |
| 1.1 | Backtest showed LGD under-predicted when the EIR is high (discounting of 18-30 month workouts); EIR added as an LGD driver. 6m loss ratio improved from 0.71 to {r['backtest_pred_vs_real']:.2f}. |
"""
    (ROOT / "docs" / "Model_Development_Document.md").write_text(txt)
    print("MDD written", len(txt.split()), "words")


if __name__ == "__main__":
    main()
