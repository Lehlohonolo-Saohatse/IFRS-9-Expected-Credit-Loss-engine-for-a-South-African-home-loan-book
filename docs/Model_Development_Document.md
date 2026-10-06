# Model Development Document: IFRS 9 ECL, South African home loans

> **SYNTHETIC DATA.** The loan portfolio is simulated with a known default and recovery process. Results demonstrate
> methodology, not real South African credit risk. Macro series are real or flagged approximations (see `docs/data_lineage.csv`).

Version 1.1 | Reporting date 2025-12 | Run time 75s | Seed 20251231

## 1. Purpose and scope
Estimate IFRS 9 expected credit loss for a variable-rate (prime-linked), 20-year residential mortgage book of 20,939 open loans
(gross exposure ZAR 24.3bn): staging, 12-month and lifetime PD, LGD, EAD, three macro scenarios, probability weighting, overlay.
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

**PD, production: multi-state discrete-time hazard model.** Three logistic transition models on loan-month data (0->30dpd; 30dpd->{cure, stay, 60dpd}; 60dpd->{30dpd, stay, default}) with seasoning hinges, bureau score, LTV, current LTV, affordability,
employment, payment shock, unemployment and its 12m change, cure-probation flag. 12m and lifetime PD = cumulative probability mass entering default when the chain is propagated with the loan's
contractual amortisation and the macro path; prepayment at 0.4% a month reduces survival. *Deviation from the original plan:* a single one-step default hazard is ~0 for current loans (default needs three consecutive roll-ups), so the plan's "1 - prod(1 - hazard)" is applied through the chain rather than to one hazard.
**Benchmark:** WoE logistic scorecard (no macro). **Challenger:** gradient boosting on the 12m default flag.

**LGD.** LGD = (1 - P(cure)) x LGD given no cure. Cure: logistic on current LTV, unemployment, score. LGD given no cure: linear model of 1/cLTV, cLTV, unemployment and EIR on realised economic LGD (recoveries discounted at the EIR over the workout). Point-in-time, no downturn add-on. Cured loans assumed to lose nothing.

**EAD.** Contractual amortisation at the prime-linked rate path; EAD at default month j = balance at the start of month j.

**Staging.** Stage 3: 90+dpd, or within 12 months of curing from default. Stage 2: 30+dpd backstop, or SICR = 12m PD now >= 2.5x origination 12m PD AND an absolute rise >= 0.50pp (both on flat-macro PD so staging is scenario-independent). Otherwise Stage 1.

**ECL.** sum over months of PD mass x LGD x EAD x DF at the loan's EIR. Stage 1: 12 months; Stage 2 and probation: remaining term; Stage 3: PD = 1, EAD x LGD x DF(12 months to sale).
Scenarios: base / upside / downside over 36 months then linear reversion to long-run values over 24 months (`src/scenarios.py`). **Weights 50%/20%/30% are illustrative.** Overlay: Municipal tariff / cost-of-living affordability overlay (illustrative), +10% on Stage 1-2 loans with instalment/income above 33%.

## 4. Results
Total ECL ZAR 290m, coverage 1.19% of gross exposure.

| stage | loans | ead | ecl_base | ecl_upside | ecl_downside | ecl_final | coverage_pct |
|---|---|---|---|---|---|---|---|
| 1 | 18,057 | 20,168,671,232 | 50,143,815 | 44,418,653 | 62,433,219 | 52,746,683 | 0.2615 |
| 2 | 1,967 | 2,921,496,320 | 51,818,748 | 45,034,306 | 69,651,450 | 56,050,359 | 1.919 |
| 3 | 915 | 1,252,712,320 | 179,092,437 | 175,575,346 | 187,625,010 | 180,948,791 | 14.44 |
| Total | 20,939 | 24,342,880,256 | 281,055,001 | 265,028,304 | 319,709,680 | 289,745,832 | 1.19 |

Scenario ECL (ZAR m): base 281, upside 265, downside 320. See `outputs/sensitivity_*.csv` for weights and SICR threshold sensitivity.

## 5. Performance (out-of-time)
| sample | model | n | default_rate | gini | ks |
|---|---|---|---|---|---|
| train (2013-2021) | Hazard (Markov, macro-aware) | 347590 | 0.01678 | 0.5886 | 0.4182 |
| train (2013-2021) | WoE scorecard (benchmark) | 347590 | 0.01678 | 0.5735 | 0.4148 |
| train (2013-2021) | Gradient boosting (challenger) | 347590 | 0.01678 | 0.7249 | 0.5363 |
| validation (2022) | Hazard (Markov, macro-aware) | 212210 | 0.02227 | 0.5678 | 0.428 |
| validation (2022) | WoE scorecard (benchmark) | 212210 | 0.02227 | 0.5566 | 0.4221 |
| validation (2022) | Gradient boosting (challenger) | 212210 | 0.02227 | 0.5735 | 0.4275 |
| out-of-time test (2023-24) | Hazard (Markov, macro-aware) | 478798 | 0.03113 | 0.6459 | 0.4879 |
| out-of-time test (2023-24) | WoE scorecard (benchmark) | 478798 | 0.03113 | 0.6174 | 0.4627 |
| out-of-time test (2023-24) | Gradient boosting (challenger) | 478798 | 0.03113 | 0.6173 | 0.4662 |

Calibration by observation year (hazard model, flat macro, no look-ahead):

| obs_year | n | predicted | observed | ratio |
|---|---|---|---|---|
| 2022 | 212,210 | 0.02288 | 0.02227 | 1.028 |
| 2023 | 231,634 | 0.03153 | 0.03543 | 0.89 |
| 2024 | 247,164 | 0.02844 | 0.02711 | 1.049 |

LGD backtest (resolved defaults after the training cut-off; cure-rate predicted 22.4% vs observed 24.5%):

| band | n | predicted_lgd | realised_lgd | true_fsd |
|---|---|---|---|---|
| <60% | 191 | 0.1528 | 0.1651 | 0.1857 |
| 60-80% | 422 | 0.2105 | 0.1921 | 0.1887 |
| 80-100% | 263 | 0.2679 | 0.2814 | 0.199 |
| 100-120% | 4 | 0.3401 | 0.3941 | 0.2001 |
| ALL | 880 | 0.2157 | 0.2138 | 0.1912 |

6-month expected loss at Dec-2022 vs realised loss on defaults of Jan-Jun 2023 (ALL row: predicted/realised = 0.82):

| vintage | exp_defaults | exp_loss | exposure | real_defaults | real_loss | loss_ratio_pred_vs_real |
|---|---|---|---|---|---|---|
| 2020 | 39.1 | 10,787,109 | 2,604,185,344 | 50 | 16,304,450 | 0.6616 |
| 2021 | 68.44 | 19,900,039 | 4,015,154,432 | 72 | 22,774,715 | 0.8738 |
| 2022 | 42.99 | 13,463,936 | 3,871,344,128 | 48 | 19,559,949 | 0.6883 |
| ALL | 291.8 | 64,483,064 | 21,975,865,344 | 271 | 78,741,525 | 0.8189 |

Recovery check against the true process: Spearman(fitted arrears-entry probability, true latent risk) = 0.89. Coefficients:

| feature | fitted | true_dgp |
|---|---|---|
| score_z | -0.5483 | -0.55 |
| dti | 4.77 | 5 |
| self_emp | 0.3383 | 0.35 |
| contract | 0.4486 | 0.45 |
| prime_shock | 0.01733 | 0.25 |
| unemp | 0.183 | 0.3 |
| du12 | -0.115 | 0 |
| probation | 0.4992 | 0.7 |

Stability (PSI/CSI):

| item | psi | flag |
|---|---|---|
| PD score (hazard 12m) | 0.2907 | shift |
| CSI score_z | 0.0008894 | stable |
| CSI cltv | 0.3368 | shift |
| CSI dti | 0.0718 | stable |
| CSI age | 0.4191 | shift |
| CSI unemp | 6.651 | shift |
| CSI ltv0 | 0.0003234 | stable |
| CSI prime_shock | 3.259 | shift |

## 6. Assumptions
Prepayment constant; probability-weighted scenario weights illustrative; staging uses flat-macro PD; cure loss zero; Stage 3 sale timing 12 months; HPI is a CPI-linked assumption unless `data/raw/hpi.csv` is supplied; income grows with CPI less 0.5pp.

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
| 1.1 | Backtest showed LGD under-predicted when the EIR is high (discounting of 18-30 month workouts); EIR added as an LGD driver. 6m loss ratio improved from 0.71 to 0.82. |
