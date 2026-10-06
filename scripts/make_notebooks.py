import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
HEAD = "import sys; sys.path.insert(0, '..')\nimport pandas as pd, json\nfrom IPython.display import Image, display\npd.set_option('display.width', 200)\nO = '../outputs/'\nshow = lambda f: display(pd.read_csv(O + f).round(4))\n"

def save(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nbf.write(nb, ROOT / "notebooks" / name)

save("01_data_and_calibration.ipynb", [
    md("# 01 Data and calibration\n**SYNTHETIC portfolio.** Real inputs: SA macro series. Everything loan-level is simulated with a known default process.\nRun `python -m src.pipeline` first."),
    code(HEAD + "show_l = pd.read_csv('../docs/data_lineage.csv'); display(show_l)"),
    md("## Macro inputs"), code("display(Image('../docs/figures/macro.png'))"),
    md("## Calibration to plausibility targets\nTargets (assumptions to verify against SARB / bank disclosures): annual default rate roughly 1-2% in normal years, 3-4% in stress (2020, 2023-24); 30+ dpd share 2-6%; 90+ stock 2-5%; LGD on sold properties 15-30%."),
    code("show('calibration_by_year.csv'); display(Image('../docs/figures/calibration_targets.png'))")])

save("02_model_development.ipynb", [
    md("# 02 Model development\nTime-ordered splits only. Models: WoE scorecard (benchmark), multi-state hazard model (production), GBM challenger, LGD (cure x loss-given-no-cure)."),
    code(HEAD + "print(open('../config.yaml').read().split('splits:')[1].split('staging:')[0])"),
    md("## Scorecard information values"), code("show('scorecard_iv.csv')"),
    md("## Scenario paths"), code("show('scenario_paths.csv')"),
    md("## ECL by stage and scenario"), code("show('ecl_by_stage.csv'); display(Image('../docs/figures/ecl_by_stage_scenario.png')); show('ecl_by_ltv_band.csv'); show('ecl_by_score_band.csv')"),
    md("## Stage migration"), code("show('stage_migration_pct.csv'); display(Image('../docs/figures/stage_migration.png'))"),
    md("## Sensitivities"), code("show('sensitivity_staging.csv'); show('sensitivity_scenario_weights.csv')")])

save("03_independent_validation.ipynb", [
    md("# 03 Independent validation\nWritten as if by a validator who did not build the models: out-of-time data only, plus a recovery check against the known synthetic DGP."),
    code(HEAD),
    md("## 1 Discrimination (Gini / KS)"), code("show('validation_discrimination.csv')"),
    md("## 2 Calibration (predicted vs observed 12m default rate)"),
    code("show('validation_calibration_year.csv'); show('validation_calibration_band.csv'); show('validation_calibration_vintage.csv'); display(Image('../docs/figures/calibration_oot.png'))"),
    md("## 3 Stability (PSI / CSI)\nPSI < 0.10 stable, 0.10-0.25 monitor, > 0.25 shift. Macro CSIs are expected to shift: the OOT period is a different rate/unemployment regime."),
    code("show('validation_stability.csv')"),
    md("## 4 Backtests"), code("show('validation_ecl_backtest.csv'); show('validation_lgd_backtest.csv')"),
    md("## 5 Recovery check vs true simulated process"),
    code("show('validation_recovery_coefficients.csv'); show('validation_recovery_deciles.csv'); display(Image('../docs/figures/recovery_check.png'))"),
    md("## 6 Headline JSON"), code("print(json.dumps(json.load(open(O + 'results.json')), indent=1)[:2500])")])
print("notebooks written")
