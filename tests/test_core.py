import numpy as np
import pandas as pd
from src.utils import pmt, discount_factor, annuity_principal, load_config
from src.ead import schedule, next_balance
from src.staging import assign_stage, migration_matrix
from src.features import make_X, FEATURES
from src.scenarios import build_future, SCENARIOS
from src.macro import build_macro

CFG = load_config()["staging"]


def test_pmt_and_annuity_roundtrip():
    inst = pmt(1_000_000, 0.10, 240)
    assert abs(float(annuity_principal(inst, 0.10, 240)) - 1_000_000) < 1e-3
    assert abs(float(inst) - 9650.22) < 1.0          # known 20y 10% instalment per R1m


def test_schedule_amortises_to_zero_and_first_month_interest():
    s = schedule(1_000_000, 0.10, 240)
    assert s[0] == 1_000_000 and s[-1] < 1e-4
    assert abs((s[0] - s[1]) - (pmt(1e6, .10, 240) - 1e6 * .10 / 12)) < 1e-6
    assert np.all(np.diff(s) <= 0)


def test_zero_rate_amortisation():
    assert abs(float(pmt(1200, 0.0, 12)) - 100) < 1e-9


def test_discounting_at_eir():
    assert abs(float(discount_factor(0.12, 12)) - 1.01 ** -12) < 1e-12
    assert float(discount_factor(0.12, 0)) == 1.0
    assert float(discount_factor(0.12, 24)) < float(discount_factor(0.08, 24))


def test_staging_rules():
    st = np.array([0, 0, 1, 2, 3, 0, 0]); msc = np.array([-1, -1, -1, -1, -1, 3, 20])
    now = np.array([0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]); orig = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    stage, reason = assign_stage(st, msc, now, orig, CFG)
    assert list(stage) == [1, 2, 2, 2, 3, 3, 1]
    assert reason[1] == "SICR" and reason[2] == "30dpd backstop" and reason[4] == "default 90dpd" and reason[5] == "cure probation"


def test_sicr_needs_both_multiple_and_absolute():
    z = np.zeros(2, dtype=int); m = -np.ones(2, dtype=int)
    stage, _ = assign_stage(z, m, np.array([0.003, 0.02]), np.array([0.001, 0.015]), CFG)   # 3x but +0.2pp ; +0.5pp but 1.3x
    assert list(stage) == [1, 1]


def test_migration_rows_sum_to_one():
    cnt, pct = migration_matrix([1, 1, 2, 3], [1, 2, 0, 3])
    assert np.allclose(pct.sum(axis=1), 1.0)


def test_make_X_shape_and_probation_flag():
    f = dict(age=[1, 40], score_z=[0, 1], ltv0=[.8, .9], cltv=[.8, .9], dti=[.2, .3], emp=[0, 2], prime_shock=[0, 1],
             unemp=[30, 32], du12=[0, 1], msc=[-1, 5])
    X = make_X(f)
    assert X.shape == (2, len(FEATURES)) and X[:, -1].tolist() == [0.0, 1.0]


def test_macro_no_lookahead_and_prime_rule():
    m, _ = build_macro(save=False)
    assert np.allclose(m["prime"], m["repo"] + 3.5)
    # QLFS Q4-2020 (32.5) is not usable before Feb-2021
    assert m.loc[pd.Period("2020-12", "M"), "unemp"] != 32.5 and m.loc[pd.Period("2021-02", "M"), "unemp"] == 32.5


def test_scenarios_ordering():
    m, _ = build_macro(save=False)
    f = {k: build_future(m, k) for k in SCENARIOS}
    assert f["downside"]["prime"].iloc[11] > f["base"]["prime"].iloc[11] > f["upside"]["prime"].iloc[11]
    assert f["downside"]["unemp"].iloc[23] > f["upside"]["unemp"].iloc[23]
    assert abs(f["downside"]["prime"].iloc[-1] - 10.5) < 1e-6      # mean reversion
