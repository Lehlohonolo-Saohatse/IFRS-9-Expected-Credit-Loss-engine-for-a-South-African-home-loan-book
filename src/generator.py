"""Synthetic SA home-loan portfolio with a KNOWN default and recovery process.

States: 0 current, 1 = 30dpd, 2 = 60dpd, 3 = 90+dpd (default). Variable-rate (prime + spread) 20y loans.
Outputs (data/processed): loans.csv, panel.csv.gz, defaults.csv, macro_monthly.csv
"""
import numpy as np
import pandas as pd
from .utils import load_config, ROOT, pmt, annuity_principal, sigmoid, logit
from .macro import build_macro


def seasoning_shape(age):
    """Hump peaking at 36 months, value in [0,1]."""
    x = np.asarray(age, dtype=float) / 36.0
    return x * np.exp(1 - x)


def dgp_eta(cfg_d, score_z, dti, cltv, emp, age, u, shock, covid, frailty, probation):
    """True monthly logit of arrears entry (current -> 30dpd)."""
    return (cfg_d["b0"] - cfg_d["b_score"] * score_z
            + cfg_d["b_dti"] * (dti - 0.25) / 0.08
            + cfg_d["b_cltv"] * np.maximum(cltv - 0.9, 0) / 0.10
            + np.where(emp == 1, cfg_d["b_self_emp"], 0) + np.where(emp == 2, cfg_d["b_contract"], 0)
            + cfg_d["b_seasoning"] * (seasoning_shape(age) - 0.4)
            + cfg_d["b_unemp"] * (u - 30.0) / 4.0
            + cfg_d["b_prime_shock"] * shock / 2.0
            + covid + frailty + cfg_d["b_probation"] * probation)


def simulate(cfg=None, macro=None, save=True):
    cfg = cfg or load_config()
    p, d = cfg["portfolio"], cfg["dgp"]
    rng = np.random.default_rng(cfg["seed"])
    if macro is None:
        macro, _ = build_macro(cfg, save=save)
    M = len(macro)
    idx = macro.index
    prime = macro["prime"].values / 100.0
    unemp = macro["unemp"].values
    hpi = macro["hpi"].values
    inc = macro["income_idx"].values
    n, term = p["n_loans"], p["term_months"]
    n_orig = len(pd.period_range(p["first_orig"], p["last_orig"], freq="M"))

    # ---------------- static attributes ----------------
    w = np.array([1 + 0.04 * (idx[i].year - 2012) for i in range(n_orig)])
    for i in range(n_orig):
        if idx[i].year == 2020 and idx[i].month in (4, 5, 6):
            w[i] *= 0.4
    orig = rng.choice(n_orig, size=n, p=w / w.sum())
    score = np.clip(rng.normal(670, 75, n), 450, 850)
    score_z = (score - 670) / 75
    emp = rng.choice(3, size=n, p=[0.72, 0.16, 0.12])         # 0 permanent, 1 self-employed, 2 contract
    prov_names = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State",
                  "Limpopo", "Mpumalanga", "North West", "Northern Cape"]
    prov = rng.choice(9, size=n, p=np.array([.33, .22, .17, .07, .04, .05, .05, .05, .02]) / 1.0)
    income0 = np.exp(rng.normal(np.log(32000), 0.5, n)) * inc[orig] / inc[0]
    spread = np.interp(score, [450, 550, 650, 750, 850], [3.0, 1.75, 0.5, -0.25, -0.75]) / 100.0
    rate0 = prime[orig] + spread
    dti0 = 0.15 + 0.20 * rng.beta(2, 2.2, n)
    loan = np.clip(annuity_principal(dti0 * income0, rate0, term), 150_000, 6_000_000).round(-3)
    ltv0 = np.clip(rng.normal(0.85, 0.12, n) + 0.03 * (emp > 0), 0.40, 1.00)
    prop0 = (loan / ltv0).round(-3)
    dti0 = pmt(loan, rate0, term) / income0

    # ---------------- state ----------------
    bal = loan.copy().astype(float)
    st = np.zeros(n, dtype=np.int8)
    closed = np.zeros(n, dtype=bool)
    mid = np.zeros(n, dtype=np.int16)           # months in default
    msc = np.full(n, -1, dtype=np.int16)         # months since cure from default (-1 never cured)
    ever_cured = np.zeros(n, dtype=bool)
    lag = np.zeros(n, dtype=np.int16)
    vol = np.zeros(n, dtype=bool)
    fsd = np.zeros(n); legal = np.zeros(n)
    def_row = -np.ones(n, dtype=np.int64)        # row into defaults list for open default
    frailty = 0.0
    panel_parts, d_rows = [], []

    covid_m = {i for i in range(M) if idx[i].year == 2020 and 4 <= idx[i].month <= 9}

    for m in range(1, M):
        frailty = d["frailty_rho"] * frailty + rng.normal(0, d["frailty_sd"])
        act = np.flatnonzero((orig < m) & ~closed)
        if act.size == 0:
            continue
        age = m - orig[act]
        rate = prime[m] + spread[act]
        s = st[act]
        b = bal[act]
        remaining = np.maximum(term - age + 1, 1)
        inst = pmt(b, rate, remaining)
        # amortise paying loans; arrears loans capitalise interest; defaults frozen
        interest = b * rate / 12
        b_new = np.where(s == 0, b - (inst - interest), np.where(s < 3, b + interest, b))
        bal[act] = np.maximum(b_new, 0)
        # prepayment / maturity (paying loans only)
        prepay = (s == 0) & (age >= 6) & (rng.random(act.size) < p["smm_prepay"])
        mature = age >= term
        gone = prepay | mature
        closed[act[gone]] = True
        keep = ~gone
        act, age, rate, s, inst = act[keep], age[keep], rate[keep], s[keep], inst[keep]
        if act.size == 0:
            continue
        b = bal[act]
        cltv = b / (prop0[act] * hpi[m] / hpi[orig[act]])
        dti_t = inst / (income0[act] * inc[m] / inc[orig[act]])
        shock = (prime[m] - prime[orig[act]]) * 100
        eta = dgp_eta(d, score_z[act], dti_t, cltv, emp[act], age, unemp[m], shock,
                      d["covid_shock"] if m in covid_m else 0.0, frailty, ever_cured[act] & (msc[act] <= 12) & (msc[act] >= 0))
        sh = eta - d["b0"]
        u = rng.random(act.size)
        new_s = s.copy()
        # state 0
        k = s == 0
        new_s[k & (u < sigmoid(eta))] = 1
        # state 1
        k = s == 1
        pu = sigmoid(logit(d["p_up_30"]) + 0.5 * sh); pc = np.clip(d["p_cure_30"] * np.exp(-0.3 * sh), 0, 0.9 - pu)
        new_s[k & (u < pu)] = 2
        new_s[k & (u >= pu) & (u < pu + pc)] = 0
        # state 2
        k = s == 2
        pu = sigmoid(logit(d["p_up_60"]) + 0.5 * sh); pd_ = np.clip(d["p_down_60"] * np.exp(-0.3 * sh), 0, 0.9 - pu)
        new_s[k & (u < pu)] = 3
        new_s[k & (u >= pu) & (u < pu + pd_)] = 1
        # state 3: cure window / resolution
        k3 = np.flatnonzero(s == 3)
        newdef = np.zeros(act.size, dtype=np.int8)
        closing = np.zeros(act.size, dtype=bool)
        if k3.size:
            li = act[k3]
            mid[li] += 1
            can = (mid[li] <= d["cure_window"])
            cure = can & (rng.random(k3.size) < d["p_cure_default"])
            new_s[k3[cure]] = 0
            ci = li[cure]
            ever_cured[ci] = True; msc[ci] = 0; mid[ci] = 0
            rows = def_row[ci]
            for r in rows:
                d_rows[r]["outcome"] = "cured"; d_rows[r]["res_m"] = m
            res = (~cure) & (mid[li] >= lag[li])
            ri = li[res]
            if ri.size:
                price = prop0[ri] * hpi[m] / hpi[orig[ri]] * (1 - fsd[ri])
                cash = np.minimum(np.maximum(price * (1 - d["agent_cost"]) - legal[ri] * d_ead(d_rows, def_row[ri]), 0),
                                  d_ead(d_rows, def_row[ri]))
                for j, r in enumerate(def_row[ri]):
                    d_rows[r]["outcome"] = "vol_sale" if vol[ri[j]] else "sale_exec"
                    d_rows[r]["res_m"] = m; d_rows[r]["recovery_cash"] = float(cash[j])
                closed[ri] = True
                closing[k3[res]] = True
        # entries into default
        ent = (s < 3) & (new_s == 3)
        if ent.any():
            ei = act[ent]
            mid[ei] = 0
            vol[ei] = rng.random(ei.size) < d["vol_sale_share"]
            lo, hi = d["sale_exec_lag"]; vlo, vhi = d["vol_sale_lag"]
            lag[ei] = np.where(vol[ei], rng.integers(vlo, vhi + 1, ei.size), rng.integers(lo, hi + 1, ei.size))
            mean = np.where(vol[ei], d["vol_fsd_mean"], d["fsd_mean"] + d["fsd_unemp_slope"] * max(unemp[m] - 28, 0))
            sd = d["fsd_sd"]
            kk = mean * (1 - mean) / sd ** 2 - 1
            fsd[ei] = rng.beta(np.maximum(mean * kk, 0.5), np.maximum((1 - mean) * kk, 0.5))
            legal[ei] = rng.uniform(*d["legal_cost"], ei.size)
            base = len(d_rows)
            for j, li in enumerate(ei):
                d_rows.append(dict(loan_idx=int(li), def_m=m, ead=float(bal[li]), rate_def=float(prime[m] + spread[li]),
                                   cltv_def=float(bal[li] / (prop0[li] * hpi[m] / hpi[orig[li]])), unemp_def=float(unemp[m]),
                                   hpi_chg_def=float(hpi[m] / hpi[orig[li]] - 1), age_def=int(m - orig[li]),
                                   outcome="open", res_m=-1, recovery_cash=np.nan, lag=int(lag[li]),
                                   true_fsd=float(fsd[li]), true_legal=float(legal[li]), vol=bool(vol[li])))
                def_row[li] = base + j
            newdef[ent] = 1
        # cured-loan probation counter
        pr = (msc[act] >= 0) & (~closing)
        msc[act[pr]] += 1
        st[act] = new_s
        live = ~closing
        panel_parts.append(pd.DataFrame({
            "loan_idx": act[live].astype(np.int32), "m": np.int16(m), "age": age[live].astype(np.int16),
            "bal": bal[act[live]].astype(np.float32), "rate": rate[live].astype(np.float32),
            "inst": inst[live].astype(np.float32), "st": new_s[live], "msc": msc[act[live]],
            "new_def": newdef[live], "true_eta": eta[live].astype(np.float32)}))

    panel = pd.concat(panel_parts, ignore_index=True)
    defaults = pd.DataFrame(d_rows)
    # realised economic LGD, discounted at the EIR over the workout lag
    sale = defaults["outcome"].isin(["vol_sale", "sale_exec"])
    t = (defaults["res_m"] - defaults["def_m"]).clip(lower=0)
    pv = defaults["recovery_cash"] * (1 + defaults["rate_def"] / 12) ** (-t)
    defaults["realised_lgd"] = np.where(sale, np.clip(1 - pv / defaults["ead"], 0, 1), np.nan)

    loans = pd.DataFrame({
        "loan_idx": np.arange(n), "orig_m": orig, "orig_month": idx[orig].astype(str), "loan_amount": loan,
        "property_value": prop0, "ltv0": ltv0, "income0": income0.round(0), "dti0": dti0,
        "bureau_score": score.round(0), "bureau_band": pd.cut(score, [0, 580, 640, 700, 760, 900],
                                                              labels=["E", "D", "C", "B", "A"]).astype(str),
        "emp_type": np.array(["permanent", "self_employed", "contract"])[emp], "emp_code": emp,
        "province": np.array(prov_names)[prov], "spread": spread, "rate0": rate0})
    if save:
        out = ROOT / cfg["paths"]["processed"]
        loans.to_csv(out / "loans.csv", index=False)
        panel.to_csv(out / "panel.csv.gz", index=False)
        defaults.to_csv(out / "defaults.csv", index=False)
    return loans, panel, defaults, macro


def d_ead(d_rows, rows):
    return np.array([d_rows[r]["ead"] for r in rows])


if __name__ == "__main__":
    loans, panel, defaults, macro = simulate()
    print(len(loans), len(panel), len(defaults))
