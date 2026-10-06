"""Forward projection of loan-level PD / EAD / discount factors under a macro path."""
import numpy as np
import pandas as pd
from .ead import next_balance
from .features import make_X
from .pd_model import chain_pd
from .utils import pmt

TERM = 240


class MacroPath:
    """Absolute-month-indexed arrays: history then forecast (month 0 = 2012-01)."""

    def __init__(self, prime, unemp, hpi, income_idx):
        self.prime, self.unemp, self.hpi, self.income_idx = map(np.asarray, (prime, unemp, hpi, income_idx))

    @classmethod
    def from_history(cls, macro, future=None):
        h = macro[["prime", "unemp", "hpi", "income_idx"]]
        if future is not None:
            h = pd.concat([h, future[["prime", "unemp", "hpi", "income_idx"]]])
        return cls(h["prime"].values, h["unemp"].values, h["hpi"].values, h["income_idx"].values)

    def at(self, t, m0, flat):
        t = m0 if flat else min(t, len(self.prime) - 1)
        return t


def snapshot(loans, ff, m0):
    """Loans open at month m0 (from feature frame rows)."""
    s = ff[ff["m"] == m0].reset_index(drop=True)
    li = s["loan_idx"].values
    for c, src in [("spread", "spread"), ("prop0", "property_value"), ("income0", "income0"), ("orig_m", "orig_m")]:
        s[c] = loans[src].values[li]
    return s


def project(model, snap, path, m0, macro_hist, horizon=240, flat=False, smm=0.004, horizon_cap=None):
    """Returns dict with pd12, pd_life, mass (N,H), ead (N,H), dfac (N,H)."""
    n = len(snap)
    age0 = snap["age"].values.astype(float)
    bal = snap["bal"].values.astype(float)
    spread = snap["spread"].values
    om = snap["orig_m"].values.astype(int)
    prime_orig = path.prime[om]
    score_z, ltv0, emp = snap["score_z"].values, snap["ltv0"].values, snap["emp"].values
    prop0, inc0 = snap["prop0"].values, snap["income0"].values
    msc0 = snap["msc"].values
    st0 = np.minimum(snap["st"].values, 2)
    H = horizon
    ead = np.zeros((n, H), dtype=np.float32)
    dfac = np.zeros((n, H), dtype=np.float32)
    ratem = np.zeros((n, H), dtype=np.float32)
    state = {"bal": bal.copy(), "cum_df": np.ones(n)}
    n_rem_total = np.maximum(TERM - age0, 0)

    def inputs(j):
        t = m0 + j
        tt = path.at(t, m0, flat)
        tm12 = path.at(t - 12, m0, flat) if not flat else m0
        # in flat mode du12 is frozen at the m0 observed value
        du12 = (path.unemp[tt] - path.unemp[max(tm12, 0)]) if not flat else (path.unemp[m0] - path.unemp[max(m0 - 12, 0)])
        age = age0 + j
        rate = path.prime[tt] / 100.0 + spread
        b = state["bal"]
        rem = np.maximum(TERM - age, 1)
        inst = pmt(b, rate, rem)
        cltv = b / (prop0 * path.hpi[tt] / path.hpi[om])
        dti = inst / (inc0 * path.income_idx[tt] / path.income_idx[om])
        if j == 0 and "dti" in snap:
            dti = snap["dti"].values
        ead[:, j] = b
        ratem[:, j] = rate
        state["cum_df"] = state["cum_df"] / (1 + rate / 12.0)
        dfac[:, j] = state["cum_df"]
        out = dict(age=age, score_z=score_z, ltv0=ltv0, cltv=cltv, dti=dti, emp=emp,
                   prime_shock=path.prime[tt] - path.prime[om], unemp=np.full(n, path.unemp[tt]),
                   du12=np.full(n, du12), msc=np.where(msc0 >= 0, msc0 + j, -1))
        # advance balance one month at next month's rate
        rn = path.prime[path.at(t + 1, m0, flat)] / 100.0 + spread
        state["bal"] = next_balance(b, rn, rem)
        return out

    pd_h, mass = chain_pd(model, inputs, st0, msc0, H, smm=smm)
    # horizon mask: loans cannot default after contractual maturity
    steps = np.arange(H)[None, :]
    mass = np.where(steps < n_rem_total[:, None], mass, 0.0)
    pd12 = mass[:, :12].sum(axis=1)
    return dict(pd12=pd12, pd_life=mass.sum(axis=1), mass=mass, ead=ead, dfac=dfac, rate=ratem)
