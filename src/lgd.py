"""LGD = (1 - P(cure)) x LGD_no_cure, point-in-time (NOT downturn), discounted at the EIR.

P(cure): logistic on cLTV at default, unemployment, bureau score.
LGD_no_cure: linear model on realised economic LGD of resolved sales (recoveries discounted at EIR
over the workout lag), features 1/cLTV, cLTV, unemployment, EIR (v1.1: added after the backtest exposed it). Cured loans are assumed to lose nothing.
Known bias (documented): only RESOLVED workouts have an observed LGD, so recent defaults are under-represented.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from .utils import sigmoid


def _lgd_X(cltv, unemp, rate):
    """rate = EIR (decimal): a higher discount rate lowers the PV of a 1-3 year workout and so raises economic LGD."""
    c = np.clip(cltv, 0.2, 2.0)
    c, u, r = np.broadcast_arrays(c, np.asarray(unemp, dtype=float), np.asarray(rate, dtype=float))
    return np.stack([1.0 / c, c, (u - 28.0) / 4.0, (r * 100 - 10.0) / 3.0], axis=-1)


def _cure_X(cltv, unemp, score_z):
    c = np.clip(cltv, 0.2, 2.0)
    c, u, s = np.broadcast_arrays(c, np.asarray(unemp, dtype=float), np.asarray(score_z, dtype=float))
    return np.stack([c, (u - 28.0) / 4.0, s], axis=-1)


class LGDModel:
    def fit(self, defaults, loans, macro, train_end_m, cure_window=9, last_m=167):
        d = defaults.merge(loans[["loan_idx", "bureau_score"]], on="loan_idx")
        d["score_z"] = (d["bureau_score"] - 670) / 75
        # cure model: defaults whose cure window is complete and whose default date is in the training period
        c = d[(d["def_m"] <= train_end_m) & (d["def_m"] + cure_window <= last_m)]
        yc = (c["outcome"] == "cured").astype(int).values
        Xc = _cure_X(c["cltv_def"].values, c["unemp_def"].values, c["score_z"].values)
        self.sc_c = StandardScaler().fit(Xc)
        self.cure = LogisticRegression(C=10).fit(self.sc_c.transform(Xc), yc)
        # LGD given no cure: resolved sales known by the training cut-off
        s = d[d["outcome"].isin(["vol_sale", "sale_exec"]) & (d["res_m"] <= train_end_m)]
        Xl = _lgd_X(s["cltv_def"].values, s["unemp_def"].values, s["rate_def"].values)
        self.lgd = Ridge(alpha=1.0).fit(Xl, s["realised_lgd"].values)
        self.n_cure, self.n_lgd = len(c), len(s)
        return self

    def p_cure(self, cltv, unemp, score_z):
        X = _cure_X(cltv, unemp, score_z)
        sh = X.shape
        return self.cure.predict_proba(self.sc_c.transform(X.reshape(-1, 3)))[:, 1].reshape(sh[:-1])

    def lgd_no_cure(self, cltv, unemp, rate):
        X = _lgd_X(cltv, unemp, rate)
        sh = X.shape
        return np.clip(self.lgd.predict(X.reshape(-1, 4)).reshape(sh[:-1]), 0.0, 1.0)

    def expected_lgd(self, cltv, unemp, score_z, rate, include_cure=True):
        l = self.lgd_no_cure(cltv, unemp, rate)
        return (1 - self.p_cure(cltv, unemp, score_z)) * l if include_cure else l
