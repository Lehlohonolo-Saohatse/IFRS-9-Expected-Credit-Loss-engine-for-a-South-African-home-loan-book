"""PD models.

1. WoE logistic SCORECARD (benchmark, 12m PD, no macro).
2. Multi-state discrete-time HAZARD model (production): three logistic transition models
   0->1 (arrears entry), 1->{0,1,2}, 2->{1,2,3}. Default needs three consecutive monthly
   roll-ups, so a one-step 'default next month' hazard is ~0 for current loans; the chain is
   what turns monthly hazards into 12m / lifetime PD:  PD_H = sum_j mass entering default at step j.
3. Gradient-boosting CHALLENGER on the 12m target.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from .features import FEATURES, make_X
from .utils import sigmoid


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks_stat(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return ks_2samp(p[y == 1], p[y == 0]).statistic


# ------------------------------------------------------------------ scorecard
class WoEScorecard:
    NUM = ["score_z", "cltv", "ltv0", "dti", "age"]
    CAT = ["emp", "st"]

    def __init__(self, n_bins=8, C=1.0):
        self.n_bins, self.C = n_bins, C

    def _prep(self, df):
        return df[self.NUM + self.CAT]

    def fit(self, df, y):
        X = self._prep(df).copy(); y = np.asarray(y)
        self.edges, self.woe, rows = {}, {}, []
        bad, good = y.sum(), (1 - y).sum()
        for c in self.NUM:
            e = np.unique(np.quantile(X[c], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges[c] = e
            self.woe[c] = self._woe(np.digitize(X[c], e), y, bad, good, rows, c)
        for c in self.CAT:
            self.woe[c] = self._woe(X[c].values, y, bad, good, rows, c)
        self.iv_table = pd.DataFrame(rows, columns=["variable", "bin", "n", "bad_rate", "woe", "iv_part"])
        Z = self._transform(df)
        self.lr = LogisticRegression(C=self.C, max_iter=500).fit(Z, y)
        return self

    @staticmethod
    def _woe(codes, y, bad, good, rows, name):
        out = {}
        for k in np.unique(codes):
            mk = codes == k
            b, g = y[mk].sum() + 0.5, (1 - y[mk]).sum() + 0.5
            w = np.log((g / (good + 1)) / (b / (bad + 1)))
            out[k] = w
            rows.append((name, int(k), int(mk.sum()), float(y[mk].mean()), float(w),
                         float((g / (good + 1) - b / (bad + 1)) * w)))
        return out

    def _transform(self, df):
        X = self._prep(df); Z = []
        for c in self.NUM:
            codes = np.digitize(X[c], self.edges[c])
            Z.append(np.array([self.woe[c].get(k, 0.0) for k in codes]))
        for c in self.CAT:
            Z.append(np.array([self.woe[c].get(k, 0.0) for k in X[c].values]))
        return np.column_stack(Z)

    def predict(self, df):
        return self.lr.predict_proba(self._transform(df))[:, 1]

    def iv(self):
        return self.iv_table.groupby("variable")["iv_part"].sum().sort_values(ascending=False)


# ------------------------------------------------------------------ Markov hazard model
class _Logit:
    """Standardised (multinomial) logistic regression with manual prediction (fast, vectorised)."""

    def __init__(self, C=10.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(C=self.C, max_iter=300).fit(self.sc.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.m.classes_
        return self

    def proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))


class MarkovPD:
    def __init__(self, neg_sample=0.3, seed=1):
        self.neg_sample, self.seed = neg_sample, seed

    def fit(self, df):
        """df: feature_frame rows (train period). Uses st_next."""
        rng = np.random.default_rng(self.seed)
        d = df[df["st_next"].notna() & (df["st"] < 3)]
        # --- state 0: arrears entry, negatives subsampled, intercept corrected
        s0 = d[d["st"] == 0]
        y0 = (s0["st_next"] == 1).astype(int).values
        keep = (y0 == 1) | (rng.random(len(s0)) < self.neg_sample)
        self.m0 = _Logit().fit(make_X(s0[keep]), y0[keep])
        self._corr0 = np.log(self.neg_sample)
        # --- states 1 and 2: multinomial
        self.m1 = _Logit().fit(make_X(d[d["st"] == 1]), d.loc[d["st"] == 1, "st_next"].astype(int).values)
        self.m2 = _Logit().fit(make_X(d[d["st"] == 2]), d.loc[d["st"] == 2, "st_next"].astype(int).values)
        return self

    # transition probabilities for arrays -------------------------------------------------
    def p01(self, X):
        z = self.m0.m.decision_function(self.m0.sc.transform(X)) + self._corr0
        return sigmoid(z)

    def p1(self, X):   # returns (to0, stay1, to2)
        P = self.m1.proba(X); cl = list(self.m1.classes_)
        g = lambda c: P[:, cl.index(c)] if c in cl else np.zeros(len(X))
        return g(0), g(1), g(2)

    def p2(self, X):   # returns (to1 (incl 0), stay2, to3)
        P = self.m2.proba(X); cl = list(self.m2.classes_)
        g = lambda c: P[:, cl.index(c)] if c in cl else np.zeros(len(X))
        return g(0) + g(1), g(2), g(3)


def chain_pd(model, step_inputs, st0, msc0, horizon, smm=0.0):
    """Propagate the 3-state chain `horizon` steps. step_inputs(j) -> dict of arrays (features at obs month t0+j)
    given the *current* msc handled by the caller. Returns cumulative PD and per-step default mass (N x H)."""
    n = len(st0)
    v = np.zeros((n, 3)); v[np.arange(n), st0.astype(int)] = 1.0
    mass = np.zeros((n, horizon))
    for j in range(horizon):
        f = step_inputs(j)
        X = make_X(f)
        a = model.p01(X); c1, s1, u1 = model.p1(X); d2, s2, u2 = model.p2(X)
        v0 = v[:, 0] * (1 - smm)
        n0 = v0 * (1 - a) + v[:, 1] * c1 + v[:, 2] * 0.0
        n1 = v0 * a + v[:, 1] * s1 + v[:, 2] * d2
        n2 = v[:, 1] * u1 + v[:, 2] * s2
        mass[:, j] = v[:, 2] * u2
        v = np.column_stack([n0, n1, n2])
    return mass.sum(axis=1), mass


# ------------------------------------------------------------------ challenger
def fit_gbm(df_train, y):
    cols = FEATURES
    X = make_X(df_train)
    gb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                                        random_state=0)
    gb.fit(np.column_stack([X, df_train["st"].values]), y)
    return gb


def gbm_predict(gb, df):
    return gb.predict_proba(np.column_stack([make_X(df), df["st"].values]))[:, 1]
