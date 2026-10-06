"""Shared helpers: config, month indexing, annuity maths."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def month_index(start="2012-01", end="2025-12"):
    return pd.period_range(start, end, freq="M")


def pmt(balance, annual_rate, n_remaining):
    """Level instalment that amortises `balance` over `n_remaining` months."""
    r = np.asarray(annual_rate, dtype=float) / 12.0
    n = np.asarray(n_remaining, dtype=float)
    bal = np.asarray(balance, dtype=float)
    small = np.abs(r) < 1e-12
    r_safe = np.where(small, 1.0, r)
    out = bal * r_safe / (1.0 - (1.0 + r_safe) ** (-n))
    return np.where(small, bal / np.maximum(n, 1), out)


def annuity_principal(instalment, annual_rate, n):
    r = np.asarray(annual_rate, dtype=float) / 12.0
    return instalment * (1.0 - (1.0 + r) ** (-np.asarray(n, dtype=float))) / r


def discount_factor(annual_rate, months):
    """Monthly-compounded discount factor at the effective interest rate."""
    return (1.0 + np.asarray(annual_rate, dtype=float) / 12.0) ** (-np.asarray(months, dtype=float))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))
