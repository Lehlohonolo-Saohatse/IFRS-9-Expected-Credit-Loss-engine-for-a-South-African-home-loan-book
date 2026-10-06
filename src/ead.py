"""EAD: contractual amortisation on a variable-rate (prime-linked) loan + prepayment assumption.

Prepayment does NOT reduce EAD of a loan that defaults (the loan is still there); it reduces the
probability the loan survives to default. It is applied to the survival mass in pd_model.chain_pd.
"""
import numpy as np
from .utils import pmt


def next_balance(balance, annual_rate, n_remaining):
    """Balance after the next level payment, with `n_remaining` payments left (incl. this one)."""
    bal = np.asarray(balance, dtype=float)
    inst = pmt(bal, annual_rate, n_remaining)
    interest = bal * np.asarray(annual_rate, dtype=float) / 12.0
    return np.maximum(bal - (inst - interest), 0.0)


def schedule(balance, annual_rate, n):
    """Full contractual balance path at constant rate: array of length n+1 (opening balance first)."""
    path = [float(balance)]
    b = float(balance)
    for k in range(n):
        b = float(next_balance(b, annual_rate, n - k))
        path.append(b)
    return np.array(path)
