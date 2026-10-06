"""IFRS 9 staging. Thresholds in config.yaml (justified via the sensitivity table in outputs/)."""
import numpy as np
import pandas as pd


def assign_stage(st, msc, pd12_now, pd12_orig, cfg_st):
    """Vectorised. st: dpd state 0..3, msc: months since cure from default (-1 never).
    Returns (stage array 1/2/3, reason array)."""
    st, msc = np.asarray(st), np.asarray(msc)
    in_default = st >= cfg_st["default_dpd_state"]
    probation = (msc >= 0) & (msc <= cfg_st["probation_months"]) & ~in_default
    backstop = (st >= cfg_st["backstop_dpd_state"]) & ~in_default
    sicr = ((np.asarray(pd12_now) >= cfg_st["sicr_pd_multiple"] * np.asarray(pd12_orig))
            & (np.asarray(pd12_now) - np.asarray(pd12_orig) >= cfg_st["sicr_pd_absolute"]))
    stage = np.ones(len(st), dtype=int)
    reason = np.full(len(st), "performing", dtype=object)
    s2 = backstop | sicr | probation
    stage[s2] = 2
    reason[sicr & ~backstop] = "SICR"
    reason[backstop] = "30dpd backstop"
    stage[in_default | probation] = 3
    reason[in_default] = "default 90dpd"
    reason[probation] = "cure probation"
    # probation loans are Stage 3 (cured from default but not yet out of probation)
    return stage, reason


def migration_matrix(stage_from, stage_to):
    """Stage_to may contain 0 for exited (prepaid / matured / resolved). Row-normalised."""
    t = pd.crosstab(pd.Series(stage_from, name="from"), pd.Series(stage_to, name="to"))
    labels = {0: "Exited", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}
    t = t.rename(index=labels, columns=labels)
    return t, t.div(t.sum(axis=1), axis=0)
