"""Fold and experiment ensembling (T2.4 / T3.6)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ARTIFACTS
from .labels import CL_COLS, EV_COLS


def _renorm(df, cols):
    p = df[cols].to_numpy()
    df[cols] = p / p.sum(1, keepdims=True)
    return df


def average(frames, weights=None, cols=None):
    """Weighted mean of probability frames aligned on datapointID (same ID order as frames[0])."""
    if not frames:
        raise ValueError("nothing to average")
    weights = np.ones(len(frames)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    base = frames[0][["datapointID"]].copy()
    cols = cols or [c for c in EV_COLS + CL_COLS if all(c in f.columns for f in frames)]
    acc = np.zeros((len(base), len(cols)))
    for w, f in zip(weights, frames):
        f = base.merge(f, on="datapointID", how="left")
        if f[cols].isna().any().any():
            raise ValueError("frames do not cover the same datapointIDs")
        acc += w * f[cols].to_numpy()
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(acc, columns=cols)], axis=1)
    out = _renorm(out, EV_COLS)
    if all(c in cols for c in CL_COLS):
        out = _renorm(out, CL_COLS)
    return out


def exp_dir(exp, root=None):
    return Path(root or ARTIFACTS) / exp


def load_oof(exp, root=None):
    return pd.read_csv(exp_dir(exp, root) / "oof.csv")


def load_test_avg(exp, root=None):
    """Average of all test_fold*.csv of one experiment."""
    files = sorted(exp_dir(exp, root).glob("test_fold*.csv"))
    if not files:
        raise FileNotFoundError(f"no test_fold*.csv in {exp_dir(exp, root)}")
    return average([pd.read_csv(f) for f in files])


def combine(exps, weights=None, root=None):
    """(oof, test) averaged over experiments. OOF uses only rows present in every experiment."""
    oofs = [load_oof(e, root) for e in exps]
    common = set(oofs[0]["datapointID"])
    for o in oofs[1:]:
        common &= set(o["datapointID"])
    oofs = [o[o["datapointID"].isin(common)].sort_values("datapointID").reset_index(drop=True) for o in oofs]
    tests = [load_test_avg(e, root) for e in exps]
    return average(oofs, weights), average(tests, weights)
