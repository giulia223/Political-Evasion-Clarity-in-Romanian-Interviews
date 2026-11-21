"""Fold/experiment ensembling, consistent decoders and per-class bias tuning (T2.4).

Decoders (all return a consistent (evasion, clarity) pair):
  a  evasion = argmax p_ev; clarity = CLARITY_OF[evasion]
  b  clarity = argmax of p_ev summed per clarity group; evasion = argmax p_ev within that group
  c  like b, but the group scores are alpha * p_cl (clarity head) + (1 - alpha) * grouped p_ev

Bias tuning adds a per-class offset to log p_ev (Explicit fixed at 0) chosen by coordinate ascent
to maximise evasion macro-F1 + clarity macro-F1. Its honest estimate is *nested*: offsets for
fold k are fitted on the other folds' OOF rows only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from polero.labels import CLARITY, CLARITY_TO_ID, EVASION, EVASION_GROUP, EVASION_TO_ID
from polero.metrics import CL_COLS, EV_COLS

DECODERS = ("a", "b", "c")
GROUP = np.array(EVASION_GROUP)                       # evasion index -> clarity index
MASK = np.stack([GROUP == c for c in range(3)])       # [3, 9]
BIAS_GRID = np.linspace(-3, 3, 25)
ALPHA_GRID = np.round(np.linspace(0, 1, 11), 2)


# ---------- loading and averaging ----------
def load_oof(exp_dir):
    return pd.read_csv(Path(exp_dir) / "oof.csv")


def load_test(exp_dir):
    """Mean of the experiment's test_fold*.csv probabilities (fold ensemble)."""
    parts = sorted(Path(exp_dir).glob("test_fold*.csv"))
    if not parts:
        raise FileNotFoundError(f"no test_fold*.csv in {exp_dir}")
    return blend([pd.read_csv(p) for p in parts])


def blend(frames, weights=None):
    """Weighted average of probability frames aligned on datapointID (same id set required)."""
    weights = np.ones(len(frames)) if weights is None else np.asarray(weights, dtype=float)
    if len(weights) != len(frames) or (weights < 0).any() or weights.sum() == 0:
        raise ValueError("need one non-negative weight per frame, not all zero")
    weights = weights / weights.sum()
    base = frames[0].sort_values("datapointID").reset_index(drop=True)
    cols = [c for c in EV_COLS + CL_COLS if all(c in f.columns for f in frames)]
    out = base[["datapointID"]].copy()
    for c in cols:
        out[c] = 0.0
    for f, w in zip(frames, weights):
        f = f.sort_values("datapointID").reset_index(drop=True)
        if not f.datapointID.equals(base.datapointID):
            raise ValueError("frames cover different datapointIDs")
        out[cols] += w * f[cols].to_numpy()
    return out


# ---------- decoding ----------
def decode(pe, pc=None, decoder="a", alpha=0.5, bias=None):
    """pe [n,9] evasion probs, pc [n,3] clarity-head probs. Returns (evasion idx, clarity idx)."""
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}")
    if bias is not None:
        logits = np.log(np.clip(pe, 1e-12, None)) + bias
        pe = np.exp(logits - logits.max(1, keepdims=True))
        pe /= pe.sum(1, keepdims=True)
    if decoder == "a":
        ev = pe.argmax(1)
        return ev, GROUP[ev]
    group = pe @ MASK.T.astype(float)                  # [n, 3]
    if decoder == "c":
        if pc is None:
            raise ValueError("decoder c needs clarity-head probabilities")
        group = alpha * pc + (1 - alpha) * group
    cl = group.argmax(1)
    ev = np.where(MASK[cl], pe, -1).argmax(1)          # best evasion label inside the chosen group
    return ev, cl


def macro_f1_idx(y, p, n):
    """sklearn-equivalent macro-F1 (union of classes, zero_division=0) on integer labels, fast."""
    cm = np.bincount(y * n + p, minlength=n * n).reshape(n, n)
    tp, fp, fn = np.diag(cm), cm.sum(0) - np.diag(cm), cm.sum(1) - np.diag(cm)
    present = (cm.sum(0) + cm.sum(1)) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return float(f1[present].mean())


def objective(y_ev, ev, cl):
    """Evasion macro-F1 + clarity macro-F1 (gold clarity derived from gold evasion)."""
    return macro_f1_idx(y_ev, ev, 9) + macro_f1_idx(GROUP[y_ev], cl, 3)


# ---------- tuning ----------
def tune_bias(pe, y_ev, pc=None, decoder="a", alpha=0.5, rounds=3):
    bias = np.zeros(9)
    best = objective(y_ev, *decode(pe, pc, decoder, alpha, bias))
    for _ in range(rounds):
        improved = False
        for j in range(1, 9):  # Explicit (index 0) is the reference
            for b in BIAS_GRID:
                trial = bias.copy()
                trial[j] = b
                s = objective(y_ev, *decode(pe, pc, decoder, alpha, trial))
                if s > best + 1e-12:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    return bias


def tune_alpha(pe, pc, y_ev, bias=None):
    scores = [objective(y_ev, *decode(pe, pc, "c", a, bias)) for a in ALPHA_GRID]
    return float(ALPHA_GRID[int(np.argmax(scores))])


def fit_params(pe, pc, y_ev, decoder, use_bias):
    alpha = tune_alpha(pe, pc, y_ev) if decoder == "c" else 0.5
    bias = tune_bias(pe, y_ev, pc, decoder, alpha) if use_bias else None
    if decoder == "c" and use_bias:
        alpha = tune_alpha(pe, pc, y_ev, bias)
    return {"alpha": alpha, "bias": bias}


def nested_predict(pe, pc, y_ev, folds, decoder, use_bias):
    """Predictions where each fold's tuned parameters come only from the other folds."""
    ev, cl = np.zeros(len(y_ev), int), np.zeros(len(y_ev), int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        prm = fit_params(pe[tr], None if pc is None else pc[tr], y_ev[tr], decoder, use_bias)
        ev[va], cl[va] = decode(pe[va], None if pc is None else pc[va], decoder, prm["alpha"], prm["bias"])
    return ev, cl


def oof_arrays(oof, train):
    """Align an OOF frame with gold labels and folds. Returns pe, pc (or None), y_ev, folds."""
    g = train.set_index("datapointID").loc[oof.datapointID]
    pe = oof[EV_COLS].to_numpy()
    pc = oof[CL_COLS].to_numpy() if all(c in oof for c in CL_COLS) else None
    folds = oof["fold"].to_numpy() if "fold" in oof else g["fold"].to_numpy()
    return pe, pc, g.evasion_label.map(EVASION_TO_ID).to_numpy(), folds


def compare_decoders(oof, train):
    """Table of OOF scores for every decoder, without and with nested bias tuning."""
    pe, pc, y, folds = oof_arrays(oof, train)
    rows = []
    for dec in DECODERS:
        if dec == "c" and pc is None:
            continue
        for use_bias in (False, True):
            if dec == "c" or use_bias:
                ev, cl = nested_predict(pe, pc, y, folds, dec, use_bias)
            else:
                ev, cl = decode(pe, pc, dec)
            rows.append({"decoder": dec, "bias": "nested" if use_bias else "none",
                         "evasion_f1": macro_f1_idx(y, ev, 9), "clarity_f1": macro_f1_idx(GROUP[y], cl, 3)})
    df = pd.DataFrame(rows)
    df["sum"] = df.evasion_f1 + df.clarity_f1
    return df


def labels_from_idx(ev, cl):
    return [EVASION[i] for i in ev], [CLARITY[i] for i in cl]
