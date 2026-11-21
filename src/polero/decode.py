"""Decoders (T2.4) from 9-class evasion probabilities, plus per-class bias tuning for macro-F1."""
import numpy as np

from .labels import CLARITY, CLARITY_OF, EV_TO_CL_IDX, EVASION
from .metrics import clarity_f1, evasion_f1

EV_TO_CL = np.array(EV_TO_CL_IDX)


def group_probs(P):
    """Sum 9-class probabilities into the 3 clarity groups."""
    G = np.zeros((len(P), len(CLARITY)))
    for j, c in enumerate(EV_TO_CL):
        G[:, c] += P[:, j]
    return G


def decode_a(P, bias=None):
    """(a) argmax evasion, then map up."""
    S = np.log(np.clip(P, 1e-9, 1)) + (0 if bias is None else bias)
    ev = [EVASION[i] for i in S.argmax(1)]
    return ev, [CLARITY_OF[e] for e in ev]


def decode_b(P, cl_bias=None):
    """(b) clarity = argmax of group sums; evasion = argmax within that group (always consistent)."""
    G = np.log(np.clip(group_probs(P), 1e-9, 1)) + (0 if cl_bias is None else cl_bias)
    cl_idx = G.argmax(1)
    ev = []
    for i, c in enumerate(cl_idx):
        masked = np.where(EV_TO_CL == c, P[i], -1)
        ev.append(EVASION[int(masked.argmax())])
    return ev, [CLARITY[c] for c in cl_idx]


def tune_bias(S, y_idx, n_classes, score_fn, grid=np.linspace(-3, 3, 25), rounds=3):
    """Coordinate ascent on additive per-class biases to maximise score_fn(y_idx, pred_idx)."""
    b = np.zeros(n_classes)
    best = score_fn(y_idx, (S + b).argmax(1))
    for _ in range(rounds):
        improved = False
        for k in range(n_classes):
            for v in grid:
                trial = b.copy()
                trial[k] = v
                s = score_fn(y_idx, (S + trial).argmax(1))
                if s > best + 1e-9:
                    best, b, improved = s, trial, True
        if not improved:
            break
    return b, best


def _ev_score(y, p):
    return evasion_f1([EVASION[i] for i in y], [EVASION[i] for i in p])


def _cl_score(y, p):
    return clarity_f1([CLARITY[i] for i in y], [CLARITY[i] for i in p])


def fit_biases(P, y_ev):
    """Fit evasion bias (9) and clarity-group bias (3) on probabilities P with labels y_ev (strings)."""
    y9 = np.array([EVASION.index(v) for v in y_ev])
    y3 = EV_TO_CL[y9]
    S9 = np.log(np.clip(P, 1e-9, 1))
    S3 = np.log(np.clip(group_probs(P), 1e-9, 1))
    b9, _ = tune_bias(S9, y9, 9, _ev_score)
    b3, _ = tune_bias(S3, y3, 3, _cl_score)
    return b9, b3


def decode_tuned(P, b9, b3):
    """Evasion = argmax(logP + b9); clarity = argmax(log group(P) + b3). May be inconsistent."""
    ev, _ = decode_a(P, b9)
    _, cl = decode_b(P, b3)
    return ev, cl


def nested_bias_eval(P, y_ev, folds):
    """Honest estimate of bias tuning: fit on OOF of the other folds, apply to the held-out fold."""
    y_ev = np.asarray(y_ev)
    ev_out = np.empty(len(P), dtype=object)
    cl_out = np.empty(len(P), dtype=object)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        b9, b3 = fit_biases(P[tr], y_ev[tr])
        ev, cl = decode_tuned(P[va], b9, b3)
        ev_out[va], cl_out[va] = ev, cl
    y_cl = [CLARITY_OF[v] for v in y_ev]
    return {
        "E_nested_bias": evasion_f1(list(y_ev), list(ev_out)),
        "C_nested_bias": clarity_f1(y_cl, list(cl_out)),
        "consistent_frac": float(np.mean([CLARITY_OF[e] == c for e, c in zip(ev_out, cl_out)])),
    }
