"""Consistent decoding and per-class bias tuning (T2.4).

Decoders (all return evasion and clarity label strings that are consistent):
  a: evasion = argmax p_ev; clarity mapped up from it.
  b: clarity = argmax of p_ev summed per clarity group; evasion = argmax inside that group.
  c: like b, but the group scores are blended with the clarity head:
     s = alpha * p_cl + (1 - alpha) * group_sum(p_ev).
Bias tuning adds a per-class offset to log p_ev before decoding, fitted to maximise
evasion macro-F1 + clarity macro-F1 on OOF (nested CV for honest reporting).
"""
import numpy as np

from .labels import CLARITY, CLARITY_IDX_OF, EVASION, GROUP
from .metrics import clarity_f1, evasion_f1

EV = np.array(EVASION, dtype=object)
CL = np.array(CLARITY, dtype=object)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_bias(pe, bias):
    if bias is None:
        return pe
    return _softmax(np.log(np.clip(pe, 1e-9, 1)) + np.asarray(bias)[None, :])


def decode_idx(pe, pc=None, decoder="a", alpha=0.5, bias=None):
    """Return (evasion_idx, clarity_idx) arrays."""
    pe = apply_bias(np.asarray(pe, dtype=float), bias)
    if decoder == "a":
        e = pe.argmax(1)
        return e, CLARITY_IDX_OF[e]
    g = pe @ GROUP.T  # [n, 3] group sums
    if decoder == "c":
        if pc is None:
            raise ValueError("decoder c needs clarity probabilities")
        g = alpha * np.asarray(pc, dtype=float) + (1 - alpha) * g
    elif decoder != "b":
        raise ValueError(f"unknown decoder {decoder}")
    c = g.argmax(1)
    masked = np.where(GROUP[c] > 0, pe, -1.0)
    return masked.argmax(1), c


def decode(pe, pc=None, decoder="a", alpha=0.5, bias=None):
    e, c = decode_idx(pe, pc, decoder, alpha, bias)
    return EV[e].tolist(), CL[c].tolist()


def score(y_e, y_c, pe, pc=None, decoder="a", alpha=0.5, bias=None):
    ev, cl = decode(pe, pc, decoder, alpha, bias)
    return evasion_f1(y_e, ev) + clarity_f1(y_c, cl)


def fit_bias(y_e, y_c, pe, pc=None, decoder="a", alpha=0.5,
             grid=np.linspace(-2.0, 2.0, 21), rounds=3, l2=0.0):
    """Coordinate ascent over per-class log-offsets (Explicit stays at 0)."""
    bias = np.zeros(len(EVASION))
    best = score(y_e, y_c, pe, pc, decoder, alpha, bias)
    for _ in range(rounds):
        improved = False
        for j in range(1, len(EVASION)):
            for v in grid:
                b = bias.copy()
                b[j] = v
                s = score(y_e, y_c, pe, pc, decoder, alpha, b) - l2 * np.sum(b ** 2)
                if s > best + 1e-9:
                    best, bias, improved = s, b, True
        if not improved:
            break
    return bias


def nested_bias_oof(y_e, y_c, pe, pc, fold, decoder="a", alpha=0.5, **kw):
    """Honest OOF estimate of bias tuning: for each fold, fit on the others."""
    y_e, y_c, fold = np.asarray(y_e, dtype=object), np.asarray(y_c, dtype=object), np.asarray(fold)
    ev = np.empty(len(fold), dtype=object)
    cl = np.empty(len(fold), dtype=object)
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        b = fit_bias(y_e[tr], y_c[tr], pe[tr], None if pc is None else pc[tr], decoder, alpha, **kw)
        e, c = decode(pe[va], None if pc is None else pc[va], decoder, alpha, b)
        ev[va], cl[va] = e, c
    return ev.tolist(), cl.tolist()
