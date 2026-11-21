"""Macro-F1 metrics and OOF evaluation.

Macro-F1 follows sklearn's default exactly: averaged over the union of classes in y_true
and y_pred, with a never-predicted class scoring 0 (zero_division=0).

OOF probability files: column `datapointID`, then `p_ev_<label>` for all 9 evasion labels
and optionally `p_cl_<label>` for the 3 clarity labels (exact label strings).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from polero.labels import CLARITY, CLARITY_OF, EVASION, EVASION_GROUP

EV_COLS = [f"p_ev_{e}" for e in EVASION]
CL_COLS = [f"p_cl_{c}" for c in CLARITY]


def _check(labels, allowed, name):
    bad = set(labels) - set(allowed)
    if bad:
        raise ValueError(f"{name}: unknown labels {sorted(bad)}")


def macro_f1(y_true, y_pred):
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def evasion_f1(y, p):
    _check(y, EVASION, "y"), _check(p, EVASION, "p")
    return macro_f1(y, p)


def clarity_f1(y, p):
    _check(y, CLARITY, "y"), _check(p, CLARITY, "p")
    return macro_f1(y, p)


def mapped_clarity_f1(y_evasion, p_evasion):
    """Clarity macro-F1 after mapping both gold and predicted evasion labels up the taxonomy."""
    _check(y_evasion, EVASION, "y"), _check(p_evasion, EVASION, "p")
    return macro_f1([CLARITY_OF[e] for e in y_evasion], [CLARITY_OF[e] for e in p_evasion])


def per_class_report(y, p, labels):
    prec, rec, f1, sup = precision_recall_fscore_support(list(y), list(p), labels=list(labels), zero_division=0)
    pred = pd.Series(list(p)).value_counts()
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1, "support": sup,
                         "predicted": [int(pred.get(l, 0)) for l in labels]}, index=list(labels))


def confusion(y, p, labels):
    """Rows = gold, columns = predicted."""
    return pd.DataFrame(confusion_matrix(list(y), list(p), labels=list(labels)), index=list(labels), columns=list(labels))


def _probs(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    p = df[cols].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name}: probabilities must be finite and non-negative")
    if not np.allclose(p.sum(1), 1, atol=1e-3):
        raise ValueError(f"{name}: rows must sum to 1 (got range {p.sum(1).min():.4f}-{p.sum(1).max():.4f})")
    return p


def evaluate_oof(probs_df, train):
    """Score OOF probabilities against gold labels in `train` (rows matched on datapointID).

    Returns a dict with scalar metrics plus per-class reports and confusion matrices.
      evasion_f1          argmax over the 9 evasion probabilities
      mapped_clarity_f1   that argmax mapped up to clarity (decoder a)
      grouped_clarity_f1  argmax of evasion probabilities summed per clarity group (decoder b)
      clarity_head_f1     argmax of p_cl_* if the file has a clarity head
    """
    if probs_df.datapointID.duplicated().any():
        raise ValueError("duplicate datapointID in OOF file")
    gold = train.set_index("datapointID")
    unknown = set(probs_df.datapointID) - set(gold.index)
    if unknown:
        raise ValueError(f"{len(unknown)} OOF ids not in train, e.g. {sorted(unknown)[:5]}")
    g = gold.loc[probs_df.datapointID]
    y_ev, y_cl = g.evasion_label.tolist(), g.clarity_label.tolist()

    pe = _probs(probs_df, EV_COLS, "evasion")
    p_ev = [EVASION[i] for i in pe.argmax(1)]
    grouped = np.stack([pe[:, [i for i, gi in enumerate(EVASION_GROUP) if gi == c]].sum(1) for c in range(3)], 1)
    p_cl_grouped = [CLARITY[i] for i in grouped.argmax(1)]

    out = {
        "n": len(probs_df),
        "coverage": len(probs_df) / len(train),
        "evasion_f1": evasion_f1(y_ev, p_ev),
        "mapped_clarity_f1": mapped_clarity_f1(y_ev, p_ev),
        "grouped_clarity_f1": clarity_f1(y_cl, p_cl_grouped),
        "clarity_head_f1": None,
        "evasion_report": per_class_report(y_ev, p_ev, EVASION),
        "evasion_confusion": confusion(y_ev, p_ev, EVASION),
        "clarity_report": per_class_report(y_cl, [CLARITY_OF[e] for e in p_ev], CLARITY),
    }
    if all(c in probs_df.columns for c in CL_COLS):
        pc = _probs(probs_df, CL_COLS, "clarity")
        out["clarity_head_f1"] = clarity_f1(y_cl, [CLARITY[i] for i in pc.argmax(1)])
    return out


SCALARS = ["n", "coverage", "evasion_f1", "mapped_clarity_f1", "grouped_clarity_f1", "clarity_head_f1"]


def to_markdown(metrics):
    """Human-readable summary: scalar table, evasion per-class report, confusion matrix."""
    fmt = lambda v: "–" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["| metric | value |", "|---|---|"] + [f"| {k} | {fmt(metrics[k])} |" for k in SCALARS]
    rep = metrics["evasion_report"].round(3)
    lines += ["", "Evasion per class:", "", rep.to_markdown() if _has_tabulate() else rep.to_string()]
    cm = metrics["evasion_confusion"]
    lines += ["", "Evasion confusion (rows = gold, cols = predicted):", "", cm.to_markdown() if _has_tabulate() else cm.to_string()]
    return "\n".join(lines)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
