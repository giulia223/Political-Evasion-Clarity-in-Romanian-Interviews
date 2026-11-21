"""Macro-F1 metrics and OOF evaluation (T0.5).

Classes that are never predicted count as F1 = 0 (all labels are always passed).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .labels import CL_COLS, CLARITY, CLARITY_OF, EV_COLS, EVASION


def evasion_f1(y, p):
    return f1_score(y, p, labels=EVASION, average="macro", zero_division=0)


def clarity_f1(y, p):
    return f1_score(y, p, labels=CLARITY, average="macro", zero_division=0)


def mapped_clarity_f1(y_evasion, p_evasion):
    return clarity_f1([CLARITY_OF[e] for e in y_evasion], [CLARITY_OF[e] for e in p_evasion])


def per_class(y, p, labels):
    return classification_report(y, p, labels=labels, output_dict=True, zero_division=0)


def confusion(y, p, labels):
    return pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels)


def evaluate_oof(probs_df, train, decoder="a"):
    """Evaluate OOF probabilities against train labels.

    probs_df: datapointID + 9 `p_ev_*` columns (+ optional 3 `p_cl_*`).
    Returns (metrics dict, markdown table, {name: confusion DataFrame}).
    """
    from .decode import decode  # local import avoids a cycle

    missing = [c for c in EV_COLS if c not in probs_df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = train[["datapointID", "evasion_label", "clarity_label"]].merge(probs_df, on="datapointID", how="inner")
    if len(df) != len(probs_df):
        raise ValueError("some datapointIDs are not in train")
    pe = df[EV_COLS].to_numpy()
    pc = df[CL_COLS].to_numpy() if all(c in df.columns for c in CL_COLS) else None
    ev_pred, cl_pred = decode(pe, pc, decoder=decoder)
    y_e, y_c = df["evasion_label"].tolist(), df["clarity_label"].tolist()
    m = {
        "n": len(df),
        "decoder": decoder,
        "evasion_f1": evasion_f1(y_e, ev_pred),
        "clarity_f1": clarity_f1(y_c, cl_pred),
        "mapped_clarity_f1": mapped_clarity_f1(y_e, ev_pred),
    }
    if pc is not None:
        m["direct_clarity_f1"] = clarity_f1(y_c, [CLARITY[i] for i in pc.argmax(1)])
    m["score"] = m["evasion_f1"] + m["clarity_f1"]
    rep = per_class(y_e, ev_pred, EVASION)
    rows = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for l in EVASION:
        r = rep[l]
        rows.append(f"| {l} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    m["per_class_f1"] = {l: rep[l]["f1-score"] for l in EVASION}
    cms = {"evasion": confusion(y_e, ev_pred, EVASION), "clarity": confusion(y_c, cl_pred, CLARITY)}
    return m, "\n".join(rows), cms


def fold_metrics(probs_df, train, folds, decoder="a"):
    """Per-fold metrics (to report variance)."""
    out = []
    df = probs_df.merge(folds, on="datapointID")
    for k, g in df.groupby("fold"):
        m, _, _ = evaluate_oof(g.drop(columns="fold"), train, decoder)
        out.append({"fold": int(k), "evasion_f1": m["evasion_f1"], "clarity_f1": m["clarity_f1"]})
    return pd.DataFrame(out)


def bootstrap_ci(y, p, fn, n=1000, seed=42):
    """95% bootstrap CI of a metric fn(y, p)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
