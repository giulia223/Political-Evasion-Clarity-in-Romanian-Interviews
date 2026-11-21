"""Metrics (T0.5): macro-F1 for evasion, clarity, and clarity mapped up from evasion."""
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from . import REPO_ROOT
from .labels import CL_COLS, CLARITY, CLARITY_OF, EV_COLS, EVASION

RESULTS_MD = REPO_ROOT / "docs" / "RESULTS.md"


def evasion_f1(y, p):
    return f1_score(y, p, labels=EVASION, average="macro", zero_division=0)


def clarity_f1(y, p):
    return f1_score(y, p, labels=CLARITY, average="macro", zero_division=0)


def mapped_clarity_f1(y_evasion, p_evasion):
    return clarity_f1([CLARITY_OF[v] for v in y_evasion], [CLARITY_OF[v] for v in p_evasion])


def per_class_f1(y, p, labels):
    return dict(zip(labels, f1_score(y, p, labels=labels, average=None, zero_division=0)))


def evaluate_preds(y_ev, p_ev, p_cl=None):
    """Metrics from label predictions. If p_cl is None, clarity is mapped from p_ev."""
    y_cl = [CLARITY_OF[v] for v in y_ev]
    p_cl = [CLARITY_OF[v] for v in p_ev] if p_cl is None else list(p_cl)
    return {
        "evasion_f1": evasion_f1(y_ev, p_ev),
        "clarity_f1": clarity_f1(y_cl, p_cl),
        "per_class_evasion": per_class_f1(y_ev, p_ev, EVASION),
        "per_class_clarity": per_class_f1(y_cl, p_cl, CLARITY),
    }


def evaluate_oof(probs_df, train_df=None):
    """Evaluate an OOF probability frame (datapointID + p_ev_* [+ p_cl_*]) against train labels."""
    from .data import load_train
    from .decode import decode_a, decode_b

    train_df = load_train() if train_df is None else train_df
    df = probs_df.merge(train_df[["datapointID", "evasion_label"]], on="datapointID", how="inner")
    y = df["evasion_label"].tolist()
    P = df[EV_COLS].to_numpy()
    out = {"n": len(df)}
    ev_a, _ = decode_a(P)
    ev_b, cl_b = decode_b(P)
    out["E_argmax"] = evasion_f1(y, ev_a)
    out["M_argmax"] = mapped_clarity_f1(y, ev_a)
    out["E_groupdec"] = evasion_f1(y, ev_b)
    out["M_groupdec"] = clarity_f1([CLARITY_OF[v] for v in y], cl_b)
    if all(c in df for c in CL_COLS):
        out["C_head"] = clarity_f1([CLARITY_OF[v] for v in y], [CLARITY[i] for i in df[CL_COLS].to_numpy().argmax(1)])
    from .decode import nested_bias_eval
    from .folds import get_folds

    folds = df[["datapointID"]].merge(get_folds(), on="datapointID", how="left")["fold"].to_numpy()
    if len(np.unique(folds)) > 2:
        nb = nested_bias_eval(P, y, folds)
        out["E_nbias"], out["M_nbias"] = nb["E_nested_bias"], nb["C_nested_bias"]
    out["per_class"] = per_class_f1(y, ev_a, EVASION)
    out["confusion"] = pd.DataFrame(confusion_matrix(y, ev_a, labels=EVASION), index=EVASION, columns=EVASION)
    return out


def append_result(exp_id, metrics, note=""):
    RESULTS_MD.parent.mkdir(exist_ok=True)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(
            "# Results (OOF, 7-fold CV on train_data.csv)\n\n"
            "E = evasion macro-F1, M = clarity macro-F1 mapped from evasion. All numbers computed by code.\n\n"
            "E nested-bias / M nested-bias: per-class logit biases fitted on the other 6 folds' OOF, applied to the held-out fold.\n\n"
            "| date | exp | n | E argmax | M argmax | E groupdec | M groupdec | C head | E nested-bias | M nested-bias | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    f = lambda k: f"{metrics[k]:.4f}" if k in metrics else "–"
    row = f"| {date.today()} | {exp_id} | {metrics['n']} | {f('E_argmax')} | {f('M_argmax')} | {f('E_groupdec')} | {f('M_groupdec')} | {f('C_head')} | {f('E_nbias')} | {f('M_nbias')} | {note} |\n"
    with open(RESULTS_MD, "a", encoding="utf-8") as fh:
        fh.write(row)


def print_report(m):
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
    print("per-class evasion F1:", {k: round(float(v), 3) for k, v in m["per_class"].items()})
    with pd.option_context("display.width", 250):
        print(m["confusion"])


if __name__ == "__main__":
    # python -m polero.metrics artifacts/<exp>/oof.csv [note]
    path = sys.argv[1]
    m = evaluate_oof(pd.read_csv(path))
    print_report(m)
    from pathlib import Path

    append_result(Path(path).parent.name, m, " ".join(sys.argv[2:]))
