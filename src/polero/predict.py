"""Ensemble experiments, pick a decoder on OOF, write a validated submission (T2.4).

python -m polero.predict --exp A [--exp B --weights 0.6 0.4] [--decoder auto|a|b|bias] --out submissions/x.csv

- test probabilities of an experiment = mean of its test_fold*.csv (or test.csv for LLM runs);
- OOF rows = intersection of the experiments' OOF rows (the LLM covers folds 1-6 only);
- `auto` picks, per subtask, the decoder with the best honest OOF score (bias tuning is scored with nested CV).
"""
import argparse

import numpy as np
import pandas as pd

from . import ARTIFACTS
from .data import load_test, load_train
from .decode import decode_a, decode_b, decode_tuned, fit_biases, nested_bias_eval
from .folds import get_folds
from .labels import CLARITY_OF, EV_COLS
from .metrics import clarity_f1, evasion_f1
from .submission import write_submission


def load_exp(exp):
    d = ARTIFACTS / exp
    oof = pd.read_csv(d / "oof.csv") if (d / "oof.csv").exists() else None
    tests = sorted(d.glob("test_fold*.csv")) or [d / "test.csv"]
    te = pd.concat([pd.read_csv(p) for p in tests]).groupby("datapointID")[EV_COLS].mean()
    if oof is not None:
        oof = oof.set_index("datapointID")[EV_COLS]
    return oof, te


def combine(exps, weights=None, mode="prob"):
    """Weighted mean of probabilities (mode='prob') or of log-probabilities (mode='log', renormalised)."""
    weights = np.ones(len(exps)) / len(exps) if weights is None else np.asarray(weights) / np.sum(weights)
    loaded = [load_exp(e) for e in exps]
    oof_ids = None
    for o, _ in loaded:
        if o is not None:
            oof_ids = o.index if oof_ids is None else oof_ids.intersection(o.index)
    te_ids = load_test()["datapointID"].to_numpy()

    def mix(frames, ids):
        if mode == "log":
            S = sum(w * np.log(np.clip(f.loc[ids].to_numpy(), 1e-6, 1)) for w, f in zip(weights, frames))
            P = np.exp(S - S.max(1, keepdims=True))
        else:
            P = sum(w * f.loc[ids].to_numpy() for w, f in zip(weights, frames))
        return P / P.sum(1, keepdims=True)

    P_oof = mix([o for o, _ in loaded], oof_ids) if oof_ids is not None else None
    P_te = mix([t for _, t in loaded], te_ids)
    return oof_ids, P_oof, te_ids, P_te


def choose_and_decode(oof_ids, P_oof, P_te, decoder="auto", verbose=True):
    tr = load_train().set_index("datapointID").loc[oof_ids]
    y = tr["evasion_label"].tolist()
    y_cl = [CLARITY_OF[v] for v in y]
    folds = get_folds().set_index("datapointID").loc[oof_ids, "fold"].to_numpy()
    ev_a, cl_a = decode_a(P_oof)
    ev_b, cl_b = decode_b(P_oof)
    nb = nested_bias_eval(P_oof, y, folds)
    scores = {
        "a": (evasion_f1(y, ev_a), clarity_f1(y_cl, cl_a)),
        "b": (evasion_f1(y, ev_b), clarity_f1(y_cl, cl_b)),
        "bias": (nb["E_nested_bias"], nb["C_nested_bias"]),
    }
    if verbose:
        print(f"OOF rows: {len(y)}")
        for k, (e, c) in scores.items():
            print(f"  decoder {k:>4}: E {e:.4f}  C {c:.4f}")
    if decoder == "auto":
        dec_e = max(scores, key=lambda k: scores[k][0])
        dec_c = max(scores, key=lambda k: scores[k][1])
    else:
        dec_e = dec_c = decoder
    b9, b3 = fit_biases(P_oof, y)
    out = {"a": decode_a(P_te), "b": decode_b(P_te), "bias": decode_tuned(P_te, b9, b3)}
    ev, cl = out[dec_e][0], out[dec_c][1]
    info = {"decoder_evasion": dec_e, "decoder_clarity": dec_c, "oof_E": scores[dec_e][0], "oof_C": scores[dec_c][1],
            "n_oof": len(y), "bias9": b9.round(2).tolist(), "bias3": b3.round(2).tolist()}
    if verbose:
        print(f"chosen: evasion via '{dec_e}' (OOF E {scores[dec_e][0]:.4f}), clarity via '{dec_c}' (OOF C {scores[dec_c][1]:.4f})")
    return ev, cl, info


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", action="append", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--mode", choices=["prob", "log"], default="prob")
    p.add_argument("--decoder", choices=["auto", "a", "b", "bias"], default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    oof_ids, P_oof, te_ids, P_te = combine(a.exp, a.weights, a.mode)
    ev, cl, info = choose_and_decode(oof_ids, P_oof, P_te, a.decoder)
    rep = write_submission(te_ids, ev, a.out, clarity_labels=cl)
    tr = load_train()
    print("train evasion dist:", tr["evasion_label"].value_counts(normalize=True).round(3).to_dict())
    zero = [l for l in EV_COLS if l[5:] not in rep["evasion_dist"]]
    if zero:
        print("WARNING classes never predicted on test:", [z[5:] for z in zero])
    print(info)


if __name__ == "__main__":
    main()
