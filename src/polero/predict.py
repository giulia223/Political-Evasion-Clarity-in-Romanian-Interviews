"""Ensemble experiments, compare decoders on OOF, and write a validated submission (T2.4).

python -m polero.predict --exp <id> [--exp <id2> --weights 0.6,0.4] [--decoder auto|a|b|c]
                         [--bias auto|none|tune] --out submissions/<name>.csv

Every decoder/bias combination is scored on OOF; bias tuning is reported with
nested CV (fit on 6 folds, scored on the 7th), never in-sample. With `auto`, the
combination with the best nested OOF E+C is used for the test predictions.
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, load_test, load_train
from .decode import decode, fit_bias, nested_bias_oof
from .ensemble import combine
from .evaluate import RESULTS_MD
from .folds import load_folds
from .labels import CL_COLS, EV_COLS
from .metrics import clarity_f1, evasion_f1
from .submission import validate_submission, write_submission


def compare_decoders(oof, train, folds, alpha=0.5, decoders=None):
    df = oof.merge(train[["datapointID", "evasion_label", "clarity_label"]], on="datapointID")
    df = df.merge(folds, on="datapointID")
    pe = df[EV_COLS].to_numpy()
    pc = df[CL_COLS].to_numpy() if all(c in df.columns for c in CL_COLS) else None
    y_e, y_c, fold = df["evasion_label"].tolist(), df["clarity_label"].tolist(), df["fold"].to_numpy()
    decoders = decoders or (["a", "b", "c"] if pc is not None else ["a", "b"])
    rows = []
    for d in decoders:
        ev, cl = decode(pe, pc, d, alpha)
        rows.append({"decoder": d, "bias": "none", "E": evasion_f1(y_e, ev), "C": clarity_f1(y_c, cl)})
        if len(np.unique(fold)) > 1:
            ev, cl = nested_bias_oof(y_e, y_c, pe, pc, fold, d, alpha)
            rows.append({"decoder": d, "bias": "tune (nested)", "E": evasion_f1(y_e, ev), "C": clarity_f1(y_c, cl)})
    res = pd.DataFrame(rows)
    res["E+C"] = res["E"] + res["C"]
    return res, (y_e, y_c, pe, pc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--decoder", default="auto", choices=["auto", "a", "b", "c"])
    ap.add_argument("--bias", default="auto", choices=["auto", "none", "tune"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--no-append", action="store_true")
    a = ap.parse_args(argv)

    weights = None if a.weights is None else [float(w) for w in a.weights.split(",")]
    oof, test_p = combine(a.exp, weights, a.artifacts)
    train, test = load_train(), load_test()
    folds = load_folds(train)
    res, (y_e, y_c, pe, pc) = compare_decoders(oof, train, folds, a.alpha)
    print(f"OOF rows: {len(oof)}")
    print(res.round(4).to_string(index=False))

    cand = res.copy()
    if a.decoder != "auto":
        cand = cand[cand["decoder"] == a.decoder]
    if a.bias != "auto":
        cand = cand[cand["bias"].str.startswith(a.bias)]
    best = cand.sort_values("E+C", ascending=False).iloc[0]
    d, use_bias = best["decoder"], best["bias"] != "none"
    bias = fit_bias(y_e, y_c, pe, pc, d, a.alpha) if use_bias else None
    print(f"chosen: decoder {d}, bias {'tuned ' + np.array2string(np.round(bias, 2)) if use_bias else 'none'} "
          f"(nested OOF E {best['E']:.4f} C {best['C']:.4f})")

    tp = test[["datapointID"]].merge(test_p, on="datapointID", how="left")
    if tp[EV_COLS].isna().any().any():
        raise SystemExit("test probabilities missing for some rows")
    tpc = tp[CL_COLS].to_numpy() if all(c in tp.columns for c in CL_COLS) else None
    ev, cl = decode(tp[EV_COLS].to_numpy(), tpc, d, a.alpha, bias)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(test["datapointID"], ev, out, clarity_labels=cl)
    info = validate_submission(out, test)
    dist = pd.DataFrame({
        "test_pred": pd.Series(ev).value_counts(normalize=True),
        "train": train["evasion_label"].value_counts(normalize=True),
    }).fillna(0).round(3)
    print(dist.to_string())
    zero = [l for l in dist.index if dist.loc[l, "test_pred"] == 0]
    if zero:
        print(f"WARNING: classes never predicted on test: {zero}")
    print(f"wrote {out} ({info['rows']} rows, valid, {info['inconsistent']} inconsistent)")

    if not a.no_append:
        if not RESULTS_MD.exists():
            from .evaluate import HEADER
            RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_MD.write_text(HEADER, encoding="utf-8")
        name = "+".join(a.exp) + (f" w={a.weights}" if a.weights else "")
        with open(RESULTS_MD, "a", encoding="utf-8") as f:
            f.write(f"\n**Decoder comparison** ({dt.date.today()}, {name}, {len(oof)} OOF rows; "
                    f"bias rows are nested-CV OOF):\n\n| decoder | bias | E | C | E+C |\n|---|---|---|---|---|\n")
            for r in res.itertuples(index=False):
                f.write(f"| {r.decoder} | {r.bias} | {r.E:.4f} | {r.C:.4f} | {r._4:.4f} |\n")
            f.write(f"\nChosen for `{out.as_posix()}`: decoder {d}, bias {'tuned' if use_bias else 'none'}.\n\n")


if __name__ == "__main__":
    main()
