"""Build a submission from one or more experiments (T2.4).

python -m polero.predict --exp robert-large_mh [--exp other --weights 0.6,0.4]
       [--decoder auto|a|b|c] [--bias] [--out submissions/sub.csv] [--log]

Steps: blend OOF and fold-averaged test probabilities -> compare decoders on OOF (nested
for tuned ones) -> fit the chosen decoder's parameters on all OOF rows -> decode test ->
write and validate the submission. --log appends the decoder table to docs/RESULTS.md.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from polero import ARTIFACTS, SUBMISSIONS
from polero.data import load_train
from polero.ensemble import blend, compare_decoders, decode, fit_params, labels_from_idx, load_oof, load_test, oof_arrays
from polero.evaluate import RESULTS_MD
from polero.folds import load_folds
from polero.metrics import CL_COLS
from polero.submission import write_submission


def build(exp_dirs, weights=None, decoder="b", use_bias=False, allow_partial=False):
    train = load_train().merge(load_folds(), on="datapointID")
    oofs = [load_oof(d) for d in exp_dirs]
    common = set.intersection(*(set(o.datapointID) for o in oofs))
    if not allow_partial and len(common) != len(train):
        raise ValueError(f"OOF covers {len(common)}/{len(train)} train rows; run all folds or pass --allow-partial")
    oof = blend([o[o.datapointID.isin(common)] for o in oofs], weights)
    test = blend([load_test(d) for d in exp_dirs], weights)

    table = compare_decoders(oof, train)
    if decoder == "auto":  # best nested OOF sum (ties -> first row: simpler decoder, no bias)
        best = table.loc[table["sum"].idxmax()]
        decoder, use_bias = best.decoder, best.bias == "nested"
    pe, pc, y, _ = oof_arrays(oof, train)
    if decoder == "c" and pc is None:
        raise ValueError("decoder c needs clarity-head columns in every experiment")
    params = fit_params(pe, pc, y, decoder, use_bias)
    te_pc = test[CL_COLS].to_numpy() if all(c in test for c in CL_COLS) else None
    ev, cl = decode(test.filter(like="p_ev_").to_numpy(), te_pc, decoder, params["alpha"], params["bias"])
    ev_labels, cl_labels = labels_from_idx(ev, cl)
    return {"table": table, "params": params, "decoder": decoder, "use_bias": use_bias, "test_ids": test.datapointID.tolist(),
            "evasion": ev_labels, "clarity": cl_labels, "train": train}


def distribution(pred, train):
    t = pd.Series(pred).value_counts(normalize=True)
    tr = train.evasion_label.value_counts(normalize=True)
    return pd.DataFrame({"test_pred": t, "train": tr}).fillna(0.0).sort_values("train", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", action="append", required=True, help="experiment id or artifacts directory (repeatable)")
    ap.add_argument("--weights", help="comma-separated blend weights, one per --exp")
    ap.add_argument("--decoder", default="auto", choices=["auto", "a", "b", "c"],
                    help="auto = best nested-OOF row of the decoder table (default)")
    ap.add_argument("--bias", action="store_true", help="tune per-class biases on OOF (ignored with --decoder auto)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--log", action="store_true", help="append the decoder table to docs/RESULTS.md")
    ap.add_argument("--results", type=Path, default=RESULTS_MD)
    a = ap.parse_args(argv)

    dirs = [Path(e) if Path(e).is_dir() else ARTIFACTS / e for e in a.exp]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else None
    r = build(dirs, weights, a.decoder, a.bias, a.allow_partial)
    name = "+".join(d.name for d in dirs)
    print(f"# {name}\n\nDecoders on OOF (tuned rows are nested-CV):\n\n{r['table'].round(4).to_string(index=False)}\n")
    print(f"chosen: decoder {r['decoder']}, bias {'tuned' if r['use_bias'] else 'none'}, params "
          f"{json.dumps({'alpha': r['params']['alpha'], 'bias': None if r['params']['bias'] is None else [round(x, 2) for x in r['params']['bias']]})}\n")
    print("Predicted evasion distribution (test) vs train:\n")
    print(distribution(r["evasion"], r["train"]).round(3).to_string(), "\n")

    out = a.out or SUBMISSIONS / f"sub_{name}_{r['decoder']}{'_bias' if r['use_bias'] else ''}.csv"
    res = write_submission(r["test_ids"], r["evasion"], out, r["clarity"])
    print(f"wrote {out}: {res.report()}")
    if a.log:
        rows = "\n".join(f"| {dt.date.today()} | {name} | {x.decoder} | {x.bias} | {x.evasion_f1:.4f} | {x.clarity_f1:.4f} | {x.sum:.4f} |"
                         for x in r["table"].itertuples())
        with Path(a.results).open("a") as fh:
            fh.write(f"\n### Decoders: {name}\n\n| date | experiments | decoder | bias | evasion | clarity | sum |\n"
                     f"|---|---|---|---|---|---|---|\n{rows}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
