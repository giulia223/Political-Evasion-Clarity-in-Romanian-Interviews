"""OOF evaluation CLI (T0.5).

python -m polero.evaluate artifacts/<exp>/oof.csv [--decoder a|b|c] [--no-append]
Prints metrics and appends one row to docs/RESULTS.md.
"""
import argparse
import datetime as dt

import pandas as pd

from .data import ROOT, load_train
from .folds import load_folds
from .labels import CL_COLS, EV_COLS
from .metrics import evaluate_oof, fold_metrics

RESULTS_MD = ROOT / "docs" / "RESULTS.md"
HEADER = (
    "# Results (OOF macro-F1 on the fixed 7-fold CV)\n\n"
    "Append-only. E = evasion macro-F1, C = clarity macro-F1 from the decoder, "
    "M = clarity mapped from argmax evasion, D = direct clarity head.\n\n"
    "| date | experiment | rows | decoder | E | C | M | D | E+C | source |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def check_schema(df):
    need = ["datapointID"] + EV_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"oof schema: missing {miss}")
    p = df[EV_COLS].to_numpy()
    if (p < -1e-6).any() or abs(p.sum(1) - 1).max() > 1e-3:
        raise ValueError("oof schema: evasion probabilities must be >= 0 and sum to 1")
    if all(c in df.columns for c in CL_COLS):
        q = df[CL_COLS].to_numpy()
        if abs(q.sum(1) - 1).max() > 1e-3:
            raise ValueError("oof schema: clarity probabilities must sum to 1")
    if df["datapointID"].duplicated().any():
        raise ValueError("oof schema: duplicate datapointID")


def append_result(exp, m, source, note=""):
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(HEADER, encoding="utf-8")
    d = m.get("direct_clarity_f1")
    row = (f"| {dt.date.today()} | {exp}{(' ' + note) if note else ''} | {m['n']} | {m['decoder']} | "
           f"{m['evasion_f1']:.4f} | {m['clarity_f1']:.4f} | {m['mapped_clarity_f1']:.4f} | "
           f"{'' if d is None else f'{d:.4f}'} | {m['score']:.4f} | `{source}` |\n")
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(row)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--decoder", default="a", choices=["a", "b", "c"])
    ap.add_argument("--no-append", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.oof)
    check_schema(df)
    train = load_train()
    m, table, cms = evaluate_oof(df, train, a.decoder)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "per_class_f1"})
    print(table)
    print(cms["evasion"])
    print(fold_metrics(df, train, load_folds(train), a.decoder).round(4).to_string(index=False))
    if not a.no_append:
        exp = pd.io.common.stringify_path(a.oof).replace("\\", "/").split("/")[-2]
        append_result(exp, m, a.oof.replace("\\", "/"))


if __name__ == "__main__":
    main()
