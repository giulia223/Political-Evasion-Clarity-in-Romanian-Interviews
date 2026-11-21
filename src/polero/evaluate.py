"""Score an OOF probability file and log it to docs/RESULTS.md.

CLI: python -m polero.evaluate artifacts/<exp_id>/oof.csv [--note "..."] [--no-log]
Writes artifacts/<exp_id>/oof_metrics.json next to the OOF file.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from polero import REPO_ROOT
from polero.data import load_train
from polero.metrics import SCALARS, evaluate_oof, to_markdown

RESULTS_MD = REPO_ROOT / "docs" / "RESULTS.md"
HEADER = (
    "# Results\n\n"
    "Append-only log of OOF evaluations (written by `python -m polero.evaluate`). "
    "All scores are macro-F1 on out-of-fold predictions over the fixed folds (D-006, D-019).\n"
    "- evasion: 9-class argmax · mapped: clarity from evasion argmax (decoder a) · "
    "grouped: clarity from summed group probabilities (decoder b) · head: direct clarity head\n\n"
    "| date | experiment | n | coverage | evasion | mapped clarity | grouped clarity | clarity head | note |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def append_result(metrics, exp_id, note="", path=RESULTS_MD):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER)
    f = lambda v: "–" if v is None else f"{v:.4f}"
    row = (f"| {dt.date.today().isoformat()} | {exp_id} | {metrics['n']} | {metrics['coverage']:.3f} | "
           f"{f(metrics['evasion_f1'])} | {f(metrics['mapped_clarity_f1'])} | {f(metrics['grouped_clarity_f1'])} | "
           f"{f(metrics['clarity_head_f1'])} | {note.replace('|', '/')} |\n")
    with path.open("a") as fh:
        fh.write(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("oof", type=Path)
    ap.add_argument("--exp", help="experiment id (default: the OOF file's folder name)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true", help="don't append to docs/RESULTS.md")
    ap.add_argument("--results", type=Path, default=RESULTS_MD)
    args = ap.parse_args(argv)

    metrics = evaluate_oof(pd.read_csv(args.oof), load_train())
    exp_id = args.exp or args.oof.resolve().parent.name
    print(f"# {exp_id}\n\n{to_markdown(metrics)}")
    (args.oof.parent / "oof_metrics.json").write_text(json.dumps({k: metrics[k] for k in SCALARS}, indent=2))
    if not args.no_log:
        append_result(metrics, exp_id, args.note, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
