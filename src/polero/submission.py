"""Write and validate submissions in the platform format (sample_output.csv, D-002).

Format: columns subtaskID,datapointID,answer; 296 clarity rows (subtaskID 1) then
296 evasion rows (subtaskID 2), each block in test_data.csv order.

CLI: python -m polero.submission validate <csv>
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from polero import TEST_CSV
from polero.labels import CLARITY, CLARITY_OF, EVASION

COLUMNS = ["subtaskID", "datapointID", "answer"]
SUBTASK_LABELS = {1: set(CLARITY), 2: set(EVASION)}


@dataclass
class ValidationResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors

    def report(self):
        lines = [f"ERROR: {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines) or "OK"


def _test_ids():
    return pd.read_csv(TEST_CSV, usecols=["datapointID"]).datapointID.tolist()


def validate_submission(path, test_ids=None):
    """Check a submission file. Errors make it invalid; warnings (e.g. inconsistency) don't."""
    test_ids = list(test_ids) if test_ids is not None else _test_ids()
    n = len(test_ids)
    res = ValidationResult()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # unreadable / empty file
        res.errors.append(f"cannot read CSV: {exc}")
        return res

    if list(df.columns) != COLUMNS:
        res.errors.append(f"columns must be exactly {COLUMNS}, got {list(df.columns)}")
        return res
    if len(df) != 2 * n:
        res.errors.append(f"expected {2 * n} rows (2 x {n}), got {len(df)}")

    sub = pd.to_numeric(df.subtaskID, errors="coerce")
    ids = pd.to_numeric(df.datapointID, errors="coerce")
    if sub.isna().any() or ids.isna().any():
        res.errors.append("subtaskID and datapointID must be integers")
        return res
    df = df.assign(subtaskID=sub.astype(int), datapointID=ids.astype(int))

    if set(df.subtaskID) - {1, 2}:
        res.errors.append(f"unknown subtaskID values: {sorted(set(df.subtaskID) - {1, 2})}")
    dups = df[df.duplicated(["subtaskID", "datapointID"], keep=False)]
    if len(dups):
        res.errors.append(f"duplicate (subtaskID, datapointID) rows: {dups[['subtaskID', 'datapointID']].drop_duplicates().values.tolist()[:5]}")

    expected_order = [1] * n + [2] * n
    if len(df) == 2 * n and df.subtaskID.tolist() != expected_order:
        res.errors.append("rows must be all subtaskID 1 first, then all subtaskID 2")

    for k in (1, 2):
        block = df[df.subtaskID == k]
        got = block.datapointID.tolist()
        missing = set(test_ids) - set(got)
        extra = set(got) - set(test_ids)
        if missing:
            res.errors.append(f"subtask {k}: {len(missing)} test IDs missing, e.g. {sorted(missing)[:5]}")
        if extra:
            res.errors.append(f"subtask {k}: {len(extra)} IDs not in test_data.csv, e.g. {sorted(extra)[:5]}")
        if not missing and not extra and len(got) == n and got != test_ids:
            res.errors.append(f"subtask {k}: rows are not in test_data.csv order")
        bad = sorted(set(block.answer) - SUBTASK_LABELS[k])
        if bad:
            res.errors.append(f"subtask {k}: invalid labels {bad}")

    if res.ok:
        cl = df[df.subtaskID == 1].set_index("datapointID").answer
        ev = df[df.subtaskID == 2].set_index("datapointID").answer
        incons = [i for i in test_ids if CLARITY_OF[ev[i]] != cl[i]]
        if incons:
            res.warnings.append(f"{len(incons)} rows where clarity != CLARITY_OF[evasion], e.g. IDs {incons[:5]}")
    return res


def write_submission(test_ids, evasion_labels, path, clarity_labels=None):
    """Write a submission. Clarity is derived from evasion unless given explicitly.

    test_ids may be in any order; rows are written in test_data.csv order.
    Raises ValueError if the result does not validate.
    """
    test_ids, evasion_labels = list(test_ids), list(evasion_labels)
    if clarity_labels is None:
        clarity_labels = [CLARITY_OF.get(e, e) for e in evasion_labels]
    clarity_labels = list(clarity_labels)
    if not len(test_ids) == len(evasion_labels) == len(clarity_labels):
        raise ValueError("test_ids, evasion_labels and clarity_labels must have the same length")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("test_ids contains duplicates")

    order = _test_ids()
    if set(test_ids) != set(order):
        raise ValueError("test_ids must be exactly the IDs in test_data.csv")
    ev = dict(zip(test_ids, evasion_labels))
    cl = dict(zip(test_ids, clarity_labels))
    out = pd.concat([
        pd.DataFrame({"subtaskID": 1, "datapointID": order, "answer": [cl[i] for i in order]}),
        pd.DataFrame({"subtaskID": 2, "datapointID": order, "answer": [ev[i] for i in order]}),
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    res = validate_submission(path, order)
    if not res.ok:
        path.unlink()
        raise ValueError(f"submission invalid, not written:\n{res.report()}")
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "validate":
        print("usage: python -m polero.submission validate <csv>", file=sys.stderr)
        return 2
    res = validate_submission(argv[1])
    print(res.report())
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
