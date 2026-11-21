"""Submission writer and validator (T0.3, D-002).

Format: `subtaskID,datapointID,answer`, 592 rows: the 296 clarity rows (subtask 1)
in the order of test_data.csv, then the 296 evasion rows (subtask 2) in the same order.

CLI: python -m polero.submission validate <csv>
"""
import sys
import warnings

import pandas as pd

from .data import load_test
from .labels import CLARITY, CLARITY_OF, EVASION

COLUMNS = ["subtaskID", "datapointID", "answer"]


class SubmissionError(ValueError):
    pass


def write_submission(test_ids, evasion_labels, path, clarity_labels=None):
    test_ids = list(test_ids)
    evasion_labels = list(evasion_labels)
    if clarity_labels is None:
        clarity_labels = [CLARITY_OF[e] for e in evasion_labels]
    clarity_labels = list(clarity_labels)
    if not (len(test_ids) == len(evasion_labels) == len(clarity_labels)):
        raise SubmissionError("length mismatch between ids and labels")
    df = pd.concat([
        pd.DataFrame({"subtaskID": 1, "datapointID": test_ids, "answer": clarity_labels}),
        pd.DataFrame({"subtaskID": 2, "datapointID": test_ids, "answer": evasion_labels}),
    ], ignore_index=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df


def validate_submission(path, test=None):
    """Raise SubmissionError on any format problem. Returns a dict summary.

    Evasion/clarity inconsistency is reported as a warning, never fixed.
    """
    test = load_test() if test is None else test
    ids = test["datapointID"].tolist()
    n = len(ids)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise SubmissionError(f"cannot read {path}: {e}") from e
    if list(df.columns) != COLUMNS:
        raise SubmissionError(f"columns must be {COLUMNS} in this order, got {list(df.columns)}")
    if len(df) != 2 * n:
        raise SubmissionError(f"expected {2 * n} rows, got {len(df)}")
    if df.duplicated(["subtaskID", "datapointID"]).any():
        raise SubmissionError("duplicate (subtaskID, datapointID) rows")
    s1, s2 = df.iloc[:n], df.iloc[n:]
    if not (s1["subtaskID"] == "1").all() or not (s2["subtaskID"] == "2").all():
        raise SubmissionError("first half must be subtaskID 1 (clarity), second half subtaskID 2 (evasion)")
    want = [str(i) for i in ids]
    for name, part in (("subtask 1", s1), ("subtask 2", s2)):
        got = part["datapointID"].tolist()
        if got != want:
            if set(got) != set(want):
                extra, miss = set(got) - set(want), set(want) - set(got)
                raise SubmissionError(f"{name}: wrong IDs (unknown {sorted(extra)[:5]}, missing {sorted(miss)[:5]})")
            raise SubmissionError(f"{name}: IDs not in test_data.csv order")
    bad_c = sorted(set(s1["answer"]) - set(CLARITY))
    bad_e = sorted(set(s2["answer"]) - set(EVASION))
    if bad_c:
        raise SubmissionError(f"invalid clarity labels: {bad_c}")
    if bad_e:
        raise SubmissionError(f"invalid evasion labels: {bad_e}")
    incons = int(sum(CLARITY_OF[e] != c for e, c in zip(s2["answer"], s1["answer"])))
    if incons:
        warnings.warn(f"{incons} rows have clarity inconsistent with evasion", stacklevel=2)
    return {
        "rows": len(df),
        "inconsistent": incons,
        "clarity_counts": s1["answer"].value_counts().to_dict(),
        "evasion_counts": s2["answer"].value_counts().to_dict(),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "validate":
        print("usage: python -m polero.submission validate <csv>")
        return 2
    try:
        info = validate_submission(argv[1])
    except SubmissionError as e:
        print(f"INVALID: {e}")
        return 1
    print(f"OK: {info['rows']} rows, {info['inconsistent']} inconsistent pairs")
    print(f"clarity: {info['clarity_counts']}")
    print(f"evasion: {info['evasion_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
