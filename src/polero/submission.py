"""Submission writer and validator (T0.3). Format: subtaskID,datapointID,answer; 296 clarity rows then 296 evasion rows."""
import sys

import pandas as pd

from .data import load_test
from .labels import CLARITY, CLARITY_OF, EVASION

COLUMNS = ["subtaskID", "datapointID", "answer"]


def write_submission(test_ids, evasion_labels, path, clarity_labels=None):
    test_ids = list(test_ids)
    evasion_labels = list(evasion_labels)
    if clarity_labels is None:
        clarity_labels = [CLARITY_OF[e] for e in evasion_labels]
    s1 = pd.DataFrame({"subtaskID": 1, "datapointID": test_ids, "answer": list(clarity_labels)})
    s2 = pd.DataFrame({"subtaskID": 2, "datapointID": test_ids, "answer": evasion_labels})
    out = pd.concat([s1, s2], ignore_index=True)
    out.to_csv(path, index=False, lineterminator="\n")
    return validate_submission(path)


def validate_submission(path, verbose=True):
    """Raise ValueError on any format problem. Returns a dict with a consistency report."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    ids = load_test()["datapointID"].astype(str).tolist()
    n = len(ids)
    if len(df) != 2 * n:
        raise ValueError(f"expected {2 * n} rows, got {len(df)}")
    if df.duplicated(["subtaskID", "datapointID"]).any():
        raise ValueError("duplicate (subtaskID, datapointID) rows")
    s1, s2 = df.iloc[:n], df.iloc[n:]
    if (s1["subtaskID"] != "1").any() or (s2["subtaskID"] != "2").any():
        raise ValueError("first half must be subtaskID 1, second half subtaskID 2")
    for name, part in (("subtask 1", s1), ("subtask 2", s2)):
        if part["datapointID"].tolist() != ids:
            raise ValueError(f"{name}: datapointIDs differ from test_data.csv order/set")
    bad1 = sorted(set(s1["answer"]) - set(CLARITY))
    bad2 = sorted(set(s2["answer"]) - set(EVASION))
    if bad1:
        raise ValueError(f"subtask 1: invalid labels {bad1}")
    if bad2:
        raise ValueError(f"subtask 2: invalid labels {bad2}")
    consistent = [CLARITY_OF[e] == c for e, c in zip(s2["answer"], s1["answer"])]
    report = {
        "ok": True,
        "n_inconsistent": int(len(consistent) - sum(consistent)),
        "clarity_dist": s1["answer"].value_counts().to_dict(),
        "evasion_dist": s2["answer"].value_counts().to_dict(),
    }
    if verbose:
        print(f"[valid] {path}: {len(df)} rows; inconsistent pairs: {report['n_inconsistent']}")
        print("  clarity:", report["clarity_dist"])
        print("  evasion:", report["evasion_dist"])
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "validate":
        sys.exit("usage: python -m polero.submission validate <csv>")
    validate_submission(sys.argv[2])
