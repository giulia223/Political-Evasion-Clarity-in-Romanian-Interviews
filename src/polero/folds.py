"""Fixed 7-fold CV split shared by every experiment (D-006).

Stratified on evasion_label, seed 42; exact-duplicate Q/A pairs stay in one fold.
CLI: python -m polero.folds   (writes artifacts/folds.csv)
"""
import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from polero import ARTIFACTS, SEED
from polero.data import load_train, pair_groups

N_FOLDS = 7
FOLDS_CSV = ARTIFACTS / "folds.csv"
# sha256 of the sorted (datapointID, fold) table (D-019). Every member and every GPU run must match it.
EXPECTED_FINGERPRINT = "fe46fbf52f31d85849ce7bf1946b1c0359f1a6eb530e65084c97d776fa27c2fb"


def make_folds(train=None, n_folds=N_FOLDS, seed=SEED):
    train = load_train() if train is None else train
    fold = pd.Series(-1, index=train.index, dtype="int64")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val) in enumerate(cv.split(train, train.evasion_label, groups=pair_groups(train))):
        fold.iloc[val] = k
    assert (fold >= 0).all()
    return pd.DataFrame({"datapointID": train.datapointID, "fold": fold})


def fingerprint(folds):
    """sha256 of the sorted (datapointID, fold) assignment; identical across machines."""
    s = folds.sort_values("datapointID").to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()


def load_folds(path=FOLDS_CSV):
    """Load the saved folds, building them first if the file doesn't exist."""
    if not path.exists():
        save_folds(path=path)
    return pd.read_csv(path)


def save_folds(path=FOLDS_CSV):
    folds = make_folds()
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return folds


if __name__ == "__main__":
    f = save_folds()
    print(f"wrote {FOLDS_CSV} ({len(f)} rows, sizes {f.fold.value_counts().sort_index().tolist()})")
    print("fingerprint", fingerprint(f))
