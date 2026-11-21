"""Fixed 7-fold CV (T0.4, D-006).

StratifiedKFold on `evasion_label`, seed 42, over unique Q/A pairs; exact
duplicates inherit the fold of their first occurrence so they never straddle folds.
Run `python -m polero.folds` to (re)build `artifacts/folds.csv`.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import ARTIFACTS, load_train, qa_key

N_FOLDS = 7
SEED = 42
FOLDS_CSV = ARTIFACTS / "folds.csv"


def build_folds(train, n_folds=N_FOLDS, seed=SEED):
    key = qa_key(train)
    first = ~key.duplicated(keep="first")
    reps = train[first]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rep_fold = np.empty(len(reps), dtype=int)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(reps)), reps["evasion_label"])):
        rep_fold[va] = k
    fold_of_key = dict(zip(key[first], rep_fold))
    return pd.DataFrame({"datapointID": train["datapointID"].values,
                         "fold": key.map(fold_of_key).astype(int).values})


def load_folds(train=None, path=None):
    """Load the fold file, building it first if missing. Returns datapointID,fold."""
    path = path or FOLDS_CSV
    if not path.exists():
        folds = build_folds(train if train is not None else load_train())
        path.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(path, index=False)
    return pd.read_csv(path)


if __name__ == "__main__":
    f = build_folds(load_train())
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(FOLDS_CSV, index=False)
    print(f"wrote {FOLDS_CSV}: {f.fold.value_counts().sort_index().to_dict()}")
