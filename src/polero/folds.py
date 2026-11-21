"""Fixed 7-fold CV (T0.4): StratifiedKFold on evasion_label, seed 42; exact-duplicate Q/A pairs share a fold."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import ARTIFACTS, CONFIGS, SEED
from .data import load_train

N_FOLDS = 7
FOLDS_CSV = ARTIFACTS / "folds.csv"


def build_folds(df=None, n_folds=N_FOLDS, seed=SEED):
    df = load_train() if df is None else df
    key = df["interview_question"].str.strip() + "\x00" + df["interview_answer"].str.strip()
    groups = key.factorize()[0]
    y = df["evasion_label"].to_numpy()
    if len(set(groups)) < len(df):
        # duplicates exist: group them so they never straddle folds
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = cv.split(df, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = cv.split(df, y)
    fold = pd.Series(-1, index=df.index, dtype="int64")
    for k, (_, va) in enumerate(splits):
        fold.iloc[va] = k
    return pd.DataFrame({"datapointID": df["datapointID"].to_numpy(), "fold": fold.to_numpy()})


COMMITTED = CONFIGS / "folds.csv"  # tracked copy so Kaggle/Colab (other sklearn versions) use identical folds


def get_folds():
    """Load the committed fold file if present, else artifacts/folds.csv (built on first use)."""
    if COMMITTED.exists():
        return pd.read_csv(COMMITTED)
    if not FOLDS_CSV.exists():
        FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
        build_folds().to_csv(FOLDS_CSV, index=False)
    return pd.read_csv(FOLDS_CSV)


if __name__ == "__main__":
    f = get_folds()
    tr = load_train().merge(f, on="datapointID")
    print(pd.crosstab(tr["evasion_label"], tr["fold"]))
