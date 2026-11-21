"""Load train/test data with a fixed schema."""
import pandas as pd

from polero import TEST_CSV, TRAIN_CSV
from polero.labels import CLARITY_OF, EVASION

TEXT_COLS = ["title", "date", "president", "url", "context", "interview_question", "interview_answer"]
INPUT_COLS = ["datapointID", *TEXT_COLS[:5], "question_order", *TEXT_COLS[5:]]
LABEL_COLS = ["clarity_label", "evasion_label"]
DTYPES = {"datapointID": "int64", "question_order": "int64", **{c: "string" for c in TEXT_COLS + LABEL_COLS}}


def _load(path, cols):
    df = pd.read_csv(path, dtype={c: DTYPES[c] for c in cols}, keep_default_na=False)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df = df[cols]
    empty = [c for c in ("interview_question", "interview_answer") if (df[c].str.strip() == "").any()]
    if empty:
        raise ValueError(f"{path.name}: empty text in {empty}")
    if df.datapointID.duplicated().any():
        raise ValueError(f"{path.name}: duplicate datapointID")
    return df.reset_index(drop=True)


def load_train(path=TRAIN_CSV):
    df = _load(path, INPUT_COLS + LABEL_COLS)
    bad = set(df.evasion_label) - set(EVASION)
    if bad:
        raise ValueError(f"unknown evasion labels {sorted(bad)}")
    if (df.evasion_label.map(CLARITY_OF) != df.clarity_label).any():
        raise ValueError("clarity_label inconsistent with evasion_label")
    return df


def load_test(path=TEST_CSV):
    return _load(path, INPUT_COLS)


def pair_groups(df):
    """Integer group id per row; exact-duplicate (question, answer) pairs share an id."""
    return df.groupby(["interview_question", "interview_answer"], sort=False).ngroup().to_numpy()


def event_dates(df):
    """Parse `date`, which mixes YYYY-MM-DD and DD.MM.YYYY (presidency.ro 2015+). Raises if any fail."""
    iso = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    dotted = pd.to_datetime(df.date, format="%d.%m.%Y", errors="coerce")
    out = iso.fillna(dotted)
    if out.isna().any():
        raise ValueError(f"unparseable dates: {df.date[out.isna()].unique()[:5].tolist()}")
    return out
