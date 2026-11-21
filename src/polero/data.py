"""Data loading (T0.4) and neighbour-turn helpers."""
import pandas as pd

from . import TEST_CSV, TRAIN_CSV

DTYPES = {
    "datapointID": "int64",
    "title": "string",
    "date": "string",
    "president": "string",
    "url": "string",
    "context": "string",
    "question_order": "int64",
    "interview_question": "string",
    "interview_answer": "string",
}


def _load(path, labels):
    dt = dict(DTYPES)
    if labels:
        dt.update(clarity_label="string", evasion_label="string")
    df = pd.read_csv(path, dtype=dt, keep_default_na=False)
    return df


def load_train():
    return _load(TRAIN_CSV, True)


def load_test():
    return _load(TEST_CSV, False)


def load_all():
    """Train + test rows together (inputs only), with an `is_test` flag."""
    tr, te = load_train(), load_test()
    tr["is_test"], te["is_test"] = False, True
    return pd.concat([tr, te], ignore_index=True)


def add_neighbours(df):
    """Previous/next question and answer within the same transcript (by url, question_order).

    Uses text only (never labels), so it is safe for both train and test rows.
    """
    df = df.copy()
    df = df.sort_values(["url", "question_order"]).reset_index(drop=True)
    g = df.groupby("url", sort=False)
    for col, short in (("interview_question", "q"), ("interview_answer", "a")):
        df[f"prev_{short}"] = g[col].shift(1).fillna("")
        df[f"next_{short}"] = g[col].shift(-1).fillna("")
    df["prev_gap"] = (df["question_order"] - g["question_order"].shift(1)).fillna(-1)
    df["turn_rank"] = g.cumcount()
    df["n_turns"] = g["question_order"].transform("size")
    return df
