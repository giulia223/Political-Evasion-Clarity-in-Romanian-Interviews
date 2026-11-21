"""Data loading with a fixed schema (T0.4)."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("POLERO_ROOT", Path(__file__).resolve().parents[2]))
TRAIN_CSV = ROOT / "train_data.csv"
TEST_CSV = ROOT / "test_data.csv"
SAMPLE_CSV = ROOT / "sample_output.csv"
ARTIFACTS = ROOT / "artifacts"

_DTYPES = {
    "datapointID": "int64",
    "title": "string",
    "date": "string",
    "president": "string",
    "url": "string",
    "context": "string",
    "question_order": "int64",
    "interview_question": "string",
    "interview_answer": "string",
    "clarity_label": "string",
    "evasion_label": "string",
}


def _load(path):
    df = pd.read_csv(path, dtype={k: v for k, v in _DTYPES.items()}, keep_default_na=False)
    for c in ("interview_question", "interview_answer"):
        df[c] = df[c].fillna("").astype(str)
    return df


def load_train(path=None):
    return _load(path or TRAIN_CSV)


def load_test(path=None):
    return _load(path or TEST_CSV)


def qa_key(df):
    """Exact-duplicate key over the stripped question/answer pair."""
    return df["interview_question"].str.strip() + "\x00" + df["interview_answer"].str.strip()
