"""T0.3: submission writer and validator."""
import pandas as pd
import pytest

from polero.data import SAMPLE_CSV
from polero.labels import EVASION
from polero.submission import SubmissionError, main, validate_submission, write_submission


def test_sample_output_valid(test_df):
    # the platform sample is a format example: it is valid, but 34 of its pairs are
    # inconsistent (random-looking labels), which the validator reports as a warning
    with pytest.warns(UserWarning, match="inconsistent"):
        info = validate_submission(SAMPLE_CSV, test_df)
    assert info["rows"] == 592 and info["inconsistent"] == 34


def test_roundtrip(tmp_path, test_df):
    ev = [EVASION[i % 9] for i in range(len(test_df))]
    p = tmp_path / "s.csv"
    write_submission(test_df["datapointID"], ev, p)
    info = validate_submission(p, test_df)
    assert info["inconsistent"] == 0
    assert main(["validate", str(p)]) == 0


def _good(tmp_path, test_df):
    p = tmp_path / "good.csv"
    write_submission(test_df["datapointID"], ["Explicit"] * len(test_df), p)
    return pd.read_csv(p, dtype=str)


@pytest.mark.parametrize("corrupt", ["missing_row", "wrong_id", "typo", "col_order", "dup_row", "swap_order"])
def test_rejects(tmp_path, test_df, corrupt):
    df = _good(tmp_path, test_df)
    if corrupt == "missing_row":
        df = df.drop(index=5)
    elif corrupt == "wrong_id":
        df.loc[3, "datapointID"] = "999999"
    elif corrupt == "typo":
        df.loc[len(df) - 1, "answer"] = "Partial/half answer"
    elif corrupt == "col_order":
        df = df[["datapointID", "subtaskID", "answer"]]
    elif corrupt == "dup_row":
        df.loc[1] = df.loc[0]
    elif corrupt == "swap_order":
        df.iloc[[0, 1]] = df.iloc[[1, 0]].to_numpy()
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(SubmissionError):
        validate_submission(p, test_df)


def test_inconsistent_warns(tmp_path, test_df):
    p = tmp_path / "inc.csv"
    write_submission(test_df["datapointID"], ["Explicit"] * len(test_df), p,
                     clarity_labels=["Ambivalent"] * len(test_df))
    with pytest.warns(UserWarning):
        info = validate_submission(p, test_df)
    assert info["inconsistent"] == len(test_df)
