"""T0.3: submission writer and validator."""
import pandas as pd
import pytest

from polero import SAMPLE_CSV, TEST_CSV
from polero.labels import EVASION
from polero.submission import main, validate_submission, write_submission

IDS = pd.read_csv(TEST_CSV).datapointID.tolist()


@pytest.fixture
def sample():
    return pd.read_csv(SAMPLE_CSV)


def _write(df, tmp_path):
    p = tmp_path / "sub.csv"
    df.to_csv(p, index=False)
    return p


def test_sample_output_is_valid(sample):
    res = validate_submission(SAMPLE_CSV)
    assert res.ok, res.report()
    # The platform sample is a placeholder with 34 inconsistent clarity/evasion pairs:
    # the format allows inconsistency, so it is reported as a warning, not an error.
    assert len(res.warnings) == 1 and res.warnings[0].startswith("34 rows")


def test_roundtrip(tmp_path):
    ev = [EVASION[i % 9] for i in range(len(IDS))]
    p = tmp_path / "out" / "sub.csv"
    res = write_submission(IDS[::-1], ev[::-1], p)  # any input order is accepted
    assert res.ok and not res.warnings
    df = pd.read_csv(p)
    assert df[df.subtaskID == 2].datapointID.tolist() == IDS
    assert df[df.subtaskID == 2].answer.tolist() == ev


@pytest.mark.parametrize("corrupt, msg", [
    (lambda d: d.drop(index=5), "expected 592 rows"),
    (lambda d: d.assign(datapointID=d.datapointID.where(d.index != 3, 999999)), "not in test_data.csv"),
    (lambda d: d.assign(answer=d.answer.where(d.index != 400, "Partial/half answer")), "invalid labels"),
    (lambda d: d[["datapointID", "subtaskID", "answer"]], "columns must be exactly"),
    (lambda d: pd.concat([d.iloc[:10], d.iloc[[9]], d.iloc[11:]]), "duplicate"),
    (lambda d: pd.concat([d.iloc[296:], d.iloc[:296]]), "subtaskID 1 first"),
    (lambda d: pd.concat([d.iloc[:296].iloc[::-1], d.iloc[296:]]), "not in test_data.csv order"),
    (lambda d: d.assign(answer=d.answer.where(d.index != 0, "Explicit")), "invalid labels"),  # evasion label in clarity block
])
def test_corrupted_files_rejected(sample, tmp_path, corrupt, msg):
    res = validate_submission(_write(corrupt(sample).reset_index(drop=True), tmp_path))
    assert not res.ok
    assert msg in res.report(), res.report()


def test_inconsistency_is_warning_not_error(sample, tmp_path):
    df = sample.copy()
    df.loc[0, "answer"] = "Ambivalent"  # evasion for this ID is Explicit
    res = validate_submission(_write(df, tmp_path))
    assert res.ok and res.warnings


def test_writer_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_submission(IDS[:-1], ["Explicit"] * (len(IDS) - 1), tmp_path / "a.csv")
    with pytest.raises(ValueError):
        write_submission(IDS, ["explicit"] * len(IDS), tmp_path / "b.csv")
    assert not (tmp_path / "b.csv").exists()


def test_cli(capsys, tmp_path, sample):
    assert main(["validate", str(SAMPLE_CSV)]) == 0
    assert main(["validate", str(_write(sample.drop(index=0), tmp_path))]) == 1
    assert main([]) == 2
