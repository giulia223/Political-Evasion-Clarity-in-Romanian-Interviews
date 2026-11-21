"""T0.4: data loading and the fixed 7-fold split."""
import pandas as pd
import pytest

from polero import TRAIN_CSV
from polero.data import load_test, load_train, pair_groups
from polero.folds import N_FOLDS, fingerprint, load_folds, make_folds

# Fold assignment fingerprint (sklearn 1.9.1, seed 42). If this changes, every member's
# OOF numbers stop being comparable: do not update it without a DECISIONS.md entry.
EXPECTED_FINGERPRINT = "fe46fbf52f31d85849ce7bf1946b1c0359f1a6eb530e65084c97d776fa27c2fb"


@pytest.fixture(scope="module")
def train():
    return load_train()


@pytest.fixture(scope="module")
def folds(train):
    return make_folds(train)


def test_loaders_schema(train):
    test = load_test()
    assert len(train) == 3278 and len(test) == 296
    assert train.datapointID.dtype == "int64" and train.question_order.dtype == "int64"
    assert str(train.interview_answer.dtype) == "string"
    assert "evasion_label" not in test.columns
    assert not set(train.datapointID) & set(test.datapointID)


def test_loader_rejects_bad_labels(tmp_path):
    df = pd.read_csv(TRAIN_CSV)
    df.loc[0, "evasion_label"] = "Evasive"
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="unknown evasion labels"):
        load_train(p)


def test_deterministic_and_fingerprint(train, folds):
    assert make_folds(train).equals(folds)
    assert fingerprint(folds) == EXPECTED_FINGERPRINT


def test_partition(train, folds):
    assert folds.datapointID.is_unique and len(folds) == len(train)
    assert set(folds.fold) == set(range(N_FOLDS))


def test_every_fold_has_every_class_within_one(train, folds):
    ct = pd.crosstab(train.evasion_label, folds.fold)
    assert (ct > 0).all().all()
    expected = train.evasion_label.value_counts() / N_FOLDS
    assert (ct.sub(expected, axis=0).abs() <= 1).all().all()


def test_duplicates_share_a_fold(train, folds):
    g = pd.Series(pair_groups(train))
    dup = g[g.duplicated(keep=False)]
    assert len(dup) == 12  # 6 duplicate pairs in train (docs/RESEARCH.md §5)
    assert (folds.fold[dup.index].groupby(dup).nunique() == 1).all()


def test_load_folds_matches(tmp_path, folds):
    saved = load_folds(tmp_path / "folds.csv")  # builds the file when missing
    assert saved.equals(folds)
