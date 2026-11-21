"""T0.4: fixed folds."""
import numpy as np

from polero.data import qa_key
from polero.folds import N_FOLDS, build_folds


def test_deterministic(train):
    assert build_folds(train).equals(build_folds(train))


def test_every_row_one_fold(train):
    f = build_folds(train)
    assert len(f) == len(train) and not f["datapointID"].duplicated().any()
    assert set(f["fold"]) == set(range(N_FOLDS))


def test_class_balance(train):
    f = build_folds(train).merge(train[["datapointID", "evasion_label"]])
    ct = f.groupby(["evasion_label", "fold"]).size().unstack(fill_value=0)
    expected = ct.sum(1).to_numpy()[:, None] / N_FOLDS
    assert (ct.to_numpy() > 0).all(), "every fold must contain every class"
    # +-1 around the expected share; duplicates moved to their twin's fold may add 1 more
    assert np.abs(ct.to_numpy() - expected).max() <= 2


def test_duplicates_share_fold(train):
    f = build_folds(train)
    key = qa_key(train)
    dup = key.duplicated(keep=False)
    assert dup.sum() >= 12
    g = f[dup.to_numpy()].assign(k=key[dup].to_numpy()).groupby("k")["fold"].nunique()
    assert (g == 1).all()
