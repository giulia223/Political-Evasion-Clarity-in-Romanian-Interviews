"""T0.3: label taxonomy."""
import numpy as np

from polero.labels import CLARITY, CLARITY_IDX_OF, CLARITY_OF, EVASION, GROUP


def test_mapping_complete():
    assert len(EVASION) == 9 and len(CLARITY) == 3
    assert set(CLARITY_OF) == set(EVASION)
    assert set(CLARITY_OF.values()) == set(CLARITY)


def test_mapping_matches_train(train):
    pairs = set(zip(train["evasion_label"], train["clarity_label"]))
    assert all(CLARITY_OF[e] == c for e, c in pairs)
    assert set(train["evasion_label"]) == set(EVASION)


def test_group_matrix():
    assert GROUP.sum() == 9 and np.all(GROUP.sum(0) == 1)
    for e, c in enumerate(CLARITY_IDX_OF):
        assert CLARITY[c] == CLARITY_OF[EVASION[e]]
