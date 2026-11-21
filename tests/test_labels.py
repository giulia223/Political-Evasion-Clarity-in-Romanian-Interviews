"""T0.3: label taxonomy."""
import pandas as pd

from polero import TRAIN_CSV
from polero.labels import CLARITY, CLARITY_OF, EVASION, EVASION_GROUP, EVASION_TO_ID, to_clarity


def test_taxonomy_shape():
    assert len(EVASION) == 9 and len(set(EVASION)) == 9
    assert set(CLARITY_OF.values()) == set(CLARITY) and len(CLARITY) == 3
    assert [EVASION[i] for i in range(9)] == list(EVASION_TO_ID)
    assert all(CLARITY[EVASION_GROUP[i]] == CLARITY_OF[e] for i, e in enumerate(EVASION))


def test_mapping_matches_every_train_row():
    tr = pd.read_csv(TRAIN_CSV, usecols=["evasion_label", "clarity_label"])
    assert set(tr.evasion_label) == set(EVASION)
    assert set(tr.clarity_label) == set(CLARITY)
    assert to_clarity(tr.evasion_label) == tr.clarity_label.tolist()
