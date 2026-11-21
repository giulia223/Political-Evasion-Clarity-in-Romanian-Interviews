"""T0.5: metrics."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from polero.labels import CLARITY, CLARITY_OF, EV_COLS, EVASION
from polero.metrics import clarity_f1, evaluate_oof, evasion_f1, mapped_clarity_f1


def test_hand_computed():
    y = ["Explicit", "Explicit", "Dodging", "General"]
    p = ["Explicit", "Dodging", "Dodging", "Explicit"]
    # Explicit P=1/2 R=1/2 F=.5 ; Dodging P=1/2 R=1 F=2/3 ; General F=0 ; others (absent) F=0
    want = (0.5 + 2 / 3 + 0) / 9
    assert evasion_f1(y, p) == pytest.approx(want)


def test_never_predicted_counts_zero():
    y = ["Explicit", "Clarification"]
    p = ["Explicit", "Explicit"]
    assert evasion_f1(y, p) == pytest.approx(f1_score(y, p, labels=EVASION, average="macro", zero_division=0))
    assert evasion_f1(y, p) < 1 / 9 * 1.0 + 1e-9


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.choice(EVASION, 200).tolist()
    p = rng.choice(EVASION, 200).tolist()
    assert evasion_f1(y, p) == f1_score(y, p, labels=EVASION, average="macro", zero_division=0)
    yc, pc = rng.choice(CLARITY, 50).tolist(), rng.choice(CLARITY, 50).tolist()
    assert clarity_f1(yc, pc) == f1_score(yc, pc, labels=CLARITY, average="macro", zero_division=0)


def test_mapped_equals_explicit_mapping():
    rng = np.random.default_rng(1)
    y = rng.choice(EVASION, 100).tolist()
    p = rng.choice(EVASION, 100).tolist()
    assert mapped_clarity_f1(y, p) == clarity_f1([CLARITY_OF[e] for e in y], [CLARITY_OF[e] for e in p])


def test_evaluate_oof(train):
    sub = train.head(50)
    probs = np.eye(9)[[EVASION.index(e) for e in sub["evasion_label"]]]
    df = pd.DataFrame(probs, columns=EV_COLS)
    df.insert(0, "datapointID", sub["datapointID"].to_numpy())
    m, table, cms = evaluate_oof(df, train)
    present = set(sub["evasion_label"])
    assert m["evasion_f1"] == pytest.approx(len(present) / 9)
    assert "| Explicit |" in table and cms["evasion"].to_numpy().trace() == 50
