"""T2.3: patience logic."""
from polero.train import EarlyStopper


def run(scores, patience=3, min_epochs=1):
    s = EarlyStopper(patience, min_epochs)
    for i, x in enumerate(scores, 1):
        _, stop = s.update(x)
        if stop:
            return i, s.best_epoch
    return None, s.best_epoch


def test_stops_after_patience():
    assert run([0.1, 0.2, 0.19, 0.18, 0.2]) == (5, 2)


def test_improvement_resets():
    assert run([0.1, 0.1, 0.1, 0.3, 0.2, 0.2, 0.2]) == (7, 4)


def test_min_epochs():
    assert run([0.5, 0.4, 0.3, 0.2, 0.1], patience=1, min_epochs=4) == (4, 1)


def test_no_stop():
    assert run([0.1, 0.2, 0.3]) == (None, 3)
