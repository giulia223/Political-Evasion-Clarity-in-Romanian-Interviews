"""Stacking is fitted out-of-fold: on pure-noise inputs the stacked OOF score stays at chance level."""
import numpy as np
import pandas as pd

import polero.predict as PR
import polero.stack as ST
from polero.data import load_test, load_train
from polero.labels import EV_COLS
from polero.metrics import evasion_f1


def _fake_exp(root, name, seed):
    rng = np.random.RandomState(seed)
    d = root / name
    d.mkdir(parents=True)
    tr, te = load_train(), load_test()
    for ids, fname in ((tr["datapointID"], "oof.csv"), (te["datapointID"], "test.csv")):
        P = rng.dirichlet(np.ones(9) * 0.3, size=len(ids))
        df = pd.DataFrame(P, columns=EV_COLS)
        df.insert(0, "datapointID", ids.to_numpy())
        df.to_csv(d / fname, index=False)


def test_stack_no_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(ST, "ARTIFACTS", tmp_path)
    _fake_exp(tmp_path, "n1", 1)
    _fake_exp(tmp_path, "n2", 2)
    oof, te = ST.stack(["n1", "n2"], C=10.0, out="s")
    y = load_train().set_index("datapointID").loc[oof["datapointID"], "evasion_label"].tolist()
    pred = [c[5:] for c in np.array(EV_COLS)[oof[EV_COLS].to_numpy().argmax(1)]]
    assert evasion_f1(y, pred) < 0.15  # noise in, noise out
    assert te.shape == (296, 10) and np.allclose(te[EV_COLS].sum(1), 1)
