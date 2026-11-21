"""T0.5: metrics and OOF evaluation."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from polero.data import load_train
from polero.evaluate import main
from polero.labels import CLARITY, CLARITY_OF, EVASION
from polero.metrics import (CL_COLS, EV_COLS, clarity_f1, evaluate_oof, evasion_f1,
                            mapped_clarity_f1, per_class_report)

Y = ["Explicit", "Explicit", "General", "Dodging"]
P = ["Explicit", "General", "General", "Explicit"]


@pytest.fixture(scope="module")
def train():
    return load_train()


def test_hand_computed_evasion():
    # Explicit P=.5 R=.5 F=.5; General P=.5 R=1 F=2/3; Dodging never predicted -> F=0
    assert evasion_f1(Y, P) == pytest.approx((0.5 + 2 / 3 + 0) / 3)


def test_never_predicted_class_counts_as_zero():
    rep = per_class_report(Y, P, EVASION)
    assert rep.loc["Dodging", "f1"] == 0 and rep.loc["Dodging", "predicted"] == 0
    assert evasion_f1(["Dodging", "Explicit"], ["Explicit", "Explicit"]) == pytest.approx((2 / 3 + 0) / 2)


def test_mapped_equals_explicit_mapping():
    # gold clarity CR,CR,Amb,Amb vs predicted CR,Amb,Amb,CR -> both classes F=.5
    assert mapped_clarity_f1(Y, P) == pytest.approx(0.5)
    assert mapped_clarity_f1(Y, P) == clarity_f1([CLARITY_OF[e] for e in Y], [CLARITY_OF[e] for e in P])


def test_matches_sklearn_on_random_labels():
    rng = np.random.default_rng(0)
    y, p = rng.choice(EVASION, 500), rng.choice(EVASION[:6], 500)
    assert evasion_f1(y, p) == f1_score(y, p, average="macro", zero_division=0)


def test_rejects_unknown_labels():
    with pytest.raises(ValueError):
        evasion_f1(["Explicit"], ["explicit"])


def _oof(train, rng, head=True, ids=None):
    df = train if ids is None else train[train.datapointID.isin(ids)]
    pe = rng.dirichlet(np.ones(9), len(df))
    out = pd.DataFrame(pe, columns=EV_COLS).assign(datapointID=df.datapointID.to_numpy())
    if head:
        out[CL_COLS] = rng.dirichlet(np.ones(3), len(df))
    return out


def test_evaluate_oof_perfect_and_partial(train):
    onehot = pd.get_dummies(train.evasion_label).reindex(columns=EVASION, fill_value=0).astype(float)
    perfect = pd.DataFrame(onehot.to_numpy(), columns=EV_COLS).assign(datapointID=train.datapointID.to_numpy())
    m = evaluate_oof(perfect, train)
    assert m["evasion_f1"] == m["mapped_clarity_f1"] == m["grouped_clarity_f1"] == 1.0
    assert m["clarity_head_f1"] is None and m["coverage"] == 1.0

    part = _oof(train, np.random.default_rng(1), ids=train.datapointID[:500])
    m = evaluate_oof(part, train)
    assert m["n"] == 500 and m["clarity_head_f1"] is not None
    g = train.set_index("datapointID").loc[part.datapointID]
    expected = f1_score(g.evasion_label, [EVASION[i] for i in part[EV_COLS].to_numpy().argmax(1)], average="macro", zero_division=0)
    assert m["evasion_f1"] == pytest.approx(expected)


def test_grouped_decoder_differs_from_argmax(train):
    # Explicit .4 vs three Ambivalent labels at .2 each: argmax -> Clear Reply, grouped -> Ambivalent
    row = dict.fromkeys(EV_COLS, 0.0) | {"p_ev_Explicit": 0.4, "p_ev_General": 0.2, "p_ev_Dodging": 0.2, "p_ev_Implicit": 0.2}
    gid = train[train.clarity_label == "Ambivalent"].datapointID.iloc[0]
    m = evaluate_oof(pd.DataFrame([row | {"datapointID": gid}]), train)
    assert m["grouped_clarity_f1"] == 1.0 and m["mapped_clarity_f1"] == 0.0


@pytest.mark.parametrize("break_it", [
    lambda d: d.drop(columns=EV_COLS[0]),
    lambda d: d.assign(**{EV_COLS[0]: d[EV_COLS[0]] + 0.5}),
    lambda d: pd.concat([d, d.iloc[:1]]),
    lambda d: d.assign(datapointID=d.datapointID + 10**7),
])
def test_evaluate_oof_rejects_bad_files(train, break_it):
    with pytest.raises(ValueError):
        evaluate_oof(break_it(_oof(train, np.random.default_rng(2), ids=train.datapointID[:50])), train)


def test_cli_writes_json_and_appends_results(train, tmp_path, capsys):
    exp = tmp_path / "toy_exp"
    exp.mkdir()
    _oof(train, np.random.default_rng(3)).to_csv(exp / "oof.csv", index=False)
    results = tmp_path / "RESULTS.md"
    for _ in range(2):
        assert main([str(exp / "oof.csv"), "--note", "smoke", "--results", str(results)]) == 0
    metrics = json.loads((exp / "oof_metrics.json").read_text())
    assert set(metrics) >= {"evasion_f1", "mapped_clarity_f1", "grouped_clarity_f1", "clarity_head_f1"}
    rows = [l for l in results.read_text().splitlines() if l.startswith("| 20")]
    assert len(rows) == 2 and "toy_exp" in rows[0] and "smoke" in rows[0]
    assert "Evasion per class" in capsys.readouterr().out
