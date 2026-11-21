"""Labels, submission, folds, metrics, decoding."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from polero import SAMPLE_CSV
from polero.data import load_test, load_train
from polero.decode import decode_b, fit_biases, group_probs, tune_bias
from polero.folds import build_folds
from polero.labels import CLARITY, CLARITY_OF, EVASION
from polero.metrics import clarity_f1, evasion_f1, mapped_clarity_f1
from polero.submission import validate_submission, write_submission


def test_mapping_matches_train():
    tr = load_train()
    assert set(CLARITY_OF) == set(EVASION) and set(CLARITY_OF.values()) == set(CLARITY)
    assert all(CLARITY_OF[e] == c for e, c in zip(tr["evasion_label"], tr["clarity_label"]))


def test_sample_output_valid():
    rep = validate_submission(SAMPLE_CSV, verbose=False)
    assert rep["ok"]  # the sample mixes labels, so it is not evasion/clarity consistent; that is only reported


def test_roundtrip_and_rejections(tmp_path):
    ids = load_test()["datapointID"].tolist()
    ev = [EVASION[i % 9] for i in range(len(ids))]
    p = tmp_path / "s.csv"
    write_submission(ids, ev, p)
    good = pd.read_csv(p, dtype=str)
    bad_cases = {
        "missing": good.drop(index=5),
        "wrong_id": good.assign(datapointID=good["datapointID"].where(good.index != 3, "999999")),
        "typo": good.assign(answer=good["answer"].replace("Partial/half-answer", "Partial/half answer")),
        "col_order": good[["datapointID", "subtaskID", "answer"]],
        "dup": pd.concat([good.iloc[:-1], good.iloc[[0]]]),
    }
    for name, bad in bad_cases.items():
        q = tmp_path / f"{name}.csv"
        bad.to_csv(q, index=False)
        with pytest.raises(ValueError):
            validate_submission(q, verbose=False)


def test_folds_deterministic_and_complete():
    tr = load_train()
    f1, f2 = build_folds(tr), build_folds(tr)
    assert f1.equals(f2)
    assert f1["datapointID"].is_unique and set(f1["fold"]) == set(range(7))
    m = tr.merge(f1, on="datapointID")
    assert (pd.crosstab(m["evasion_label"], m["fold"]) > 0).all().all()
    key = m["interview_question"].str.strip() + "|" + m["interview_answer"].str.strip()
    assert (m.groupby(key)["fold"].nunique() == 1).all()


def test_metrics_match_sklearn_and_unpredicted_class():
    y = ["Explicit", "General", "Dodging", "Clarification", "Explicit"]
    p = ["Explicit", "General", "General", "Explicit", "Explicit"]
    assert evasion_f1(y, p) == pytest.approx(f1_score(y, p, labels=EVASION, average="macro", zero_division=0))
    # mapped clarity equals clarity F1 after explicit mapping
    assert mapped_clarity_f1(y, p) == clarity_f1([CLARITY_OF[v] for v in y], [CLARITY_OF[v] for v in p])


def test_decoder_b_consistent():
    P = np.random.RandomState(0).dirichlet(np.ones(9), size=200)
    ev, cl = decode_b(P)
    assert all(CLARITY_OF[e] == c for e, c in zip(ev, cl))
    assert np.allclose(group_probs(P).sum(1), 1)


def test_bias_tuning_helps_imbalanced():
    rng = np.random.RandomState(0)
    y = rng.choice(3, 600, p=[0.8, 0.15, 0.05])
    S = rng.normal(size=(600, 3)) * 0.5
    S[np.arange(600), y] += 0.6
    S[:, 0] += 1.0  # bias toward the majority
    score = lambda a, b: f1_score(a, b, average="macro")
    b, best = tune_bias(S, y, 3, score)
    assert best > score(y, S.argmax(1))
