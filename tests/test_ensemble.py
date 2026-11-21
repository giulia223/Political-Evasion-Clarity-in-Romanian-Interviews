"""T2.4: ensembling, decoders, bias tuning, and an end-to-end submission from smoke artifacts."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from polero import CONFIGS
from polero.ensemble import (GROUP, blend, compare_decoders, decode, load_test, macro_f1_idx, nested_predict,
                             objective, tune_bias)
from polero.labels import CLARITY_OF, CLARITY, EVASION
from polero.metrics import CL_COLS, EV_COLS
from polero.predict import main as predict_main
from polero.submission import validate_submission
from polero.train import SMOKE, TrainConfig, run


def frame(ids, pe, pc=None):
    df = pd.DataFrame(pe, columns=EV_COLS).assign(datapointID=ids)
    if pc is not None:
        df[CL_COLS] = pc
    return df


def test_blend_weighted_average():
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(9), 4), rng.dirichlet(np.ones(9), 4)
    out = blend([frame([1, 2, 3, 4], a), frame([4, 3, 2, 1], b[::-1])], [3, 1])
    assert np.allclose(out[EV_COLS].to_numpy(), 0.75 * a + 0.25 * b)
    with pytest.raises(ValueError):
        blend([frame([1, 2, 3, 4], a), frame([1, 2, 3, 5], b)])


def test_fold_test_average(tmp_path):
    rng = np.random.default_rng(1)
    ps = [rng.dirichlet(np.ones(9), 3) for _ in range(3)]
    for k, p in enumerate(ps):
        frame([7, 8, 9], p).to_csv(tmp_path / f"test_fold{k}.csv", index=False)
    assert np.allclose(load_test(tmp_path)[EV_COLS].to_numpy(), np.mean(ps, 0))


def test_decoders_consistent_and_differ():
    rng = np.random.default_rng(2)
    pe, pc = rng.dirichlet(np.ones(9) * 0.5, 500), rng.dirichlet(np.ones(3), 500)
    for dec in ("a", "b", "c"):
        ev, cl = decode(pe, pc, dec, alpha=0.5)
        assert all(CLARITY_OF[EVASION[e]] == CLARITY[c] for e, c in zip(ev, cl))
    # Explicit .4 vs three Ambivalent labels at .2: a -> Explicit, b -> best Ambivalent label
    p = np.zeros((1, 9)); p[0, [0, 1, 2, 4]] = [0.4, 0.2, 0.25, 0.15]
    assert decode(p, decoder="a")[0][0] == 0
    assert decode(p, decoder="b")[0][0] == 2 and decode(p, decoder="b")[1][0] == 1
    assert decode(p, np.array([[1.0, 0, 0]]), "c", alpha=1.0)[0][0] == 0  # clarity head decides


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y, p = rng.integers(0, 9, 300), rng.integers(0, 6, 300)
        assert macro_f1_idx(y, p, 9) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def _imbalanced(n=1500, seed=4):
    """Synthetic problem where a model under-predicts rare classes: biases should help."""
    rng = np.random.default_rng(seed)
    y = rng.choice(9, n, p=[0.55, 0.08, 0.13, 0.02, 0.09, 0.05, 0.05, 0.02, 0.01])
    logits = rng.normal(0, 1, (n, 9)) + 1.6 * np.eye(9)[y]
    logits[:, 0] += 1.5  # majority bias
    pe = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return pe, y


def test_bias_tuning_improves_macro_f1_in_sample_and_nested():
    pe, y = _imbalanced()
    base = objective(y, *decode(pe))
    bias = tune_bias(pe, y)
    assert bias[0] == 0 and objective(y, *decode(pe, bias=bias)) > base + 0.05
    folds = np.arange(len(y)) % 5
    assert objective(y, *nested_predict(pe, None, y, folds, "a", True)) > base + 0.03


def test_nested_never_sees_its_own_fold(monkeypatch):
    """The parameters used for fold k must be fitted without fold k's rows."""
    import polero.ensemble as E
    pe, y = _imbalanced(400)
    folds = np.arange(400) % 4
    seen = []
    orig = E.fit_params
    monkeypatch.setattr(E, "fit_params", lambda p, c, yy, d, u: seen.append(len(yy)) or orig(p, c, yy, d, u))
    nested_predict(pe, None, y, folds, "a", True)
    assert seen == [300, 300, 300, 300]


def test_compare_decoders_table():
    pe, y = _imbalanced(700)
    pc = np.stack([pe[:, GROUP == c].sum(1) for c in range(3)], 1)
    oof = frame(np.arange(700), pe, pc).assign(fold=np.arange(700) % 7)
    train = pd.DataFrame({"datapointID": np.arange(700), "evasion_label": [EVASION[i] for i in y]})
    t = compare_decoders(oof, train)
    assert set(zip(t.decoder, t.bias)) == {(d, b) for d in "abc" for b in ("none", "nested")}
    assert ((t.evasion_f1 >= 0) & (t.clarity_f1 <= 1)).all()


def test_end_to_end_submission_from_smoke(tmp_path):
    cfg = dataclasses.replace(TrainConfig.from_json(CONFIGS / "robert-large_mh.json"), **SMOKE)
    cfg = dataclasses.replace(cfg, n_train_rows=None, n_test_rows=None, max_steps=1, batch_chunks=64,
                              batch_examples=64, eval_batch_chunks=256)
    exp = tmp_path / "smoke_full"
    run(cfg, "smoke_full", None, exp)  # all 7 folds, all rows, 1 step each
    out = tmp_path / "sub.csv"
    results = tmp_path / "RESULTS.md"
    assert predict_main(["--exp", str(exp), "--decoder", "b", "--bias", "--out", str(out), "--log",
                         "--results", str(results)]) == 0
    res = validate_submission(out)
    assert res.ok and not res.warnings  # consistent by construction
    assert "### Decoders: smoke_full" in results.read_text()


def test_auto_decoder_picks_best_nested_row(tmp_path):
    from polero.predict import build
    from polero.data import load_train, load_test as load_test_rows
    from polero.folds import load_folds
    train = load_train().merge(load_folds(), on="datapointID")
    rng = np.random.default_rng(6)
    # OOF where the model under-predicts rare classes -> tuned biases should win
    y_true = train.evasion_label.map({e: i for i, e in enumerate(EVASION)}).to_numpy()
    logits = rng.normal(0, 1, (len(train), 9)) + 2.0 * np.eye(9)[y_true]
    logits[:, 0] += 2.0
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    exp = tmp_path / "e"
    exp.mkdir()
    frame(train.datapointID.to_numpy(), p).assign(fold=train.fold.to_numpy()).to_csv(exp / "oof.csv", index=False)
    te = load_test_rows()
    frame(te.datapointID.to_numpy(), rng.dirichlet(np.ones(9), len(te))).to_csv(exp / "test_fold0.csv", index=False)
    r = build([exp], decoder="auto")
    best = r["table"].loc[r["table"]["sum"].idxmax()]
    assert (r["decoder"], r["use_bias"]) == (best.decoder, best.bias == "nested") and r["use_bias"]
