"""T2.3 / T2.4: end-to-end smoke run on CPU (tiny model, no downloads)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from polero import predict, train
from polero.labels import CL_COLS, EV_COLS
from polero.submission import validate_submission

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "robert-large_base.json"


def _run(out, folds):
    train.main(["--config", str(CFG), "--smoke", "--out", str(out), "--folds", folds])
    return out / "robert-large_base_smoke"


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("a"), "0,1")


def test_outputs(smoke_dir, test_df):
    oof = pd.read_csv(smoke_dir / "oof.csv")
    assert list(oof.columns) == ["datapointID"] + EV_COLS + CL_COLS
    assert len(oof) > 0 and abs(oof[EV_COLS].sum(1) - 1).max() < 1e-4
    t = pd.read_csv(smoke_dir / "test_fold0.csv")
    assert len(t) == len(test_df) and t["datapointID"].tolist() == test_df["datapointID"].tolist()
    m = json.loads((smoke_dir / "fold0" / "metrics.json").read_text())
    assert m["history"]


def test_deterministic(smoke_dir, tmp_path):
    other = _run(tmp_path, "0,1")
    a = json.loads((smoke_dir / "fold0" / "metrics.json").read_text())["history"]
    b = json.loads((other / "fold0" / "metrics.json").read_text())["history"]
    for x, y in zip(a, b):
        assert x["train_loss"] == pytest.approx(y["train_loss"], rel=1e-5)
        assert x["val_score"] == pytest.approx(y["val_score"])


def test_resume_merges(tmp_path):
    d = _run(tmp_path, "0")
    n0 = len(pd.read_csv(d / "oof.csv"))
    d = _run(tmp_path, "1")
    n01 = len(pd.read_csv(d / "oof.csv"))
    assert n01 > n0


def test_predict_writes_valid_submission(smoke_dir, tmp_path, test_df):
    out = tmp_path / "sub.csv"
    predict.main(["--exp", smoke_dir.name, "--artifacts", str(smoke_dir.parent), "--out", str(out),
                  "--no-append", "--bias", "none"])
    assert validate_submission(out, test_df)["inconsistent"] == 0
