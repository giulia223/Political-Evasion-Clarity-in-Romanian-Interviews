"""T2.3: end-to-end smoke training on CPU (offline tiny backbone)."""
import dataclasses
import json

import pandas as pd
import pytest

from polero import CONFIGS
from polero.data import load_train
from polero.evaluate import main as evaluate_main
from polero.metrics import CL_COLS, EV_COLS, evaluate_oof
from polero.train import SMOKE, TrainConfig, main, run

CONFIG = CONFIGS / "robert-large_mh.json"


def smoke_cfg():
    return dataclasses.replace(TrainConfig.from_json(CONFIG), **SMOKE)


def test_cli_smoke_end_to_end(tmp_path):
    out = tmp_path / "exp"
    assert main(["--config", str(CONFIG), "--smoke", "--folds", "0,1", "--out-dir", str(out)]) == 0
    oof = pd.read_csv(out / "oof.csv")
    assert list(oof.columns) == ["datapointID", *EV_COLS, *CL_COLS, "fold"]
    assert set(oof.fold) == {0, 1} and oof.datapointID.is_unique
    for k in (0, 1):
        test = pd.read_csv(out / f"test_fold{k}.csv")
        assert len(test) == SMOKE["n_test_rows"] and list(test.columns) == ["datapointID", *EV_COLS, *CL_COLS]
        m = json.loads((out / f"fold{k}" / "metrics.json").read_text())
        assert m["steps"] == 2 and m["history"]
    assert json.loads((out / "config.json").read_text())["backbone"] == "smoke"
    evaluate_oof(oof, load_train())  # valid OOF file for polero.evaluate
    assert evaluate_main([str(out / "oof.csv"), "--no-log"]) == 0


def test_deterministic(tmp_path):
    a = run(smoke_cfg(), "a", [0], tmp_path / "a")
    b = run(smoke_cfg(), "b", [0], tmp_path / "b")
    assert [r["history"][0]["train_loss"] for r in a] == [r["history"][0]["train_loss"] for r in b]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "oof.csv"), pd.read_csv(tmp_path / "b" / "oof.csv"))


def test_resumed_folds_merge(tmp_path):
    out = tmp_path / "exp"
    run(smoke_cfg(), "exp", [0], out)
    n0 = len(pd.read_csv(out / "oof.csv"))
    run(smoke_cfg(), "exp", [2, 3], out)
    oof = pd.read_csv(out / "oof.csv")
    assert set(oof.fold) == {0, 2, 3} and len(oof) > n0 and oof.datapointID.is_unique


def test_class_weight_power_option(tmp_path):
    cfg = dataclasses.replace(smoke_cfg(), loss={"class_weight_power": 0.5})
    (r,) = run(cfg, "w", [0], tmp_path / "w")
    assert r["steps"] == 2


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"backbone": "x", "learning_rate": 1e-5}))
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_json(p)
