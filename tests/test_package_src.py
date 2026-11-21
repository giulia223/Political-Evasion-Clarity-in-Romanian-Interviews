"""T2.5: code package contents, and the Kaggle notebook + runner executed top to bottom in smoke mode."""
import json
import os
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from polero import REPO_ROOT

REQUIRED = {"VERSION", "pyproject.toml", "requirements.txt", "train_data.csv", "test_data.csv", "sample_output.csv",
            "artifacts/folds.csv", "scripts/kaggle_run.py", "src/polero/train.py", "configs/robert-large_mh.json"}


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist") / "polero-src.zip"
    subprocess.run(["bash", "scripts/package_src.sh", str(out)], cwd=REPO_ROOT, check=True,
                   env=dict(os.environ, PY=sys.executable), capture_output=True)
    return out


def test_zip_contents(package):
    names = set(zipfile.ZipFile(package).namelist())
    assert REQUIRED <= names
    assert not [n for n in names if n.startswith((".venv", ".git/", "artifacts/")) and n != "artifacts/folds.csv"]
    assert not [n for n in names if "__pycache__" in n or n.startswith(("tests/", "docs/", ".claude/"))]


def test_notebook_runs_top_to_bottom_in_smoke_mode(package, tmp_path):
    """Execute the notebook's code cells with SRC pointing at the zip and a temp work dir."""
    nb = json.loads((REPO_ROOT / "notebooks" / "kaggle_train.ipynb").read_text())
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    params = (code[0].replace('SMOKE = False', 'SMOKE = True').replace('INSTALL = "pinned"', 'INSTALL = "none"')
              .replace('SRC = None', f'SRC = {str(package)!r}').replace('FOLDS = "0,1,2,3,4,5,6"', 'FOLDS = "0,1,2"'))
    os.environ["POLERO_WORK"] = str(tmp_path)
    try:
        ns = {}
        for cell in [params, *code[1:]]:
            exec(compile(cell, "<notebook>", "exec"), ns)
    finally:
        os.environ.pop("POLERO_WORK")
    out = tmp_path / "robert-large_mh_smoke_folds0-1-2.zip"
    assert out.exists()
    names = zipfile.ZipFile(out).namelist()
    assert "robert-large_mh_smoke/oof.csv" in names and "robert-large_mh_smoke/test_fold2.csv" in names
    oof = pd.read_csv(tmp_path / "polero" / "artifacts" / "robert-large_mh_smoke" / "oof.csv")
    assert set(oof.fold) == {0, 1, 2}


def test_runner_rejects_wrong_folds(package, tmp_path):
    repo = tmp_path / "polero"
    zipfile.ZipFile(package).extractall(repo)
    f = pd.read_csv(repo / "artifacts" / "folds.csv")
    f.loc[0, "fold"] = (f.loc[0, "fold"] + 1) % 7
    f.to_csv(repo / "artifacts" / "folds.csv", index=False)
    r = subprocess.run([sys.executable, str(repo / "scripts" / "kaggle_run.py"), "--src", str(repo), "--config",
                        "robert-large_mh", "--smoke", "--install", "none", "--folds", "0"], capture_output=True, text=True)
    assert r.returncode != 0 and "does not match the fixed folds" in r.stderr


def test_runner_queues_several_configs(package, tmp_path):
    repo = tmp_path / "polero"
    zipfile.ZipFile(package).extractall(repo)
    r = subprocess.run([sys.executable, str(repo / "scripts" / "kaggle_run.py"), "--src", str(repo), "--config",
                        "robert-large_mh,ro-bert-base_mh", "--folds", "0,1", "--smoke", "--install", "none"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "4 jobs (2 configs x 2 folds)" in r.stdout
    names = zipfile.ZipFile(tmp_path / "robert-large_mh_smoke+ro-bert-base_mh_smoke_folds0-1.zip").namelist()
    for exp in ("robert-large_mh_smoke", "ro-bert-base_mh_smoke"):
        assert f"{exp}/oof.csv" in names and f"{exp}/run_{exp[:-6]}_fold1.log" in names


def test_runner_rejects_unknown_config(package, tmp_path):
    repo = tmp_path / "polero"
    zipfile.ZipFile(package).extractall(repo)
    r = subprocess.run([sys.executable, str(repo / "scripts" / "kaggle_run.py"), "--src", str(repo), "--config",
                        "nope", "--install", "none"], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown config" in r.stderr
