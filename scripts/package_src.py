"""Build dist/polero-src.zip for upload as a Kaggle Dataset (T2.5).

python scripts/package_src.py
Contains the code, configs, scripts, notebooks, data CSVs and artifacts/folds.csv.
Never contains .venv, artifacts (other than folds.csv), .git or caches.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "configs", "scripts", "notebooks", "tests", "prompts"]
INCLUDE_FILES = ["pyproject.toml", "pytest.ini", "requirements.txt", "train_data.csv", "test_data.csv",
                 "sample_output.csv", "artifacts/folds.csv"]
SKIP_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", "artifacts", "dist"}


def files():
    for d in INCLUDE_DIRS:
        p = ROOT / d
        if p.exists():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_PARTS) and f.suffix != ".pyc":
                    yield f
    for name in INCLUDE_FILES:
        f = ROOT / name
        if f.exists():
            yield f


def build(out=None):
    out = Path(out or ROOT / "dist" / "polero-src.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, f.relative_to(ROOT).as_posix())
    return out


if __name__ == "__main__":
    p = build(sys.argv[1] if len(sys.argv) > 1 else None)
    with zipfile.ZipFile(p) as z:
        n = len(z.namelist())
    print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB, {n} files)")
