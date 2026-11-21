"""Generate notebooks/kaggle_train.ipynb and notebooks/colab_train.ipynb.

Both are thin wrappers around scripts/kaggle_run.py (logic stays in src/polero).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = """# ---- settings ----
SMOKE = False   # True = 2-minute setup check (2 folds x 2 steps, tiny model). Then set back to False.
CONFIGS = [
    "configs/robert-large_base.json",   # main model (RoBERT-large, research prior)
    "configs/robert-large_ctx.json",    # + previous question and president/context metadata
    "configs/xlmr-large_base.json",     # XLM-R-large, for ensemble diversity
]
FOLDS = "0,1,2,3,4,5,6"
"""

INSTALL = """subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.23.0', 'sentencepiece==0.2.3'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', '.'], check=True)
import torch, transformers
assert torch.cuda.is_available(), 'No GPU: enable a GPU accelerator first (see the first cell).'
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory / 1e9:.0f} GB')
print('torch', torch.__version__, '| transformers', transformers.__version__)
"""

RUN = """cmd = [sys.executable, '-u', 'scripts/kaggle_run.py', '--folds', FOLDS, '--out-dir', OUT, '--artifacts', ARTIFACTS]
for c in CONFIGS:
    cmd += ['--config', c]
if SMOKE:
    cmd.append('--smoke')
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='')
print('exit code', p.wait())
"""

SHOW = """# per-epoch progress of every fold, and the files to download
for f in sorted(glob.glob(ARTIFACTS + '/*/train.log')):
    print('=' * 20, f)
    print(open(f, encoding='utf-8').read()[-2500:])
print(sorted(glob.glob(OUT + '/*')))
"""

KAGGLE = [
    ("markdown", """# PolERo: train on a free Kaggle GPU

**Settings (right panel) before running:**
1. **Accelerator: GPU T4 x2** (two 16 GB T4s; two folds train at the same time). *GPU P100* also works, but is slower and only one GPU.
2. **Internet: On** (needed to download the public models from Hugging Face; requires a phone-verified Kaggle account).
3. **Add Input** → your dataset `polero-src` (the uploaded `polero-src.zip`).

Then **Save Version → Save & Run All (Commit)**: the run continues in the background (max 12 h) and you can close the browser.
When it finishes, open the version → **Output** → download `out/submission.csv` and the `out/*.zip` files.

Precision and batch size are chosen automatically: fp16 on T4/P100, bf16 on L4/A100; the per-step token budget follows GPU memory."""),
    ("code", SETTINGS),
    ("code", """import glob, os, shutil, subprocess, sys
src = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/pyproject.toml', recursive=True)]
assert src, 'Add the polero-src dataset as input (Add Input -> Your Datasets).'
WORK = '/kaggle/working/polero'
shutil.copytree(src[0], WORK, dirs_exist_ok=True)
os.chdir(WORK)
OUT = '/kaggle/working/out'
ARTIFACTS = '/kaggle/working/polero/artifacts'
""" + INSTALL),
    ("code", RUN),
    ("code", SHOW),
]

COLAB = [
    ("markdown", """# PolERo: train on Google Colab

**Before running:** menu **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → Save.
(With Colab Pro, *L4* or *A100* are faster; the batch size and precision adapt automatically: fp16 on T4, bf16 on L4/A100.)

1. Put `polero-src.zip` in the root of your Google Drive (`My Drive/polero-src.zip`).
2. **Runtime → Run all** and allow access to Google Drive.
3. Results are saved to `My Drive/polero_artifacts` after every fold. If Colab disconnects, just **Run all** again: finished folds are skipped.
4. At the end: `My Drive/polero_out/submission.csv` and one zip per experiment.

Free Colab has a single T4 and limited session time, so the three experiments may need two or three sessions; the order in `CONFIGS` is the priority order."""),
    ("code", SETTINGS),
    ("code", """import glob, os, shutil, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
D = '/content/drive/MyDrive'
ZIP = D + '/polero-src.zip'
if not os.path.exists(ZIP):
    from google.colab import files
    print('polero-src.zip not found in My Drive; upload it now')
    ZIP = '/content/' + list(files.upload())[0]
WORK = '/content/polero'
shutil.rmtree(WORK, ignore_errors=True)
zipfile.ZipFile(ZIP).extractall(WORK)
os.chdir(WORK)
OUT = D + '/polero_out'
ARTIFACTS = D + '/polero_artifacts'   # on Drive: survives disconnects, enables resume
os.makedirs(ARTIFACTS, exist_ok=True)
shutil.copy('artifacts/folds.csv', ARTIFACTS + '/folds.csv')
""" + INSTALL),
    ("code", RUN),
    ("code", SHOW),
]


def write(cells, name):
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    write(KAGGLE, "kaggle_train.ipynb")
    write(COLAB, "colab_train.ipynb")
