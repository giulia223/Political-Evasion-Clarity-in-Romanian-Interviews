"""Generate the thin Kaggle/Colab runner notebooks. Run: python scripts/make_notebooks.py"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "https://github.com/giulia223/Political-Evasion-Clarity-in-Romanian-Interviews.git"
BRANCH = "member/giulia"


def nb(cells):
    return {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = f"""
# Settings: Kaggle -> Accelerator "GPU T4 x2", Internet ON.
import os, subprocess, sys
if not os.path.exists("repo"):
    subprocess.run(["git", "clone", "-b", "{BRANCH}", "--depth", "1", "{REPO}", "repo"], check=True)
else:
    subprocess.run(["git", "-C", "repo", "pull"], check=True)
os.chdir("repo")
# Kaggle already has torch/transformers/sklearn/pandas; install only our package (no dependency changes).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
import torch; print(torch.__version__, torch.cuda.device_count(), "GPUs")
"""

train = nb([
    ("markdown", """
# PolERo encoder training (multi-head, sliding window, 7-fold CV)
Runs `python -m polero.train` with the folds split across the 2 T4 GPUs (one process per GPU).
Set `CONFIG` below, run all cells, then download `/kaggle/working/<exp>.zip` and unzip it into `artifacts/` locally.
"""),
    ("code", SETUP),
    ("code", """
CONFIG = "configs/robert-large_mh.json"   # <- change per experiment
SMOKE = False                              # True = quick CPU/GPU sanity run
EXP = os.path.splitext(os.path.basename(CONFIG))[0] + ("_smoke" if SMOKE else "")
n_gpu = max(1, torch.cuda.device_count())
groups = [",".join(str(k) for k in range(7) if k % n_gpu == g) for g in range(n_gpu)]
procs = []
for g, folds in enumerate(groups):
    cmd = [sys.executable, "-m", "polero.train", "--config", CONFIG, "--folds", folds]
    if torch.cuda.is_available(): cmd += ["--device", f"cuda:{g}"]
    if SMOKE: cmd += ["--smoke"]
    print(" ".join(cmd))
    procs.append(subprocess.Popen(cmd, stdout=open(f"/kaggle/working/log_gpu{g}.txt", "w"), stderr=subprocess.STDOUT))
"""),
    ("code", """
# Progress (re-run this cell to refresh; training continues in the background)
for g in range(len(procs)):
    print(f"--- GPU {g} (running={procs[g].poll() is None})")
    print("".join(open(f"/kaggle/working/log_gpu{g}.txt").readlines()[-6:]))
"""),
    ("code", """
# Wait for both, merge OOF, evaluate, zip for download
for p in procs: p.wait()
from polero.train import merge_oof
from polero import ARTIFACTS
print("oof folds merged:", merge_oof(ARTIFACTS / EXP))
subprocess.run([sys.executable, "-m", "polero.metrics", f"artifacts/{EXP}/oof.csv", "kaggle"])
import shutil; shutil.make_archive(f"/kaggle/working/{EXP}", "zip", f"artifacts/{EXP}")
print("download /kaggle/working/" + EXP + ".zip")
"""),
])

llm = nb([
    ("markdown", """
# PolERo LLM few-shot classifier (Ollama, localhost only)
Serves an open-weight model locally with Ollama on the 2 T4 GPUs and runs `python -m polero.llm`.
Few-shot examples come from fold 0 only, so outputs on folds 1-6 are valid OOF.
Run order: setup -> server -> (1) oof_sample to measure quality -> (2) test -> (3, optional) full oof for stacking.
Download `/kaggle/working/<exp>.zip` and unzip into `artifacts/` locally.
"""),
    ("code", SETUP),
    ("code", """
# Install and start Ollama (localhost:11434). 2 GPUs are used automatically for big models.
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
env = dict(os.environ, OLLAMA_NUM_PARALLEL="8", OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_KEEP_ALIVE="-1", OLLAMA_FLASH_ATTENTION="1")
server = subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/kaggle/working/ollama.log", "w"), stderr=subprocess.STDOUT)
import time; time.sleep(8)
MODEL = "qwen3.6:35b-a3b"   # if this tag does not exist, try "qwen3:30b-a3b" (smaller/faster) or check ollama.com/library
r = subprocess.run(["ollama", "pull", MODEL]); print("pull rc", r.returncode)
"""),
    ("code", """
EXP = "llm_" + MODEL.replace(":", "_").replace(".", "")
base = ["--base-url", "http://localhost:11434/v1", "--model", MODEL, "--exp", EXP, "--workers", "8"]
# (1) quality check on a stratified sample (<=20 per class from folds 1-6), 1 sample each
subprocess.run([sys.executable, "-m", "polero.llm", *base, "--split", "oof_sample", "--n", "1"])
"""),
    ("code", """
# (2) test set, 3 samples each (self-consistency votes -> probabilities)
subprocess.run([sys.executable, "-m", "polero.llm", *base, "--split", "test", "--n", "3"])
"""),
    ("code", """
# (3) optional, long: all of folds 1-6 (~2800 rows) for stacking with the encoder
# subprocess.run([sys.executable, "-m", "polero.llm", *base, "--split", "oof", "--n", "1"])
"""),
    ("code", """
import shutil; shutil.make_archive(f"/kaggle/working/{EXP}", "zip", f"artifacts/{EXP}")
print("download /kaggle/working/" + EXP + ".zip")
"""),
])

NB.mkdir(exist_ok=True)
(NB / "kaggle_train.ipynb").write_text(json.dumps(train, indent=1), encoding="utf-8")
(NB / "kaggle_llm.ipynb").write_text(json.dumps(llm, indent=1), encoding="utf-8")
print("wrote", NB / "kaggle_train.ipynb", NB / "kaggle_llm.ipynb")
