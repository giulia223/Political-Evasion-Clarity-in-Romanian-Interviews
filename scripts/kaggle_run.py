"""Run polero training on a Kaggle/Colab GPU box (T2.5). Standard library only, because it runs
before the package's dependencies are installed.

python scripts/kaggle_run.py --src /kaggle/working/polero --config robert-large_mh --folds 0,1,2,3
       [--smoke] [--install pinned|minimal|none]

1. installs the pinned requirements (keeping the platform's CUDA torch)
2. checks that artifacts/folds.csv matches the fixed fold fingerprint (D-019)
3. splits the folds across the visible GPUs and trains them in parallel (one process per GPU)
4. merges the OOF files, prints the OOF metrics and zips artifacts/<exp_id> for download
"""
import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sh(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def py(src, code):
    """Run python code against the repo at `src` (its src/ on PYTHONPATH, cwd = repo)."""
    return subprocess.run([sys.executable, "-c", code], cwd=src, env=env_for(src), check=True,
                          capture_output=True, text=True).stdout.strip()


def env_for(src, gpu=None):
    env = dict(os.environ, PYTHONPATH=str(Path(src) / "src"), TOKENIZERS_PARALLELISM="false")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def install(src, mode):
    if mode == "none":
        return
    reqs = [l.strip() for l in (Path(src) / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("torch")]
    if mode == "minimal":  # only what the platform image is most likely to lack
        reqs = [r for r in reqs if r.split("==")[0] in ("transformers", "accelerate", "sentencepiece")]
    sh([sys.executable, "-m", "pip", "install", "-q", *reqs])


def check_folds(src):
    out = py(src, "import pandas as pd; from polero.folds import FOLDS_CSV, EXPECTED_FINGERPRINT, fingerprint;"
                  "f = fingerprint(pd.read_csv(FOLDS_CSV)); print(f == EXPECTED_FINGERPRINT, f)")
    ok, fp = out.split()
    if ok != "True":
        raise SystemExit(f"artifacts/folds.csv fingerprint {fp} does not match the fixed folds; re-package the source")
    print("folds OK", fp[:12])


def n_gpus(src):
    return int(py(src, "import torch; print(torch.cuda.device_count())"))


def split(folds, n):
    groups = [folds[i::n] for i in range(n)]
    return [g for g in groups if g]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="extracted polero-src directory")
    ap.add_argument("--config", required=True, help="config name in configs/ (without .json)")
    ap.add_argument("--folds", default="0,1,2,3,4,5,6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--install", default="pinned", choices=["pinned", "minimal", "none"])
    ap.add_argument("--max-gpus", type=int, default=8)
    a = ap.parse_args(argv)
    src = a.src.resolve()
    exp_id = a.config + ("_smoke" if a.smoke else "")
    folds = [int(f) for f in a.folds.split(",")]

    install(src, a.install)
    print((src / "VERSION").read_text().strip() if (src / "VERSION").exists() else "VERSION: (working tree)")
    check_folds(src)
    gpus = min(n_gpus(src), a.max_gpus)
    groups = split(folds, max(gpus, 1))
    print(f"GPUs: {gpus}; fold groups: {groups}", flush=True)

    t0, procs = time.time(), []
    for i, g in enumerate(groups):
        cmd = [sys.executable, "-m", "polero.train", "--config", f"configs/{a.config}.json",
               "--folds", ",".join(map(str, g)), "--no-merge"] + (["--smoke"] if a.smoke else [])
        logf = open(src / f"run_gpu{i}.log", "w")
        print("$", " ".join(cmd), f"(GPU {i if gpus else 'none'}, log run_gpu{i}.log)", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=src, env=env_for(src, i if gpus else None),
                                       stdout=logf, stderr=subprocess.STDOUT), logf, i))
    failed = []
    for p, logf, i in procs:
        p.wait()
        logf.close()
        tail = (src / f"run_gpu{i}.log").read_text().splitlines()[-15:]
        print(f"--- GPU {i} exit {p.returncode} ---\n" + "\n".join(tail), flush=True)
        if p.returncode:
            failed.append(i)
    if failed:
        raise SystemExit(f"training failed on GPU(s) {failed}; see run_gpu*.log")
    print(f"training took {(time.time() - t0) / 60:.1f} min", flush=True)

    exp_dir = src / "artifacts" / exp_id
    py(src, f"from pathlib import Path; from polero.train import merge_oof; merge_oof(Path({str(exp_dir)!r}))")
    sh([sys.executable, "-m", "polero.evaluate", str(exp_dir / "oof.csv"), "--no-log"], cwd=src, env=env_for(src))
    for i, _ in enumerate(groups):
        log = src / f"run_gpu{i}.log"
        log.replace(exp_dir / log.name)
    out = src.parent / f"{exp_id}_folds{'-'.join(map(str, folds))}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in exp_dir.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(exp_dir.parent))
    print(f"\nDONE. Download {out} and unzip it into artifacts/ (it contains {exp_id}/...).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
