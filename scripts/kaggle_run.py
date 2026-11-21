"""Run polero training on a Kaggle/Colab GPU box (T2.5). Standard library only, because it runs
before the package's dependencies are installed.

python scripts/kaggle_run.py --src /kaggle/working/polero --config robert-large_mh[,xlmr-large_mh,...]
       --folds 0,1,2 [--smoke] [--install pinned|minimal|none]

1. installs the pinned requirements (keeping the platform's CUDA torch)
2. checks that artifacts/folds.csv matches the fixed fold fingerprint (D-019)
3. queues every (config, fold) job; one worker per visible GPU takes the next job (one process per job)
4. merges each config's OOF files, prints its OOF metrics and zips all artifacts/<exp_id> for download
"""
import argparse
import os
import subprocess
import sys
import threading
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


def run_jobs(src, jobs, gpus, smoke):
    """Run (config, fold) jobs from a shared queue: one worker per GPU, each job a separate process."""
    queue, lock, results = list(jobs), threading.Lock(), []

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                cfg, k = queue.pop(0)
            name = f"run_{cfg}_fold{k}.log"
            cmd = [sys.executable, "-m", "polero.train", "--config", f"configs/{cfg}.json", "--folds", str(k),
                   "--no-merge"] + (["--smoke"] if smoke else [])
            t0 = time.time()
            with lock:
                print(f"[GPU {gpu}] start {cfg} fold {k}", flush=True)
            with open(src / name, "w") as logf:
                rc = subprocess.run(cmd, cwd=src, env=env_for(src, gpu), stdout=logf, stderr=subprocess.STDOUT).returncode
            tail = [l for l in (src / name).read_text().splitlines() if "done:" in l or "Error" in l][-3:]
            with lock:
                print(f"[GPU {gpu}] {cfg} fold {k}: exit {rc}, {(time.time() - t0) / 60:.1f} min. " + " | ".join(tail), flush=True)
                results.append((cfg, k, rc, name))

    threads = [threading.Thread(target=worker, args=(g,)) for g in (range(gpus) if gpus else [None])]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="extracted polero-src directory")
    ap.add_argument("--config", required=True, help="config name(s) in configs/ without .json, comma-separated")
    ap.add_argument("--folds", default="0,1,2,3,4,5,6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--install", default="pinned", choices=["pinned", "minimal", "none"])
    ap.add_argument("--max-gpus", type=int, default=8)
    a = ap.parse_args(argv)
    src = a.src.resolve()
    configs = [c.strip() for c in a.config.split(",") if c.strip()]
    missing = [c for c in configs if not (src / "configs" / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"unknown config(s) {missing}; available: {sorted(p.stem for p in (src / 'configs').glob('*.json'))}")
    folds = [int(f) for f in a.folds.split(",")]

    install(src, a.install)
    print((src / "VERSION").read_text().strip() if (src / "VERSION").exists() else "VERSION: (working tree)")
    check_folds(src)
    gpus = min(n_gpus(src), a.max_gpus)
    jobs = [(c, k) for c in configs for k in folds]  # config-major: each config finishes early
    print(f"GPUs: {gpus}; {len(jobs)} jobs ({len(configs)} configs x {len(folds)} folds)", flush=True)

    t0 = time.time()
    results = run_jobs(src, jobs, gpus, a.smoke)
    print(f"training took {(time.time() - t0) / 60:.1f} min", flush=True)

    exp_dirs = []
    for c in configs:
        exp_id = c + ("_smoke" if a.smoke else "")
        exp_dir = src / "artifacts" / exp_id
        for cfg, k, rc, name in results:
            if cfg == c and (src / name).exists():
                exp_dir.mkdir(parents=True, exist_ok=True)
                (src / name).replace(exp_dir / name)
        if any((exp_dir / f"fold{k}" / "oof.csv").exists() for k in folds):
            py(src, f"from pathlib import Path; from polero.train import merge_oof; merge_oof(Path({str(exp_dir)!r}))")
            print(f"\n===== {exp_id} =====", flush=True)
            sh([sys.executable, "-m", "polero.evaluate", str(exp_dir / "oof.csv"), "--no-log"], cwd=src, env=env_for(src))
            exp_dirs.append(exp_dir)

    tag = "+".join(d.name for d in exp_dirs) or "none"
    out = src.parent / f"{tag}_folds{'-'.join(map(str, folds))}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in exp_dirs:
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(d.parent))
    failed = [(c, k) for c, k, rc, _ in results if rc]
    print(f"\nwrote {out} (unzip into artifacts/)")
    if failed:
        raise SystemExit(f"FAILED jobs {failed}: see run_<config>_fold<k>.log inside the zip; finished jobs are saved")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
