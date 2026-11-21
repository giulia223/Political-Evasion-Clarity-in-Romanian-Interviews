"""Train experiments on all available GPUs, zip artifacts, and write a submission (T2.5).

python scripts/kaggle_run.py --config configs/a.json [--config configs/b.json] [--folds 0,1,2]
                             [--smoke] [--out-dir DIR] [--artifacts DIR]

- Folds of each experiment are spread over the GPUs (one training process per GPU;
  Kaggle T4 x2 -> 2 folds at a time, Colab -> 1).
- Folds that already finished in --artifacts are skipped, so a run that was cut off
  (Colab disconnect, Kaggle 12 h limit) resumes where it stopped when re-run.
- Each finished experiment is zipped to <out-dir>/<exp_id>.zip right away.
- At the end, all finished experiments are ensembled into <out-dir>/submission.csv
  (decoder and bias chosen on nested OOF; see polero.predict).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def n_gpus():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def exp_id(cfg_path, smoke):
    exp = json.loads(Path(cfg_path).read_text(encoding="utf-8")).get("exp_id", Path(cfg_path).stem)
    return exp + ("_smoke" if smoke else "")


def run_experiment(cfg_path, folds, gpus, smoke, artifacts):
    exp = exp_id(cfg_path, smoke)
    devices = [f"cuda:{i}" for i in range(gpus)] or ["cpu"]
    groups = [folds[i::len(devices)] for i in range(len(devices))]
    procs = []
    for dev, fs in zip(devices, groups):
        if not fs:
            continue
        cmd = [sys.executable, "-m", "polero.train", "--config", str(cfg_path), "--folds", ",".join(map(str, fs)),
               "--device", dev, "--out", str(artifacts), "--skip-done"] + (["--smoke"] if smoke else [])
        (Path(artifacts) / exp).mkdir(parents=True, exist_ok=True)
        log = open(Path(artifacts) / exp / f"stdout_{dev.replace(':', '')}.txt", "a", encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {exp}: folds {fs} on {dev}", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, dev))
    failed = False
    for p, log, dev in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed = True
            print(f"!! {exp} on {dev} exited with code {rc}; last lines of {log.name}:", flush=True)
            print("".join(open(log.name, encoding="utf-8").readlines()[-25:]), flush=True)
    subprocess.run([sys.executable, "-m", "polero.train", "--config", str(cfg_path), "--merge-only",
                    "--out", str(artifacts)] + (["--smoke"] if smoke else []), cwd=ROOT, check=True)
    return exp, failed


def complete(artifacts, exp, folds):
    d = Path(artifacts) / exp
    return all((d / f"test_fold{k}.csv").exists() and (d / f"fold{k}" / "oof_part.csv").exists() for k in folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "out"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    if a.smoke:
        folds = folds[:2]
    artifacts = Path(a.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    if not (artifacts / "folds.csv").exists() and (ROOT / "artifacts" / "folds.csv").exists():
        shutil.copy(ROOT / "artifacts" / "folds.csv", artifacts / "folds.csv")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = n_gpus()
    print(f"GPUs: {g}", flush=True)
    if g == 0 and not a.smoke:
        sys.exit("No GPU found. Kaggle: Settings -> Accelerator -> GPU T4 x2. "
                 "Colab: Runtime -> Change runtime type -> T4 GPU.")
    done = []
    for cfg in a.config:
        t0 = time.time()
        exp, failed = run_experiment(cfg, folds, g, a.smoke, artifacts)
        zip_path = shutil.make_archive(str(out / exp), "zip", root_dir=artifacts, base_dir=exp)
        ok = complete(artifacts, exp, folds)
        print(f"[{time.strftime('%H:%M:%S')}] {exp} {'complete' if ok else 'INCOMPLETE'} in "
              f"{(time.time() - t0) / 60:.1f} min -> {zip_path}", flush=True)
        if (artifacts / exp / "oof.csv").exists():
            subprocess.run([sys.executable, "-m", "polero.evaluate", str(artifacts / exp / "oof.csv"),
                            "--no-append"], cwd=ROOT)
        if ok:
            done.append(exp)
    if done and not a.smoke:
        cmd = [sys.executable, "-m", "polero.predict", "--artifacts", str(artifacts),
               "--out", str(out / "submission.csv"), "--no-append"]
        for e in done:
            cmd += ["--exp", e]
        print(f"ensembling {done} -> {out / 'submission.csv'}", flush=True)
        subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
