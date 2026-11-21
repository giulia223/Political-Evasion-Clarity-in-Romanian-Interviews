# GPU Runbook: training on a free Kaggle GPU

This is for members without a local GPU (CLAUDE.md §5). Everything is free. No tokens or API keys are needed: the model and packages download from public sources.

## One-time setup
1. Create a Kaggle account at https://www.kaggle.com.
2. Verify your phone number (Settings → Phone verification). Kaggle requires this before it allows GPUs and internet access in notebooks.
3. Free quota is about **30 GPU-hours per week**, with sessions of at most **12 hours**. The quota resets weekly; check it under *Settings → Quotas*.

## Each run

### 1. Package the code (on your machine)
```bash
.venv/bin/python -m pytest -q      # everything must pass first
bash scripts/package_src.sh        # -> dist/polero-src.zip (~1.5 MB)
```
- The zip holds the code, configs, scripts, the three CSVs and the fixed `artifacts/folds.csv`.
- `VERSION` inside the zip records the commit. `+dirty` means it includes uncommitted changes, so commit first if you want the run to be traceable.

### 2. Upload it as a Kaggle Dataset
- **First time:** kaggle.com → **Create → New Dataset** → upload `dist/polero-src.zip`. Title it `polero-src` and keep it **Private**. Kaggle may unpack the zip; that's fine, because the notebook handles both forms.
- **After code changes:** open the dataset → **New Version** → upload the new zip.

### 3. Open the notebook
1. **Create → New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_train.ipynb` from the repo.
3. In the right panel, **Add Input → Your Work → Datasets → polero-src**.
4. Under **Settings → Accelerator**, choose **GPU T4 ×2**. The runner trains two fold groups in parallel, one per GPU. P100 also works, but it has only one GPU.
5. Under **Settings → Internet**, choose **On**.

### 4. Set the parameters (first code cell)
| parameter | meaning |
|---|---|
| `CONFIG` | config file name in `configs/` without `.json`, e.g. `robert-large_mh` |
| `FOLDS` | which folds to train, e.g. `"0,1,2,3,4,5,6"`; split a large run across sessions with `"0,1,2,3"` then `"4,5,6"` |
| `SMOKE` | `True` = a tiny offline model that only checks the setup (about 2–3 minutes including installs) |
| `INSTALL` | `"pinned"` (default). If pip fails on Kaggle's image, use `"minimal"` (transformers, accelerate and sentencepiece only) |

**Always do a `SMOKE = True` run first** after uploading a new package version.

### 5. Run it in the background
- Click **Save Version → Save & Run All (Commit)**. The run continues after you close the browser, for up to 12 hours.
- Interactive **Run All** also works, but it stops if the session goes idle.
- The output shows the fold groups per GPU, per-epoch validation scores, the total time, and the merged OOF metrics.

### 6. Download the results
1. Open the finished version → **Output** tab → download `<config>_folds<ids>.zip`, e.g. `robert-large_mh_folds0-1-2-3-4-5-6.zip`.
2. On your machine, from the repo root:
   ```bash
   unzip robert-large_mh_folds0-1-2-3-4-5-6.zip -d artifacts/
   .venv/bin/python -m polero.train --help >/dev/null   # sanity: env OK
   .venv/bin/python -c "from pathlib import Path; from polero.train import merge_oof; merge_oof(Path('artifacts/robert-large_mh'))"
   .venv/bin/python -m polero.evaluate artifacts/robert-large_mh/oof.csv --note "kaggle T4x2"
   .venv/bin/python -m polero.predict --exp robert-large_mh --log
   ```
   - `merge_oof` matters when folds came from several sessions: unzip them all into `artifacts/` first, since each zip adds its own `fold{k}/` files.
   - `evaluate` appends the OOF scores to `docs/RESULTS.md`.
   - `predict` writes a validated submission to `submissions/`.
3. Tell your agent the run finished. It verifies the artifacts and updates STATUS.md.

## Expected runtime (estimate, not yet measured)
The first real run measures this, and its numbers should replace these rough guesses.
- **RoBERT-large on a T4:** about 1–3 minutes per epoch per fold, plus about 20 seconds of evaluation. Most inputs are a single short chunk (median about 113 tokens, docs/EDA.md).
- **A fold:** with early stopping, 5–10 epochs.
- **All 7 folds on T4×2:** about 1–3 hours. This fits in one session.

## Troubleshooting
| symptom | fix |
|---|---|
| `polero-src.zip not found` | add the dataset as notebook input (step 3.3) |
| `does not match the fixed folds` | the package has a different `artifacts/folds.csv`; rebuild it with `python -m polero.folds` locally and re-package |
| pip errors during install | set `INSTALL = "minimal"` |
| CUDA out of memory | lower `batch_chunks` in the config (e.g. 8 → 4); keep `gradient_checkpointing: true` |
| `no GPU visible` | Settings → Accelerator → GPU, and check the phone verification |
| a run stopped at 12 h | run the remaining folds in a new session with `FOLDS`, then merge as in step 6 |
