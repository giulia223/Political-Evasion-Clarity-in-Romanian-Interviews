# GPU runbook (Kaggle, free tier)

Both notebooks `git clone` the public branch `member/giulia`, so you only upload the notebook file — no datasets, no tokens.

## One-time Kaggle setup
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload the `.ipynb` from `notebooks/`.
2. Right panel → **Session options**: Accelerator **GPU T4 x2**, **Internet ON** (needs a phone-verified account).
3. **Run All**. You can close the tab only if you use **Save Version → Save & Run All (Commit)** — that runs in the background (max 12 h) and keeps `/kaggle/working` as output.

## A. Encoder: `notebooks/kaggle_train.ipynb`
- Default `CONFIG = "configs/robert-large_mh.json"` (RoBERT-large, multi-head, max-pool over chunks, 7 folds).
- Folds are split over the 2 GPUs (0,2,4,6 on GPU 0; 1,3,5 on GPU 1). Expected ≈ 2–3 h total (estimate; first run will tell).
- Cell 3 shows progress; rerun it any time. The last cell prints OOF metrics and makes `/kaggle/working/<exp>.zip`.
- Bring it back: download the zip → unzip into `artifacts/<exp>/` in the repo (e.g. `artifacts/robert-large_mh/oof.csv`).
- Next configs to try (one per session): `robert-large_mh_prevq.json` (previous question as context), `xlmr-large_mh.json`.

## B. LLM: `notebooks/kaggle_llm.ipynb`
- Installs Ollama, pulls `MODEL` (default `qwen3.6:35b-a3b`, ~20 GB; if the tag doesn't exist use `qwen3:30b-a3b`).
- Cell order: (1) stratified OOF sample (≤20 per class, ~170 rows) → prints macro-F1; (2) test set with 3 votes each;
  (3, optional, several hours) all of folds 1–6 for stacking.
- Bring back `/kaggle/working/llm_<model>.zip` → unzip into `artifacts/llm_<model>/`.
- Everything is cached in `cache.jsonl`, so a crashed session can be resumed by re-running the same cell
  (copy the old `cache.jsonl` back into `artifacts/llm_<model>/` first).

## After either run
`python -m polero.predict --exp <exp> [--exp <exp2>] --out submissions/<name>.csv` → validated submission.
