# STATUS

Update this file when you claim a task, finish it, or get blocked (rules in CLAUDE.md §3). Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `AWAITING_USER`, `DONE`, `DROPPED`.

## Current best
| Field | Value |
|---|---|
| Submission | none yet (waiting for the first GPU run) |
| System | none yet |
| OOF evasion macro-F1 | – |
| OOF mapped clarity macro-F1 | – |
| Decoder | – |
| Public leaderboard (clarity / evasion) | – |

## Tasks
| ID | Task | Status | Owner | Updated | Notes / result |
|---|---|---|---|---|---|
| T0.1 | Environment and dependencies | DONE | setup | 2026-11-21 | torch 2.15.0+cpu, transformers 5.23.0 and others (D-015). Windows paths: D-023 |
| T0.2 | Repository skeleton | DONE | ar/model_nitro | 2026-11-21 | `src/polero`, `configs/`, `notebooks/`, `scripts/`, `pyproject.toml`, `pytest.ini`; CSV checksums in `tests/data_checksums.json` |
| T0.3 | Labels and submission writer/validator | DONE | ar/model_nitro | 2026-11-21 | `polero.labels`, `polero.submission` (+CLI). sample_output.csv is valid but has 34 inconsistent pairs (D-021) |
| T0.4 | Data loading and fixed CV folds | DONE | ar/model_nitro | 2026-11-21 | `artifacts/folds.csv`, fold sizes 467–471 (D-017) |
| T0.5 | Metrics and OOF evaluation | DONE | ar/model_nitro | 2026-11-21 | `polero.metrics`, `python -m polero.evaluate` → docs/RESULTS.md |
| T1.1 | EDA with tokenizers | DONE | ar/model_nitro | 2026-11-21 | `docs/EDA.md` from `scripts/eda.py` (tokenizers only, no weights) |
| T1.2 | Text normalization | DONE | ar/model_nitro | 2026-11-21 | `polero.text`; modes per backbone in D-019 |
| T2.1 | Sliding-window encoding | DONE | ar/model_nitro | 2026-11-21 | `polero.chunking` (D-018) |
| T2.2 | Multi-head model | DONE | ar/model_nitro | 2026-11-21 | `polero.model`: max/mean/attn pooling, CE/weights/focal/smoothing |
| T2.3 | Training loop with CV | DONE | ar/model_nitro | 2026-11-21 | `python -m polero.train`; `--smoke` runs on CPU in ~11 s; `--skip-done` resume |
| T2.4 | Prediction, ensembling, decoding | DONE (code) | ar/model_nitro | 2026-11-21 | `polero.decode`, `polero.ensemble`, `python -m polero.predict` (D-020). Decoder table on real OOF comes with the first GPU run |
| T2.5 | GPU runner notebook and runbook | AWAITING_USER | ar/model_nitro | 2026-11-21 | `notebooks/kaggle_train.ipynb`, `notebooks/colab_train.ipynb`, `scripts/kaggle_run.py`, `docs/GPU_RUNBOOK.md`. User runs it (D-022) |
| T2.6 | Backbone screening (GPU) | AWAITING_USER | ar/model_nitro | 2026-11-21 | Folded into the first run: RoBERT-large and XLM-R-large on all 7 folds |
| T2.7 | Full 7-fold run (GPU) | AWAITING_USER | ar/model_nitro | 2026-11-21 | configs `robert-large_base`, `robert-large_ctx`, `xlmr-large_base` |
| T2.8 | First real submission | AWAITING_USER | ar/model_nitro | 2026-11-21 | The notebook writes `out/submission.csv` (ensemble of finished experiments) |
| T3.1 | Loss/imbalance ablation (GPU) | TODO | – | 2026-11-21 | Options already in the code (`class_weights`, `focal_gamma`, `label_smoothing`, `w_c`) |
| T3.2 | Context and metadata features (GPU) | IN_PROGRESS | ar/model_nitro | 2026-11-21 | `polero.context` + `configs/robert-large_ctx.json`; result comes with the first GPU run |
| T3.3 | Cross-lingual EN+RO training | DROPPED | – | 2026-11-21 | No external data (D-012). Parked as docs/IDEAS.md I-001 |
| T3.4 | Open-weight LLM few-shot (Kaggle GPU) | TODO | – | 2026-11-21 | Runs on Kaggle (D-013) |
| T3.5 | Encoder-guided LLM reranking | TODO | – | 2026-11-21 | |
| T3.6 | Final ensemble and stacking | TODO | – | 2026-11-21 | Weighted averaging is in `polero.ensemble`; stacking not yet |
| T3.7 | Seed ensemble (GPU) | TODO | – | 2026-11-21 | |
| T4.1 | Final submission | TODO | – | 2026-11-21 | |
| T4.2 | Write-up | TODO | – | 2026-11-21 | |

## Waiting on the user
- Run `notebooks/kaggle_train.ipynb` (Kaggle, GPU T4 x2) or `notebooks/colab_train.ipynb` (Colab, T4) with `dist/polero-src.zip`, following docs/GPU_RUNBOOK.md. Bring back `out/*.zip` and `out/submission.csv`.

## Log
Append-only, newest at the bottom. Format: `YYYY-MM-DD HH:MM · <agent> · <task> · <what happened, with numbers and artifact paths>`
- 2026-11-21 · setup · none · Installed skills (D-010), created `.venv` (D-009). Built and removed the TF-IDF baseline (D-007). Wrote the roadmap, CLAUDE.md, DECISIONS.md, STATUS.md and docs/RESEARCH.md.
- 2026-11-21 · setup · T0.1 · The user approved the installs. Pinned the stack in requirements.txt (D-015). tests/test_env.py: 2 passed. Recorded the user's answers as D-011 to D-014. T3.3 dropped, T3.4 moved to Kaggle. Initial commit pushed to GitHub.
- 2026-11-21 · setup · none · The user asked to commit the data CSVs (D-016).
- 2026-11-21 · claude-setup · T0.2 · Claimed.
- 2026-11-21 · ar/model_nitro · T0.1 · New machine (Windows, i5-12450H, 7.7 GB RAM, no GPU). User approved the installs; `.venv` created with the pinned requirements (torch 2.15.0+cpu, transformers 5.23.0). Work is on local branch `ar/model_nitro`, not pushed (user's request).
- 2026-11-21 · ar/model_nitro · T0.2–T2.4 · Implemented the `polero` package, folds (`artifacts/folds.csv`), EDA (`docs/EDA.md`) and tests: `python -m pytest -q` → 66 passed. CPU smoke training end to end in ~11 s.
- 2026-11-21 · ar/model_nitro · T2.5 · Measured CPU training as impractical (~47 s per RoBERT-small step); user asked to run on Kaggle/Colab only (D-022). Added auto GPU budgets, per-fold resume, Kaggle and Colab notebooks, runbook and `scripts/package_src.py`. No real-data metrics exist yet.
