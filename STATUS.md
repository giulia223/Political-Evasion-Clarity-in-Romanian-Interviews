# STATUS

On `main` this file is the shared starting point (T0.1–T0.2 done). Each member's `member/<name>` branch keeps its own copy for their version (D-017).

Update this file when you claim a task, finish it, or get blocked (rules in CLAUDE.md §3). Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `AWAITING_USER`, `DONE`, `DROPPED`.

## Current best
| Field | Value |
|---|---|
| Submission | submissions/sub_lr_baseline.csv (CPU baseline) |
| System | none yet |
| OOF evasion macro-F1 | – |
| OOF mapped clarity macro-F1 | – |
| Decoder | – |
| Public leaderboard (clarity / evasion) | – |

## Tasks
| ID | Task | Status | Owner | Updated | Notes / result |
|---|---|---|---|---|---|
| T0.1 | Environment and dependencies | DONE | setup | 2026-11-21 | torch 2.15.0+cpu, transformers 5.23.0 and others (D-015). `tests/test_env.py` passes |
| T0.2 | Repository skeleton | DONE | claude-setup | 2026-11-21 | `pip install -e .` works. Data checksums in tests/data_checksums.json. 4 tests pass |
| T0.3 | Labels and submission writer/validator | DONE | claude-giulia | 2026-11-21 | labels.py, submission.py (+CLI). tests/test_core.py |
| T0.4 | Data loading and fixed CV folds | DONE | claude-giulia | 2026-11-21 | 7-fold StratifiedGroupKFold (dups grouped), seed 42 → artifacts/folds.csv; every class in every fold |
| T0.5 | Metrics and OOF evaluation | DONE | claude-giulia | 2026-11-21 | metrics.py: E, M, C, per-class, confusion, nested-CV bias decoding; `python -m polero.metrics <oof.csv>` appends to docs/RESULTS.md |
| T1.1 | EDA with tokenizers | DONE (light) | claude-giulia | 2026-11-21 | RoBERT-large/base: uncased BertTokenizer, comma form best, UNK 0. XLM-R: cedilla best. 512/128 chunking: 3.3% of rows >1 chunk, p50 113 / p90 341 tokens. No separate EDA.md (D-018) |
| T1.2 | Text normalization | DONE | claude-giulia | 2026-11-21 | chunking.normalize + automatic comma/cedilla pick per tokenizer (lowest UNK, then fewest tokens) |
| T2.1 | Sliding-window encoding | DONE | claude-giulia | 2026-11-21 | chunking.py; pair template inferred (transformers 5 dropped build_inputs_with_special_tokens) |
| T2.2 | Multi-head model | DONE | claude-giulia | 2026-11-21 | model.py: max/mean/attn/first pooling, CE/LS/focal/class weights |
| T2.3 | Training loop with CV | DONE | claude-giulia | 2026-11-21 | train.py; smoke test passes on CPU. CPU too slow for real runs (RoBERT-base ≈1 ex/s) |
| T2.4 | Prediction, ensembling, decoding | IN_PROGRESS | claude-giulia | 2026-11-21 | decoders a/b + nested bias tuning done (decode.py); predict CLI next |
| T2.5 | GPU runner notebook and runbook | DONE | claude-giulia | 2026-11-21 | notebooks/kaggle_train.ipynb clones the public branch; folds split over 2 GPUs |
| T2.6 | Backbone screening (GPU) | TODO | – | 2026-11-21 | |
| T2.7 | Full 7-fold run (GPU) | TODO | – | 2026-11-21 | |
| T2.8 | First real submission | TODO | – | 2026-11-21 | |
| T3.1 | Loss/imbalance ablation (GPU) | TODO | – | 2026-11-21 | |
| T3.2 | Context and metadata features (GPU) | TODO | – | 2026-11-21 | |
| T3.3 | Cross-lingual EN+RO training | DROPPED | – | 2026-11-21 | No external data (D-012). Parked as docs/IDEAS.md I-001 |
| T3.4 | Open-weight LLM few-shot (Kaggle GPU) | AWAITING_USER | claude-giulia | 2026-11-21 | polero/llm.py + notebooks/kaggle_llm.ipynb (Ollama). Moved up (D-019) |
| T3.5 | Encoder-guided LLM reranking | TODO | – | 2026-11-21 | |
| T3.6 | Final ensemble and stacking | TODO | – | 2026-11-21 | |
| T3.7 | Seed ensemble (GPU) | TODO | – | 2026-11-21 | |
| T4.1 | Final submission | TODO | – | 2026-11-21 | |
| T4.2 | Write-up | TODO | – | 2026-11-21 | |

## Waiting on the user
- (nothing pending)

## Log
Append-only, newest at the bottom. Format: `YYYY-MM-DD HH:MM · <agent> · <task> · <what happened, with numbers and artifact paths>`
- 2026-11-21 · setup · none · Installed skills (D-010), created `.venv` (D-009). Built and removed the TF-IDF baseline (D-007). Wrote the roadmap, CLAUDE.md, DECISIONS.md, STATUS.md and docs/RESEARCH.md.
- 2026-11-21 · setup · T0.1 · The user approved the installs. Pinned the stack in requirements.txt (D-015). tests/test_env.py: 2 passed. Recorded the user's answers as D-011 to D-014. T3.3 dropped, T3.4 moved to Kaggle. Initial commit pushed to GitHub.
- 2026-11-21 · setup · none · The user asked to commit the data CSVs (D-016).
- 2026-11-21 · claude-setup · T0.2 · Claimed.
- 2026-11-21 · claude-setup · T0.2 · Done: pyproject, pytest.ini, src/polero, folder layout, .gitattributes (CSV byte-identical), checksum test. Full suite: 4 passed. Added docs/TEAM.md (3 tracks) and a Track column.
- 2026-11-21 · claude-setup · none · The user dropped the 3-track split. Each member builds their own version on `member/<name>` (D-017). Removed docs/TEAM.md and the Track column.
- 2026-11-21 · claude-giulia · none · Started version on member/giulia (cloned over HTTPS; SSH key not set up). Plan: docs/PLAN_giulia.md.
- 2026-11-21 · claude-giulia · T0.3–T2.5 · Built core package, CPU baseline, encoder trainer, LLM client and Kaggle notebooks. pytest: 18 passed.
- 2026-11-21 · claude-giulia · baseline · lr_tfidf_feats_prior OOF: E .275 / M .486 argmax; nested-bias E .323 / M .543 (docs/RESULTS.md).
