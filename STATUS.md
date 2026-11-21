# STATUS

On `main` this file is the shared starting point (T0.1–T0.2 done). Each member's `member/<name>` branch keeps its own copy for their version (D-017).

Update this file when you claim a task, finish it, or get blocked (rules in CLAUDE.md §3). Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `AWAITING_USER`, `DONE`, `DROPPED`.

## Current best
| Field | Value |
|---|---|
| Submission | none yet |
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
| T0.3 | Labels and submission writer/validator | DONE | claude-raluca | 2026-11-21 | `polero.labels`, `polero.submission` (+ CLI). sample_output.csv validates. 19 tests pass |
| T0.4 | Data loading and fixed CV folds | DONE | claude-raluca | 2026-11-21 | `artifacts/folds.csv` (`python -m polero.folds`). Fold sizes 468–469, every class within ±0.86 of expected per fold. Fingerprint `fe46fbf5…` |
| T0.5 | Metrics and OOF evaluation | IN_PROGRESS | claude-raluca | 2026-11-21 | |
| T1.1 | EDA with tokenizers | TODO | – | 2026-11-21 | |
| T1.2 | Text normalization | TODO | – | 2026-11-21 | |
| T2.1 | Sliding-window encoding | TODO | – | 2026-11-21 | |
| T2.2 | Multi-head model | TODO | – | 2026-11-21 | |
| T2.3 | Training loop with CV | TODO | – | 2026-11-21 | |
| T2.4 | Prediction, ensembling, decoding | TODO | – | 2026-11-21 | |
| T2.5 | GPU runner notebook and runbook | TODO | – | 2026-11-21 | |
| T2.6 | Backbone screening (GPU) | TODO | – | 2026-11-21 | |
| T2.7 | Full 7-fold run (GPU) | TODO | – | 2026-11-21 | |
| T2.8 | First real submission | TODO | – | 2026-11-21 | |
| T3.1 | Loss/imbalance ablation (GPU) | TODO | – | 2026-11-21 | |
| T3.2 | Context and metadata features (GPU) | TODO | – | 2026-11-21 | |
| T3.3 | Cross-lingual EN+RO training | DROPPED | – | 2026-11-21 | No external data (D-012). Parked as docs/IDEAS.md I-001 |
| T3.4 | Open-weight LLM few-shot (Kaggle GPU) | TODO | – | 2026-11-21 | Runs on Kaggle (D-013) |
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
- 2026-11-21 · claude-raluca · none · Started Raluca's version on branch `member/raluca`.
- 2026-11-21 · claude-raluca · T0.3 · Claimed.
- 2026-11-21 · claude-raluca · T0.3 · Done. labels.py (EVASION, CLARITY, CLARITY_OF, index maps, EVASION_GROUP), submission.py (write_submission, validate_submission, CLI). The validator rejects: wrong columns or order, row count, non-integer IDs, unknown subtask, duplicates, wrong block or row order, missing or extra IDs, invalid labels. Finding: sample_output.csv has 34 inconsistent clarity/evasion pairs, so the platform accepts inconsistency. It is a warning only (D-018). Full suite: 19 passed.
- 2026-11-21 · claude-raluca · T0.4 · Claimed.
- 2026-11-21 · claude-raluca · T0.4 · Done. polero.data (load_train/load_test with fixed dtypes and label checks, pair_groups) and polero.folds (StratifiedGroupKFold 7, seed 42, grouped by exact Q/A pair). 6 duplicate pairs (12 rows) share folds. The fold fingerprint is pinned in tests/test_folds.py (D-019). Full suite: 26 passed.
- 2026-11-21 · claude-raluca · none · Restored src/polero/submission.py to the committed version, with the user's OK (a stray indent broke the import).
- 2026-11-21 · claude-raluca · T0.5 · Claimed.
