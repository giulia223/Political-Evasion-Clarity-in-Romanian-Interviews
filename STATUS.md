# STATUS

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
| T0.2 | Repository skeleton | TODO | – | 2026-11-21 | Git repo exists (D-014) |
| T0.3 | Labels and submission writer/validator | TODO | – | 2026-11-21 | |
| T0.4 | Data loading and fixed CV folds | TODO | – | 2026-11-21 | |
| T0.5 | Metrics and OOF evaluation | TODO | – | 2026-11-21 | |
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
