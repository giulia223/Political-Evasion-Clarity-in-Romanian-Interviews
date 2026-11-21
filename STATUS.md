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
| ID | Track | Task | Status | Owner | Updated | Notes / result |
|---|---|---|---|---|---|---|
| T0.1 | – | Environment and dependencies | DONE | setup | 2026-11-21 | torch 2.15.0+cpu, transformers 5.23.0 and others (D-015). `tests/test_env.py` passes |
| T0.2 | – | Repository skeleton | DONE | claude-setup | 2026-11-21 | `pip install -e .` works. Data checksums in tests/data_checksums.json. 4 tests pass |
| T0.3 | A | Labels and submission writer/validator | TODO | – | 2026-11-21 | |
| T0.4 | B | Data loading and fixed CV folds | TODO | – | 2026-11-21 | |
| T0.5 | A | Metrics and OOF evaluation | TODO | – | 2026-11-21 | |
| T1.1 | A | EDA with tokenizers | TODO | – | 2026-11-21 | |
| T1.2 | A | Text normalization | TODO | – | 2026-11-21 | |
| T2.1 | B | Sliding-window encoding | TODO | – | 2026-11-21 | |
| T2.2 | B | Multi-head model | TODO | – | 2026-11-21 | |
| T2.3 | B | Training loop with CV | TODO | – | 2026-11-21 | |
| T2.4 | A | Prediction, ensembling, decoding | TODO | – | 2026-11-21 | |
| T2.5 | C | GPU runner notebook and runbook | TODO | – | 2026-11-21 | |
| T2.6 | C | Backbone screening (GPU) | TODO | – | 2026-11-21 | |
| T2.7 | C | Full 7-fold run (GPU) | TODO | – | 2026-11-21 | |
| T2.8 | A | First real submission | TODO | – | 2026-11-21 | |
| T3.1 | B | Loss/imbalance ablation (GPU) | TODO | – | 2026-11-21 | |
| T3.2 | B | Context and metadata features (GPU) | TODO | – | 2026-11-21 | |
| T3.3 | – | Cross-lingual EN+RO training | DROPPED | – | 2026-11-21 | No external data (D-012). Parked as docs/IDEAS.md I-001 |
| T3.4 | C | Open-weight LLM few-shot (Kaggle GPU) | TODO | – | 2026-11-21 | Runs on Kaggle (D-013) |
| T3.5 | C | Encoder-guided LLM reranking | TODO | – | 2026-11-21 | |
| T3.6 | A | Final ensemble and stacking | TODO | – | 2026-11-21 | |
| T3.7 | B | Seed ensemble (GPU) | TODO | – | 2026-11-21 | |
| T4.1 | A | Final submission | TODO | – | 2026-11-21 | |
| T4.2 | C | Write-up | TODO | – | 2026-11-21 | |

## Waiting on the user
- (nothing pending)

## Log
Append-only, newest at the bottom. Format: `YYYY-MM-DD HH:MM · <agent> · <task> · <what happened, with numbers and artifact paths>`
- 2026-11-21 · setup · none · Installed skills (D-010), created `.venv` (D-009). Built and removed the TF-IDF baseline (D-007). Wrote the roadmap, CLAUDE.md, DECISIONS.md, STATUS.md and docs/RESEARCH.md.
- 2026-11-21 · setup · T0.1 · The user approved the installs. Pinned the stack in requirements.txt (D-015). tests/test_env.py: 2 passed. Recorded the user's answers as D-011 to D-014. T3.3 dropped, T3.4 moved to Kaggle. Initial commit pushed to GitHub.
- 2026-11-21 · setup · none · The user asked to commit the data CSVs (D-016).
- 2026-11-21 · claude-setup · T0.2 · Claimed.
- 2026-11-21 · claude-setup · T0.2 · Done: pyproject, pytest.ini, src/polero, folder layout, .gitattributes (CSV byte-identical), checksum test. Full suite: 4 passed. Added docs/TEAM.md (3 tracks) and a Track column.
