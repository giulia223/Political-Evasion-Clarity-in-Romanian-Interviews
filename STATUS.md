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
| T0.5 | Metrics and OOF evaluation | DONE | claude-raluca | 2026-11-21 | `polero.metrics`, `python -m polero.evaluate <oof.csv>` → oof_metrics.json + row in docs/RESULTS.md. 38 tests pass |
| T1.1 | EDA with tokenizers | DONE | claude-raluca | 2026-11-21 | docs/EDA.md. 96% of pairs ≤512 tokens. Recommended: question cap 192, overlap 128, max_chunks 8, keep comma diacritics, map –→- and ť→ț |
| T1.2 | Text normalization | DONE | claude-raluca | 2026-11-21 | `polero.text.normalize`, mode `clean` for all backbones. ro-BERT UNK 0.041% → 0.001%. Diacritics kept |
| T2.1 | Sliding-window encoding | DONE | claude-raluca | 2026-11-21 | `polero.chunking`. RoBERT-large on train+test: 3,767 chunks, 124 rows multi-chunk, 2 capped at 8, 59 questions truncated |
| T2.2 | Multi-head model | DONE | claude-raluca | 2026-11-21 | `polero.model.MultiHeadChunkClassifier`: CLS per chunk → max/mean/attention pool → dropout → heads 3+9. CE (default), weighted, focal, label smoothing |
| T2.3 | Training loop with CV | DONE | claude-raluca | 2026-11-21 | `python -m polero.train --config configs/robert-large_mh.json [--folds] [--smoke]`. Smoke: 7 folds in ~6 s on CPU. GPU/AMP path not yet run (T2.5/T2.6) |
| T2.4 | Prediction, ensembling, decoding | DONE | claude-raluca | 2026-11-21 | `python -m polero.predict --exp <id> [--decoder auto]`: decoders a/b/c and nested bias tuning on OOF, validated submission. Real decoder table pending first GPU run |
| T2.5 | GPU runner notebook and runbook | DONE | claude-raluca | 2026-11-21 | notebooks/kaggle_train.ipynb + scripts/kaggle_run.py (folds split across GPUs) + scripts/package_src.sh + docs/GPU_RUNBOOK.md. Smoke-tested from the zip on CPU. First real Kaggle run happens in T2.6 |
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
- 2026-11-21 · claude-raluca · T0.5 · Done. polero.metrics: evasion_f1, clarity_f1, mapped_clarity_f1, per_class_report (with a predicted-count column), confusion, evaluate_oof. evaluate_oof reports evasion, mapped clarity (decoder a), grouped clarity (decoder b) and the clarity head. It validates columns, probability sums, duplicate and unknown IDs. Macro-F1 = sklearn default (union of classes, never-predicted = 0; D-020). polero.evaluate CLI. docs/RESULTS.md created (header only, no results yet). Full suite: 38 passed.
- 2026-11-21 · claude-raluca · T1.1 · Claimed.
- 2026-11-21 · claude-raluca · T1.1 · Done. src/polero/eda.py + scripts/eda.py → docs/EDA.md (recommendations kept in docs/eda_recommendations.md). Tokenizers downloaded (RoBERT-large, ro-BERT-base-cased, XLM-R-large; no weights). Pairs over 512 tokens: 4% (RoBERT/XLM-R), 2% (ro-BERT). Raw comma diacritics: 0 UNK for RoBERT and XLM-R; the cedilla form gives 9.3% UNK for ro-BERT, so diacritics are kept (D-021). Small fix to T0.4: added polero.data.event_dates, because `date` mixes YYYY-MM-DD and DD.MM.YYYY (1,402 train rows). Full suite: 41 passed.
- 2026-11-21 · claude-raluca · T1.2 · Claimed.
- 2026-11-21 · claude-raluca · T1.2 · Done. polero.text: normalize(text, mode) with raw / clean / clean+quotes, BACKBONE_MODE and mode_for(). clean maps dashes to '-', ť to ț, collapses all whitespace, and never touches diacritics. Measured in docs/EDA.md §8: clean for every backbone (D-022). Full suite: 46 passed.
- 2026-11-21 · claude-raluca · T2.1 · Claimed.
- 2026-11-21 · claude-raluca · T2.1 · Done. polero.chunking: ChunkConfig (512 / q_cap 192 / overlap 128 / max_chunks 8), PairTemplate (special tokens learned from the tokenizer), encode_pair, encode_examples (applies T1.2 normalization), collate_chunks (per-batch padding plus example_index). Built manually rather than with return_overflowing_tokens (D-023). Matches tokenizer pair encoding exactly on single-chunk rows for all 3 tokenizers. Full-data counts match the EDA. Full suite: 57 passed.
- 2026-11-21 · claude-raluca · T2.2 · Claimed.
- 2026-11-21 · claude-raluca · T2.2 · Done. polero.model: MultiHeadChunkClassifier (from_backbone loads AutoModel without the pooler, optional activation checkpointing), pool_chunks (max/mean/attention via scatter, variable chunk counts), LossConfig (ce/focal, label smoothing, per-head weights w_c/w_e, per-class weights), class_weights (inverse frequency ** power), focal_loss. Tested with a tiny random BERT: shapes, pooling equals the manual computation, overfits 8 examples to loss < 0.1, gradients reach the encoder, every loss variant is finite, end to end with collate_chunks. Full suite: 74 passed.
- 2026-11-21 · claude-raluca · T2.3 · Claimed.
- 2026-11-21 · claude-raluca · T2.3 · Done. polero.train: TrainConfig (JSON, unknown keys rejected), AdamW with no decay on bias/LayerNorm, linear warmup 10% and decay, clip 1.0, AMP (auto: bf16 or fp16 with GradScaler on CUDA, fp32 on CPU), chunk-budget micro-batches with gradient accumulation to batch_examples, early stopping (patience 3) on val evasion F1 + mapped clarity F1, best state kept in memory. Per-fold oof/test/metrics files; oof.csv merged across runs. polero.smoke: offline deterministic tiny tokenizer and BERT. configs/robert-large_mh.json is the default config. The GPU precision path is untested until the first Kaggle run (T2.5/T2.6). Full suite: 83 passed (D-024).
- 2026-11-21 · claude-raluca · T2.4 · Claimed.
- 2026-11-21 · claude-raluca · T2.4 · Done. polero.ensemble: blend (weighted, aligned on datapointID), load_test (fold average), decode a/b/c (consistent by construction), a fast sklearn-equivalent macro-F1, per-class log-prob bias tuning (coordinate ascent, Explicit fixed at 0), alpha tuning for c, nested_predict (parameters from the other folds only), compare_decoders. polero.predict CLI: decoder table plus test-vs-train prediction distribution, write_submission and validate, --log to RESULTS.md, --decoder auto picks the best nested row. Verified end to end on a full-size smoke experiment. Real numbers come after T2.7. Full suite: 92 passed (D-025).
- 2026-11-21 · claude-raluca · T2.5 · Claimed.
- 2026-11-21 · claude-raluca · T2.5 · Done. package_src.sh: zip of the working-tree code, configs, scripts, data and artifacts/folds.csv, with VERSION (commit, +dirty flag). kaggle_run.py (stdlib only): pinned or minimal install keeping CUDA torch, fold-fingerprint check, one training process per GPU (--no-merge), then merge, evaluate and zip. polero.train gained --no-merge and per-fold-group log files. The notebook is a thin wrapper and auto-detects the dataset zip or an unpacked directory. EXPECTED_FINGERPRINT moved into polero.folds. Runtime in the runbook is an estimate until measured. Full suite: 95 passed (D-026).
