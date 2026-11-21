# ROADMAP: PolERo (Nitro NLP hackathon)

Rules for working this roadmap are in `CLAUDE.md`. Each team member works through the **whole** roadmap on their own branch `member/<name>` (D-017). Live progress goes in `STATUS.md`, decisions in `DECISIONS.md`, and evidence in `docs/RESEARCH.md`.

**Goal:** maximise macro-F1 on Subtask 1 (clarity, 3 classes) and Subtask 2 (evasion, 9 classes) for the 296 test rows. Each subtask is worth 50 points.

**Core strategy (from the research):**
1. Train a multi-head encoder (RoBERT-large) with sliding-window max-pooling.
2. Ensemble it over 7 folds.
3. Derive clarity from evasion so the two predictions are always consistent.
4. Add a free open-weight LLM (few-shot, reasoning) and combine it with the encoder.

**Metric:** all model selection uses **out-of-fold (OOF) macro-F1** on the fixed 7-fold CV of `train_data.csv` (T0.4). There is no other validation set. Test labels are never available.

**Notation:**
- `GPU` marks a task that needs a free GPU. The agent builds it and smoke-tests it on CPU. The user runs it on Kaggle or Colab (see CLAUDE.md §5).
- `USER` marks a task that needs a user decision or approval first.
- Tests are run with `.venv/bin/python -m pytest -q`, which must pass.

---

## Phase 0: Foundations

### T0.1 Environment and dependencies (DONE: D-015)
- **Description:** Create `requirements.txt` and install the minimal stack into `.venv`, which is Python 3.14 and already has scikit-learn and pandas. The stack is CPU torch, transformers, sentencepiece, accelerate and pytest, with pinned versions. Check that wheels exist for Python 3.14. If any are missing, propose a fallback interpreter such as 3.12 and log it as a decision.
- **Depends on:** none.
- **Acceptance:**
  - The user approved the package list before installation.
  - `requirements.txt` pins exact versions.
  - `import torch, transformers` works.
  - `torch.__version__` and `transformers.__version__` are recorded in STATUS.md.
- **Tests:** `tests/test_env.py` imports every package listed in requirements.txt and asserts the versions match.

### T0.2 Repository skeleton
- **Description:** The git repository already exists (D-014). Create this layout inside it:
  ```
  src/polero/      # package: labels, data, folds, metrics, chunking, model, train, predict, ensemble, submission
  configs/         # JSON experiment configs
  tests/
  notebooks/       # GPU runner notebooks (thin wrappers that import src/polero)
  artifacts/       # (ignored) OOF/test probabilities, checkpoints, logs: artifacts/<exp_id>/
  submissions/     # validated submission CSVs
  docs/
  ```
  Also add `pyproject.toml` (installed as an editable package) and `pytest.ini`. `.gitignore` already exists; extend it if needed.
- **Depends on:** T0.1.
- **Acceptance:**
  - `.venv/bin/pip install -e .` works.
  - `python -c "import polero"` works.
  - The data CSVs stay unmodified at the repo root.
- **Tests:** `tests/test_package.py` imports `polero`. The sha256 of the three CSVs is recorded in `tests/data_checksums.json` and asserted unchanged.

### T0.3 Label taxonomy and submission writer/validator
- **Description:** In `polero/labels.py`, define `EVASION` (9 exact strings), `CLARITY` (3), and `CLARITY_OF` (the evasion→clarity map) with index mappings.

  In `polero/submission.py`:
  - `write_submission(test_ids, evasion_labels, path, clarity_labels=None)` writes the format of `sample_output.csv`: `subtaskID,datapointID,answer`, Subtask 1 rows first, then Subtask 2, in the order of `test_data.csv`, 592 rows. When `clarity_labels` is None, clarity is derived from evasion.
  - `validate_submission(path)` checks columns, row count, ID set and order against `test_data.csv`, the exact label strings, and evasion/clarity consistency (a warning if inconsistent, never silently fixed).
  - A CLI: `python -m polero.submission validate <csv>`.
- **Depends on:** T0.2.
- **Acceptance:**
  - `sample_output.csv` passes validation, with the consistency check reporting it (all Explicit means consistent).
  - Corrupted variants fail with clear messages.
- **Tests:** `tests/test_labels.py` and `tests/test_submission.py`:
  - the mapping covers all 9 labels and matches every train row's `(evasion, clarity)` pair;
  - a round-trip write then validate passes;
  - each of these is rejected: a missing row, a wrong ID, a typo label (e.g. `"Partial/half answer"`), a wrong column order, a duplicate row.

### T0.4 Data loading and fixed CV folds
- **Description:**
  - `polero/data.py` loads train and test with a fixed dtype schema.
  - `polero/folds.py` builds **7-fold StratifiedKFold on `evasion_label`, seed 42**, saved to `artifacts/folds.csv` (`datapointID,fold`). Every experiment uses this file.
  - Stratify on evasion, not clarity, so each fold gets about 5 Clarification examples. Log this as a decision.
  - Exact-duplicate Q/A pairs must land in the same fold.
- **Depends on:** T0.2.
- **Acceptance:**
  - Folds are deterministic.
  - Every fold contains every evasion class.
  - The 6 duplicate pairs never straddle folds.
- **Tests:** `tests/test_folds.py`:
  - rebuilding gives identical output;
  - per-fold class counts are within ±1 of the expected proportion;
  - no `datapointID` is in two folds;
  - duplicates share a fold.

### T0.5 Metrics and OOF evaluation
- **Description:** `polero/metrics.py` provides:
  - `evasion_f1(y, p)`;
  - `clarity_f1(y, p)`;
  - `mapped_clarity_f1(y_evasion, p_evasion)`;
  - per-class report and confusion matrix;
  - `evaluate_oof(probs_df)`, which takes a dataframe of `datapointID` plus 9 evasion probability columns (plus optional 3 clarity columns) and returns a metrics dict, a markdown table and confusion matrices.

  `python -m polero.evaluate artifacts/<exp>/oof.csv` appends a row to `docs/RESULTS.md`.
- **Depends on:** T0.3, T0.4.
- **Acceptance:**
  - Results match `sklearn.metrics.f1_score(average="macro")` exactly.
  - Classes that are never predicted count as F1 0, not dropped.
- **Tests:** `tests/test_metrics.py` uses hand-computed toy cases, including a class that is never predicted, plus a check that the mapped clarity metric equals clarity F1 computed after an explicit mapping.

---

## Phase 1: Data understanding

### T1.1 EDA with tokenizers
- **Description:** Write `docs/EDA.md` covering:
  - token-length distributions of Q, A and Q+A under the candidate tokenizers: `readerbench/RoBERT-large`, `dumitrescustefan/bert-base-romanian-cased-v1`, `FacebookAI/xlm-roberta-large`;
  - the fraction of rows above 256 and 512 tokens, and per-label length;
  - **diacritics handling**: whether each vocabulary contains comma-below `ș ț` or cedilla `ş ţ`, the UNK rate and subword fertility under each form;
  - whether each tokenizer is fast, which `return_overflowing_tokens` needs;
  - duplicate and near-duplicate Q/A pairs between train and test;
  - neighbour-turn structure (per `url`, the gaps in `question_order`; how many test rows have labeled neighbours in train);
  - label rates per president and per context.

  Tokenizer downloads are public and small. Do not download model weights here.
- **Depends on:** T0.1, T0.4.
- **Acceptance:** `docs/EDA.md` exists with tables and the script that generated it (`scripts/eda.py`, reproducible), and ends with explicit recommendations for max length, stride and normalization.
- **Tests:** `tests/test_eda_smoke.py` runs the EDA functions on 50 rows without error.

### T1.2 Text normalization
- **Description:** `polero/text.py` provides `normalize(text, mode)`, which handles whitespace, quotes and dashes, and maps diacritics to the form the chosen tokenizer prefers (cedilla↔comma, based on T1.1). It must never strip diacritics unless T1.1 shows the tokenizer is diacritic-free. Log the chosen mode per backbone in DECISIONS.md.
- **Depends on:** T1.1.
- **Acceptance:** for each candidate tokenizer, the chosen mode gives an UNK rate ≤ the raw text's UNK rate, and subword fertility no worse than raw.
- **Tests:** `tests/test_text.py`: idempotence (`normalize(normalize(x)) == normalize(x)`), exact mapping tables, and an empty-string case.

---

## Phase 2: Core model (multi-head chunked encoder)

### T2.1 Sliding-window encoding of Q+A
- **Description:** `polero/chunking.py` encodes each example as one or more chunks:
  - the question segment is repeated in every chunk and capped (default 128 tokens, truncated from the start so the end of the question survives);
  - the answer is split with `max_length=512` and `stride=128` (configurable) using the fast tokenizer's pair encoding (`truncation="only_second"`, `return_overflowing_tokens=True`);
  - it returns `input_ids`, `attention_mask` and `example_index` per chunk;
  - an optional `max_chunks` cap (default 8, keeping first, last and evenly spaced chunks) bounds memory.

  This differs from the paper, which chunks the concatenated sequence so the question only appears in chunk 1. Log the choice in DECISIONS.md. An ablation in T3.2 may revisit it.
- **Depends on:** T1.2.
- **Acceptance:**
  - every answer token appears in at least one chunk (unless `max_chunks` truncates, which is reported);
  - the question appears in every chunk;
  - short examples produce exactly one chunk with no padding waste (dynamic padding happens in the collator).
- **Tests:** `tests/test_chunking.py`, run with a local tiny tokenizer or the real fast tokenizer from cache:
  - an answer token coverage check;
  - a 3,000-word answer gives the expected chunk count;
  - a 1-word answer gives 1 chunk;
  - the question is present in every chunk;
  - `max_chunks` behaviour.

### T2.2 Multi-head model
- **Description:** `polero/model.py` defines `MultiHeadChunkClassifier(backbone_name)`:
  - an `AutoModel` encoder;
  - the position-0 hidden state per chunk;
  - **element-wise max-pool over the chunks of each example** (mean and attention pooling selectable for ablation);
  - dropout 0.1;
  - two linear heads (clarity 3, evasion 9).

  The loss is `w_c·CE(clarity) + w_e·CE(evasion)` with defaults 1 and 1. It supports optional class weights, focal loss, and label smoothing. The forward pass takes the flattened chunks plus `example_index` and returns per-example logits. Gradient checkpointing is toggleable.
- **Depends on:** T2.1.
- **Acceptance:**
  - pooling is correct for variable chunk counts in a batch;
  - logits have shape `[n_examples, 3]` and `[n_examples, 9]`;
  - the loss decreases on a tiny overfit test.
- **Tests:** `tests/test_model.py` builds a tiny random `BertConfig` locally (no download) and checks:
  - output shapes for a batch with 1, 3 and 1 chunks;
  - max-pool equality against a manual computation;
  - the model overfits 8 examples to loss < 0.1 within 100 steps on CPU;
  - gradients reach the encoder.

### T2.3 Training loop with CV
- **Description:** `polero/train.py` provides the CLI `python -m polero.train --config configs/<exp>.json [--folds 0,1] [--smoke]`.
  - **Optimizer and schedule:** AdamW with weight decay 0.01, lr from the config (default 2e-5 for RoBERT-large, 1e-5 for XLM-R-large), 10% warmup with linear decay, gradient clip 1.0.
  - **Precision:** AMP fp16 on T4, bf16 where supported.
  - **Batching:** batches are built by **chunk budget**, not example count, so long answers don't cause out-of-memory errors. Gradient accumulation reaches an effective batch of about 16.
  - **Epochs and early stopping:** max 15 epochs. Early stopping (patience 3) on validation **evasion macro-F1 + mapped clarity macro-F1**, evaluated each epoch on the held-out fold.
  - **Seeding:** seed 42 for Python, NumPy and torch, plus a per-fold offset.
  - **Outputs** in `artifacts/<exp_id>/`:
    - the config copy;
    - `fold{k}/metrics.json`;
    - `oof.csv` (OOF probabilities for both heads);
    - `test_fold{k}.csv` (test probabilities from each fold's best checkpoint);
    - the training log.
  - Checkpoints are optional and off by default; they are large.
  - **`--smoke`:** 32 rows, 2 steps, tiny model. It must run on CPU in under 2 minutes.
- **Depends on:** T0.5, T2.2.
- **Acceptance:**
  - the smoke run finishes on CPU and produces correctly-shaped `oof.csv` and test files;
  - two smoke runs with the same seed produce identical metrics;
  - a resumed run (`--folds` subset) merges into the same `oof.csv`.
- **Tests:** `tests/test_train_smoke.py` runs `--smoke` end to end. `tests/test_early_stopping.py` unit-tests the patience logic.

### T2.4 Prediction, fold ensembling and consistent decoding
- **Description:** `polero/predict.py` and `polero/ensemble.py`:
  - average fold test probabilities (and probabilities across experiments with optional weights);
  - implement and compare on OOF the following **decoders**:
    - (a) argmax evasion, then map up;
    - (b) clarity = argmax of evasion probabilities summed per clarity group, then evasion = argmax within that group (consistent by construction);
    - (c) the clarity head blended with (b);
  - **per-class logit bias tuning** for macro-F1, fitted on OOF with nested CV to avoid overfitting.

  The final pick must keep clarity consistent with evasion. If the best clarity decoder were inconsistent, log a decision with numbers and ask the user.

  `python -m polero.predict --exp <id> [--exp <id2> --weights ...] --decoder b --out submissions/<name>.csv` writes the file and runs the validator.
- **Depends on:** T2.3, T0.3.
- **Acceptance:**
  - every decoder is evaluated on OOF and the table is written to `docs/RESULTS.md`;
  - the produced submission passes `validate_submission`;
  - the bias tuning is reported with nested-CV OOF, not in-sample.
- **Tests:** `tests/test_ensemble.py`:
  - averaging is correct;
  - decoder (b) always gives a consistent pair;
  - bias tuning on a synthetic imbalanced problem raises macro-F1;
  - an end-to-end run on smoke artifacts produces a valid submission.

### T2.5 GPU runner notebook `GPU`
- **Description:** `notebooks/kaggle_train.ipynb` (it also works on Colab) is a thin notebook that:
  - installs pinned requirements;
  - loads the repo code (uploaded as a Kaggle Dataset `polero-src` zip built by `scripts/package_src.sh`) and the three data CSVs;
  - runs `polero.train` for the given config and folds;
  - zips `artifacts/<exp_id>` for download.

  Write `docs/GPU_RUNBOOK.md` with step-by-step instructions for the user: upload, select GPU T4×2 or P100, run, download, and unzip into `artifacts/`.
- **Depends on:** T2.3.
- **Acceptance:**
  - the notebook runs top to bottom on CPU with `SMOKE=True`;
  - the runbook is clear enough for a human to follow without reading code;
  - the packaged zip excludes `.venv` and `artifacts`.
- **Tests:** `tests/test_package_src.py` checks the zip contents. Manually run the notebook in smoke mode with `jupyter nbconvert --execute` if jupyter is installed (ask first), otherwise run the extracted script.

### T2.6 Backbone screening `GPU`
- **Description:** On folds 0–2 only, compare:
  - `readerbench/RoBERT-large`;
  - `FacebookAI/xlm-roberta-large`;
  - `dumitrescustefan/bert-base-romanian-cased-v1`, as a fast base-size reference;
  - optionally `readerbench/RoBERT-base`.

  Use multi-head with default settings. Report OOF on those folds: E, M, C and per-class F1.
- **Depends on:** T2.5.
- **Acceptance:**
  - a results table in `docs/RESULTS.md`;
  - a backbone decision logged in DECISIONS.md;
  - the research prior is RoBERT-large; deviating from it requires a gain of at least 0.02 evasion F1.
- **Tests:** `artifacts/<exp>/oof.csv` for each run passes the schema check in `polero.evaluate`.

### T2.7 Full 7-fold run of the chosen backbone `GPU`
- **Description:** Train all 7 folds of the chosen backbone and produce `oof.csv` and 7 test probability files.
- **Depends on:** T2.6.
- **Acceptance:**
  - full OOF metrics are recorded;
  - **target OOF evasion macro-F1 ≥ 0.50 and mapped clarity ≥ 0.70** (the paper reports test .545/.763 for RoBERT-large multi-head, and CV is usually a little lower);
  - if the target is missed, record the analysis in STATUS.md before moving on.
- **Tests:** schema check; per-fold metric variance is reported.

### T2.8 First real submission
- **Description:** Use `polero.predict` with the T2.7 experiment and the best decoder from T2.4 to write `submissions/sub_v1_<exp>.csv`, and copy it to `submission.csv` at the repo root. The user uploads it and records the public-leaderboard score in STATUS.md (D-011).
- **Depends on:** T2.4, T2.7.
- **Acceptance:**
  - the validator passes;
  - the predicted label distribution on test is reported next to the train distribution, flagging any class predicted 0 times;
  - the user is told that the file is ready (agents do not upload).
- **Tests:** `python -m polero.submission validate submission.csv`.

---

## Phase 3: Improvements

Each improvement is kept only if it beats the current best on OOF. The gain must be at least 0.01 on the target metric, or else beat it with a consistent sign across at least 5 of 7 folds. Log every result, including failures, in `docs/RESULTS.md`.

### T3.1 Loss and imbalance ablation `GPU`
- **Description:** Using the current best setup, compare:
  - plain CE (the default; the research shows class weights hurt);
  - sqrt-inverse-frequency class weights;
  - focal γ=2;
  - label smoothing 0.1;
  - clarity loss weight `w_c ∈ {0.5, 1}`.

  Also compare post-hoc logit bias tuning (T2.4) against training-time reweighting.
- **Depends on:** T2.7.
- **Acceptance:** a results table, and the winner adopted via a DECISIONS.md entry.
- **Tests:** unit tests for each loss variant in `tests/test_model.py` (finite values, correct reduction).

### T3.2 Context and metadata features `GPU`
- **Description:** Test whether neighbouring turns help. These are reconstructed by `url` and sorted by `question_order`, covering train and test rows together, since both are inputs.
  - **Variants:**
    - (a) the previous question prepended as context (capped at 64 tokens);
    - (b) previous and next questions;
    - (c) metadata prefix tokens: president, context type, year bucket;
    - (d) the question-in-every-chunk choice versus the paper's chunk-once input.
  - Never use the labels of neighbouring rows inside CV unless those rows are in the training fold. Prefer not to use neighbour labels at all.
- **Depends on:** T2.7.
- **Acceptance:** OOF comparison against the T2.7 baseline using the same folds; adopted only by the Phase 3 rule.
- **Tests:** `tests/test_context.py`:
  - neighbour retrieval is correct on a hand-built 5-row interview with gaps;
  - there is no label leakage (a test proves validation-fold labels are never read).

### T3.3 (dropped) Cross-lingual joint training
- **Status:** DROPPED by the user (D-012). No external training data is used. The idea is parked in `docs/IDEAS.md` (I-001). **Do not implement it.**

### T3.4 Open-weight LLM few-shot classifier `GPU`
- **Description:** Implement `polero/llm.py`. Ask the user before installing any new package.
  - **Backend:** run on a **free Kaggle GPU** (D-013), using llama.cpp (`llama-server`) or vLLM inside the Kaggle notebook. The endpoint must be **localhost only**.
  - **Runs:** `notebooks/kaggle_llm.ipynb` runs the prompts and writes the cache and predictions. Package it the same way as T2.5.
  - **Local checks:** use the fake backend for development and tests.
  - **Model:** Qwen3.6-35B-A3B in Q4, the best open model on Romanian evasion (.682). It must fit Kaggle T4×2 (2×16 GB). If it doesn't fit or is too slow, fall back to a smaller open instruct model and log that as a decision.
  - The weights download inside Kaggle, not to this machine.
  - **Prompt** (written in Romanian, stored in `prompts/`):
    - the task and all 9 definitions plus the 3 boundary distinctions from the task brief;
    - few-shot examples drawn **only from training folds**: at least 1 per class, with extra examples for the Dodging/Deflection and Implicit/General pairs;
    - reasoning allowed, then JSON output `{"evasion": "<label>"}` parsed strictly.
  - A hierarchical variant in the style of TeleAI: a coarse pass, then a fine pass with label-specific guidelines.
  - **Results caching:** cache by hash of prompt and model, in `artifacts/llm/<run>/cache.jsonl`, so reruns cost nothing.
  - **Evaluation:** first on a stratified 180-row OOF sample (20 per class, from a held-out fold so the few-shot pool excludes it), then on all 296 test rows.
- **Depends on:** T0.4 and T0.5. The Kaggle packaging also needs T2.5. Prompting and parsing code can be built in parallel with Phase 2.
- **Acceptance:**
  - parse failure rate < 1% (with a retry, then a fallback label of "Explicit" that is logged);
  - sample-level evasion macro-F1 is reported with a bootstrap 95% CI;
  - throughput and total runtime are reported;
  - no paid API is ever called.
- **Tests:** `tests/test_llm.py` uses a fake backend to check prompt construction, strict parsing (malformed JSON, near-miss labels such as lowercase or a missing slash rejected or normalized by an explicit table), cache hits, and that few-shot examples exclude the evaluated rows.

### T3.5 Encoder-guided LLM reranking
- **Description:** For each item, give the LLM the encoder ensemble's top-k labels (k=3), optionally with their probabilities, and ask it to choose among them. Route high-confidence encoder predictions (max probability ≥ τ, with τ tuned on OOF) around the LLM to save compute. This is the 2nd-place SemEval evasion approach.
- **Depends on:** T2.7, T3.4.
- **Acceptance:** compared on the same OOF sample as T3.4; reports the share of items routed to the LLM.
- **Tests:** routing logic unit tests; the candidate list always contains the encoder argmax.

### T3.6 Final ensemble and stacking
- **Description:** Combine the encoder experiments (seeds and backbones) and the LLM outputs:
  - weighted probability averaging, with the LLM label as a one-hot or a soft vote;
  - weights or a logistic-regression stacker fitted on OOF with nested CV;
  - then the T2.4 decoder and bias tuning.
- **Depends on:** T2.7, plus any of T3.1–T3.5 that are DONE.
- **Acceptance:**
  - the chosen ensemble beats the best single component on nested-CV OOF;
  - the evasion/clarity consistency is preserved.
- **Tests:** `tests/test_stacking.py`: nested-CV prevents in-sample fitting (with a synthetic leakage detector), and weights sum to 1.

### T3.7 Seed ensemble `GPU`
- **Description:** Rerun the best encoder configuration with seeds 7 and 123. The fold split stays fixed and only the training seed changes. Average with seed 42.
- **Depends on:** T2.7 (preferably after T3.1 and T3.2).
- **Acceptance:** OOF of the seed-averaged model is at least as good as single-seed; the variance across seeds is reported.
- **Tests:** schema checks on the artifacts.

---

## Phase 4: Finalisation

### T4.1 Final submission
- **Description:** Choose the final system by OOF, regenerate the submission from saved artifacts with one command (`scripts/make_final_submission.sh`), validate it, and copy it to `submission.csv`.
- **Depends on:** T2.8, plus any Phase 3 tasks kept.
- **Acceptance:**
  - the command reproduces a byte-identical file;
  - the validator passes;
  - STATUS.md "Current best" lists the exact experiments, weights, decoder and OOF metrics.
- **Tests:** `tests/test_final_repro.py` regenerates the file into a temp path and diffs it with `submission.csv`.

### T4.2 Write-up
- **Description:** Write `docs/REPORT.md` covering the approach, the OOF results table, ablations (what helped and what didn't), compute used (all free), and known limitations. Keep it short enough for hackathon slides.
- **Depends on:** T4.1.
- **Acceptance:** every number in the report traces to `docs/RESULTS.md`.
- **Tests:** none (documentation). A reviewer checks the numbers against RESULTS.md.

---

## Dependency overview
```
T0.1 → T0.2 → T0.3 ─┐
             └ T0.4 ─┴→ T0.5 ──────────────┐
T0.1+T0.4 → T1.1 → T1.2 → T2.1 → T2.2 → T2.3 → T2.4 → T2.8 → T4.1 → T4.2
                                         └ T2.5 → T2.6 → T2.7 ┬→ T3.1, T3.2, T3.7
T0.4+T0.5 (+T2.5 for packaging) → T3.4 (parallel track) ──────┴→ T3.5 → T3.6 → T4.1
(T3.3 dropped: see docs/IDEAS.md)
```
**Critical path to a first real submission:** T0.1 → T0.2 → T0.3/T0.4 → T0.5 → T1.1 → T1.2 → T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T2.7 → T2.8.
