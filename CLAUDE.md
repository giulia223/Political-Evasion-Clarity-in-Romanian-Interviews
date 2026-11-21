# CLAUDE.md: PolERo hackathon project

This project is our team's entry for the Nitro NLP hackathon task **PolERo: Political Evasion & Clarity in Romanian Presidential Interviews**. Claude Code agents build it by following `ROADMAP.md`.

## 1. The task, in brief
- **Input:** a Romanian journalist question and the president's answer, plus metadata. Columns: `datapointID, title, date, president, url, context, question_order, interview_question, interview_answer`. `train_data.csv` (3,278 rows) also has `clarity_label` and `evasion_label`. `test_data.csv` has 296 rows and no labels.
- **Subtask 1 (clarity, 3 classes, 50 points):** `Clear Reply`, `Ambivalent`, `Clear Non-Reply`.
- **Subtask 2 (evasion, 9 classes, 50 points):** `Explicit`, `Implicit`, `General`, `Partial/half-answer`, `Dodging`, `Deflection`, `Declining to answer`, `Claims ignorance`, `Clarification`.
- **Taxonomy:** each evasion label maps to exactly one clarity label.
  - Explicit → Clear Reply
  - Implicit, General, Partial/half-answer, Dodging, Deflection → Ambivalent
  - Declining to answer, Claims ignorance, Clarification → Clear Non-Reply
- **Metric:** macro-F1 per subtask. Test labels come from 3 annotators. For evasion, a prediction counts as correct if it matches any annotator. For clarity, it must match the exact majority label.
- **Submission:** the format of `sample_output.csv`, which is `subtaskID,datapointID,answer`. It has 592 rows: the 296 Subtask 1 rows first, then the 296 Subtask 2 rows, in the order of `test_data.csv`. Labels must be the exact strings above. See D-002.
- **Background and evidence:** `docs/RESEARCH.md`. Read it before designing anything.
- **Data files:** the three CSVs are not in git (D-014). Each collaborator copies them to the repo root. If they are missing, stop and ask the user.
- **Environment setup:**
  ```
  python3 -m venv .venv
  .venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r requirements.txt
  ```

## 2. Hard constraints (never break these)
1. **Free compute only.** No paid services, no Hugging Face Jobs, Inference Endpoints or Providers, and no paid LLM APIs. Local CPU, plus free Kaggle or Colab GPUs run by the user (D-001).
2. **No credentials.** Never ask for, read, store or use tokens or API keys (HF, Kaggle, OpenAI and so on). Public models and datasets download without a token. If something seems to need one, stop and ask.
3. **Ask before installing** any package or tool, and before any single download over 2 GB, such as LLM weights. Pin versions in `requirements.txt`. Never touch system Python. Use `.venv/bin/python` and `.venv/bin/pip`.
4. **Never modify** `train_data.csv`, `test_data.csv` or `sample_output.csv`.
5. **No test-set tuning.** Test rows are inputs only. All model selection, thresholds, weights and few-shot example choices come from training folds and OOF. Never put test rows, or OOF rows being evaluated, into few-shot prompts.
6. **Reproducibility.** Seed 42 by default. The fixed folds are `artifacts/folds.csv` (T0.4). Every result must be traceable to a config plus artifacts.
7. **Honest reporting.** Never invent or estimate a metric. Report a number only if code computed it, and say how: OOF over which folds, and which file. If a run failed or was skipped, say so.
8. **Don't copy unlicensed code.** The SG-UniBuc reference repo has no license. Use it for ideas only and write our own implementation.
9. **Never upload or submit** anything to the hackathon platform or anywhere else. The user submits. Pushing code to our own GitHub repo is fine (§7).
10. **No external training data** (D-012). Train only on `train_data.csv`; public pretrained models are fine. Ideas in `docs/IDEAS.md` are parked: never implement them unless the user moves them into the roadmap.
11. **Leaderboard:** the public leaderboard is a sanity check, never the selection criterion (D-011).

## 3. How to work the roadmap
1. **Start of every session:** read `CLAUDE.md`, then `STATUS.md`, the relevant `ROADMAP.md` task, `DECISIONS.md`, and `docs/RESEARCH.md` if you are designing.
2. **Pick a task:** the lowest-numbered `TODO` task whose dependencies are all `DONE`. Skip tasks marked `USER` until the user has answered. Parallel agents must pick tasks that don't edit the same files.
3. **Claim it before working:** in `STATUS.md`, set it to `IN_PROGRESS` with your agent name and the date, and add a log line. Work on one task at a time.
4. **Scope:** do what the task describes. If you find the task needs a change (to scope, dependencies or acceptance), don't silently change it. Propose the change in DECISIONS.md and ask the user. Small fixes to earlier code are fine; note them in the log.
5. **A task is Done only when all of these hold:**
   - every acceptance criterion in the roadmap is met;
   - the task's tests exist and `.venv/bin/python -m pytest -q` passes in full (not just the new tests);
   - `STATUS.md` is updated (status `DONE`, a result summary with numbers and artifact paths, and a log line);
   - any decisions made are in `DECISIONS.md`;
   - new experiment results are appended to `docs/RESULTS.md`.
6. **Blocked:** set the status to `BLOCKED` (or `AWAITING_USER`) with the exact reason and what is needed. Add it to "Waiting on the user" in STATUS.md. Then move to another eligible task.
7. **Ending a session mid-task:** leave it `IN_PROGRESS` with a log line saying what's done, what's next, and any half-finished files.

## 4. Recording decisions (`DECISIONS.md`)
- **Log a decision** for anything that affects results, interfaces or constraints: model or backbone choices, hyperparameter defaults, fold or metric design, normalization, decoding, any deviation from the roadmap or the user's plan, and any rejected approach together with its evidence.
- **Format:** use the template at the top of `DECISIONS.md`. The log is append-only. Supersede an entry instead of editing it.
- **User approval** is needed for anything involving the hard constraints, installs, large downloads, external data, roadmap changes, or the final submission choice. Record it as `Proposed` and ask the user.
- **Open questions** for the user go in the "Open questions" list as `Q-0NN`.

## 5. GPU tasks (marked `GPU` in the roadmap)
There is no GPU locally (16 CPU cores, 30 GB RAM).

**Agents:**
1. Implement and smoke-test on CPU (`--smoke`).
2. Package the code (`scripts/package_src.sh`).
3. Write the exact run instructions: which notebook, config, folds and GPU type, and the expected runtime.
4. Set the task to `AWAITING_USER`.

**The user:** runs the notebook on Kaggle or Colab and unzips the results into `artifacts/<exp_id>/`. An agent then verifies the artifacts (schema plus `polero.evaluate`) and completes the task.

LLM inference (T3.4) also runs this way on Kaggle (D-013). Plan runs to fit the free tiers: Kaggle sessions are 12 hours and roughly 30 GPU-hours a week. Split folds across sessions if needed with `--folds`.

## 6. Code conventions
- **Package and configs:** the code lives in `src/polero/`, installed editable. Experiments are JSON files in `configs/`, named `<backbone>_<variant>.json`. The experiment ID is the config name.
- **CLIs:** `python -m polero.train | polero.predict | polero.evaluate | polero.submission validate`.
- **Notebooks** are thin wrappers that import `polero`. Logic never lives only in a notebook.
- **Artifacts:**
  - `artifacts/<exp_id>/` holds `config.json`, `oof.csv` (`datapointID`, `p_ev_<label>`… for 9 labels, `p_cl_<label>`… for 3), `test_fold{k}.csv`, `fold{k}/metrics.json` and `train.log`.
  - Checkpoints are off by default.
  - `artifacts/` and `.venv/` are never committed.
- **Submissions** go in `submissions/` and must pass `python -m polero.submission validate`. Only T2.8 and T4.1 write the root `submission.csv`.
- **Tests:** pytest in `tests/`. They are fast, CPU-only, and run without network access. Use tiny locally-built models (e.g. a small `BertConfig`) in unit tests. Tests that need a cached tokenizer must skip cleanly when it is not cached.
- **Text:** keep Romanian diacritics. Normalization follows T1.2 and is logged per backbone.
- **Style:** keep code short and readable. Comment only where something is not obvious.

## 7. Git workflow (shared repo; several collaborators and agents)
- **Remote:** `git@github.com:giulia223/Political-Evasion-Clarity-in-Romanian-Interviews.git`, branch `main`.
- **Claiming a task:** run `git pull --rebase` first. Then commit and push the STATUS.md claim right away (`STATUS: claim T2.1`), so two agents never take the same task.
- **Task work:** do it on a branch named `task/<ID>-<slug>` (e.g. `task/T2.1-chunking`).
- **Finishing a task:**
  1. Rebase on `main`.
  2. Make sure the full test suite passes.
  3. Merge into `main` (fast-forward) or open a PR if the user asks for reviews.
  4. Push.

  The commit message starts with the task ID, e.g. `T2.1: sliding-window encoder with question repeated per chunk`.
- **STATUS.md, DECISIONS.md and RESULTS.md conflicts:** keep both sides. These files are append-only or one row per task.
- **Never:** force-push, rewrite `main` history, or commit data CSVs, `.venv/`, `artifacts/`, model weights, notebook outputs with large blobs, or anything containing a secret.

## 8. Skills available in `.claude/skills/`
- `transformers`: loading models and tokenizers, the Trainer, fine-tuning. Ignore its Hub login and upload sections; we never log in or upload.
- `scikit-learn`: splits, metrics, logistic-regression stacking.
- `ml-model-training`: general training hygiene (seeds, early stopping, leakage).
