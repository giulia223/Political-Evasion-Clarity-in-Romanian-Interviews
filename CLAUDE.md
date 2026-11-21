# CLAUDE.md: PolERo hackathon project

This project is our team's entry for the Nitro NLP hackathon task **PolERo: Political Evasion & Clarity in Romanian Presidential Interviews**. Claude Code agents build it by following `ROADMAP.md`.

**Team setup (D-017):** each of the 3 team members builds **their own complete version** of the whole roadmap, on their own git branch `member/<name>`, and trains their own models. Everyone follows the same instructions, roadmap and rules. The members don't split tasks between them. `main` holds only the shared instructions and the common skeleton (T0.1–T0.2). In this file, "the user" means the member who owns the branch you are working on.

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
- **Data files:** the three CSVs are committed at the repo root (D-016). Never modify them.
- **Environment setup:**
  ```
  python3 -m venv .venv
  .venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install -e .
  ```

## 2. Hard constraints (never break these)
1. **Free compute only.** No paid services, no Hugging Face Jobs, Inference Endpoints or Providers, and no paid LLM APIs. Allowed: the member's own machine (CPU, or a local GPU if they have one), plus free Kaggle or Colab GPUs run by the member (D-001, D-017).
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
1. **Start of every session:**
   - check you are on the user's own `member/<name>` branch. If you are on `main` or the branch doesn't exist yet, follow §7 first;
   - read `CLAUDE.md`, then `STATUS.md`, the relevant `ROADMAP.md` task, `DECISIONS.md`, and `docs/RESEARCH.md` if you are designing;
   - `STATUS.md` and `DECISIONS.md` on a member branch belong to that member's version only.
2. **Pick a task:** the lowest-numbered `TODO` task whose dependencies are all `DONE`. Skip tasks marked `USER` until the user has answered. If several agents work on the same branch at once, they must pick tasks that don't edit the same files.
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
Each member chooses where their GPU work runs. Check with `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` and ask the member if unsure.

**Local GPU** (if the member has one):
- install the CUDA build of torch with the member's approval, and record it in STATUS.md;
- run `python -m polero.train` directly;
- the task completes without a hand-off.

**No local GPU:** the work runs on a free Kaggle (or Colab) GPU under the member's own account.
- **Agents:**
  1. Implement and smoke-test on CPU (`--smoke`).
  2. Package the code (`scripts/package_src.sh`).
  3. Write the exact run instructions: which notebook, config, folds and GPU type, and the expected runtime.
  4. Set the task to `AWAITING_USER`.
- **The member:** runs the notebook and unzips the results into `artifacts/<exp_id>/`. An agent then verifies the artifacts (schema plus `polero.evaluate`) and completes the task.

LLM inference (T3.4) runs on Kaggle (D-013) unless the member has a local GPU big enough for it. Plan Kaggle runs to fit the free tier: Kaggle sessions are 12 hours and roughly 30 GPU-hours a week. Split folds across sessions if needed with `--folds`.

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

## 7. Git workflow (one branch per member; D-017)
- **Remote:** `git@github.com:giulia223/Political-Evasion-Clarity-in-Romanian-Interviews.git`.
- **`main`:** shared instructions and skeleton only: `CLAUDE.md`, `ROADMAP.md`, `docs/RESEARCH.md`, `docs/IDEAS.md`, the setup, and the template `STATUS.md` and `DECISIONS.md`. Nobody does roadmap work on `main`.
- **Creating your branch:** ask the member for their name if you don't know it. Then:
  ```
  git switch main
  git pull
  git switch -c member/<name>
  git push -u origin member/<name>
  ```
  Then add a log line in STATUS.md saying the version was started.
- **Daily work:** commit to `member/<name>`, or use short-lived task branches off it that you merge back. Push often, since the remote is the backup. The commit message starts with the task ID, e.g. `T2.1: sliding-window encoder with question repeated per chunk`.
- **Changing shared instructions** (a roadmap fix, a CLAUDE.md rule, research notes):
  1. Make the change on a branch off `main` (`docs/<slug>`).
  2. Open a PR or merge it after the member approves.
  3. Each member pulls it into their own branch with `git merge main`.

  Never merge a `member/*` branch into `main` or into another member's branch.
- **Reading other versions** is fine (`git fetch`, `git diff main..origin/member/<other>`). Never commit to someone else's branch.
- **Never:** force-push to `main` or someone else's branch, rewrite shared history, or commit `.venv/`, `artifacts/`, model weights, notebook outputs with large blobs, or anything containing a secret.

## 8. Skills available in `.claude/skills/`
- `transformers`: loading models and tokenizers, the Trainer, fine-tuning. Ignore its Hub login and upload sections; we never log in or upload.
- `scikit-learn`: splits, metrics, logistic-regression stacking.
- `ml-model-training`: general training hygiene (seeds, early stopping, leakage).
