# Team Plan (3 people)

Each person owns one **track**: a set of roadmap tasks that mostly touch different files. Each person runs their own Claude Code session in their own clone of the repo. That agent works **only on its owner's track**. Edit the names below once, and the agents read them from here.

| Track | Person | Focus | Main files |
|---|---|---|---|
| **A: Data, evaluation, submission** | _name_ | correctness of everything that turns probabilities into a score or a file | `labels.py`, `submission.py`, `metrics.py`, `evaluate.py`, `text.py`, `predict.py`, `ensemble.py`, `docs/EDA.md`, `docs/RESULTS.md` |
| **B: Encoder model** | _name_ | the multi-head chunked encoder and its training | `data.py`, `folds.py`, `chunking.py`, `model.py`, `train.py`, `configs/` |
| **C: GPU runs and LLM** | _name_ | Kaggle runs, backbone experiments, the LLM track, final write-up | `notebooks/`, `scripts/package_src.sh`, `docs/GPU_RUNBOOK.md`, `llm.py`, `prompts/`, `docs/REPORT.md` |

## Task assignment
| Phase | Track A | Track B | Track C |
|---|---|---|---|
| 0 | T0.3 labels/submission · T0.5 metrics | T0.4 folds | (T0.1, T0.2 done) · start T3.4 prompt + parser with the fake backend |
| 1 | T1.1 EDA · T1.2 normalization | (waits on T1.2: review T0.3/T0.5) | T3.4 continued |
| 2 | T2.4 predict/ensemble/decoders | T2.1 chunking · T2.2 model · T2.3 training loop | T2.5 Kaggle notebook + runbook · T2.6 backbone screen · T2.7 full 7-fold run |
| 2→3 | T2.8 first submission | T3.1 loss ablation · T3.2 context features | T3.4 LLM run on Kaggle · T3.5 LLM rerank |
| 3–4 | T3.6 ensemble/stacking · T4.1 final submission | T3.7 seed ensemble | T4.2 write-up |

The load is balanced by effort, not by task count. Track C has fewer coding tasks but carries all the Kaggle runs and the LLM track, which is the largest single task.

## Working rules
1. **Before every session:** `git pull --rebase`. Then tell your agent your track: "You are on Track A (see docs/TEAM.md). Continue the roadmap."
2. **Picking work:** the agent picks the lowest-numbered eligible `TODO` task **in its track** (CLAUDE.md §3). If its track is blocked, it may take another track's task only after asking its owner and the other track's owner. Take it only if it is unclaimed in STATUS.md.
3. **Interfaces first:** where tracks meet, agree on the function signature and file format before coding. The shared formats are fixed in CLAUDE.md §6: the `oof.csv` and `test_fold{k}.csv` columns, and the `artifacts/<exp_id>/` layout. If you need to change one, add a DECISIONS.md entry and tell the other two.
4. **Review:** model-affecting changes (anything in `model.py`, `train.py`, `metrics.py`, `submission.py`, `ensemble.py`) go through a PR reviewed by one teammate. Docs and tests can go straight to `main`.
5. **GPU quota:** each person has their own free Kaggle account, about 30 GPU-hours a week each, so about 90 in total. Split fold runs across accounts with `--folds` (e.g. person 1 runs folds 0–2, person 2 folds 3–4, person 3 folds 5–6). Push the artifacts to a shared place (a Kaggle Dataset or a shared drive, never git) and record who ran what in STATUS.md.
6. **Sync:** a 10-minute check-in at the start of each work block. Go through STATUS.md "Waiting on the user", new DECISIONS.md entries, and the current best OOF.
7. **Submissions:** Track A owns `submission.csv`. Anyone can propose a candidate, but A validates and uploads it, and records the leaderboard score in STATUS.md.
