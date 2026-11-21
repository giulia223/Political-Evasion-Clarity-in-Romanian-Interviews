# Decision Log

Append-only. Never edit an accepted decision. To change it, add a new entry that supersedes it, and set the old entry's status to `Superseded by D-xxx`.

Entry template:
```
## D-0NN: <title>
- Date: YYYY-MM-DD · Task: <Txx or none> · Status: Proposed | Accepted | Superseded by D-xxx | Rejected
- Decided by: user | agent (<name>) with user approval | agent
- Context: why a decision was needed
- Decision: what was chosen
- Alternatives: what was rejected and why
- Evidence: numbers (OOF metric, fold count) or sources (docs/RESEARCH.md §)
- Consequences: what this changes in the code or the roadmap
```
Agents may accept decisions inside their task's scope on their own, such as hyperparameters or a code design. Anything touching the hard constraints in CLAUDE.md, external data, installs, large downloads, the submission content, or a roadmap change needs user approval. Record it as `Proposed` and ask the user.

---

## D-001: Free compute only
- Date: 2026-11-21 · Task: none · Status: Accepted · Decided by: user
- Decision: no paid services, no Hugging Face Jobs, Inference Endpoints or Providers, and no paid LLM APIs (OpenAI, Anthropic, Google and so on). Allowed: local CPU, plus free Kaggle and Colab GPUs run by the user, and public model and dataset downloads without a token.
- Consequences: GPU tasks are built and smoke-tested locally and run by the user (CLAUDE.md §5). LLM work uses open weights only.

## D-002: Submission format follows `sample_output.csv`
- Date: 2026-11-21 · Task: T0.3 · Status: Accepted · Decided by: agent, confirmed by the platform sample file
- Context: the initial brief assumed the columns `datapointID, clarity_label, evasion_label`. The platform sample file has a different format.
- Decision: columns `subtaskID,datapointID,answer`, 592 rows. The 296 Subtask 1 (clarity) rows come first, then the 296 Subtask 2 (evasion) rows, in the order of `test_data.csv`.

## D-003: One model trained on evasion; clarity derived consistently
- Date: 2026-11-21 · Task: T2.4 · Status: Accepted · Decided by: user (initial plan) and backed by research
- Decision: the 9-class evasion prediction is primary. Clarity is derived through the fixed taxonomy so every submitted pair is consistent. A multi-head model with an auxiliary clarity head is allowed, since it helps evasion by about 0.02–0.03 in the papers. The final clarity output must still be consistent with evasion, using decoder (a) or (b) from T2.4.
- Evidence: RESEARCH.md §1 (mapped beats direct for RoBERT-large) and §3 (upward derivation was the most effective approach at SemEval).

## D-004: Unweighted cross-entropy by default; class weights only as an ablation
- Date: 2026-11-21 · Task: T2.2 / T3.1 · Status: Accepted · Decided by: agent, based on evidence (this deviates from the user's initial plan, which asked for class weights)
- Context: the initial plan specified class weights for the imbalance.
- Decision: plain CE by default. Class weights, focal loss and post-hoc logit bias tuning are compared in T3.1, and whichever wins on OOF is adopted.
- Evidence: SG-UniBuc 7-fold CV results: CE .70/.45, weighted CE .66/.41, focal .69/.44 (C/E). RESEARCH.md §2.

## D-005: Pool hidden states, not logits, across chunks
- Date: 2026-11-21 · Task: T2.2 · Status: Accepted · Decided by: agent
- Context: the initial plan said "max-pooling over the chunk logits".
- Decision: element-wise max-pool of each chunk's position-0 hidden state, followed by the heads, as in the reference papers. Mean pooling and attention pooling remain as ablation options.
- Evidence: RESEARCH.md §2, pooling ablation (max .70/.45 > mean .68/.43 > first-chunk .67/.42).

## D-006: Fixed 7-fold CV with OOF macro-F1 as the selection metric
- Date: 2026-11-21 · Task: T0.4 · Status: Accepted · Decided by: agent
- Decision: 7-fold StratifiedKFold on `evasion_label`, seed 42, in `artifacts/folds.csv`, shared by all experiments. Fold models are ensembled for test predictions. This replaces the single 15% holdout from the initial plan, because a holdout has too few rare-class examples (e.g. 5 Clarification) for a stable macro-F1.
- Evidence: RESEARCH.md §1–2 (the 7-fold ensemble gives the best results; seed variance is ±0.02–0.04).

## D-007: TF-IDF baseline dropped
- Date: 2026-11-21 · Task: none · Status: Accepted · Decided by: user
- Context: a TF-IDF + LR baseline was built (val evasion macro-F1 .205, mapped clarity .459), then removed at the user's request.
- Decision: no classical baseline in the roadmap. The paper's TF-IDF numbers (RESEARCH.md §1) serve as the reference floor.

## D-008: Primary backbone prior is RoBERT-large
- Date: 2026-11-21 · Task: T2.6 · Status: Accepted (prior, confirmed or overturned by T2.6) · Decided by: agent
- Decision: `readerbench/RoBERT-large` is the default backbone. XLM-R-large is kept for the cross-lingual track and diversity in the ensemble. Switching the default needs an OOF gain of at least 0.02 evasion F1 in T2.6.
- Evidence: RESEARCH.md §1 (RoBERT-large multi-head is the best in-language encoder: .545 E / .763 M).

## D-009: Local environment is a project `.venv` on system Python 3.14
- Date: 2026-11-21 · Task: T0.1 · Status: Accepted · Decided by: user (approved the venv)
- Decision: `.venv` was created with `python3 -m venv` and has scikit-learn 1.9.1 and pandas 3.1.1. Every further package needs user approval (T0.1).

## D-010: Installed Claude skills
- Date: 2026-11-21 · Task: none · Status: Accepted · Decided by: user
- Decision: `.claude/skills/` contains `ml-model-training` (secondsky), plus `transformers` and `scikit-learn` (K-Dense-AI). All were reviewed: no hooks and no network code. huggingface/skills was not installed because it relies on paid Jobs.

## D-011: OOF is the primary metric; the public leaderboard is a secondary check
- Date: 2026-11-21 · Task: all · Status: Accepted · Decided by: user (answered Q-001) and agent
- Context: submissions are unlimited (as far as the user knows) and the leaderboard is public.
- Decision: model selection stays on OOF macro-F1 (D-006). The public leaderboard is used as a sanity check after each real submission, and its score is recorded in STATUS.md. Never choose a system on the leaderboard score alone. If the leaderboard and OOF disagree by more than 0.05, investigate (for example a format bug or distribution shift) and log the finding.
- Rationale: the test set is small (296 rows), and repeatedly tuning to the public leaderboard overfits it.

## D-012: No external training data; QEvasion joint training dropped
- Date: 2026-11-21 · Task: T3.3 · Status: Accepted · Decided by: user (answered Q-002)
- Decision: only `train_data.csv` is used for training. T3.3 is DROPPED. The idea is parked in `docs/IDEAS.md` (I-001) and must not be implemented. Public pretrained models are still allowed.

## D-013: The LLM runs on a free Kaggle GPU
- Date: 2026-11-21 · Task: T3.4 · Status: Accepted · Decided by: user (answered Q-003)
- Decision: LLM inference runs inside a Kaggle notebook on a free GPU, following the GPU workflow in CLAUDE.md §5. LLM weights are downloaded inside Kaggle, not on the local machine.

## D-014: Shared GitHub repository; data files are not committed
- Date: 2026-11-21 · Task: T0.2 · Status: Repo part Accepted; data exclusion Superseded by D-016 · Decided by: user (repo) and agent (data exclusion)
- Decision: the remote is `git@github.com:giulia223/Political-Evasion-Clarity-in-Romanian-Interviews.git`, branch `main`. The hackathon CSVs (`train_data.csv`, `test_data.csv`, `sample_output.csv`) are excluded through `.gitignore`. Each collaborator places them at the repo root. Also never committed: `.venv/`, `artifacts/`, model weights.
- Rationale: competition data may not be redistributable, and the repo could be public. The user can reverse this.

## D-015: Pinned local environment (T0.1)
- Date: 2026-11-21 · Task: T0.1 · Status: Accepted · Decided by: user approved the installs
- Decision: Python 3.14 `.venv` with torch 2.15.0+cpu (CPU index), transformers 5.23.0, sentencepiece 0.2.3, accelerate 1.16.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, pytest 9.2.0, all pinned in `requirements.txt`. On Kaggle and Colab, keep the preinstalled CUDA torch and install the rest. Note that the transformers skill was written for 5.18; check for API differences.

## D-016: Data CSVs are committed to the repo
- Date: 2026-11-21 · Task: T0.2 · Status: Accepted · Decided by: user (answered Q-005)
- Decision: `train_data.csv`, `test_data.csv` and `sample_output.csv` are committed at the repo root so all collaborators share the same files. They are read-only (CLAUDE.md §2.4), and T0.2 adds checksum tests that enforce this. This supersedes the data-exclusion part of D-014.

## D-017: Folds are built over unique Q/A pairs
- Date: 2026-11-21 · Task: T0.4 · Status: Accepted · Decided by: agent (ar/model_nitro)
- Decision: 7-fold StratifiedKFold on `evasion_label`, seed 42, over the first occurrence of each exact (question, answer) pair; the 6 duplicates inherit their twin's fold. Stratifying on evasion gives every fold every class (Clarification: 5 per fold).
- Evidence: fold sizes 467–471; every fold has every class; per-class counts are within ±1 of the expected share except where a duplicate adds one (`tests/test_folds.py`).

## D-018: Chunking repeats the question in every chunk; template inferred from the tokenizer
- Date: 2026-11-21 · Task: T2.1 · Status: Accepted · Decided by: agent
- Decision: question capped at `q_max` tokens (truncated from the start), repeated in every chunk; answer windows of `max_length - specials - len(q)` tokens overlapping by `stride`; at most `max_chunks` chunks (first, last, evenly spaced). The special-token layout is inferred by encoding a probe pair, so BERT-style (RoBERT) and RoBERTa-style (XLM-R) tokenizers share one code path. This deviates from the paper (question only in chunk 1); T3.2 (d) may ablate it.
- Evidence: EDA (docs/EDA.md): with RoBERT-large 3.6% of Q+A exceed 512 tokens; questions exceed 128 tokens in 5.6% of rows.

## D-019: Normalization mode per backbone
- Date: 2026-11-21 · Task: T1.2 · Status: Accepted · Decided by: agent
- Decision: RoBERT-large `basic` (its tokenizer lowercases but keeps diacritics: `strip_accents=False`; UNK 0% in every mode); bert-base-romanian `basic` (cedilla would raise UNK to 9.8%); XLM-R-large `cedilla` (UNK 0% in all modes, fertility 1.493 vs 1.498). Diacritics are never stripped.
- Evidence: docs/EDA.md tokenizer table (all 3,278 train rows).

## D-020: Decoder and bias are chosen on nested OOF at submission time
- Date: 2026-11-21 · Task: T2.4 · Status: Accepted · Decided by: agent
- Decision: `polero.predict --decoder auto --bias auto` scores decoders a/b/c with and without per-class log-offset tuning; tuning is always evaluated with nested CV over the 7 folds (fit on 6, score the 7th). The best nested E+C is used for test, with the bias refit on all OOF rows. Every option keeps clarity consistent with evasion.

## D-021: The platform sample file is not internally consistent
- Date: 2026-11-21 · Task: T0.3 · Status: Accepted · Decided by: agent
- Context: T0.3 assumed `sample_output.csv` is all Explicit. It is not: 278 Explicit plus other labels, and 34 rows where clarity does not match the evasion taxonomy.
- Decision: the validator accepts it with a warning (format valid). Our own submissions are always consistent.

## D-022: All training runs on free GPUs; local CPU training is not used
- Date: 2026-11-21 · Task: T2.5 · Status: Accepted · Decided by: user (asked for Kaggle/Colab only), with agent evidence
- Context: this machine (i5-12450H, 8 cores, 7.7 GB RAM, no NVIDIA GPU) is not the 16-core/30 GB machine in RESEARCH.md §6.
- Evidence: one training step of RoBERT-small (19M parameters, 20×512 tokens) took ~47 s on this CPU, even though plain matmul runs at ~280 GFLOPS; a full CV run would take days.
- Decision: everything is trained with `notebooks/kaggle_train.ipynb` (GPU T4 x2) or `notebooks/colab_train.ipynb` (T4), see docs/GPU_RUNBOOK.md. `token_budget: "auto"` sizes micro-batches from GPU memory (2,048 tokens on 16 GB for large models); precision is fp16 on T4/P100 and bf16 on Ampere+. XLM-R-large freezes its 250k-row word-embedding matrix (`freeze_embeddings`) to save ~4 GB of optimizer state on 16 GB GPUs. Interrupted runs resume per fold (`--skip-done`).

## D-023: Windows paths for the local environment
- Date: 2026-11-21 · Task: T0.1 · Status: Accepted · Decided by: agent
- Decision: on Windows the venv tools are `.venv\Scripts\python.exe` / `pip.exe` (CLAUDE.md shows the Linux `.venv/bin/...` form). The same pinned `requirements.txt` installs on Windows with Python 3.14. Data checksums are computed with CRLF normalised to LF so they match with or without git `core.autocrlf`.

---

## Open questions for the user
- ~~Q-001~~ Answered: unlimited submissions, public leaderboard (D-011).
- ~~Q-002~~ Answered: no external data (D-012).
- ~~Q-003~~ Answered: Kaggle GPU (D-013).
- ~~Q-004~~ Answered: existing GitHub repo (D-014).
- ~~Q-005~~ Answered: yes, commit them (D-016).
