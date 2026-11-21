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

## D-017: Each team member builds an independent version on their own branch
- Date: 2026-11-21 · Task: all · Status: Accepted · Decided by: user
- Context: a split of the roadmap into three per-person tracks was drafted (docs/TEAM.md) and then rejected by the user.
- Decision:
  - all 3 members follow the same CLAUDE.md and ROADMAP.md, and each builds the whole roadmap on their own branch `member/<name>`;
  - each trains their own models, either locally (CPU, or their own GPU if they have one) or on free Kaggle GPUs under their own account;
  - `main` holds only shared instructions and the T0.1–T0.2 skeleton;
  - STATUS.md and DECISIONS.md on each branch track that member's version;
  - instruction changes go to `main` and are merged into member branches.
- Consequences:
  - docs/TEAM.md is removed;
  - CLAUDE.md §3, §5 and §7 are rewritten;
  - D-001 is extended to allow a member's own local GPU;
  - because everyone uses the same fixed folds (T0.4, seed 42) and the same artifact format (CLAUDE.md §6), the members' OOF and test probabilities stay directly comparable.

## D-018: Clarity/evasion inconsistency is a validator warning, not an error
- Date: 2026-11-21 · Task: T0.3 · Status: Accepted · Decided by: agent (claude-raluca)
- Context: the roadmap assumed sample_output.csv was fully consistent. In fact it has 34 rows where the clarity label differs from CLARITY_OF[evasion] (e.g. Explicit with Ambivalent), so the platform format allows inconsistent pairs.
- Decision: `validate_submission` reports inconsistency as a warning. Our own submissions stay consistent by construction (D-003). The writer derives clarity from evasion unless clarity labels are passed explicitly.
- Evidence: crosstab of sample_output.csv subtask 1 against subtask 2 (34 of 296 inconsistent).

## D-019: Folds built with StratifiedGroupKFold and pinned by fingerprint
- Date: 2026-11-21 · Task: T0.4 · Status: Accepted · Decided by: agent (claude-raluca)
- Decision: implement D-006 with `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`, stratified on `evasion_label`, with groups = exact (question, answer) pair, so the 6 duplicate pairs never straddle folds. The sha256 of the sorted (datapointID, fold) table is `fe46fbf52f31d85849ce7bf1946b1c0359f1a6eb530e65084c97d776fa27c2fb`. It is pinned in tests/test_folds.py so that every member's folds are identical (D-017).
- Evidence: fold sizes 468–469. Every class is within ±0.86 of its expected per-fold count (Clarification has 5–6 per fold).
- Consequences: a change in sklearn's splitter (e.g. a version bump) would change the folds. The test catches that, and any such change needs a new decision.

## D-020: Macro-F1 definition and OOF file format
- Date: 2026-11-21 · Task: T0.5 · Status: Accepted · Decided by: agent (claude-raluca)
- Decision:
  - macro-F1 is exactly `sklearn.metrics.f1_score(average="macro", zero_division=0)`: averaged over the union of gold and predicted classes, with a never-predicted class scoring 0. This is the standard definition and the most likely platform scorer;
  - OOF and test probability columns are `datapointID`, `p_ev_<exact evasion label>` ×9 and, optionally, `p_cl_<exact clarity label>` ×3. Rows must sum to 1 (tolerance 1e-3);
  - `evaluate_oof` also reports `grouped_clarity_f1` (decoder b from T2.4) so the decoders can be compared from the first experiment onward.
- Alternatives: averaging over all 9 classes even when absent from both gold and predictions. Rejected because it differs from sklearn and only matters for partial or toy files.

## D-021: Chunking parameters and normalization direction from EDA
- Date: 2026-11-21 · Task: T1.1 → T1.2, T2.1 · Status: Accepted · Decided by: agent (claude-raluca)
- Decision:
  - **T2.1 defaults:** `max_length` 512; question cap **192** tokens, keeping the end (the roadmap said 128); answer overlap **128**; `max_chunks` 8;
  - **T1.2 direction:** keep the comma-below diacritics (no cedilla conversion); map `–` and `‑` to `-`, `ť Ť` to `ț Ț`, nbsp and tab to a space, and collapse repeated spaces. Quote mapping is optional.
- Evidence (docs/EDA.md):
  - **question cap:** 128 truncates 199 questions, 192 truncates 59, for +16 chunks in total;
  - **overlap:** 256 needs 3,931 chunks against 3,776 for 128;
  - **cedilla:** raises ro-BERT's UNK rate from 0.04% to 9.3%;
  - **ro-BERT UNKs:** the en dash accounts for 183 of the 194 unknown characters.
- Consequences: T2.1 uses these defaults, all configurable. T1.2 implements and verifies the mapping.

## D-022: Normalization mode `clean` for every backbone
- Date: 2026-11-21 · Task: T1.2 · Status: Accepted · Decided by: agent (claude-raluca)
- Decision: all backbones (RoBERT-large/base, ro-BERT-base-cased, XLM-R-large) use `normalize(text, "clean")`. It maps dashes to `-` and `ť Ť` to `ț Ț`, collapses every whitespace run (nbsp, tab, newline) to one space, and strips. Diacritics and case are never changed. Quotes are left alone.
- Alternatives:
  - `clean+quotes`: no measurable effect, so the simpler mode was kept;
  - cedilla conversion: rejected in D-021;
  - lowercasing: RoBERT does it internally, and the cased models need the case.
- Evidence (docs/EDA.md §8):
  - ro-BERT UNK rate 0.041% → 0.001%; RoBERT and XLM-R stay at 0 UNKs;
  - fertility changes by at most 0.00001. The acceptance test therefore allows a 0.1% fertility tolerance instead of a strict ≤, because the dash→hyphen mapping can shift a token boundary;
  - newlines are collapsed: BERT-style tokenizers already treat them as whitespace, so no information reaches the model.

## D-023: Chunks built from a learned pair template instead of `return_overflowing_tokens`
- Date: 2026-11-21 · Task: T2.1 · Status: Accepted · Decided by: agent (claude-raluca)
- Context: the roadmap suggested HF pair encoding with `truncation="only_second"`, `return_overflowing_tokens=True`. That can't cap the question from its start, keep the end-aligned last window, or apply `max_chunks` selection. Also, transformers 5 fast tokenizers no longer have `build_inputs_with_special_tokens`.
- Decision: `PairTemplate` learns the prefix, middle and suffix special tokens and token-type ids from one dummy pair encoding. Chunks are assembled from token ids:
  - the question is capped at 192 tokens, keeping the end;
  - answer windows overlap by 128 tokens, and the last window ends at the end of the answer;
  - when there are more than 8 windows, the first, last and evenly spaced ones are kept.
- Evidence: on 300 real train rows, the output is identical to `tok(q, a)` for every single-chunk row, for RoBERT-large, ro-BERT-base-cased and XLM-R-large. Full-data chunk counts match docs/EDA.md.
- Consequences: the question appears in every chunk. The paper's chunk-once variant stays an ablation in T3.2(d).

## D-024: Training-loop design details
- Date: 2026-11-21 · Task: T2.3 · Status: Accepted · Decided by: agent (claude-raluca)
- Decision:
  - **Batching:** micro-batches are capped by chunk count (`batch_chunks`, default 8 at 512 tokens). Gradients accumulate until `batch_examples` (16) examples, and each micro-batch's loss is weighted by its share of examples.
  - **Ordering:** training order is shuffled per epoch with seed + fold, then sorted by length within windows of 64 to cut padding.
  - **Early-stopping score:** validation evasion macro-F1 + mapped clarity macro-F1, the two quantities we submit. Ties are not improvements. The best weights are kept in memory, and checkpoints are not written.
  - **Class weights**, when enabled through `loss.class_weight_power`, are computed from each fold's training rows only.
  - **Smoke backbone:** a deterministic WordPiece vocabulary built from the smoke rows. HF `WordPieceTrainer` made runs non-reproducible, because vocab ids changed between builds.
  - **Default config** `configs/robert-large_mh.json`: lr 2e-5 (the paper's RoBERT-large rate), batch 16, 15 epochs, patience 3, activation checkpointing on.
- Consequences: the fp16 and bf16 paths only run on GPU, so they are first exercised in T2.5/T2.6.

---

## Open questions for the user
- ~~Q-001~~ Answered: unlimited submissions, public leaderboard (D-011).
- ~~Q-002~~ Answered: no external data (D-012).
- ~~Q-003~~ Answered: Kaggle GPU (D-013).
- ~~Q-004~~ Answered: existing GitHub repo (D-014).
- ~~Q-005~~ Answered: yes, commit them (D-016).
