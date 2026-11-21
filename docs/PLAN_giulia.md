# Giulia's version: plan and how it differs from the shared roadmap

## Constraints of this machine
Intel i7-12700H (20 threads), 16 GB RAM, **no CUDA GPU** (Intel Iris Xe). Large-encoder fine-tuning and LLM inference run on
free Kaggle GPUs (T4×2) via `notebooks/kaggle_train.ipynb` and `notebooks/kaggle_llm.ipynb`, which `git clone` this
branch (the repo is public), so no zip upload is needed. CPU work: tests, smoke runs, classical models, RoBERT-base.

## What the evidence says (docs/RESEARCH.md + my own search)
- Best published Romanian numbers: few-shot **Qwen3.6-35B-A3B E .682 / M .760**; best encoder (RoBERT-large multi-head,
  7-fold ensemble) **E .545 / M .763**. LLMs are far ahead on 9-class evasion; encoders match them on clarity.
- SemEval-2026 Task 6 (English): hierarchical LLM prompting and upward evasion→clarity derivation won. The #2 evasion
  system used an encoder's top-k labels as candidates for the LLM (ChulaNLP). LoRA-SFT of open LLMs ("Evaluators")
  reached only E .54 — no better than a good encoder — so I don't prioritise LoRA-tuning a decoder.
- Class weights hurt encoders (SG-UniBuc); post-hoc bias tuning is the cheaper way to buy macro-F1 on rare classes.

## My plan (priority order) vs. the roadmap
| # | My step | Roadmap equivalent | Change / why |
|---|---|---|---|
| 1 | Core utils, fixed folds, metrics, validator, **all decoders + nested-CV bias tuning** in one go | T0.3–T0.5, T2.4 | Same content, built together to get to a submission faster |
| 2 | **CPU baseline** (TF-IDF + Romanian refusal/ignorance/clarification lexicon + transcript label prior) | none (D-007 dropped TF-IDF) | Gives a valid submission on day 1 and a cheap, diverse stacking member |
| 3 | **LLM few-shot (Kaggle)** with self-consistency votes as probabilities | T3.4 (phase 3) | **Moved up**: biggest expected gain (+.13 E in the paper). Ollama on Kaggle = no compile, OpenAI-compatible localhost API |
| 4 | RoBERT-large multi-head chunked, 7 folds, 2 GPUs in parallel (folds split across GPUs) | T2.1–T2.7 | Same model; skipped separate screening run — XLM-R config is ready if time allows |
| 5 | **Transcript label prior** (labels of *training-fold* turns of the same interview) | T3.2 says "prefer not" | Leakage-safe (validation labels never read; tested) and mirrors test conditions (182/194 test transcripts have labelled train turns) |
| 6 | Stacking: LR on [encoder probs, LLM votes, baseline probs] with nested CV, then bias-tuned decoding | T3.5/T3.6 | Same idea; the LLM covers folds 1–6 only, so stacking is fitted on those |
| 7 | Allow **inconsistent** clarity/evasion pairs if OOF says so | D-003 forces consistency | The two subtasks are scored independently; consistency costs points if a separate clarity decoder is better. Decided by OOF |

Dropped for speed: per-task STATUS ceremony, separate EDA doc (key facts are in the training log), T2.6 3-backbone screen.
